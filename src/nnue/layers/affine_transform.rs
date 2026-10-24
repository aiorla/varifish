//! A fully connected layer (aka affine transform), for small layers: inputs
//! are processed in chunks of 4, weights are respectively transposed, and
//! accumulation happens directly to i32s.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/layers/affine_transform.h`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

#![allow(unused_unsafe)]
#![allow(clippy::needless_range_loop)]

#[cfg(use_neon)]
use std::arch::aarch64::*;
#[cfg(use_simd128)]
use std::arch::wasm32::*;
#[cfg(use_sse2)]
use std::arch::x86_64::*;

use crate::nnue::common::{
    Aligned, MAX_SIMD_WIDTH, Reader, ceil_to_multiple, write_i8s, write_i32s,
};
#[cfg(any(enable_seq_opt, use_neon))]
use crate::nnue::simd::*;

/// The layer for architectures without a sequential dot product: a row of
/// weights per output, from the inputs padded to a multiple of 16 with NEON
/// or SSE2, or the weights traversed in transpose order to skip zero inputs.
#[cfg(not(enable_seq_opt))]
#[inline(always)]
pub(super) fn affine_transform_non_ssse3<const IN: usize, const OUT: usize>(
    output: &mut [i32; OUT],
    weights: &[i8],
    biases: &[i32; OUT],
    input: &[u8],
) {
    let padded_input_dimensions = ceil_to_multiple(IN, MAX_SIMD_WIDTH);
    assert!(input.len() >= padded_input_dimensions);
    assert!(weights.len() >= OUT * padded_input_dimensions);

    #[cfg(use_neon)]
    // SAFETY: the rows and the inputs hold padded_input_dimensions bytes,
    // a multiple of 16 that NUM_CHUNKS chunks of 16 do not exceed.
    unsafe {
        let num_chunks = ceil_to_multiple(IN, 16) / 16;
        let input_vector = |j: usize| vld1_s8(input.as_ptr().add(8 * j).cast());

        for i in 0..OUT {
            let offset = i * padded_input_dimensions;

            let mut sum = vsetq_lane_s32::<0>(biases[i], vdupq_n_s32(0));
            let row = |j: usize| vld1_s8(weights.as_ptr().add(offset + 8 * j));
            for j in 0..num_chunks {
                let mut product = vmull_s8(input_vector(j * 2), row(j * 2));
                product = vmlal_s8(product, input_vector(j * 2 + 1), row(j * 2 + 1));
                sum = vpadalq_s16(sum, product);
            }
            output[i] = neon_m128_reduce_add_epi32(sum);
        }
    }

    #[cfg(use_sse2)]
    // SAFETY: as with NEON.
    unsafe {
        let num_chunks = ceil_to_multiple(IN, 16) / 16;
        let zeros = _mm_setzero_si128();
        let input_vector = |j: usize| _mm_loadu_si128(input.as_ptr().add(16 * j).cast());

        for i in 0..OUT {
            let offset = i * padded_input_dimensions;

            let mut sum_lo = _mm_cvtsi32_si128(biases[i]);
            let mut sum_hi = zeros;
            let row = |j: usize| _mm_loadu_si128(weights.as_ptr().add(offset + 16 * j).cast());
            for j in 0..num_chunks {
                let row_j = row(j);
                let input_j = input_vector(j);
                let extended_row_lo = _mm_srai_epi16::<8>(_mm_unpacklo_epi8(row_j, row_j));
                let extended_row_hi = _mm_srai_epi16::<8>(_mm_unpackhi_epi8(row_j, row_j));
                let extended_input_lo = _mm_unpacklo_epi8(input_j, zeros);
                let extended_input_hi = _mm_unpackhi_epi8(input_j, zeros);
                let product_lo = _mm_madd_epi16(extended_row_lo, extended_input_lo);
                let product_hi = _mm_madd_epi16(extended_row_hi, extended_input_hi);
                sum_lo = _mm_add_epi32(sum_lo, product_lo);
                sum_hi = _mm_add_epi32(sum_hi, product_hi);
            }
            let sum = _mm_add_epi32(sum_lo, sum_hi);
            let sum_high_64 = _mm_shuffle_epi32::<0x4E>(sum); // _MM_SHUFFLE(1, 0, 3, 2)
            let sum = _mm_add_epi32(sum, sum_high_64);
            let sum_second_32 = _mm_shufflelo_epi16::<0x4E>(sum); // _MM_SHUFFLE(1, 0, 3, 2)
            let sum = _mm_add_epi32(sum, sum_second_32);
            output[i] = _mm_cvtsi128_si32(sum);
        }
    }

    #[cfg(not(any(use_neon, use_sse2)))]
    {
        output.copy_from_slice(biases);

        // Traverse weights in transpose order to take advantage of input sparsity
        for i in 0..IN {
            if input[i] != 0 {
                let w = &weights[i..];
                let input = i32::from(input[i]);
                for j in 0..OUT {
                    output[j] =
                        output[j].wrapping_add(i32::from(w[j * padded_input_dimensions]) * input);
                }
            }
        }
    }
}

/// The layer, whose inputs come from SqrClippedReLU::propagate_pair, with
/// the 128-bit lanes it interleaves, when PAIRED_INPUT, and in their order
/// otherwise.
pub struct AffineTransform<
    const IN: usize,
    const OUT: usize,
    const WEIGHTS: usize,
    const PAIRED_INPUT: bool,
> {
    biases: Aligned<[i32; OUT]>,
    weights: Aligned<[i8; WEIGHTS]>,
}

impl<const IN: usize, const OUT: usize, const WEIGHTS: usize, const PAIRED_INPUT: bool>
    AffineTransform<IN, OUT, WEIGHTS, PAIRED_INPUT>
{
    pub const PADDED_INPUT_DIMENSIONS: usize = ceil_to_multiple(IN, MAX_SIMD_WIDTH);

    #[cfg(enable_seq_opt)]
    const NUM_CHUNKS: usize = ceil_to_multiple(IN, 8) / 4;
    #[cfg(enable_seq_opt)]
    const NUM_ACCUMS: usize = OUT / OUTPUT_SIMD_WIDTH;
    #[cfg(enable_seq_opt)]
    const NUM_REGS: usize = if cfg!(any(use_vnni, use_neon_dotprod)) {
        2 * Self::NUM_ACCUMS
    } else {
        Self::NUM_ACCUMS
    };

    const DIMENSIONS: () = {
        assert!(WEIGHTS == OUT * Self::PADDED_INPUT_DIMENSIONS);
        #[cfg(enable_seq_opt)]
        assert!(OUT == 1 || OUT.is_multiple_of(OUTPUT_SIMD_WIDTH));
        // With VNNI or dot products, the chunks go in pairs.
        #[cfg(enable_seq_opt)]
        assert!(Self::NUM_CHUNKS.is_multiple_of(2));
        #[cfg(enable_seq_opt)]
        assert!(Self::NUM_REGS <= MAX_ACCUMULATORS);
    };

    /// Hash value embedded in the evaluation file
    pub const fn get_hash_value(prev_hash: u32) -> u32 {
        let mut hash_value = 0xCC03_DAE4u32;
        hash_value = hash_value.wrapping_add(OUT as u32);
        hash_value ^= prev_hash >> 1;
        hash_value ^= prev_hash << 31;
        hash_value
    }

    const fn get_weight_index_scrambled(i: usize) -> usize {
        let mut input_index = i % Self::PADDED_INPUT_DIMENSIONS;

        // At load time, pre-permute the weights to match the per-128-bit-lane
        // interleaving that the previous layer produces via
        // SqrClippedReLU::propagate_pair(), so no shuffle is needed at runtime.
        if cfg!(use_pair_activations) && PAIRED_INPUT {
            let block = input_index / 32;
            let chunk = (input_index % 32) / 4;
            let chunk = if cfg!(use_avx512) {
                (chunk % 4) * 2 + chunk / 4
            } else {
                (chunk % 2) * 4 + chunk / 2
            };
            input_index = block * 32 + chunk * 4 + input_index % 4;
        }
        input_index / 4 * OUT * 4 + i / Self::PADDED_INPUT_DIMENSIONS * 4 + input_index % 4
    }

    const fn get_weight_index(i: usize) -> usize {
        if cfg!(enable_seq_opt) {
            Self::get_weight_index_scrambled(i)
        } else {
            i
        }
    }

    /// Read network parameters
    pub fn read_parameters(&mut self, reader: &mut Reader) -> bool {
        #[allow(clippy::let_unit_value)]
        let () = Self::DIMENSIONS;
        reader.read_i32s(&mut self.biases.0);
        if let Some(bytes) = reader.take(WEIGHTS) {
            for (i, &byte) in bytes.iter().enumerate() {
                self.weights.0[Self::get_weight_index(i)] = byte as i8;
            }
        }
        reader.good()
    }

    /// Write network parameters
    pub fn write_parameters(&self, out: &mut Vec<u8>) {
        write_i32s(out, &self.biases.0);
        let weights: Vec<i8> = (0..WEIGHTS)
            .map(|i| self.weights.0[Self::get_weight_index(i)])
            .collect();
        write_i8s(out, &weights);
    }

    /// Forward propagation of the first IN of `input`, padded to
    /// PADDED_INPUT_DIMENSIONS.
    #[inline(always)]
    pub fn propagate(&self, input: &[u8], output: &mut [i32; OUT]) {
        #[allow(clippy::let_unit_value)]
        let () = Self::DIMENSIONS;

        #[cfg(enable_seq_opt)]
        {
            assert!(input.len() >= Self::PADDED_INPUT_DIMENSIONS);
            let weights = self.weights.0.as_ptr();

            if OUT > 1 {
                const IN_VEC: usize = size_of::<InVec>();

                // SAFETY: NUM_CHUNKS chunks of 4 inputs, at most the padded
                // inputs, each with a column of OUT * 4 weights.
                unsafe {
                    let mut acc = [vec_zero_out(); MAX_ACCUMULATORS];
                    for k in 0..Self::NUM_ACCUMS {
                        acc[k] = vec_load_out(self.biases.0.as_ptr().add(k * OUTPUT_SIMD_WIDTH));
                    }

                    let input_at = |i: usize| vec_load_32(input.as_ptr().add(i * 4));
                    let column = |i: usize| weights.add(i * OUT * 4);

                    let mut i = 0;
                    #[cfg(any(use_vnni, use_neon_dotprod))]
                    {
                        while i < Self::NUM_CHUNKS {
                            let (in0, in1) = (input_at(i), input_at(i + 1));
                            let (col0, col1) = (column(i), column(i + 1));

                            for k in 0..Self::NUM_ACCUMS {
                                vec_add_dpbusd_32(
                                    &mut acc[k],
                                    in0,
                                    vec_load_in(col0.add(k * IN_VEC)),
                                );
                                vec_add_dpbusd_32(
                                    &mut acc[k + Self::NUM_ACCUMS],
                                    in1,
                                    vec_load_in(col1.add(k * IN_VEC)),
                                );
                            }
                            i += 2;
                        }

                        for k in 0..Self::NUM_ACCUMS {
                            acc[k] = vec_add_32(acc[k], acc[k + Self::NUM_ACCUMS]);
                        }
                    }
                    while i < Self::NUM_CHUNKS {
                        let in0 = input_at(i);
                        let col0 = column(i);

                        for k in 0..Self::NUM_ACCUMS {
                            vec_add_dpbusd_32(&mut acc[k], in0, vec_load_in(col0.add(k * IN_VEC)));
                        }
                        i += 1;
                    }

                    for k in 0..Self::NUM_ACCUMS {
                        vec_store_out(output.as_mut_ptr().add(k * OUTPUT_SIMD_WIDTH), acc[k]);
                    }
                }
            } else if OUT == 1 {
                // SAFETY: the padded inputs and the row of weights hold
                // NUM_CHUNKS vectors.
                #[cfg(use_avx2)]
                unsafe {
                    let input_vector = input.as_ptr().cast::<__m256i>();
                    let num_chunks = Self::PADDED_INPUT_DIMENSIONS / size_of::<__m256i>();
                    let mut sum0 = _mm256_setzero_si256();
                    let row0 = weights.cast::<__m256i>();

                    for j in 0..num_chunks {
                        let input = _mm256_loadu_si256(input_vector.add(j));
                        m256_add_dpbusd_epi32(&mut sum0, input, _mm256_loadu_si256(row0.add(j)));
                    }
                    output[0] = m256_hadd(sum0, self.biases.0[0]);
                }

                #[cfg(all(use_ssse3, not(use_avx2)))]
                unsafe {
                    let input_vector = input.as_ptr().cast::<__m128i>();
                    let num_chunks = Self::PADDED_INPUT_DIMENSIONS / size_of::<__m128i>();
                    let mut sum0 = _mm_setzero_si128();
                    let row0 = weights.cast::<__m128i>();

                    for j in 0..num_chunks {
                        let input = _mm_loadu_si128(input_vector.add(j));
                        m128_add_dpbusd_epi32(&mut sum0, input, _mm_loadu_si128(row0.add(j)));
                    }
                    output[0] = m128_hadd(sum0, self.biases.0[0]);
                }

                #[cfg(use_neon_dotprod)]
                unsafe {
                    let num_chunks = Self::PADDED_INPUT_DIMENSIONS / 16;
                    let mut sum0 = vdupq_n_s32(0);

                    for j in 0..num_chunks {
                        let input = vld1q_s8(input.as_ptr().add(16 * j).cast());
                        dotprod_m128_add_dpbusd_epi32(
                            &mut sum0,
                            input,
                            vld1q_s8(weights.add(16 * j)),
                        );
                    }
                    output[0] = neon_m128_hadd(sum0, self.biases.0[0]);
                }

                #[cfg(use_simd128)]
                unsafe {
                    let num_chunks = Self::PADDED_INPUT_DIMENSIONS / 16;
                    let mut sum0 = i32x4_splat(0);

                    for j in 0..num_chunks {
                        let input = v128_load(input.as_ptr().add(16 * j).cast());
                        simd128_add_dpbusd_epi32(
                            &mut sum0,
                            input,
                            v128_load(weights.add(16 * j).cast()),
                        );
                    }
                    output[0] = simd128_hadd(sum0, self.biases.0[0]);
                }
            }
        }

        // Use old implementation for the other architectures.
        #[cfg(not(enable_seq_opt))]
        affine_transform_non_ssse3::<IN, OUT>(output, &self.weights.0, &self.biases.0, input);
    }
}
