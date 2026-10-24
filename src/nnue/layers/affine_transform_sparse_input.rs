//! A fully connected layer (aka affine transform) with block sparse input:
//! the transformed features, mostly zero, which it reads only where the
//! feature transformer recorded non-zero chunks of four.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/layers/affine_transform_sparse_input.h`: Stockfish, a UCI chess
//! playing engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The
//! Stockfish developers (see AUTHORS), under the GNU General Public License,
//! version 3 or later.

#![allow(unused_unsafe)]
#![allow(clippy::needless_range_loop)]

#[cfg(all(use_simd128, not(use_relaxed_simd)))]
use std::arch::wasm32::*;
#[cfg(all(use_sse2, not(use_ssse3)))]
use std::arch::x86_64::*;

use crate::nnue::architecture::L1;
use crate::nnue::common::{
    Aligned, MAX_SIMD_WIDTH, Reader, ceil_to_multiple, write_i8s, write_i32s,
};
use crate::nnue::nnz_helper::NnzInfo;
#[cfg(sparse_input)]
use crate::nnue::simd::*;

#[cfg(not(sparse_input))]
use super::affine_transform::affine_transform_non_ssse3;

pub struct AffineTransformSparseInput<const IN: usize, const OUT: usize, const WEIGHTS: usize> {
    biases: Aligned<[i32; OUT]>,
    weights: Aligned<[i8; WEIGHTS]>,
}

impl<const IN: usize, const OUT: usize, const WEIGHTS: usize>
    AffineTransformSparseInput<IN, OUT, WEIGHTS>
{
    pub const PADDED_INPUT_DIMENSIONS: usize = ceil_to_multiple(IN, MAX_SIMD_WIDTH);

    /// The inputs read at a time, in chunks of 4 where the layer reads only
    /// the non-zero chunks, one at a time otherwise, when the weights stay
    /// in their order.
    #[cfg(sparse_input)]
    const CHUNK_SIZE: usize = 4;

    #[cfg(sparse_input)]
    const NUM_ACCUMS: usize = OUT / OUTPUT_SIMD_WIDTH;
    // If we're using high-latency dot product instructions, split the
    // accumulators into separate dependency chains and merge at the end
    #[cfg(sparse_input)]
    const NUM_REGS: usize = if cfg!(any(all(use_vnni, use_avx512), use_neon_dotprod)) {
        3 * Self::NUM_ACCUMS
    } else {
        Self::NUM_ACCUMS
    };

    const DIMENSIONS: () = {
        assert!(
            OUT.is_multiple_of(16),
            "Only implemented for OutputDimensions divisible by 16."
        );
        assert!(IN == L1, "The non-zero chunks are recorded for L1 inputs.");
        assert!(WEIGHTS == OUT * Self::PADDED_INPUT_DIMENSIONS);
        #[cfg(sparse_input)]
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

    #[cfg(sparse_input)]
    const fn get_weight_index_scrambled(i: usize) -> usize {
        let index = (i / Self::CHUNK_SIZE) % (Self::PADDED_INPUT_DIMENSIONS / Self::CHUNK_SIZE)
            * OUT
            * Self::CHUNK_SIZE
            + i / Self::PADDED_INPUT_DIMENSIONS * Self::CHUNK_SIZE
            + i % Self::CHUNK_SIZE;
        // Without a dot product of bytes, each 16 weights, of four outputs
        // for a chunk's four inputs, hold the first two inputs' weights of
        // the four outputs, then the last two inputs'.
        #[cfg(widening_dot)]
        let index = {
            let in_vector = index % 16;
            let (output, input) = (in_vector / 4, in_vector % 4);
            index - in_vector + input / 2 * 8 + output * 2 + input % 2
        };
        index
    }

    const fn get_weight_index(i: usize) -> usize {
        #[cfg(sparse_input)]
        return Self::get_weight_index_scrambled(i);
        #[cfg(not(sparse_input))]
        i
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

    /// Forward propagation
    #[inline(always)]
    pub fn propagate(&self, input: &[u8; IN], output: &mut [i32; OUT], nnz_info: &NnzInfo) {
        #[allow(clippy::let_unit_value)]
        let () = Self::DIMENSIONS;

        #[cfg(sparse_input)]
        // SAFETY: the loads read whole vectors of biases and weights, the
        // weights of a chunk of the inputs, and four inputs, at chunks below
        // L1 / 4, which is IN / 4; the stores write the OUT outputs.
        unsafe {
            const IN_VEC: usize = size_of::<InVec>();

            let mut acc = [vec_zero_out(); MAX_ACCUMULATORS];
            for k in 0..Self::NUM_ACCUMS {
                acc[k] = vec_load_out(self.biases.0.as_ptr().add(k * OUTPUT_SIMD_WIDTH));
            }

            // the columns of weights, OUT for each chunk of four inputs
            let weights = self.weights.0.as_ptr();
            let column = |i: usize| weights.add(i * OUT * Self::CHUNK_SIZE);

            #[cfg(use_avx512)]
            {
                let nnz = nnz_info.indices();
                let mut i = 0;

                #[cfg(use_vnni)]
                {
                    while i + 2 < nnz.len() {
                        let i0 = usize::from(nnz[i]);
                        let i1 = usize::from(nnz[i + 1]);
                        let i2 = usize::from(nnz[i + 2]);
                        i += 3;
                        let in0 = vec_load_32(input.as_ptr().add(i0 * 4));
                        let in1 = vec_load_32(input.as_ptr().add(i1 * 4));
                        let in2 = vec_load_32(input.as_ptr().add(i2 * 4));
                        let (col0, col1, col2) = (column(i0), column(i1), column(i2));
                        for k in 0..Self::NUM_ACCUMS {
                            vec_add_dpbusd_32(&mut acc[k], in0, vec_load_in(col0.add(k * IN_VEC)));
                            vec_add_dpbusd_32(
                                &mut acc[k + Self::NUM_ACCUMS],
                                in1,
                                vec_load_in(col1.add(k * IN_VEC)),
                            );
                            vec_add_dpbusd_32(
                                &mut acc[k + 2 * Self::NUM_ACCUMS],
                                in2,
                                vec_load_in(col2.add(k * IN_VEC)),
                            );
                        }
                    }

                    for k in 0..Self::NUM_ACCUMS {
                        acc[k] = vec_add_32(
                            vec_add_32(acc[k], acc[k + Self::NUM_ACCUMS]),
                            acc[k + 2 * Self::NUM_ACCUMS],
                        );
                    }
                }

                while i < nnz.len() {
                    let i0 = usize::from(nnz[i]);
                    i += 1;
                    let in0 = vec_load_32(input.as_ptr().add(i0 * 4));
                    let col0 = column(i0);
                    for k in 0..Self::NUM_ACCUMS {
                        vec_add_dpbusd_32(&mut acc[k], in0, vec_load_in(col0.add(k * IN_VEC)));
                    }
                }
            }

            #[cfg(not(use_avx512))]
            {
                const { assert!(L1.is_multiple_of(256)) };

                for k in 0..IN / 256 {
                    let bitset = &nnz_info.bitset()[k * 8..k * 8 + 8];
                    let mut bits = u64::from_le_bytes(bitset.try_into().unwrap());
                    let base = k * 64;

                    let base_addr = input.as_ptr().add(base * 4);
                    let input_at = |i: usize| vec_load_32(base_addr.add(i * 4));
                    let pop_lsb = |bits: &mut u64| {
                        let i = crate::bitboard::lowest_bit(*bits) as usize;
                        *bits &= *bits - 1;
                        i
                    };

                    #[cfg(use_neon_dotprod)]
                    while bits != 0 {
                        let i0 = pop_lsb(&mut bits);
                        if bits == 0 {
                            let in0 = input_at(i0);
                            let col0 = column(base + i0);
                            for l in 0..Self::NUM_ACCUMS {
                                vec_add_dpbusd_32(
                                    &mut acc[l],
                                    in0,
                                    vec_load_in(col0.add(l * IN_VEC)),
                                );
                            }
                            break;
                        }

                        let i1 = pop_lsb(&mut bits);
                        if bits == 0 {
                            let (in0, in1) = (input_at(i0), input_at(i1));
                            let (col0, col1) = (column(base + i0), column(base + i1));
                            for l in 0..Self::NUM_ACCUMS {
                                vec_add_dpbusd_32(
                                    &mut acc[l],
                                    in0,
                                    vec_load_in(col0.add(l * IN_VEC)),
                                );
                                vec_add_dpbusd_32(
                                    &mut acc[l + Self::NUM_ACCUMS],
                                    in1,
                                    vec_load_in(col1.add(l * IN_VEC)),
                                );
                            }
                            break;
                        }

                        let i2 = pop_lsb(&mut bits);
                        let (in0, in1, in2) = (input_at(i0), input_at(i1), input_at(i2));
                        let (col0, col1, col2) =
                            (column(base + i0), column(base + i1), column(base + i2));
                        for l in 0..Self::NUM_ACCUMS {
                            vec_add_dpbusd_32(&mut acc[l], in0, vec_load_in(col0.add(l * IN_VEC)));
                            vec_add_dpbusd_32(
                                &mut acc[l + Self::NUM_ACCUMS],
                                in1,
                                vec_load_in(col1.add(l * IN_VEC)),
                            );
                            vec_add_dpbusd_32(
                                &mut acc[l + 2 * Self::NUM_ACCUMS],
                                in2,
                                vec_load_in(col2.add(l * IN_VEC)),
                            );
                        }
                    }

                    // Without a dot product of bytes, the first two inputs,
                    // widened and repeated across a vector, take the dot
                    // product of pairs with the first 8 weights of four
                    // outputs, and the last two inputs with the next 8.
                    #[cfg(all(use_simd128, not(use_relaxed_simd)))]
                    while bits != 0 {
                        let i = pop_lsb(&mut bits);
                        let inputs = u16x8_extend_low_u8x16(input_at(i));
                        let first = i32x4_shuffle::<0, 0, 0, 0>(inputs, inputs);
                        let last = i32x4_shuffle::<1, 1, 1, 1>(inputs, inputs);
                        let col = column(base + i);
                        for l in 0..Self::NUM_ACCUMS {
                            let weights = col.add(l * IN_VEC);
                            let sum = i32x4_add(
                                i32x4_dot_i16x8(first, i16x8_load_extend_i8x8(weights)),
                                i32x4_dot_i16x8(last, i16x8_load_extend_i8x8(weights.add(8))),
                            );
                            acc[l] = i32x4_add(acc[l], sum);
                        }
                    }

                    // The same with SSE2 alone, whose weights widen by
                    // interleaving each with a byte of its sign.
                    #[cfg(all(use_sse2, not(use_ssse3)))]
                    while bits != 0 {
                        let i = pop_lsb(&mut bits);
                        let zero = _mm_setzero_si128();
                        let inputs = _mm_unpacklo_epi8(input_at(i), zero);
                        let first = _mm_shuffle_epi32::<0x00>(inputs);
                        let last = _mm_shuffle_epi32::<0x55>(inputs);
                        let col = column(base + i);
                        for l in 0..Self::NUM_ACCUMS {
                            let weights = vec_load_in(col.add(l * IN_VEC));
                            let sign = _mm_cmpgt_epi8(zero, weights);
                            let sum = _mm_add_epi32(
                                _mm_madd_epi16(first, _mm_unpacklo_epi8(weights, sign)),
                                _mm_madd_epi16(last, _mm_unpackhi_epi8(weights, sign)),
                            );
                            acc[l] = _mm_add_epi32(acc[l], sum);
                        }
                    }

                    #[cfg(not(any(use_neon_dotprod, widening_dot)))]
                    while bits != 0 {
                        let i = pop_lsb(&mut bits);
                        let in0 = input_at(i);
                        let col = column(base + i);
                        for l in 0..Self::NUM_ACCUMS {
                            vec_add_dpbusd_32(&mut acc[l], in0, vec_load_in(col.add(l * IN_VEC)));
                        }
                    }
                }

                #[cfg(use_neon_dotprod)]
                for l in 0..Self::NUM_ACCUMS {
                    acc[l] = vec_add_32(
                        vec_add_32(acc[l], acc[l + Self::NUM_ACCUMS]),
                        acc[l + 2 * Self::NUM_ACCUMS],
                    );
                }
            }

            for k in 0..Self::NUM_ACCUMS {
                vec_store_out(output.as_mut_ptr().add(k * OUTPUT_SIMD_WIDTH), acc[k]);
            }
        }

        // Use dense implementation for the other architectures.
        #[cfg(not(sparse_input))]
        {
            let _ = nnz_info;
            affine_transform_non_ssse3::<IN, OUT>(output, &self.weights.0, &self.biases.0, input);
        }
    }
}
