//! The feature transformer: the first layer, whose sums for the active
//! features the accumulators keep up to date, and its conversion of them
//! into the input of the layer stacks.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/nnue_feature_transformer.h`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

#![allow(unused_unsafe)]

#[cfg(use_neon)]
use std::arch::aarch64::*;
#[cfg(use_simd128)]
use std::arch::wasm32::*;
use std::mem::MaybeUninit;

use super::accumulator::{AccumulatorCaches, AccumulatorStack};
use super::architecture::{L1, PairFeatureSet, PsqFeatureSet, ThreatFeatureSet};
use super::common::{Aligned, FT_MAX_VAL, Reader, write_i8s, write_i16s};
use super::nnz_helper::NnzInfo;
#[cfg(vector)]
use super::simd::*;
use crate::position::Position;

/// Number of output dimensions for one side
const HALF_DIMENSIONS: usize = L1;

pub const THREAT_INPUT_DIMENSIONS: usize = ThreatFeatureSet::DIMENSIONS as usize;
pub const PAIR_INPUT_DIMENSIONS: usize = PairFeatureSet::DIMENSIONS as usize;
pub const PSQ_DIMENSIONS: usize = PsqFeatureSet::DIMENSIONS;
pub const THREAT_AND_PP_DIMENSIONS: usize = THREAT_INPUT_DIMENSIONS + PAIR_INPUT_DIMENSIONS;
pub const INPUT_DIMENSIONS: usize = PSQ_DIMENSIONS + THREAT_AND_PP_DIMENSIONS;
pub const OUTPUT_DIMENSIONS: usize = HALF_DIMENSIONS;
const THREAT_WEIGHT_SIZE: usize = THREAT_INPUT_DIMENSIONS * HALF_DIMENSIONS;
const THREAT_AND_PP_WEIGHT_SIZE: usize = THREAT_AND_PP_DIMENSIONS * HALF_DIMENSIONS;

/// The order by which the eight 8-value blocks of 64 values are permuted, so
/// that packus on adjacent vectors of 16-bit integers loaded from them gives
/// the values in their order.
pub(super) const PACKUS_EPI16_ORDER: [usize; 8] = if cfg!(use_avx512) {
    // _mm512_packus_epi16 after permutation:
    // |   0   |   2   |   4   |   6   | // Vector 0
    // |   1   |   3   |   5   |   7   | // Vector 1
    // | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | // Packed Result
    [0, 2, 4, 6, 1, 3, 5, 7]
} else if cfg!(use_avx2) {
    // _mm256_packus_epi16 after permutation:
    // |   0   |   2   |  |   4   |   6   | // Vector 0, 2
    // |   1   |   3   |  |   5   |   7   | // Vector 1, 3
    // | 0 | 1 | 2 | 3 |  | 4 | 5 | 6 | 7 | // Packed Result
    [0, 2, 1, 3, 4, 6, 5, 7]
} else {
    [0, 1, 2, 3, 4, 5, 6, 7]
};

/// Returns the inverse of a permutation
const fn invert_permutation<const N: usize>(order: &[usize; N]) -> [usize; N] {
    let mut inverse = [0; N];
    let mut i = 0;
    while i < N {
        inverse[order[i]] = i;
        i += 1;
    }
    inverse
}

pub(super) const INVERSE_PACKUS_EPI16_ORDER: [usize; 8] = invert_permutation(&PACKUS_EPI16_ORDER);

/// The int8 weight rows are consumed one full vector register
/// (size_of::<VecT>() bytes) at a time: unpack{lo,hi}_epi8 interleave the
/// bytes of two rows and maddubs against a vector of ones sums each pair into
/// an int16 lane. Both unpacks work within 128-bit lanes, so the int16
/// results of a chunk come out with its 8-byte blocks in the order {0, 2, 4,
/// 6, ...} (lo) and {1, 3, 5, 7, ...} (hi). Storing the 8-byte blocks of
/// every chunk in this order at load time makes the two output registers
/// hold consecutive accumulator lanes, so the accumulator layout is
/// unchanged.
#[cfg(use_paired_widening_add)]
pub(super) const WIDENING_PAIR_ORDER: [usize; size_of::<VecT>() / 8] = {
    const NUM_BLOCKS: usize = size_of::<VecT>() / 8;
    let mut order = [0; NUM_BLOCKS];
    let mut i = 0;
    while i < NUM_BLOCKS {
        order[i] = if i % 2 == 0 {
            i / 2
        } else {
            NUM_BLOCKS / 2 + i / 2
        };
        i += 1;
    }
    order
};

#[cfg(use_paired_widening_add)]
pub(super) const INVERSE_WIDENING_PAIR_ORDER: [usize; size_of::<VecT>() / 8] =
    invert_permutation(&WIDENING_PAIR_ORDER);

/// Divides `data` into chunks of N blocks of 8 values, and permutes the
/// blocks of each by `order`: 16 bytes of the i16 biases, or 8 of the i8
/// weights.
pub(super) fn permute<T: Copy, const N: usize>(data: &mut [T], order: &[usize; N]) {
    const BLOCK_SIZE: usize = 8;
    let (blocks, rest) = data.as_chunks_mut::<BLOCK_SIZE>();
    assert!(rest.is_empty());
    let (chunks, rest) = blocks.as_chunks_mut::<N>();
    assert!(rest.is_empty());

    for chunk in chunks {
        let buffer = *chunk;
        for (block, &o) in chunk.iter_mut().zip(order) {
            *block = buffer[o];
        }
    }
}

/// The weights of `weights`, in the order of the network's file.
fn unpermuted_weights(weights: &[i8]) -> Vec<i8> {
    let mut weights = weights.to_vec();
    #[cfg(use_paired_widening_add)]
    permute(&mut weights, &INVERSE_WIDENING_PAIR_ORDER);
    permute(&mut weights, &INVERSE_PACKUS_EPI16_ORDER);
    weights
}

/// Input feature converter
pub struct FeatureTransformer {
    pub biases: Aligned<[i16; HALF_DIMENSIONS]>,
    pub weights: Aligned<[i8; HALF_DIMENSIONS * PSQ_DIMENSIONS]>,

    /// Threats and pawn-pair features are concatenated into one array to
    /// allow for a single index to address either. The first pawn-pair
    /// feature is at index ThreatFeatureSet::DIMENSIONS.
    pub threat_and_pp_weights: Aligned<[i8; THREAT_AND_PP_WEIGHT_SIZE]>,
}

const _: () = assert!(PairFeatureSet::INDEX_BASE == ThreatFeatureSet::DIMENSIONS);

const fn combine_hash(hashes: &[u32]) -> u32 {
    let mut hash = 0u32;
    let mut i = 0;
    while i < hashes.len() {
        hash = hash.rotate_left(1);
        hash ^= hashes[i];
        i += 1;
    }
    hash
}

impl FeatureTransformer {
    /// Hash value embedded in the evaluation file
    pub const fn get_hash_value() -> u32 {
        combine_hash(&[
            ThreatFeatureSet::HASH_VALUE,
            PairFeatureSet::HASH_VALUE,
            PsqFeatureSet::HASH_VALUE,
        ]) ^ (OUTPUT_DIMENSIONS * 2) as u32
    }

    fn permute_weights(&mut self) {
        permute(&mut self.biases.0, &PACKUS_EPI16_ORDER);
        permute(&mut self.weights.0, &PACKUS_EPI16_ORDER);
        permute(&mut self.threat_and_pp_weights.0, &PACKUS_EPI16_ORDER);

        #[cfg(use_paired_widening_add)]
        {
            permute(&mut self.weights.0, &WIDENING_PAIR_ORDER);
            permute(&mut self.threat_and_pp_weights.0, &WIDENING_PAIR_ORDER);
        }
    }

    /// Read network parameters
    pub fn read_parameters(&mut self, reader: &mut Reader) -> bool {
        reader.read_i16s(&mut self.biases.0);

        let (threat_weights, pair_weights) = self
            .threat_and_pp_weights
            .0
            .split_at_mut(THREAT_WEIGHT_SIZE);
        reader.read_i8s(threat_weights);
        reader.read_i8s(pair_weights);

        reader.read_i8s(&mut self.weights.0);

        self.permute_weights();

        reader.good()
    }

    /// Write network parameters
    pub fn write_parameters(&self, out: &mut Vec<u8>) {
        let mut biases = self.biases.0;
        permute(&mut biases, &INVERSE_PACKUS_EPI16_ORDER);
        write_i16s(out, &biases);

        let threat_and_pp_weights = unpermuted_weights(&self.threat_and_pp_weights.0);
        write_i8s(out, &threat_and_pp_weights);
        drop(threat_and_pp_weights);

        write_i8s(out, &unpermuted_weights(&self.weights.0));
    }

    /// Convert input features: writes `output` and returns it.
    #[inline(always)]
    pub fn transform<'a>(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
        output: &'a mut MaybeUninit<Aligned<[u8; OUTPUT_DIMENSIONS]>>,
        nnz_info: &mut NnzInfo,
    ) -> &'a [u8; OUTPUT_DIMENSIONS] {
        accumulator_stack.evaluate(pos, self, cache);
        let accumulator_state = accumulator_stack.latest();

        let perspectives = [pos.side_to_move(), !pos.side_to_move()];
        let accumulation = &accumulator_state.accumulation;

        for (p, perspective) in perspectives.into_iter().enumerate() {
            Self::transform_perspective(&accumulation[perspective.idx()], output, p, nnz_info);
        }

        // SAFETY: each perspective wrote its half.
        unsafe { &output.assume_init_ref().0 }
    }

    /// Writes the half of `output` for perspective `p`: the products of the
    /// clipped pairs of the perspective's accumulation, divided by 512.
    #[inline(always)]
    fn transform_perspective(
        accumulation: &[i16; HALF_DIMENSIONS],
        output: &mut MaybeUninit<Aligned<[u8; OUTPUT_DIMENSIONS]>>,
        perspective: usize,
        nnz_info: &mut NnzInfo,
    ) {
        let offset = (HALF_DIMENSIONS / 2) * perspective;

        #[cfg(vector)]
        // SAFETY: the loads read vectors of the accumulation, the stores
        // write vectors of the perspective's half of the output.
        unsafe {
            let mut cursor = nnz_info.make_cursor(perspective);

            const OUTPUT_CHUNK_SIZE: usize = MAX_CHUNK_SIZE;
            const { assert!((HALF_DIMENSIONS / 2).is_multiple_of(OUTPUT_CHUNK_SIZE)) };
            const NUM_OUTPUT_CHUNKS: usize = HALF_DIMENSIONS / 2 / OUTPUT_CHUNK_SIZE;
            // The chunks go in pairs.
            const { assert!(NUM_OUTPUT_CHUNKS.is_multiple_of(2)) };

            #[cfg(not(any(use_neon, use_simd128)))]
            let (zero, ft_max) = (vec_zero(), vec_set_16(FT_MAX_VAL));
            const { assert!(FT_MAX_VAL == 255) };

            let in0 = |i: usize| vec_load(accumulation.as_ptr().add(i * LANES));
            let in1 =
                |i: usize| vec_load(accumulation.as_ptr().add(HALF_DIMENSIONS / 2 + i * LANES));
            let out = output.as_mut_ptr().cast::<u8>().add(offset);

            // Per the NNUE architecture, here we want to multiply pairs of
            // clipped elements and divide the product by 512. To do this, we
            // can naively perform min/max operation to clip each of the four
            // int16 vectors, mullo pairs together, then pack them into one
            // int8 vector. However, there exists a faster way.
            //
            // The idea here is to use the implicit clipping from packus to
            // save us two vec_max_16 instructions. This clipping works due
            // to the fact that any int16 integer below zero will be zeroed
            // on packus.
            //
            // Consider the case where the second element is negative. If we
            // do standard clipping, that element will be zero, which means
            // our pairwise product is zero. If we perform packus and remove
            // the lower-side clip for the second element, then our product
            // before packus will be negative, and is zeroed on pack. The two
            // operations produce equivalent results, but the second one
            // (using packus) saves one max operation per pair.
            //
            // But here we run into a problem: mullo does not preserve the
            // sign of the multiplication. We can get around this by doing
            // mulhi, which keeps the sign. But that requires an additional
            // tweak.
            //
            // mulhi cuts off the last 16 bits of the resulting product,
            // which is the same as performing a rightward shift of 16 bits.
            // We can use this to our advantage. Recall that we want to
            // divide the final product by 512, which is equivalent to a
            // 9-bit right shift. Intuitively, if we shift the clipped value
            // left by 7, and perform mulhi, which shifts the product right
            // by 16 bits, then we will net a right shift of 9 bits.
            let mut j = 0;
            while j < NUM_OUTPUT_CHUNKS {
                let mut packed = [in0(0); 2];
                for (k, packed) in packed.iter_mut().enumerate() {
                    let i = (j + k) * 2;

                    let acc0a = in0(i);
                    let acc0b = in0(i + 1);
                    let acc1a = in1(i);
                    let acc1b = in1(i + 1);

                    #[cfg(use_neon)]
                    let result = {
                        let mul0 = vmull_u8(vqmovun_s16(acc0a), vqmovun_s16(acc1a));
                        let mul1 = vmull_u8(vqmovun_s16(acc0b), vqmovun_s16(acc1b));

                        let uzp = vuzpq_u8(vreinterpretq_u8_u16(mul0), vreinterpretq_u8_u16(mul1));
                        let pab = vshrq_n_u8::<1>(uzp.1);
                        vreinterpretq_s16_u8(pab)
                    };

                    // WebAssembly has no mulhi, whose emulation multiplies
                    // in 32 bits: the clipped values multiply as bytes
                    // instead, as with NEON.
                    #[cfg(use_simd128)]
                    let result = {
                        let mul0 = vec_packus_16(acc0a, acc0b);
                        let mul1 = vec_packus_16(acc1a, acc1b);

                        let low = u16x8_extmul_low_u8x16(mul0, mul1);
                        let high = u16x8_extmul_high_u8x16(mul0, mul1);

                        // As NEON's vuzp2: the high byte of each product
                        #[rustfmt::skip]
                        let merged = i8x16_shuffle::<
                            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                        >(low, high);
                        u8x16_shr(merged, 1)
                    };

                    #[cfg(not(any(use_neon, use_simd128)))]
                    let result = {
                        let sum0a = vec_slli_16::<7>(vec_max_16(vec_min_16(acc0a, ft_max), zero));
                        let sum0b = vec_slli_16::<7>(vec_max_16(vec_min_16(acc0b, ft_max), zero));
                        let sum1a = vec_min_16(acc1a, ft_max);
                        let sum1b = vec_min_16(acc1b, ft_max);

                        let pa = vec_mulhi_16(sum0a, sum1a);
                        let pb = vec_mulhi_16(sum0b, sum1b);

                        vec_packus_16(pa, pb)
                    };

                    *packed = result;
                    vec_store(out.add((j + k) * OUTPUT_CHUNK_SIZE).cast(), result);
                }

                cursor.record2(packed[0], packed[1]);
                j += 2;
            }
        }

        #[cfg(not(vector))]
        {
            let _ = nnz_info;
            let output = output.as_mut_ptr().cast::<u8>();
            for j in 0..HALF_DIMENSIONS / 2 {
                let sum0 = accumulation[j].clamp(0, FT_MAX_VAL);
                let sum1 = accumulation[j + HALF_DIMENSIONS / 2].clamp(0, FT_MAX_VAL);

                // SAFETY: the perspective's half of the output.
                unsafe {
                    output
                        .add(offset + j)
                        .write(((sum0 as u32 * sum1 as u32) / 512) as u8);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permuting_and_its_inverse_restore_the_order() {
        let original: Vec<i16> = (0..256).collect();
        let mut values = original.clone();
        permute(&mut values, &PACKUS_EPI16_ORDER);
        permute(&mut values, &INVERSE_PACKUS_EPI16_ORDER);
        assert_eq!(values, original);
    }

    #[cfg(use_paired_widening_add)]
    #[test]
    fn weights_widened_in_pairs_come_out_in_order() {
        // Unpacking a vector's bytes works within 128-bit lanes: the low
        // halves of the lanes, then the high halves.
        let unpacked = |chunk: &[i8]| -> Vec<i8> {
            let lanes = chunk.chunks(16);
            let lo = lanes.clone().flat_map(|lane| &lane[..8]);
            let hi = lanes.flat_map(|lane| &lane[8..]);
            lo.chain(hi).copied().collect()
        };
        let original: Vec<i8> = (0..=127).collect();
        let mut weights = original.clone();
        permute(&mut weights, &WIDENING_PAIR_ORDER);
        let widened: Vec<i8> = weights
            .chunks(size_of::<VecT>())
            .flat_map(unpacked)
            .collect();
        assert_eq!(widened, original);

        permute(&mut weights, &INVERSE_WIDENING_PAIR_ORDER);
        assert_eq!(weights, original);
    }

    #[test]
    fn the_hash_matches_the_default_network() {
        assert_eq!(FeatureTransformer::get_hash_value(), 0x86fe_fd2f);
    }
}
