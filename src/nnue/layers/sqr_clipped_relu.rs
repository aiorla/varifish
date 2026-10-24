//! The squared clipped ReLU activation: the inputs squared, scaled down by
//! 2^(2W + 7), and clipped to 127.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/layers/sqr_clipped_relu.h`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

#![allow(unused_unsafe)]

#[cfg(use_neon)]
use std::arch::aarch64::*;
#[cfg(use_sse2)]
use std::arch::x86_64::*;

#[cfg(all(use_sse2, not(use_avx2)))]
use crate::nnue::simd::mm_packs_epi16;
#[cfg(use_avx512)]
use crate::nnue::simd::mm512_packs_epi32;
#[cfg(all(use_avx2, not(use_avx512)))]
use crate::nnue::simd::{mm256_packs_epi16, mm256_packs_epi32};

/// Without parameters, and left out of the network's hash value, as the
/// trainer leaves it out.
pub struct SqrClippedReLU<const IN: usize, const W: i32>;

impl<const IN: usize, const W: i32> SqrClippedReLU<IN, W> {
    const W_IN_RANGE: () = assert!(
        W >= 5 && W <= 8,
        "SqrClippedReLU only supports W between 5 and 8"
    );

    // After squaring we need to shift by W * 2 + 7. MulHi strips the lower
    // 16 bits (i.e. shift by 16) so we need to shift out the remaining.
    #[cfg(any(use_sse2, use_neon))]
    const SIMD_SHIFT_AMOUNT: i32 = W * 2 + 7 - 16;

    /// Produces the squared and the clipped activations together, sharing
    /// the loads and the narrowing from 32 to 16 bits. The packs leave each
    /// 128-bit lane's chunks interleaved, which the next layer's weights
    /// were permuted at load to match.
    #[cfg(use_pair_activations)]
    #[inline(always)]
    pub fn propagate_pair(
        &self,
        input: &[i32; IN],
        squared: &mut [u8; IN],
        clipped: &mut [u8; IN],
    ) {
        #[allow(clippy::let_unit_value)]
        let () = Self::W_IN_RANGE;
        const {
            assert!(
                IN.is_multiple_of(32),
                "propagate_pair() needs a tail path for other sizes"
            )
        };
        let num_chunks = IN / 32;

        // SAFETY: each chunk reads 32 inputs and writes 32 of each output.
        unsafe {
            // Shifts by a constant count compile to shifts by an immediate.
            let shift = _mm_cvtsi32_si128(Self::SIMD_SHIFT_AMOUNT);
            let clip_shift = _mm_cvtsi32_si128(W);

            #[cfg(use_avx512)]
            {
                let zero = _mm512_setzero_si512();
                for i in 0..num_chunks {
                    let at = input.as_ptr().add(32 * i);
                    let words = mm512_packs_epi32(
                        _mm512_loadu_si512(at.cast()),
                        _mm512_loadu_si512(at.add(16).cast()),
                    );
                    let sqr_words = _mm512_srl_epi16(_mm512_mulhi_epi16(words, words), shift);
                    _mm256_storeu_si256(
                        squared.as_mut_ptr().add(32 * i).cast(),
                        _mm512_cvtsepi16_epi8(sqr_words),
                    );

                    let clip_words = _mm512_srl_epi16(_mm512_max_epi16(words, zero), clip_shift);
                    _mm256_storeu_si256(
                        clipped.as_mut_ptr().add(32 * i).cast(),
                        _mm512_cvtsepi16_epi8(clip_words),
                    );
                }
            }

            #[cfg(not(use_avx512))]
            {
                let zero = _mm256_setzero_si256();
                for i in 0..num_chunks {
                    let at =
                        |j: usize| _mm256_loadu_si256(input.as_ptr().add(32 * i + 8 * j).cast());
                    let words0 = mm256_packs_epi32(at(0), at(1));
                    let words1 = mm256_packs_epi32(at(2), at(3));

                    let sqr0 = _mm256_srl_epi16(_mm256_mulhi_epi16(words0, words0), shift);
                    let sqr1 = _mm256_srl_epi16(_mm256_mulhi_epi16(words1, words1), shift);
                    _mm256_storeu_si256(
                        squared.as_mut_ptr().add(32 * i).cast(),
                        mm256_packs_epi16(sqr0, sqr1),
                    );

                    let clip0 = _mm256_srl_epi16(_mm256_max_epi16(words0, zero), clip_shift);
                    let clip1 = _mm256_srl_epi16(_mm256_max_epi16(words1, zero), clip_shift);
                    _mm256_storeu_si256(
                        clipped.as_mut_ptr().add(32 * i).cast(),
                        mm256_packs_epi16(clip0, clip1),
                    );
                }
            }
        }
    }

    /// Forward propagation
    #[cfg(not(use_pair_activations))]
    #[inline(always)]
    pub fn propagate(&self, input: &[i32; IN], output: &mut [u8; IN]) {
        #[allow(clippy::let_unit_value)]
        let () = Self::W_IN_RANGE;

        #[cfg(use_neon)]
        // SAFETY: the chunks of 16 are within IN.
        let start = unsafe {
            let num_chunks = IN / 16;
            // Neon needs to shift by one more since the used simd instruction
            // does `Saturating Doubling Multiply High` (doubling before shift
            // by 16). Shifts by a constant compile to shifts by an immediate.
            let shift = vdupq_n_s16(-(Self::SIMD_SHIFT_AMOUNT + 1) as i16);
            for i in 0..num_chunks {
                let at = |j: usize| vld1q_s32(input.as_ptr().add(16 * i + 4 * j));
                let words0 = vcombine_s16(vqmovn_s32(at(0)), vqmovn_s32(at(1)));
                let words1 = vcombine_s16(vqmovn_s32(at(2)), vqmovn_s32(at(3)));
                let r0 = vshlq_s16(vqdmulhq_s16(words0, words0), shift);
                let r1 = vshlq_s16(vqdmulhq_s16(words1, words1), shift);

                vst1q_s8(
                    output.as_mut_ptr().add(16 * i).cast(),
                    vcombine_s8(vqmovn_s16(r0), vqmovn_s16(r1)),
                );
            }
            num_chunks * 16
        };

        #[cfg(all(use_sse2, not(use_avx2)))]
        // SAFETY: the chunks of 16 are within IN.
        let start = unsafe {
            let num_chunks = IN / 16;
            // Shifts by a constant count compile to shifts by an immediate.
            let shift = _mm_cvtsi32_si128(Self::SIMD_SHIFT_AMOUNT);
            for i in 0..num_chunks {
                let at = |j: usize| _mm_loadu_si128(input.as_ptr().add(16 * i + 4 * j).cast());
                let words0 = _mm_packs_epi32(at(0), at(1));
                let words1 = _mm_packs_epi32(at(2), at(3));

                let words0 = _mm_srl_epi16(_mm_mulhi_epi16(words0, words0), shift);
                let words1 = _mm_srl_epi16(_mm_mulhi_epi16(words1, words1), shift);

                _mm_storeu_si128(
                    output.as_mut_ptr().add(16 * i).cast(),
                    mm_packs_epi16(words0, words1),
                );
            }
            num_chunks * 16
        };

        #[cfg(not(any(use_neon, all(use_sse2, not(use_avx2)))))]
        let start = 0;

        for i in start..IN {
            // Really should be /127 but we need to make it fast so we
            // right-shift by an extra 7 bits instead. Needs to be accounted
            // for in the trainer.
            let square = i64::from(input[i]) * i64::from(input[i]);
            output[i] = (square >> (2 * W + 7)).min(127) as u8;
        }
    }
}
