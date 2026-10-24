//! The clipped ReLU activation: the inputs, scaled down by 2^W, clamped to
//! 0..=127.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/layers/clipped_relu.h`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

#![allow(unused_unsafe)]

#[cfg(use_neon)]
use std::arch::aarch64::*;
#[cfg(all(use_sse2, not(use_avx2)))]
use std::arch::x86_64::*;

use crate::nnue::common::Reader;
#[cfg(all(use_sse2, not(use_avx2)))]
use crate::nnue::simd::mm_packs_epi16;
#[cfg(all(use_sse41, not(use_avx2)))]
use crate::nnue::simd::mm_packus_epi32;

/// With AVX2, SqrClippedReLU::propagate_pair computes this activation for
/// chess's network.
pub struct ClippedReLU<const IN: usize, const W: i32>;

impl<const IN: usize, const W: i32> ClippedReLU<IN, W> {
    /// Hash value embedded in the evaluation file
    pub const fn get_hash_value(prev_hash: u32) -> u32 {
        // Stockfish leaves W out on purpose, as the trainer does not write
        // it yet.
        0x538D_24C7u32.wrapping_add(prev_hash)
    }

    /// Read network parameters
    pub fn read_parameters(&mut self, _: &mut Reader) -> bool {
        true
    }

    /// Write network parameters
    pub fn write_parameters(&self, _: &mut Vec<u8>) {}

    /// Forward propagation
    #[inline(always)]
    pub fn propagate(&self, input: &[i32; IN], output: &mut [u8; IN]) {
        #[cfg(use_neon)]
        // SAFETY: the chunks of 8 are within IN.
        let start = unsafe {
            let num_chunks = IN / 8;
            let zero = vdup_n_s8(0);
            for i in 0..num_chunks {
                let at = input.as_ptr().add(8 * i);
                let shifted = vcombine_s16(
                    vqshrn_n_s32::<W>(vld1q_s32(at)),
                    vqshrn_n_s32::<W>(vld1q_s32(at.add(4))),
                );
                vst1_s8(
                    output.as_mut_ptr().add(8 * i).cast(),
                    vmax_s8(vqmovn_s16(shifted), zero),
                );
            }
            num_chunks * 8
        };

        #[cfg(all(use_sse2, not(use_avx2)))]
        // SAFETY: the chunks of 16 are within IN.
        let start = unsafe {
            let num_chunks = IN / 16;
            let at = |j: usize| _mm_loadu_si128(input.as_ptr().add(4 * j).cast());

            // Without SSE4.1's packus, saturating adds of -128 then
            // subtractions clamp the bytes below at zero.
            #[cfg(not(use_sse41))]
            let k0x80s = _mm_set1_epi8(-128);

            for i in 0..num_chunks {
                #[cfg(use_sse41)]
                let packed = {
                    let words0 = _mm_srli_epi16::<W>(mm_packus_epi32(at(4 * i), at(4 * i + 1)));
                    let words1 = _mm_srli_epi16::<W>(mm_packus_epi32(at(4 * i + 2), at(4 * i + 3)));
                    mm_packs_epi16(words0, words1)
                };
                #[cfg(not(use_sse41))]
                let packed = {
                    let words0 = _mm_srai_epi16::<W>(_mm_packs_epi32(at(4 * i), at(4 * i + 1)));
                    let words1 = _mm_srai_epi16::<W>(_mm_packs_epi32(at(4 * i + 2), at(4 * i + 3)));
                    let packed = mm_packs_epi16(words0, words1);
                    _mm_subs_epi8(_mm_adds_epi8(packed, k0x80s), k0x80s)
                };
                _mm_storeu_si128(output.as_mut_ptr().add(16 * i).cast(), packed);
            }
            num_chunks * 16
        };

        #[cfg(not(any(use_neon, all(use_sse2, not(use_avx2)))))]
        let start = 0;

        for i in start..IN {
            output[i] = (input[i] >> W).clamp(0, 127) as u8;
        }
    }
}
