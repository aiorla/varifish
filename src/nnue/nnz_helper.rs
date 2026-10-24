//! Which 4-byte chunks of the transformed features are not zero, recorded as
//! the feature transformer writes them, so that the first layer reads only
//! those: a list of their indices with AVX-512, a bitset with SSE2 to AVX2,
//! NEON and WebAssembly's SIMD. The portable layer reads every chunk and
//! records nothing.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/nnz_helper.h`: Stockfish, a UCI chess playing engine derived from
//! Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! AUTHORS), under the GNU General Public License, version 3 or later.

#![allow(unused_unsafe)]

#[cfg(use_neon)]
use std::arch::aarch64::*;
#[cfg(use_simd128)]
use std::arch::wasm32::*;
#[cfg(use_avx512)]
use std::arch::x86_64::*;
#[cfg(use_avx512)]
use std::mem::MaybeUninit;

#[cfg(any(sparse_input, test))]
use super::architecture::L1;
#[cfg(vector)]
use super::simd::VecT;

#[cfg(use_avx512)]
pub struct NnzInfo {
    count: usize,
    // indices of non-zero chunks, the first `count` written
    nnz: [MaybeUninit<u16>; L1 / 4],
}

#[cfg(all(use_avx512, not(use_avx512icl)))]
#[repr(align(64))]
struct Indices([[u32; 16]; 2]);

#[cfg(all(use_avx512, not(use_avx512icl)))]
static INDICES: Indices = {
    let mut indices = [[0; 16]; 2];
    let mut i = 0;
    while i < 2 {
        let mut m = 0;
        while m < 16 {
            indices[i][m] = (m + i * L1 / 8) as u32;
            m += 1;
        }
        i += 1;
    }
    Indices(indices)
};

// With Ice Lake's AVX-512, as in Stockfish's avx512icl builds, 16-bit
// indices, in the order in which packing two vectors of 32-bit chunks into
// one of 16 bits interleaves them: four of the first, then four of the
// second, in each 128 bits
#[cfg(use_avx512icl)]
#[repr(align(64))]
struct Indices([[u16; 32]; 2]);

#[cfg(use_avx512icl)]
static INDICES: Indices = {
    #[rustfmt::skip]
    const PACKED: [u16; 32] = [
        0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23,
        8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31,
    ];
    let mut indices = [[0; 32]; 2];
    let mut i = 0;
    while i < 2 {
        let mut m = 0;
        while m < 32 {
            indices[i][m] = PACKED[m] + (i * L1 / 8) as u16;
            m += 1;
        }
        i += 1;
    }
    Indices(indices)
};

#[cfg(use_avx512)]
pub struct NnzCursor<'a> {
    info: &'a mut NnzInfo,
    indices: __m512i,
    count: usize,
    // A perspective's L1 / 8 chunks fill L1 / 128 vectors
    vectors_left: usize,
}

#[cfg(use_avx512)]
impl NnzInfo {
    #[inline(always)]
    pub fn new() -> NnzInfo {
        // Built field by field: the list is written only up to `count`.
        let mut info = MaybeUninit::<NnzInfo>::uninit();
        // SAFETY: `count` is the only field that needs a value.
        unsafe {
            (&raw mut (*info.as_mut_ptr()).count).write(0);
            info.assume_init()
        }
    }

    /// The indices of the non-zero chunks, each less than L1 / 4.
    #[inline(always)]
    pub fn indices(&self) -> &[u16] {
        // SAFETY: the cursors write the list from index 0 up, at least
        // `count` values, and keep `count` within it.
        unsafe { std::slice::from_raw_parts(self.nnz.as_ptr().cast(), self.count) }
    }

    pub fn make_cursor(&mut self, perspective: usize) -> NnzCursor<'_> {
        let count = self.count;
        NnzCursor {
            info: self,
            // SAFETY: the row is a vector.
            indices: unsafe { _mm512_loadu_si512(INDICES.0[perspective].as_ptr().cast()) },
            count,
            vectors_left: L1 / 128,
        }
    }
}

#[cfg(use_avx512)]
impl NnzCursor<'_> {
    #[inline(always)]
    pub fn record2(&mut self, neurons1: VecT, neurons2: VecT) {
        // Keeps the stores within the list, and the indices below L1 / 4.
        assert!(self.vectors_left >= 2 && self.count <= L1 / 4 - 32);
        self.vectors_left -= 2;

        #[cfg(use_avx512icl)]
        unsafe {
            let increment = _mm512_set1_epi16(32);

            // Get a bitmask and gather non zero indices, of both vectors at
            // once: a chunk packs to 16 bits that are zero only if it is.
            let input = super::simd::mm512_packs_epi32(neurons1, neurons2);
            let nnz_mask = _mm512_test_epi16_mask(input, input);
            // Not _mm512_mask_compressstoreu_epi16(), 256 µops on Zen 4 as
            // Stockfish notes.
            let nnz_v = _mm512_maskz_compress_epi16(nnz_mask, self.indices);
            // SAFETY: the assertion above leaves room for 32 indices.
            _mm512_storeu_si512(self.info.nnz.as_mut_ptr().add(self.count).cast(), nnz_v);

            self.count += nnz_mask.count_ones() as usize;
            self.indices = _mm512_add_epi16(self.indices, increment);
        }

        #[cfg(not(use_avx512icl))]
        unsafe {
            let increment = _mm512_set1_epi32(16);

            for neurons in [neurons1, neurons2] {
                // Get a bitmask and gather non zero indices
                let nnz_mask = _mm512_test_epi32_mask(neurons, neurons);
                let nnz_v = _mm512_maskz_compress_epi32(nnz_mask, self.indices);
                // SAFETY: the assertion above leaves room for 16 indices.
                _mm512_mask_cvtepi32_storeu_epi16(
                    self.info.nnz.as_mut_ptr().add(self.count).cast(),
                    0xFFFF,
                    nnz_v,
                );

                self.count += nnz_mask.count_ones() as usize;
                self.indices = _mm512_add_epi32(self.indices, increment);
            }
        }
    }
}

#[cfg(use_avx512)]
impl Drop for NnzCursor<'_> {
    fn drop(&mut self) {
        self.info.count = self.count;
    }
}

#[cfg(all(sparse_input, not(use_avx512)))]
pub struct NnzInfo {
    // A bit per 4-byte chunk
    bitset: [u8; L1.div_ceil(32)],
}

#[cfg(all(sparse_input, not(use_avx512)))]
pub struct NnzCursor<'a> {
    out: &'a mut [u8],
}

#[cfg(all(sparse_input, not(use_avx512)))]
impl NnzInfo {
    #[inline(always)]
    pub fn new() -> NnzInfo {
        NnzInfo {
            bitset: [0; L1.div_ceil(32)],
        }
    }

    /// A bit per chunk, the lowest bit of byte `i` for chunk `8 * i`.
    #[inline(always)]
    pub fn bitset(&self) -> &[u8; L1.div_ceil(32)] {
        &self.bitset
    }

    pub fn make_cursor(&mut self, perspective: usize) -> NnzCursor<'_> {
        NnzCursor {
            out: &mut self.bitset[perspective * L1 / 64..],
        }
    }
}

#[cfg(all(sparse_input, not(use_avx512)))]
impl NnzCursor<'_> {
    #[inline(always)]
    pub fn record2(&mut self, neurons1: VecT, neurons2: VecT) {
        #[cfg(use_neon)]
        unsafe {
            static MASK8: [u16; 8] = [1, 16, 2, 32, 4, 64, 8, 128];

            let n1 = vreinterpretq_u32_s16(neurons1);
            let n2 = vreinterpretq_u32_s16(neurons2);

            let t1 = vtstq_u32(n1, n1);
            let t2 = vtstq_u32(n2, n2);

            let packed = vtrn1q_u16(vreinterpretq_u16_u32(t1), vreinterpretq_u16_u32(t2));
            let bits = vandq_u16(packed, vld1q_u16(MASK8.as_ptr()));

            self.out[0] = vaddvq_u16(bits) as u8;
            self.advance(1);
        }
        #[cfg(use_simd128)]
        {
            // A vector holds 4 chunks, so the two fill a byte
            let zero = i32x4_splat(0);
            self.out[0] = i32x4_bitmask(i32x4_ne(neurons1, zero))
                | i32x4_bitmask(i32x4_ne(neurons2, zero)) << 4;
            self.advance(1);
        }
        #[cfg(use_avx2)]
        {
            use super::simd::vec_nnz;

            // A byte of a mask per vector of 8 chunks
            self.out[0] = vec_nnz(neurons1) as u8;
            self.out[1] = vec_nnz(neurons2) as u8;
            self.advance(2);
        }
        #[cfg(all(use_sse2, not(use_avx2)))]
        {
            use super::simd::vec_nnz;

            // A vector holds 4 chunks, so the two fill a byte
            self.out[0] = (vec_nnz(neurons1) | vec_nnz(neurons2) << 4) as u8;
            self.advance(1);
        }
    }

    #[inline(always)]
    fn advance(&mut self, bytes: usize) {
        let out = std::mem::take(&mut self.out);
        self.out = &mut out[bytes..];
    }
}

#[cfg(not(sparse_input))]
pub struct NnzInfo;

#[cfg(not(sparse_input))]
impl NnzInfo {
    #[inline(always)]
    pub fn new() -> NnzInfo {
        NnzInfo
    }
}

impl Default for NnzInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl NnzInfo {
    /// The chunks of the transformed features `input` recorded as the
    /// feature transformer records them.
    #[cfg(test)]
    pub fn of(input: &[u8; L1]) -> NnzInfo {
        #[cfg_attr(not(vector), allow(unused_mut))]
        let mut info = NnzInfo::new();
        #[cfg(vector)]
        for perspective in 0..2 {
            let mut cursor = info.make_cursor(perspective);
            let bytes = &input[perspective * L1 / 2..(perspective + 1) * L1 / 2];
            for pair in bytes.as_chunks::<{ 2 * size_of::<VecT>() }>().0 {
                // SAFETY: each half of the pair holds a vector.
                let (neurons1, neurons2) = unsafe {
                    let at = pair.as_ptr().cast::<VecT>();
                    (at.read_unaligned(), at.add(1).read_unaligned())
                };
                cursor.record2(neurons1, neurons2);
            }
        }
        let _ = input;
        info
    }
}
