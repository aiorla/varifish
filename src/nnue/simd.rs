//! The vectors the network is computed with, one set of operations per
//! instruction set the build targets: AVX-512, AVX2, SSE2 to SSE4.1, NEON,
//! WebAssembly's SIMD, or a portable fallback.
//!
//! The cfgs `use_sse2`, `use_ssse3`, `use_sse41`, `use_avx2`, `use_avx512`,
//! `use_vnni`, `use_neon` and `use_neon_dotprod` stand for Stockfish's
//! `USE_*` defines, `vector` for its `VECTOR`, and `use_paired_widening_add`
//! for its `USE_PAIRED_WIDENING_ADD`, which SSSE3 and later take; build.rs
//! sets them from the target's features. Unlike Stockfish's, the portable
//! fallback keeps the accumulators in vectors of 16 lanes, which compilers
//! vectorize, rather than one value at a time.
//!
//! WebAssembly's 128-bit SIMD, `use_simd128`, takes the paths of SSE, as
//! Stockfish's `wasm32` build does through Emscripten's translation of SSE's
//! intrinsics, with WebAssembly's own operations where it has none of SSE's;
//! `use_relaxed_simd` adds relaxed SIMD's dot product of bytes, as its
//! `wasm32-relaxed-simd` build does.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/simd.h`: Stockfish, a UCI chess playing engine derived from
//! Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! AUTHORS), under the GNU General Public License, version 3 or later.

// Every operation is a function whose instructions the build targets, so
// that calling it is sound; loads and stores also need their pointers to be
// valid for a vector.
#![allow(unused_unsafe)]

#[cfg(use_neon)]
use std::arch::aarch64::*;
#[cfg(use_simd128)]
use std::arch::wasm32::*;
#[cfg(use_sse2)]
use std::arch::x86_64::*;

use super::architecture::L1;

#[cfg(use_avx512)]
mod platform {
    use std::arch::x86_64::*;

    pub type VecT = __m512i;
    pub const NUM_REGISTERS_SIMD: usize = 16;
    pub const MAX_CHUNK_SIZE: usize = 64;

    #[inline(always)]
    pub unsafe fn vec_load(a: *const i16) -> VecT {
        unsafe { _mm512_loadu_si512(a.cast()) }
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut i16, b: VecT) {
        unsafe { _mm512_storeu_si512(a.cast(), b) }
    }

    #[inline(always)]
    pub fn vec_add_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm512_add_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_sub_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm512_sub_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_mulhi_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm512_mulhi_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_zero() -> VecT {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    pub fn vec_set_16(a: i16) -> VecT {
        unsafe { _mm512_set1_epi16(a) }
    }

    #[inline(always)]
    pub fn vec_max_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm512_max_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_min_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm512_min_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_slli_16<const B: u32>(a: VecT) -> VecT {
        unsafe { _mm512_slli_epi16::<B>(a) }
    }

    #[inline(always)]
    pub fn vec_srai_16<const B: u32>(a: VecT) -> VecT {
        unsafe { _mm512_srai_epi16::<B>(a) }
    }

    /// Inverse permuted at load time
    #[inline(always)]
    pub fn vec_packus_16(a: VecT, b: VecT) -> VecT {
        super::mm512_packus_epi16(a, b)
    }

    #[inline(always)]
    pub fn vec_set_8(a: i8) -> VecT {
        unsafe { _mm512_set1_epi8(a) }
    }

    #[inline(always)]
    pub fn vec_unpacklo_8(a: VecT, b: VecT) -> VecT {
        unsafe { _mm512_unpacklo_epi8(a, b) }
    }

    #[inline(always)]
    pub fn vec_unpackhi_8(a: VecT, b: VecT) -> VecT {
        unsafe { _mm512_unpackhi_epi8(a, b) }
    }

    #[inline(always)]
    pub fn vec_maddubs_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm512_maddubs_epi16(a, b) }
    }
}

#[cfg(all(use_avx2, not(use_avx512)))]
mod platform {
    use std::arch::x86_64::*;

    pub type VecT = __m256i;
    pub const NUM_REGISTERS_SIMD: usize = 12;
    pub const MAX_CHUNK_SIZE: usize = 32;

    #[inline(always)]
    pub unsafe fn vec_load(a: *const i16) -> VecT {
        unsafe { _mm256_loadu_si256(a.cast()) }
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut i16, b: VecT) {
        unsafe { _mm256_storeu_si256(a.cast(), b) }
    }

    #[inline(always)]
    pub fn vec_add_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm256_add_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_sub_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm256_sub_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_mulhi_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm256_mulhi_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_zero() -> VecT {
        unsafe { _mm256_setzero_si256() }
    }

    #[inline(always)]
    pub fn vec_set_16(a: i16) -> VecT {
        unsafe { _mm256_set1_epi16(a) }
    }

    #[inline(always)]
    pub fn vec_max_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm256_max_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_min_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm256_min_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_slli_16<const B: i32>(a: VecT) -> VecT {
        unsafe { _mm256_slli_epi16::<B>(a) }
    }

    #[inline(always)]
    pub fn vec_srai_16<const B: i32>(a: VecT) -> VecT {
        unsafe { _mm256_srai_epi16::<B>(a) }
    }

    /// Inverse permuted at load time
    #[inline(always)]
    pub fn vec_packus_16(a: VecT, b: VecT) -> VecT {
        super::mm256_packus_epi16(a, b)
    }

    #[inline(always)]
    pub fn vec_set_8(a: i8) -> VecT {
        unsafe { _mm256_set1_epi8(a) }
    }

    #[inline(always)]
    pub fn vec_unpacklo_8(a: VecT, b: VecT) -> VecT {
        unsafe { _mm256_unpacklo_epi8(a, b) }
    }

    #[inline(always)]
    pub fn vec_unpackhi_8(a: VecT, b: VecT) -> VecT {
        unsafe { _mm256_unpackhi_epi8(a, b) }
    }

    #[inline(always)]
    pub fn vec_maddubs_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm256_maddubs_epi16(a, b) }
    }

    /// The i32 lanes greater than zero, a bit each.
    #[inline(always)]
    pub fn vec_nnz(a: VecT) -> u32 {
        unsafe {
            _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(
                a,
                _mm256_setzero_si256(),
            ))) as u32
        }
    }
}

#[cfg(all(use_sse2, not(use_avx2)))]
mod platform {
    use std::arch::x86_64::*;

    pub type VecT = __m128i;
    pub const NUM_REGISTERS_SIMD: usize = 12;
    pub const MAX_CHUNK_SIZE: usize = 16;

    #[inline(always)]
    pub unsafe fn vec_load(a: *const i16) -> VecT {
        unsafe { _mm_loadu_si128(a.cast()) }
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut i16, b: VecT) {
        unsafe { _mm_storeu_si128(a.cast(), b) }
    }

    /// Loads 8 i8 values, widened to i16 by interleaving each with a byte of
    /// its sign, for SSE2 alone: with SSSE3, the weights widen in pairs.
    /// Credit: Yoshie2000
    #[cfg(not(use_ssse3))]
    #[inline(always)]
    pub unsafe fn vec_load_convert_8_16(a: *const i8) -> VecT {
        unsafe {
            let v8 = _mm_loadl_epi64(a.cast());
            let sign = _mm_cmpgt_epi8(_mm_setzero_si128(), v8);
            _mm_unpacklo_epi8(v8, sign)
        }
    }

    #[inline(always)]
    pub fn vec_add_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm_add_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_sub_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm_sub_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_mulhi_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm_mulhi_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_zero() -> VecT {
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    pub fn vec_set_16(a: i16) -> VecT {
        unsafe { _mm_set1_epi16(a) }
    }

    #[inline(always)]
    pub fn vec_max_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm_max_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_min_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm_min_epi16(a, b) }
    }

    #[inline(always)]
    pub fn vec_slli_16<const B: i32>(a: VecT) -> VecT {
        unsafe { _mm_slli_epi16::<B>(a) }
    }

    #[cfg(use_ssse3)]
    #[inline(always)]
    pub fn vec_srai_16<const B: i32>(a: VecT) -> VecT {
        unsafe { _mm_srai_epi16::<B>(a) }
    }

    #[inline(always)]
    pub fn vec_packus_16(a: VecT, b: VecT) -> VecT {
        super::mm_packus_epi16(a, b)
    }

    #[cfg(use_ssse3)]
    #[inline(always)]
    pub fn vec_set_8(a: i8) -> VecT {
        unsafe { _mm_set1_epi8(a) }
    }

    #[cfg(use_ssse3)]
    #[inline(always)]
    pub fn vec_unpacklo_8(a: VecT, b: VecT) -> VecT {
        unsafe { _mm_unpacklo_epi8(a, b) }
    }

    #[cfg(use_ssse3)]
    #[inline(always)]
    pub fn vec_unpackhi_8(a: VecT, b: VecT) -> VecT {
        unsafe { _mm_unpackhi_epi8(a, b) }
    }

    #[cfg(use_ssse3)]
    #[inline(always)]
    pub fn vec_maddubs_16(a: VecT, b: VecT) -> VecT {
        unsafe { _mm_maddubs_epi16(a, b) }
    }

    /// The i32 lanes greater than zero, a bit each.
    #[inline(always)]
    pub fn vec_nnz(a: VecT) -> u32 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a, _mm_setzero_si128()))) as u32 }
    }
}

#[cfg(use_neon)]
mod platform {
    use std::arch::aarch64::*;

    pub type VecT = int16x8_t;
    pub const NUM_REGISTERS_SIMD: usize = 16;
    pub const MAX_CHUNK_SIZE: usize = 16;

    #[inline(always)]
    pub unsafe fn vec_load(a: *const i16) -> VecT {
        unsafe { vld1q_s16(a) }
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut i16, b: VecT) {
        unsafe { vst1q_s16(a, b) }
    }

    #[inline(always)]
    pub fn vec_add_16(a: VecT, b: VecT) -> VecT {
        unsafe { vaddq_s16(a, b) }
    }

    #[inline(always)]
    pub fn vec_sub_16(a: VecT, b: VecT) -> VecT {
        unsafe { vsubq_s16(a, b) }
    }
}

#[cfg(use_simd128)]
mod platform {
    use std::arch::wasm32::*;

    pub type VecT = v128;
    pub const NUM_REGISTERS_SIMD: usize = 8;
    pub const MAX_CHUNK_SIZE: usize = 16;

    #[inline(always)]
    pub unsafe fn vec_load(a: *const i16) -> VecT {
        unsafe { v128_load(a.cast()) }
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut i16, b: VecT) {
        unsafe { v128_store(a.cast(), b) }
    }

    /// Loads a vector's worth of i8 values, widened to i16.
    #[inline(always)]
    pub unsafe fn vec_load_convert_8_16(a: *const i8) -> VecT {
        unsafe { i16x8_load_extend_i8x8(a) }
    }

    #[inline(always)]
    pub fn vec_add_16(a: VecT, b: VecT) -> VecT {
        i16x8_add(a, b)
    }

    #[inline(always)]
    pub fn vec_sub_16(a: VecT, b: VecT) -> VecT {
        i16x8_sub(a, b)
    }

    #[inline(always)]
    pub fn vec_set_16(a: i16) -> VecT {
        i16x8_splat(a)
    }

    #[inline(always)]
    pub fn vec_min_16(a: VecT, b: VecT) -> VecT {
        i16x8_min(a, b)
    }

    /// In order: WebAssembly's narrowing does not interleave lanes.
    #[inline(always)]
    pub fn vec_packus_16(a: VecT, b: VecT) -> VecT {
        u8x16_narrow_i16x8(a, b)
    }
}

#[cfg(not(vector))]
mod platform {
    /// 16 lanes, a vector to compilers.
    pub type VecT = [i16; 16];
    pub const NUM_REGISTERS_SIMD: usize = 64;

    #[inline(always)]
    pub unsafe fn vec_load(a: *const i16) -> VecT {
        unsafe { a.cast::<VecT>().read_unaligned() }
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut i16, b: VecT) {
        unsafe { a.cast::<VecT>().write_unaligned(b) }
    }

    /// Loads a vector's worth of i8 values, widened to i16.
    #[inline(always)]
    pub unsafe fn vec_load_convert_8_16(a: *const i8) -> VecT {
        let bytes = unsafe { a.cast::<[i8; 16]>().read_unaligned() };
        bytes.map(i16::from)
    }

    #[inline(always)]
    pub fn vec_add_16(a: VecT, b: VecT) -> VecT {
        std::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    pub fn vec_sub_16(a: VecT, b: VecT) -> VecT {
        std::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    }
}

pub use platform::*;

/// Defines a function for one of x86's packs of two vectors into one of half
/// as wide lanes, the instruction alone: Rust's intrinsics for them, as of
/// 1.98, clamp each operand to the range of the result first, two more
/// instructions per operand, though the instructions saturate by themselves.
#[cfg(use_avx2)]
macro_rules! pack {
    ($(#[$doc:meta])* $name:ident, $ty:ty, $reg:ident, $instruction:literal) => {
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(a: $ty, b: $ty) -> $ty {
            let packed;
            // SAFETY: the build targets the instruction, which only computes.
            unsafe {
                std::arch::asm!(
                    concat!($instruction, " {packed}, {a}, {b}"),
                    packed = lateout($reg) packed,
                    a = in($reg) a,
                    b = in($reg) b,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            packed
        }
    };
}

/// Defines a function for one of SSE's packs, the instruction alone, as
/// `pack!` does for AVX's: SSE's take the first operand as the result.
#[cfg(all(use_sse2, not(use_avx2)))]
macro_rules! pack_sse {
    ($(#[$doc:meta])* $name:ident, $instruction:literal) => {
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(a: __m128i, b: __m128i) -> __m128i {
            let mut packed = a;
            // SAFETY: the build targets the instruction, which only computes.
            unsafe {
                std::arch::asm!(
                    concat!($instruction, " {packed}, {b}"),
                    packed = inout(xmm_reg) packed,
                    b = in(xmm_reg) b,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            packed
        }
    };
}

#[cfg(all(use_sse2, not(use_avx2)))]
pack_sse!(
    /// `_mm_packus_epi16`
    mm_packus_epi16, "packuswb"
);
#[cfg(all(use_sse2, not(use_avx2)))]
pack_sse!(
    /// `_mm_packs_epi16`
    mm_packs_epi16, "packsswb"
);
#[cfg(all(use_sse41, not(use_avx2)))]
pack_sse!(
    /// `_mm_packus_epi32`
    mm_packus_epi32, "packusdw"
);

#[cfg(all(use_avx2, not(use_avx512)))]
pack!(
    /// `_mm256_packus_epi16`
    mm256_packus_epi16, __m256i, ymm_reg, "vpackuswb"
);
#[cfg(all(use_avx2, not(use_avx512)))]
pack!(
    /// `_mm256_packs_epi16`
    mm256_packs_epi16, __m256i, ymm_reg, "vpacksswb"
);
#[cfg(all(use_avx2, not(use_avx512)))]
pack!(
    /// `_mm256_packs_epi32`
    mm256_packs_epi32, __m256i, ymm_reg, "vpackssdw"
);
#[cfg(use_avx512)]
pack!(
    /// `_mm512_packus_epi16`
    mm512_packus_epi16, __m512i, zmm_reg, "vpackuswb"
);
#[cfg(use_avx512)]
pack!(
    /// `_mm512_packs_epi32`
    mm512_packs_epi32, __m512i, zmm_reg, "vpackssdw"
);

#[cfg(use_avx512)]
#[inline(always)]
pub fn m512_add_dpbusd_epi32(acc: &mut __m512i, a: __m512i, b: __m512i) {
    #[cfg(use_vnni)]
    unsafe {
        *acc = _mm512_dpbusd_epi32(*acc, a, b);
    }
    #[cfg(not(use_vnni))]
    unsafe {
        let product0 = _mm512_maddubs_epi16(a, b);
        let product0 = _mm512_madd_epi16(product0, _mm512_set1_epi16(1));
        *acc = _mm512_add_epi32(*acc, product0);
    }
}

#[cfg(use_avx2)]
#[inline(always)]
pub fn m256_hadd(sum: __m256i, bias: i32) -> i32 {
    unsafe {
        let mut sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(sum),
            _mm256_extracti128_si256::<1>(sum),
        );
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32::<0x4E>(sum128)); // _MM_PERM_BADC
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32::<0xB1>(sum128)); // _MM_PERM_CDAB
        _mm_cvtsi128_si32(sum128) + bias
    }
}

#[cfg(use_avx2)]
#[inline(always)]
pub fn m256_add_dpbusd_epi32(acc: &mut __m256i, a: __m256i, b: __m256i) {
    #[cfg(use_vnni)]
    unsafe {
        *acc = _mm256_dpbusd_epi32(*acc, a, b);
    }
    #[cfg(not(use_vnni))]
    unsafe {
        let product0 = _mm256_maddubs_epi16(a, b);
        let product0 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
        *acc = _mm256_add_epi32(*acc, product0);
    }
}

#[cfg(all(use_ssse3, not(use_avx2)))]
#[inline(always)]
pub fn m128_hadd(sum: __m128i, bias: i32) -> i32 {
    unsafe {
        let sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0x4E>(sum)); // _MM_PERM_BADC
        let sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0xB1>(sum)); // _MM_PERM_CDAB
        _mm_cvtsi128_si32(sum) + bias
    }
}

#[cfg(all(use_ssse3, not(use_avx2)))]
#[inline(always)]
pub fn m128_add_dpbusd_epi32(acc: &mut __m128i, a: __m128i, b: __m128i) {
    unsafe {
        let product0 = _mm_maddubs_epi16(a, b);
        let product0 = _mm_madd_epi16(product0, _mm_set1_epi16(1));
        *acc = _mm_add_epi32(*acc, product0);
    }
}

#[cfg(use_neon_dotprod)]
#[inline(always)]
pub fn dotprod_m128_add_dpbusd_epi32(acc: &mut int32x4_t, a: int8x16_t, b: int8x16_t) {
    unsafe {
        *acc = vdotq_s32(*acc, a, b);
    }
}

#[cfg(use_neon)]
#[inline(always)]
pub fn neon_m128_reduce_add_epi32(s: int32x4_t) -> i32 {
    unsafe { vaddvq_s32(s) }
}

#[cfg(use_neon_dotprod)]
#[inline(always)]
pub fn neon_m128_hadd(sum: int32x4_t, bias: i32) -> i32 {
    neon_m128_reduce_add_epi32(sum) + bias
}

#[cfg(all(use_neon, not(use_neon_dotprod)))]
#[inline(always)]
pub fn neon_m128_add_dpbusd_epi32(acc: &mut int32x4_t, a: int8x16_t, b: int8x16_t) {
    unsafe {
        let product0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let product1 = vmull_high_s8(a, b);
        let sum = vpaddq_s16(product0, product1);
        *acc = vpadalq_s16(*acc, sum);
    }
}

/// Adds to each i32 lane of `acc` the products of the four u8 in `a`, each
/// at most 127, and the four i8 in `b` at its place: with relaxed SIMD's dot
/// product, whose second operand's lanes give the same result everywhere
/// within 0..=127, or else with SSE's widening and pairwise sums.
#[cfg(use_simd128)]
#[inline(always)]
pub fn simd128_add_dpbusd_epi32(acc: &mut v128, a: v128, b: v128) {
    #[cfg(use_relaxed_simd)]
    {
        *acc = i32x4_relaxed_dot_i8x16_i7x16_add(b, a, *acc);
    }
    #[cfg(not(use_relaxed_simd))]
    {
        let product0 = i32x4_dot_i16x8(u16x8_extend_low_u8x16(a), i16x8_extend_low_i8x16(b));
        let product1 = i32x4_dot_i16x8(u16x8_extend_high_u8x16(a), i16x8_extend_high_i8x16(b));
        let sum = i32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(product0, product1),
            i32x4_shuffle::<1, 3, 5, 7>(product0, product1),
        );
        *acc = i32x4_add(*acc, sum);
    }
}

#[cfg(use_simd128)]
#[inline(always)]
pub fn simd128_hadd(sum: v128, bias: i32) -> i32 {
    let sum = i32x4_add(sum, i32x4_shuffle::<2, 3, 0, 1>(sum, sum));
    let sum = i32x4_add(sum, i32x4_shuffle::<1, 0, 3, 2>(sum, sum));
    i32x4_extract_lane::<0>(sum) + bias
}

/// The vectors of the affine layers, as their `vec_*_32` macros: i32 sums of
/// products of u8 inputs and i8 weights, four at a time, in `OutVec`, and
/// the inputs and weights in `InVec`.
#[cfg(any(use_sse2, use_neon, use_simd128))]
mod affine {
    #[cfg(use_neon)]
    use std::arch::aarch64::*;
    #[cfg(use_simd128)]
    use std::arch::wasm32::*;
    #[cfg(use_sse2)]
    use std::arch::x86_64::*;

    #[cfg(use_avx512)]
    pub type OutVec = __m512i;
    #[cfg(use_avx512)]
    pub type InVec = __m512i;
    #[cfg(all(use_avx2, not(use_avx512)))]
    pub type OutVec = __m256i;
    #[cfg(all(use_avx2, not(use_avx512)))]
    pub type InVec = __m256i;
    #[cfg(all(use_sse2, not(use_avx2)))]
    pub type OutVec = __m128i;
    #[cfg(all(use_sse2, not(use_avx2)))]
    pub type InVec = __m128i;
    #[cfg(use_neon)]
    pub type OutVec = int32x4_t;
    #[cfg(use_neon)]
    pub type InVec = int8x16_t;
    #[cfg(use_simd128)]
    pub type OutVec = v128;
    #[cfg(use_simd128)]
    pub type InVec = v128;

    /// The i32 lanes of an output vector.
    pub const OUTPUT_SIMD_WIDTH: usize = size_of::<OutVec>() / 4;

    /// The most accumulators a layer keeps: 3 chains of 32 outputs with NEON.
    pub const MAX_ACCUMULATORS: usize = 24;

    #[inline(always)]
    pub unsafe fn vec_load_out(a: *const i32) -> OutVec {
        unsafe {
            #[cfg(use_avx512)]
            return _mm512_loadu_si512(a.cast());
            #[cfg(all(use_avx2, not(use_avx512)))]
            return _mm256_loadu_si256(a.cast());
            #[cfg(all(use_sse2, not(use_avx2)))]
            return _mm_loadu_si128(a.cast());
            #[cfg(use_neon)]
            return vld1q_s32(a);
            #[cfg(use_simd128)]
            return v128_load(a.cast());
        }
    }

    #[inline(always)]
    pub unsafe fn vec_store_out(a: *mut i32, b: OutVec) {
        unsafe {
            #[cfg(use_avx512)]
            _mm512_storeu_si512(a.cast(), b);
            #[cfg(all(use_avx2, not(use_avx512)))]
            _mm256_storeu_si256(a.cast(), b);
            #[cfg(all(use_sse2, not(use_avx2)))]
            _mm_storeu_si128(a.cast(), b);
            #[cfg(use_neon)]
            vst1q_s32(a, b);
            #[cfg(use_simd128)]
            v128_store(a.cast(), b);
        }
    }

    #[inline(always)]
    pub fn vec_zero_out() -> OutVec {
        unsafe {
            #[cfg(use_avx512)]
            return _mm512_setzero_si512();
            #[cfg(all(use_avx2, not(use_avx512)))]
            return _mm256_setzero_si256();
            #[cfg(all(use_sse2, not(use_avx2)))]
            return _mm_setzero_si128();
            #[cfg(use_neon)]
            return vdupq_n_s32(0);
            #[cfg(use_simd128)]
            return i32x4_splat(0);
        }
    }

    /// Merges the chains of accumulators that dot products with a high
    /// latency are split into.
    #[cfg(any(use_vnni, use_neon_dotprod))]
    #[inline(always)]
    pub fn vec_add_32(a: OutVec, b: OutVec) -> OutVec {
        unsafe {
            #[cfg(use_avx512)]
            return _mm512_add_epi32(a, b);
            #[cfg(all(use_avx2, not(use_avx512)))]
            return _mm256_add_epi32(a, b);
            #[cfg(use_neon)]
            return vaddq_s32(a, b);
        }
    }

    /// The four inputs at `a`, repeated across a vector.
    #[inline(always)]
    pub unsafe fn vec_load_32(a: *const u8) -> InVec {
        unsafe {
            let chunk = a.cast::<i32>().read_unaligned();
            #[cfg(use_avx512)]
            return _mm512_set1_epi32(chunk);
            #[cfg(all(use_avx2, not(use_avx512)))]
            return _mm256_set1_epi32(chunk);
            #[cfg(all(use_sse2, not(use_avx2)))]
            return _mm_set1_epi32(chunk);
            #[cfg(use_neon)]
            return vreinterpretq_s8_s32(vdupq_n_s32(chunk));
            #[cfg(use_simd128)]
            return i32x4_splat(chunk);
        }
    }

    /// A vector of weights.
    #[inline(always)]
    pub unsafe fn vec_load_in(a: *const i8) -> InVec {
        unsafe {
            #[cfg(use_avx512)]
            return _mm512_loadu_si512(a.cast());
            #[cfg(all(use_avx2, not(use_avx512)))]
            return _mm256_loadu_si256(a.cast());
            #[cfg(all(use_sse2, not(use_avx2)))]
            return _mm_loadu_si128(a.cast());
            #[cfg(use_neon)]
            return vld1q_s8(a);
            #[cfg(use_simd128)]
            return v128_load(a.cast());
        }
    }

    /// Adds to each i32 lane of `acc` the products of the four u8 in `a`
    /// and the four i8 in `b` at its place: not with SSE2 alone, which has
    /// no dot product of bytes.
    #[cfg(not(all(use_sse2, not(use_ssse3))))]
    #[inline(always)]
    pub fn vec_add_dpbusd_32(acc: &mut OutVec, a: InVec, b: InVec) {
        #[cfg(use_avx512)]
        super::m512_add_dpbusd_epi32(acc, a, b);
        #[cfg(all(use_avx2, not(use_avx512)))]
        super::m256_add_dpbusd_epi32(acc, a, b);
        #[cfg(all(use_ssse3, not(use_avx2)))]
        super::m128_add_dpbusd_epi32(acc, a, b);
        #[cfg(use_neon_dotprod)]
        super::dotprod_m128_add_dpbusd_epi32(acc, a, b);
        #[cfg(all(use_neon, not(use_neon_dotprod)))]
        super::neon_m128_add_dpbusd_epi32(acc, a, b);
        #[cfg(use_simd128)]
        super::simd128_add_dpbusd_epi32(acc, a, b);
    }
}

#[cfg(any(use_sse2, use_neon, use_simd128))]
pub use affine::*;

/// The largest number of vector registers, at most `max_registers`, that
/// divides the registers needed for `num_lanes` lanes of `lane_size` bytes.
pub(super) const fn best_register_count(
    register_size: usize,
    lane_size: usize,
    num_lanes: usize,
    max_registers: usize,
) -> usize {
    assert!(register_size >= lane_size && register_size.is_multiple_of(lane_size));
    assert!((num_lanes * lane_size).is_multiple_of(register_size));

    let ideal = (num_lanes * lane_size) / register_size;
    if ideal <= max_registers {
        return ideal;
    }

    // Look for the largest divisor of the ideal register count that is smaller than MaxRegisters
    let mut divisor = max_registers;
    while divisor > 1 {
        if ideal.is_multiple_of(divisor) {
            return divisor;
        }
        divisor -= 1;
    }
    1
}

/// The accumulators are updated and refreshed tile by tile, each tile
/// filling `NUM_REGS` vector registers.
pub const NUM_REGS: usize =
    best_register_count(size_of::<VecT>(), size_of::<i16>(), L1, NUM_REGISTERS_SIMD);
/// The i16 lanes of a vector.
pub const LANES: usize = size_of::<VecT>() / 2;
pub const TILE_HEIGHT: usize = NUM_REGS * LANES;

const _: () = assert!(
    L1.is_multiple_of(TILE_HEIGHT),
    "TileHeight must divide HalfDimensions"
);
#[cfg(vector)]
const _: () = assert!(
    MAX_CHUNK_SIZE == size_of::<VecT>(),
    "MaxChunkSize must equal one vector."
);
