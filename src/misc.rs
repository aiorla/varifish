//! The pseudo-random number generator behind the Zobrist keys and magics, a
//! fixed-capacity list, prefetching, the clock, and parsing as C++ input
//! streams do.
//!
//! A translation into Rust, modified for varifish in 2026, of parts of
//! Stockfish's `misc.h`: Stockfish, a UCI chess playing engine derived from
//! Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! AUTHORS), under the GNU General Public License, version 3 or later.

use std::mem::MaybeUninit;
use std::sync::OnceLock;
use std::time::Instant;

/// xorshift64star, after Sebastiano Vigna's public-domain code (2014): 64-bit
/// outputs, a single 64-bit state and a period of 2^64 - 1, with no warm-up
/// needed. It runs in constant evaluation, where all of varifish's tables are
/// built.
pub struct Prng {
    s: u64,
}

impl Prng {
    pub const fn new(seed: u64) -> Prng {
        assert!(seed != 0);
        Prng { s: seed }
    }

    pub const fn rand64(&mut self) -> u64 {
        self.s ^= self.s >> 12;
        self.s ^= self.s << 25;
        self.s ^= self.s >> 27;
        self.s.wrapping_mul(2685821657736338717)
    }

    /// A number with only 1/8th of its bits set on average, to find magics
    /// quickly.
    pub const fn sparse_rand(&mut self) -> u64 {
        self.rand64() & self.rand64() & self.rand64()
    }
}

/// A list of at most `N` values, kept uninitialized past its end.
///
/// `N` is a power of two and the index of a pushed value wraps around it, so
/// pushing needs no bounds check: overflowing a list, which callers size to
/// rule out and debug builds assert against, overwrites its first values
/// instead of memory beyond it.
pub struct ValueList<T: Copy, const N: usize> {
    values: [MaybeUninit<T>; N],
    // The number of values pushed, which exceeds N only after an overflow.
    size: usize,
}

impl<T: Copy, const N: usize> ValueList<T, N> {
    const CAPACITY_IS_A_POWER_OF_TWO: () = assert!(N.is_power_of_two());

    #[inline(always)]
    pub fn new() -> Self {
        #[allow(clippy::let_unit_value)]
        let () = Self::CAPACITY_IS_A_POWER_OF_TWO;
        // Built field by field: as a constant, the list is written whole,
        // which zeroes its N values on each construction.
        let mut list = MaybeUninit::<Self>::uninit();
        // SAFETY: `size` is the only field that needs a value.
        unsafe {
            (&raw mut (*list.as_mut_ptr()).size).write(0);
            list.assume_init()
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        if self.size < N { self.size } else { N }
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.size = 0;
    }

    #[inline(always)]
    pub fn push(&mut self, value: T) {
        debug_assert!(self.size < N, "ValueList overflow");
        self.values[self.size & (N - 1)] = MaybeUninit::new(value);
        self.size += 1;
    }

    /// Appends the first `count` of the `width` values that `store` writes
    /// in one vector store at the end of the list, as Stockfish's
    /// `make_space` lets its AVX-512 code write.
    ///
    /// # Safety
    ///
    /// `store` writes at most `width` values from the pointer it is given,
    /// the first `count` of them, of at most `width`, valid.
    #[cfg(use_avx512icl)]
    #[inline(always)]
    pub unsafe fn extend_by_store(
        &mut self,
        width: usize,
        count: usize,
        store: impl FnOnce(*mut T),
    ) {
        debug_assert!(
            count <= width && self.size + width <= N,
            "ValueList overflow"
        );
        // Within the list even after an overflow, as `push` writes
        let at = self.size.min(N - width);
        // SAFETY: the `width` values from `at` lie within the list.
        store(unsafe { self.values.as_mut_ptr().add(at).cast() });
        self.size += count;
    }

    /// Pushes `value` if it is less than `max`, without branching.
    #[inline(always)]
    pub fn push_back_if_lt(&mut self, value: T, max: T)
    where
        T: PartialOrd,
    {
        debug_assert!(self.size < N, "ValueList overflow");
        self.values[self.size & (N - 1)] = MaybeUninit::new(value);
        self.size += usize::from(value < max);
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: values are written from index 0 up, wrapping around, so the
        // first len() are initialized; MaybeUninit<T> has T's layout.
        unsafe { std::slice::from_raw_parts(self.values.as_ptr().cast::<T>(), self.len()) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as for as_slice.
        unsafe { std::slice::from_raw_parts_mut(self.values.as_mut_ptr().cast::<T>(), self.len()) }
    }

    /// Keeps the first `len` values, of at most len().
    #[inline(always)]
    pub fn truncate(&mut self, len: usize) {
        debug_assert!(len <= self.len());
        self.size = len;
    }

    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    #[inline(always)]
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().contains(value)
    }
}

impl<T: Copy, const N: usize> Default for ValueList<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> Clone for ValueList<T, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy, const N: usize> Copy for ValueList<T, N> {}

impl<T: Copy + std::fmt::Debug, const N: usize> std::fmt::Debug for ValueList<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: Copy, const N: usize> std::ops::Index<usize> for ValueList<T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &T {
        &self.as_slice()[index]
    }
}

impl<'a, T: Copy, const N: usize> IntoIterator for &'a ValueList<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Asks the processor to bring the cache line at `addr` into a low level of
/// cache, for reading soon: Stockfish's `prefetch<READ, LOW>`. Any address
/// is fine, as a prefetch never faults.
#[inline(always)]
pub fn prefetch_low<T>(addr: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: x86-64 always has SSE.
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T2, _mm_prefetch};
        _mm_prefetch::<_MM_HINT_T2>(addr.cast());
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: the instruction only hints the memory system.
    unsafe {
        std::arch::asm!(
            "prfm pldl3keep, [{addr}]",
            addr = in(reg) addr,
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = addr;
}

/// Asks the processor to bring the cache line at `addr` into the closest
/// cache, for reading soon: Stockfish's default `prefetch`. Any address is
/// fine, as a prefetch never faults.
#[inline(always)]
pub fn prefetch<T>(addr: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: x86-64 always has SSE.
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch::<_MM_HINT_T0>(addr.cast());
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: the instruction only hints the memory system.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{addr}]",
            addr = in(reg) addr,
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = addr;
}

/// A value in milliseconds
pub type TimePoint = i64;

/// The milliseconds since a fixed moment, which only differences make sense
/// of: Stockfish's `now()` on a steady clock.
pub fn now() -> TimePoint {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as TimePoint
}

/// `a * b + c` as Stockfish's compilers evaluate it: where the target has
/// fused multiply-adds they contract the expression into one, rounding once
/// instead of twice, and so does varifish.
#[inline(always)]
pub fn mul_add(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(target_feature = "fma", target_arch = "aarch64"))]
    let result = a.mul_add(b, c);
    #[cfg(not(any(target_feature = "fma", target_arch = "aarch64")))]
    let result = a * b + c;
    result
}

/// The linear interpolation at `x` of the line through (x0, y0) and
/// (x1, y1).
#[inline]
pub fn interpolate(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    debug_assert!(x0 != x1);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// `interpolate` for integer abscissas, whose differences are taken before
/// they are converted.
#[inline]
pub fn interpolate_i64(x: i64, x0: i64, x1: i64, y0: f64, y1: f64) -> f64 {
    debug_assert!(x0 != x1);
    y0 + (y1 - y0) * (x - x0) as f64 / (x1 - x0) as f64
}

/// C++'s `std::min` of two doubles: `a` unless `b` is less.
#[inline(always)]
pub fn min_f64(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// C++'s `std::max` of two doubles: `a` unless it is less than `b`.
#[inline(always)]
pub fn max_f64(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// C++'s `std::clamp` of a double.
#[inline(always)]
pub fn clamp_f64(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if hi < v {
        hi
    } else {
        v
    }
}

/// The number of threads the system can run at once.
pub fn get_hardware_concurrency() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// The C library's isspace in the "C" locale.
#[inline]
pub fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

/// The parts of a C++ input string stream that Stockfish's parsing of FENs
/// and commands relies on, so that varifish accepts and refuses the same
/// input, field by field.
pub struct IStream<'a> {
    bytes: &'a [u8],
    pos: usize,
    eof: bool,
    fail: bool,
}

impl<'a> IStream<'a> {
    pub fn new(s: &'a str) -> IStream<'a> {
        IStream {
            bytes: s.as_bytes(),
            pos: 0,
            eof: false,
            fail: false,
        }
    }

    pub fn good(&self) -> bool {
        !self.eof && !self.fail
    }

    pub fn eof(&self) -> bool {
        self.eof
    }

    pub fn fail(&self) -> bool {
        self.fail
    }

    /// Whether extraction can start: the sentry of a formatted extraction,
    /// which skips whitespace when `skip_ws`.
    fn sentry(&mut self, skip_ws: bool) -> bool {
        if !self.good() {
            self.fail = true;
            return false;
        }
        if skip_ws {
            while self.bytes.get(self.pos).is_some_and(|&c| is_space(c)) {
                self.pos += 1;
            }
            if self.pos == self.bytes.len() {
                self.eof = true;
                self.fail = true;
                return false;
            }
        }
        true
    }

    /// `ss >> c` for a character with `std::noskipws`.
    pub fn get(&mut self) -> Option<u8> {
        if !self.sentry(false) {
            return None;
        }
        match self.bytes.get(self.pos) {
            Some(&c) => {
                self.pos += 1;
                Some(c)
            }
            None => {
                self.eof = true;
                self.fail = true;
                None
            }
        }
    }

    /// `ss >> std::ws`
    pub fn skip_ws(&mut self) {
        if !self.good() {
            self.fail = true;
            return;
        }
        while let Some(&c) = self.bytes.get(self.pos) {
            if !is_space(c) {
                return;
            }
            self.pos += 1;
        }
        self.eof = true;
    }

    /// `ss >> token` for a string: the next run of non-space characters.
    pub fn token(&mut self) -> Option<&'a str> {
        if !self.sentry(true) {
            return None;
        }
        let start = self.pos;
        while self.bytes.get(self.pos).is_some_and(|&c| !is_space(c)) {
            self.pos += 1;
        }
        if self.pos == self.bytes.len() {
            self.eof = true;
        }
        // Both ends are at ASCII characters or the ends of the string, so on
        // character boundaries.
        std::str::from_utf8(&self.bytes[start..self.pos]).ok()
    }

    /// The sign and the decimal digits of a number, as `num_get` reads them,
    /// with the magnitude saturated beyond u64::MAX; None when no digits
    /// follow, after which the stream fails.
    fn number(&mut self) -> Option<(bool, u128)> {
        let negative = self.bytes[self.pos] == b'-';
        if matches!(self.bytes[self.pos], b'+' | b'-') {
            self.pos += 1;
        }
        let digits = self.pos;
        let mut magnitude: u128 = 0;
        while let Some(&c) = self.bytes.get(self.pos).filter(|c| c.is_ascii_digit()) {
            magnitude = (magnitude * 10 + (c - b'0') as u128).min(u64::MAX as u128 + 1);
            self.pos += 1;
        }
        if self.pos == self.bytes.len() {
            self.eof = true;
        }
        if self.pos == digits {
            self.fail = true;
            return None;
        }
        Some((negative, magnitude))
    }

    /// `ss >> n` for a signed integer type between `min` and `max`: None when
    /// extraction could not start, which leaves the variable as it was. A
    /// missing number reads as 0, and one out of range as the nearest bound,
    /// failing the stream.
    fn signed(&mut self, min: i128, max: i128) -> Option<i128> {
        if !self.sentry(true) {
            return None;
        }
        let Some((negative, magnitude)) = self.number() else {
            return Some(0);
        };
        let value = if negative {
            -(magnitude as i128)
        } else {
            magnitude as i128
        };
        if value < min || value > max {
            self.fail = true;
            return Some(value.clamp(min, max));
        }
        Some(value)
    }

    pub fn read_i32(&mut self) -> Option<i32> {
        self.signed(i32::MIN as i128, i32::MAX as i128)
            .map(|v| v as i32)
    }

    pub fn read_i64(&mut self) -> Option<i64> {
        self.signed(i64::MIN as i128, i64::MAX as i128)
            .map(|v| v as i64)
    }

    /// `ss >> n` for a u64, where as in C a negative number wraps around.
    pub fn read_u64(&mut self) -> Option<u64> {
        if !self.sentry(true) {
            return None;
        }
        let Some((negative, magnitude)) = self.number() else {
            return Some(0);
        };
        if magnitude > u64::MAX as u128 {
            self.fail = true;
            return Some(u64::MAX);
        }
        let magnitude = magnitude as u64;
        Some(if negative {
            magnitude.wrapping_neg()
        } else {
            magnitude
        })
    }
}
