//! Allocations in large pages, for the network, the transposition table and
//! the other large tables that are read all the time, so that fewer of their
//! pages miss the TLB.
//!
//! A translation into Rust, modified for varifish in 2026, of parts of
//! Stockfish's `memory.h` and `memory.cpp`: Stockfish, a UCI chess playing
//! engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// The alignment and granularity of the allocations: on Linux, 2 MB pages,
/// which transparent huge pages back when asked to.
#[cfg(target_os = "linux")]
const ALIGNMENT: usize = 2 * 1024 * 1024;
#[cfg(not(target_os = "linux"))]
const ALIGNMENT: usize = 4096;

/// A box whose value lives in large pages where the system has them.
pub struct LargePageBox<T> {
    ptr: NonNull<T>,
}

// SAFETY: the box owns its value, as Box does.
unsafe impl<T: Send> Send for LargePageBox<T> {}
// SAFETY: the box shares its value only through &T.
unsafe impl<T: Sync> Sync for LargePageBox<T> {}

impl<T> LargePageBox<T> {
    /// A value of zero bytes, in its own allocation.
    ///
    /// # Safety
    ///
    /// Zero bytes must make a valid `T`.
    pub unsafe fn zeroed() -> LargePageBox<T> {
        let layout = Self::layout();
        // SAFETY: the layout's size is not zero.
        let ptr = unsafe { alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            alloc::handle_alloc_error(layout);
        };
        // Asked before the pages are first touched, which is when the kernel
        // decides what backs them.
        advise_huge_pages(ptr.as_ptr(), layout.size());
        // SAFETY: the allocation holds layout.size() bytes.
        unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        LargePageBox { ptr: ptr.cast() }
    }

    fn layout() -> Layout {
        let size = size_of::<T>().max(1).next_multiple_of(ALIGNMENT);
        Layout::from_size_align(size, ALIGNMENT.max(align_of::<T>())).unwrap()
    }
}

impl<T> Drop for LargePageBox<T> {
    fn drop(&mut self) {
        // SAFETY: the box owns a valid value, allocated with this layout.
        unsafe {
            self.ptr.as_ptr().drop_in_place();
            alloc::dealloc(self.ptr.as_ptr().cast(), Self::layout());
        }
    }
}

impl<T> Deref for LargePageBox<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        // SAFETY: the box owns a valid value.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> DerefMut for LargePageBox<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: the box owns a valid value, borrowed mutably through it.
        unsafe { self.ptr.as_mut() }
    }
}

/// An array whose length is set at run time, in large pages where the system
/// has them: the transposition table and the histories threads share.
pub struct LargePageArray<T> {
    ptr: NonNull<T>,
    len: usize,
    // The bytes allocated, a whole number of pages
    size: usize,
}

// SAFETY: the array owns its values, as Box<[T]> does.
unsafe impl<T: Send> Send for LargePageArray<T> {}
// SAFETY: the array shares its values only through &[T].
unsafe impl<T: Sync> Sync for LargePageArray<T> {}

/// The size of the huge pages that large tables ask for first.
pub const HUGE_PAGE_SIZE: usize = 1 << 30;

impl<T> LargePageArray<T> {
    /// An array of no values, which allocates nothing.
    pub const fn new() -> LargePageArray<T> {
        LargePageArray {
            ptr: NonNull::dangling(),
            len: 0,
            size: 0,
        }
    }

    /// `len` values of zero bytes. With `huge_page_hint`, an array of at
    /// least `HUGE_PAGE_SIZE` bytes first asks for pages of that size, which
    /// Linux has only where the administrator reserved them. The pages are
    /// touched only when the values are, so that what backs them is decided
    /// then. Exits if the memory cannot be had, as Stockfish does.
    ///
    /// # Safety
    ///
    /// Zero bytes must make a valid `T`.
    pub unsafe fn zeroed(len: usize, huge_page_hint: bool) -> LargePageArray<T> {
        assert!(size_of::<T>() > 0 && align_of::<T>() <= 4096);
        let Some(bytes) = len.checked_mul(size_of::<T>()) else {
            out_of_memory(usize::MAX)
        };
        let (ptr, size) = pages::alloc_zeroed(bytes.max(1), huge_page_hint);
        LargePageArray {
            ptr: ptr.cast(),
            len,
            size,
        }
    }
}

impl<T> Default for LargePageArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LargePageArray<T> {
    fn drop(&mut self) {
        if self.size == 0 {
            return;
        }
        // SAFETY: the array owns its values, allocated by pages::alloc_zeroed
        // with this size.
        unsafe {
            std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len).drop_in_place();
            pages::free(self.ptr.cast(), self.size);
        }
    }
}

impl<T> Deref for LargePageArray<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: the array owns len valid values.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for LargePageArray<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: the array owns len valid values, borrowed mutably through it.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

fn out_of_memory(bytes: usize) -> ! {
    eprintln!("Failed to allocate {bytes} bytes in large pages.");
    std::process::exit(1)
}

/// Whether the system can back memory with large pages.
pub fn has_large_pages() -> bool {
    cfg!(target_os = "linux")
}

/// Zeroed allocations of whole pages: on Linux, mappings aligned to 2 MB
/// that transparent huge pages back, or 1 GB pages when asked and reserved.
#[cfg(target_os = "linux")]
mod pages {
    use std::ffi::{c_int, c_long, c_void};
    use std::ptr::NonNull;

    use super::{ALIGNMENT, HUGE_PAGE_SIZE, advise_huge_pages, out_of_memory};

    const PROT_READ: c_int = 1;
    const PROT_WRITE: c_int = 2;
    const MAP_PRIVATE: c_int = 0x02;
    const MAP_ANONYMOUS: c_int = 0x20;
    #[cfg(target_arch = "x86_64")]
    const MAP_HUGETLB: c_int = 0x40000;
    #[cfg(target_arch = "x86_64")]
    const MAP_HUGE_SHIFT: c_int = 26;
    const MAP_FAILED: *mut c_void = !0 as *mut c_void;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: c_long,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, length: usize) -> c_int;
    }

    fn map(size: usize, flags: c_int) -> Option<NonNull<u8>> {
        // SAFETY: an anonymous mapping of new memory, which aliases nothing.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | flags,
                -1,
                0,
            )
        };
        if ptr == MAP_FAILED {
            None
        } else {
            NonNull::new(ptr.cast())
        }
    }

    /// Unmaps the bytes from `ptr` on, which must be whole pages of a mapping.
    unsafe fn unmap(ptr: *mut u8, size: usize) {
        if size > 0 {
            // SAFETY: the caller's pages are mapped and no longer used.
            unsafe { munmap(ptr.cast(), size) };
        }
    }

    pub fn alloc_zeroed(bytes: usize, huge_page_hint: bool) -> (NonNull<u8>, usize) {
        // Pages of 1 GB, as Stockfish asks for them on x86-64 Linux
        #[cfg(target_arch = "x86_64")]
        if huge_page_hint && bytes >= HUGE_PAGE_SIZE {
            let size = bytes.next_multiple_of(HUGE_PAGE_SIZE);
            if let Some(ptr) = map(size, MAP_HUGETLB | (30 << MAP_HUGE_SHIFT)) {
                return (ptr, size);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (huge_page_hint, HUGE_PAGE_SIZE);

        // Otherwise a mapping aligned to 2 MB, cut from a larger one
        let size = bytes.next_multiple_of(ALIGNMENT);
        let Some(reservation) = map(size + ALIGNMENT, 0) else {
            out_of_memory(size)
        };
        let base = reservation.as_ptr();
        let prefix = base.align_offset(ALIGNMENT);
        // SAFETY: the prefix and the suffix are whole pages of the
        // reservation, outside the aligned part kept.
        let aligned = unsafe {
            unmap(base, prefix);
            unmap(base.add(prefix + size), ALIGNMENT - prefix);
            base.add(prefix)
        };
        advise_huge_pages(aligned, size);
        (NonNull::new(aligned).expect("a mapping is not null"), size)
    }

    /// # Safety
    ///
    /// `ptr` and `size` must come from `alloc_zeroed`, and the memory must no
    /// longer be used.
    pub unsafe fn free(ptr: NonNull<u8>, size: usize) {
        // SAFETY: as the caller ensures
        unsafe { unmap(ptr.as_ptr(), size) };
    }
}

#[cfg(not(target_os = "linux"))]
mod pages {
    use std::alloc::{self, Layout};
    use std::ptr::NonNull;

    use super::{ALIGNMENT, out_of_memory};

    fn layout(size: usize) -> Layout {
        Layout::from_size_align(size, ALIGNMENT).unwrap()
    }

    pub fn alloc_zeroed(bytes: usize, _huge_page_hint: bool) -> (NonNull<u8>, usize) {
        // On 32-bit targets an allocation holds at most isize::MAX bytes.
        let Some(size) = bytes
            .checked_next_multiple_of(ALIGNMENT)
            .filter(|&size| Layout::from_size_align(size, ALIGNMENT).is_ok())
        else {
            out_of_memory(bytes)
        };
        // SAFETY: the layout's size is not zero.
        let ptr = unsafe { alloc::alloc_zeroed(layout(size)) };
        match NonNull::new(ptr) {
            Some(ptr) => (ptr, size),
            None => out_of_memory(size),
        }
    }

    /// # Safety
    ///
    /// `ptr` and `size` must come from `alloc_zeroed`, and the memory must no
    /// longer be used.
    pub unsafe fn free(ptr: NonNull<u8>, size: usize) {
        // SAFETY: as the caller ensures
        unsafe { alloc::dealloc(ptr.as_ptr(), layout(size)) };
    }
}

/// Asks Linux to back the memory with transparent huge pages, as Stockfish
/// does with `madvise(MADV_HUGEPAGE)`. Failure only means small pages.
#[cfg(target_os = "linux")]
fn advise_huge_pages(ptr: *mut u8, size: usize) {
    use std::ffi::{c_int, c_void};

    const MADV_HUGEPAGE: c_int = 14;
    unsafe extern "C" {
        fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
    }
    // SAFETY: the range is an allocation of ours, aligned to a page.
    unsafe {
        madvise(ptr.cast(), size, MADV_HUGEPAGE);
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_huge_pages(_ptr: *mut u8, _size: usize) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_zeroed_and_aligned() {
        // SAFETY: zero bytes are a valid array of integers.
        let mut values = unsafe { LargePageBox::<[u64; 100_000]>::zeroed() };
        assert!(values.iter().all(|&v| v == 0));
        assert_eq!(values.as_ptr() as usize % ALIGNMENT, 0);
        values[99_999] = 1;
        assert_eq!(values.iter().sum::<u64>(), 1);
    }

    #[test]
    fn arrays_are_zeroed_and_aligned() {
        // SAFETY: zero bytes are a valid integer.
        let mut values = unsafe { LargePageArray::<u32>::zeroed(1_000_003, false) };
        assert_eq!(values.len(), 1_000_003);
        assert!(values.iter().all(|&v| v == 0));
        assert_eq!(values.as_ptr() as usize % ALIGNMENT, 0);
        values[1_000_002] = 7;
        assert_eq!(values.iter().map(|&v| u64::from(v)).sum::<u64>(), 7);
    }
}
