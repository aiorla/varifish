//! The transposition table, which all search threads share.
//!
//! There is only one global hash table for the engine and all its threads.
//! For chess in particular, we even allow racy updates between threads to and
//! from the TT, as taking the time to synchronize access would cost thinking
//! time and thus Elo. As a hash table, collisions are possible and may cause
//! chess playing issues (bizarre blunders, faulty mate reports, etc). Fixing
//! these also loses Elo; however such risk decreases with larger TT size.
//!
//! We clearly separate TTData, a local copy of an entry, from TTWriter, which
//! writes to the global table.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `tt.h` and `tt.cpp`: Stockfish, a UCI chess playing engine derived from
//! Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! AUTHORS), under the GNU General Public License, version 3 or later.

use std::sync::Mutex;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicI16, AtomicU8, AtomicU16};

use crate::memory::{HUGE_PAGE_SIZE, LargePageArray};
use crate::thread::lock;
use crate::types::*;

/// A copy of the data already in an entry (possibly collided). Probes and
/// reads are racy and non-atomic, possibly resulting in inconsistent data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TTData {
    pub mv: Move,
    pub value: Value,
    pub eval: Value,
    pub depth: Depth,
    pub bound: Bound,
    pub is_pv: bool,
}

// TTEntry struct is the 10 bytes transposition table entry, defined as:
//
// key        16 bit
// depth       8 bit
// pv node     1 bit
// bound type  2 bit
// generation  5 bit
// move       16 bit
// value      16 bit
// evaluation 16 bit
//
// These fields are in the same order as accessed by TT::probe(), since memory
// is fastest sequentially. The store order in save() matches this order.
//
// We use `depth8 != 0` as the cheap internal occupancy check, corresponding
// to `depth == DEPTH_NONE` externally, so we offset the internal depth by
// DEPTH_NONE.
//
// Pv, bound and generation are packed in a single byte.

const GENERATION_BITS: u8 = 5;
const GENERATION_MASK: u8 = (1 << GENERATION_BITS) - 1;
const BOUND_SHIFT: u8 = GENERATION_BITS;
const BOUND_MASK: u8 = 0b11 << BOUND_SHIFT;
const PV_SHIFT: u8 = BOUND_SHIFT + 2;
const PV_MASK: u8 = 1 << PV_SHIFT;

#[repr(C)]
struct TTEntry {
    key16: AtomicU16,
    depth8: AtomicU8,
    gen_bound8: AtomicU8,
    move16: AtomicU16,
    value16: AtomicI16,
    eval16: AtomicI16,
}

impl TTEntry {
    /// Extracts the data from the entry, converting its internal bitfields
    /// to external types.
    #[inline(always)]
    fn read(&self) -> TTData {
        let gen_bound8 = self.gen_bound8.load(Relaxed);
        TTData {
            mv: Move(self.move16.load(Relaxed)),
            value: Value::from(self.value16.load(Relaxed)),
            eval: Value::from(self.eval16.load(Relaxed)),
            depth: DEPTH_NONE + Depth::from(self.depth8.load(Relaxed)),
            bound: (gen_bound8 & BOUND_MASK) >> BOUND_SHIFT,
            is_pv: gen_bound8 & PV_MASK != 0,
        }
    }

    /// Whether the entry holds data
    #[inline(always)]
    fn is_occupied(&self) -> bool {
        self.depth8.load(Relaxed) != 0
    }

    /// Populates the entry with a new node's data, possibly overwriting an
    /// old position. The update is non-atomic and can be racy. We convert
    /// external types to internal bitfields.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn save(
        &self,
        k: Key,
        v: Value,
        pv: bool,
        b: Bound,
        d: Depth,
        m: Move,
        ev: Value,
        curr_generation: u8,
    ) {
        // Preserve the old ttmove if we don't have a new one
        if m != Move::none() || k as u16 != self.key16.load(Relaxed) {
            self.move16.store(m.0, Relaxed);
        }

        // Overwrite less valuable entries (cheapest checks first)
        if b == BOUND_EXACT
            || k as u16 != self.key16.load(Relaxed)
            || d - DEPTH_NONE + 2 * pv as i32 > i32::from(self.depth8.load(Relaxed)) - 4
            || self.relative_age(curr_generation) != 0
        {
            debug_assert!(d > DEPTH_NONE);
            debug_assert!(d - DEPTH_NONE < 256);
            debug_assert!(curr_generation <= GENERATION_MASK);

            self.key16.store(k as u16, Relaxed);
            self.depth8.store((d - DEPTH_NONE) as u8, Relaxed);
            self.gen_bound8.store(
                curr_generation | (b << BOUND_SHIFT) | ((pv as u8) << PV_SHIFT),
                Relaxed,
            );
            self.value16.store(v as i16, Relaxed);
            self.eval16.store(ev as i16, Relaxed);
        }
        // Secondary aging. Important for elementary mate finding.
        // (*Scaler) Secondary aging on entries relevant to singular extensions
        // generally scales poorly and requires VVLTC verification.
        else if Depth::from(self.depth8.load(Relaxed)) + DEPTH_NONE >= 5
            && (self.gen_bound8.load(Relaxed) & BOUND_MASK) >> BOUND_SHIFT != BOUND_EXACT
        {
            let v16 = Value::from(self.value16.load(Relaxed));

            // Guard against racy underflows, default to "unoccupied"
            if v16.abs() < VALUE_INFINITE && is_decisive(v16) {
                let depth8 = i32::from(self.depth8.load(Relaxed));
                self.depth8.store((depth8 - 1).max(0) as u8, Relaxed);
            }
        }
    }

    /// Returns this entry's age. We count generations like clocks count
    /// hours, i.e. we require 0 - 1 == 31. Wrapping subtraction guarantees
    /// the required borrowing regardless of the upper pv/bound bits.
    #[inline(always)]
    fn relative_age(&self, curr_generation: u8) -> u8 {
        curr_generation.wrapping_sub(self.gen_bound8.load(Relaxed)) & GENERATION_MASK
    }
}

/// This is used to make racy, non-atomic writes to the global TT. Writes are
/// not "guaranteed": for chess reasons, we may decide the new data is less
/// important than the old.
#[derive(Clone, Copy)]
pub struct TTWriter<'a> {
    entry: &'a TTEntry,
}

impl TTWriter<'_> {
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn write(
        &self,
        k: Key,
        v: Value,
        pv: bool,
        b: Bound,
        d: Depth,
        m: Move,
        ev: Value,
        generation8: u8,
    ) {
        self.entry.save(k, v, pv, b, d, m, ev, generation8);
    }

    /// Flags the entry as useless, decrementing its stored depth by the
    /// penalty.
    #[inline(always)]
    pub fn penalize(&self, penalty: i32) {
        // Guard against racy underflows, default to "unoccupied"
        let depth8 = i32::from(self.entry.depth8.load(Relaxed));
        self.entry
            .depth8
            .store((depth8 - penalty).max(0) as u8, Relaxed);
    }
}

// A TranspositionTable is an array of Cluster, of size clusterCount. Each
// cluster consists of CLUSTER_SIZE number of TTEntry. Each non-empty TTEntry
// contains information on exactly one position. The size of a Cluster should
// divide the size of a cache line for best performance, as the cacheline is
// prefetched when possible.

const CLUSTER_SIZE: usize = 3;

#[repr(C)]
struct Cluster {
    entry: [TTEntry; CLUSTER_SIZE],
    padding: [u8; 2], // Pad to 32 bytes
}

const _: () = assert!(size_of::<TTEntry>() == 10);
const _: () = assert!(size_of::<Cluster>() == 32, "Suboptimal Cluster size");

/// The table, empty until resized.
#[derive(Default)]
pub struct TranspositionTable {
    table: LargePageArray<Cluster>,
    generation8: AtomicU8,
}

impl TranspositionTable {
    /// Sets the size of the table in megabytes and clears it with `threads`
    /// threads. The table consists of clusters and each cluster consists of
    /// CLUSTER_SIZE number of TTEntry.
    pub fn resize(&mut self, mb_size: usize, threads: usize) {
        // Free the old table before allocating the new one.
        self.table = LargePageArray::new();

        let cluster_count = mb_size * 1024 * 1024 / size_of::<Cluster>();
        let tt_bytes = cluster_count * size_of::<Cluster>();

        // Request 1GB pages if we'd get at least eight, to avoid memory
        // oversubscription
        let huge_page_hint = tt_bytes >= HUGE_PAGE_SIZE.saturating_mul(8);

        // SAFETY: zero bytes are empty entries.
        self.table = unsafe { LargePageArray::zeroed(cluster_count, huge_page_hint) };

        self.clear(threads);
    }

    /// Initializes the entire table to zero, with `threads` threads each
    /// zeroing its part.
    pub fn clear(&mut self, threads: usize) {
        *self.generation8.get_mut() = 0;

        let cluster_count = self.table.len();
        let threads = threads.clamp(1, cluster_count.max(1));
        let stride = cluster_count / threads;

        let zero = |part: &mut [Cluster]| {
            // SAFETY: zero bytes are empty entries, written to clusters
            // borrowed mutably.
            unsafe { part.as_mut_ptr().write_bytes(0, part.len()) }
        };

        if threads == 1 {
            zero(&mut self.table);
            return;
        }

        // The last thread also zeroes what the stride leaves. A thread that
        // cannot start, as when a WebAssembly host has no worker free yet,
        // leaves its part to this one, which is why each part is behind a
        // lock.
        let mut rest = &mut self.table[..];
        let parts: Vec<Mutex<&mut [Cluster]>> = (0..threads)
            .map(|i| {
                let len = if i + 1 != threads { stride } else { rest.len() };
                let (part, tail) = std::mem::take(&mut rest).split_at_mut(len);
                rest = tail;
                Mutex::new(part)
            })
            .collect();
        std::thread::scope(|scope| {
            for part in &parts {
                let zero_part = move || zero(&mut lock(part));
                if std::thread::Builder::new()
                    .spawn_scoped(scope, zero_part)
                    .is_err()
                {
                    zero_part();
                }
            }
        });
    }

    /// Approximates what fraction of entries (permille) have been written to
    /// during this root search. Only counts entries which are younger than
    /// `max_age`.
    pub fn hashfull(&self, max_age: i32) -> i32 {
        let generation8 = self.generation();
        let mut cnt = 0;
        for cluster in &self.table[..1000] {
            for entry in &cluster.entry {
                cnt += (entry.is_occupied()
                    && i32::from(entry.relative_age(generation8)) <= max_age)
                    as i32;
            }
        }
        cnt / CLUSTER_SIZE as i32
    }

    /// Must be called at the beginning of each root search to track entry
    /// aging.
    pub fn new_search(&self) {
        // Don't overflow into the other bits of TTEntry::gen_bound8
        let generation8 = self.generation8.load(Relaxed).wrapping_add(1) & GENERATION_MASK;
        self.generation8.store(generation8, Relaxed);
    }

    /// The current age, used when writing new data to the TT
    #[inline(always)]
    pub fn generation(&self) -> u8 {
        self.generation8.load(Relaxed)
    }

    /// The primary method: given a board position, we look up its entry in
    /// the table, and return
    ///  1) whether the entry already had data on this position
    ///  2) a copy of the prior data, if any (may be self-inconsistent due to
    ///     races)
    ///  3) a writer to the entry
    ///
    /// The key is found if one of the cluster's entries holds its low 16
    /// bits (which may be a collision) and data. Otherwise the writer is to
    /// an empty or the least valuable entry, to be replaced later. The value
    /// of an entry is its depth minus 8 times its relative age.
    #[inline(always)]
    pub fn probe(&self, key: Key) -> (bool, TTData, TTWriter<'_>) {
        let tte = &self.table[self.cluster_index(key)].entry;
        let key16 = key as u16; // Use the low 16 bits as key inside the cluster

        for e in tte {
            if e.key16.load(Relaxed) == key16 {
                // This gap is the main place for read races.
                // After `read()` completes that copy is final, but may be
                // self-inconsistent.
                return (e.is_occupied(), e.read(), TTWriter { entry: e });
            }
        }

        // Find an entry to be replaced according to the replacement strategy
        let generation8 = self.generation();
        let worth = |e: &TTEntry| {
            i32::from(e.depth8.load(Relaxed)) - 8 * i32::from(e.relative_age(generation8))
        };
        let mut replace = &tte[0];
        for e in &tte[1..] {
            if worth(replace) > worth(e) {
                replace = e;
            }
        }

        (
            false,
            TTData {
                mv: Move::none(),
                value: VALUE_NONE,
                eval: VALUE_NONE,
                depth: DEPTH_NONE,
                bound: BOUND_NONE,
                is_pv: false,
            },
            TTWriter { entry: replace },
        )
    }

    /// The hash function: the cluster of a key
    #[inline(always)]
    fn cluster_index(&self, key: Key) -> usize {
        mul_hi64(key, self.table.len())
    }

    /// The address of the first entry for a key, to prefetch.
    #[inline(always)]
    pub fn first_entry(&self, key: Key) -> *const u8 {
        self.table
            .as_ptr()
            .wrapping_add(self.cluster_index(key))
            .cast()
    }
}

/// The high 64 bits of a times b, below b. WebAssembly has no multiply to
/// 128 bits, where u128's multiply is a call, so there it multiplies halves
/// of 32 bits, as Stockfish does.
#[inline(always)]
fn mul_hi64(a: u64, b: usize) -> usize {
    #[cfg(not(target_family = "wasm"))]
    return ((u128::from(a) * b as u128) >> 64) as usize;
    #[cfg(target_family = "wasm")]
    return mul_hi64_by_halves(a, b as u64) as usize;
}

#[cfg(any(target_family = "wasm", test))]
#[inline(always)]
fn mul_hi64_by_halves(a: u64, b: u64) -> u64 {
    let (a_lo, a_hi) = (a & 0xffff_ffff, a >> 32);
    let (b_lo, b_hi) = (b & 0xffff_ffff, b >> 32);
    let c1 = (a_lo * b_lo) >> 32;
    let c2 = a_hi * b_lo + c1;
    let c3 = a_lo * b_hi + (c2 & 0xffff_ffff);
    a_hi * b_hi + (c2 >> 32) + (c3 >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> TranspositionTable {
        let mut tt = TranspositionTable::default();
        tt.resize(1, 2);
        tt
    }

    #[test]
    fn the_halves_multiply_as_u128() {
        let mut rng = crate::misc::Prng::new(20261023);
        let edges = [0, 1, 0xffff_ffff, 1 << 32, u64::MAX];
        let pairs = edges
            .iter()
            .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
            .chain((0..100_000).map(|_| (rng.rand64(), rng.rand64())));
        for (a, b) in pairs {
            let hi = ((u128::from(a) * u128::from(b)) >> 64) as u64;
            assert_eq!(mul_hi64_by_halves(a, b), hi, "{a:#x} * {b:#x}");
        }
    }

    #[test]
    fn entries_keep_what_is_written() {
        let tt = table();
        let key = 0x1234_5678_9abc_def0;
        let (hit, data, writer) = tt.probe(key);
        assert!(!hit);
        assert_eq!(data.depth, DEPTH_NONE);

        let m = Move::new(Square::E1, Square::E2);
        writer.write(key, -123, true, BOUND_LOWER, 7, m, 45, tt.generation());
        let (hit, data, _) = tt.probe(key);
        assert!(hit);
        assert_eq!(
            data,
            TTData {
                mv: m,
                value: -123,
                eval: 45,
                depth: 7,
                bound: BOUND_LOWER,
                is_pv: true,
            }
        );
        assert_eq!(tt.hashfull(0), 0, "the first 1000 clusters are empty");
    }

    #[test]
    fn shallower_data_keeps_the_entry_but_a_new_move_replaces_its_move() {
        let tt = table();
        let key = 42;
        let (_, _, writer) = tt.probe(key);
        let m1 = Move::new(Square::A2, Square::A3);
        let m2 = Move::new(Square::B2, Square::B3);
        writer.write(key, 10, false, BOUND_UPPER, 20, m1, 0, tt.generation());
        // Not deep enough to replace the data, but a move is kept anyway.
        writer.write(key, 30, false, BOUND_UPPER, 3, m2, 0, tt.generation());
        let (_, data, _) = tt.probe(key);
        assert_eq!((data.mv, data.value, data.depth), (m2, 10, 20));
        // An exact bound always replaces.
        writer.write(
            key,
            30,
            false,
            BOUND_EXACT,
            3,
            Move::none(),
            0,
            tt.generation(),
        );
        let (_, data, _) = tt.probe(key);
        assert_eq!((data.mv, data.value, data.depth), (m2, 30, 3));
    }

    #[test]
    fn generations_age_entries_and_clear_empties_the_table() {
        let mut tt = table();
        for g in 0..40 {
            assert_eq!(tt.generation(), g % 32);
            tt.new_search();
        }
        // Entries of the first clusters, which hashfull counts.
        for key in 0..20_000_u64 {
            let key = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let (_, _, writer) = tt.probe(key);
            writer.write(
                key,
                0,
                false,
                BOUND_EXACT,
                1,
                Move::none(),
                0,
                tt.generation(),
            );
        }
        let full = tt.hashfull(0);
        assert!(full > 0);
        tt.new_search();
        assert_eq!(tt.hashfull(0), 0);
        assert_eq!(tt.hashfull(1), full);
        tt.clear(3);
        assert_eq!(tt.generation(), 0);
        assert_eq!(tt.hashfull(31), 0);
    }
}
