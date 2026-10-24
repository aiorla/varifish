//! The histories: statistics of how moves did in the search so far, which
//! order moves, decide reductions and correct static evaluations.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `history.h`: Stockfish, a UCI chess playing engine derived from Glaurung
//! 2.1, Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under
//! the GNU General Public License, version 3 or later.

use std::sync::atomic::AtomicI16;
use std::sync::atomic::Ordering::Relaxed;

use crate::memory::{LargePageArray, LargePageBox};
use crate::position::Position;
use crate::types::*;
use crate::variant::Variant;

pub const PAWN_HISTORY_BASE_SIZE: usize = 8192; // has to be a power of 2
pub const UINT_16_HISTORY_SIZE: usize = u16::MAX as usize + 1;
pub const CORRHIST_BASE_SIZE: usize = UINT_16_HISTORY_SIZE;
pub const CORRECTION_HISTORY_LIMIT: i32 = 1024;
pub const LOW_PLY_HISTORY_SIZE: usize = 5;

const _: () = assert!(PAWN_HISTORY_BASE_SIZE.is_power_of_two());
const _: () = assert!(CORRHIST_BASE_SIZE.is_power_of_two());

/// A statistic, which `update` moves towards a bonus while keeping it in
/// [-D, D]. Zero bytes make a valid entry.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct StatsEntry<const D: i32>(i16);

impl<const D: i32> StatsEntry<D> {
    const D_IS_SAFE: () = assert!(D > 0 && D <= i16::MAX as i32 && D <= i32::MAX / D);

    #[inline(always)]
    pub fn get(self) -> i32 {
        self.0 as i32
    }

    #[inline(always)]
    pub fn set(&mut self, v: i32) {
        self.0 = v as i16;
    }

    /// Stockfish's `<<`.
    #[inline(always)]
    pub fn update(&mut self, bonus: i32) {
        #[allow(clippy::let_unit_value)]
        let () = Self::D_IS_SAFE;
        // Make sure that bonus is in range [-D, D]
        let clamped_bonus = bonus.clamp(-D, D);
        let val = self.get();
        self.set(val + clamped_bonus - val * clamped_bonus.abs() / D);

        debug_assert!(self.get().abs() <= D);
    }
}

/// A statistic that threads share, read and written without ordering:
/// another thread's update in between may be lost, as in Stockfish.
#[derive(Default, Debug)]
#[repr(transparent)]
pub struct AtomicStatsEntry<const D: i32>(AtomicI16);

impl<const D: i32> AtomicStatsEntry<D> {
    const D_IS_SAFE: () = assert!(D > 0 && D <= i16::MAX as i32 && D <= i32::MAX / D);

    #[inline(always)]
    pub fn get(&self) -> i32 {
        self.0.load(Relaxed) as i32
    }

    #[inline(always)]
    pub fn set(&self, v: i32) {
        self.0.store(v as i16, Relaxed);
    }

    /// Stockfish's `<<`.
    #[inline(always)]
    pub fn update(&self, bonus: i32) {
        #[allow(clippy::let_unit_value)]
        let () = Self::D_IS_SAFE;
        let clamped_bonus = bonus.clamp(-D, D);
        let val = self.get();
        self.set(val + clamped_bonus - val * clamped_bonus.abs() / D);

        debug_assert!(self.get().abs() <= D);
    }
}

/// Sets every entry of a table of statistics.
pub trait Fill {
    fn fill(&mut self, v: i32);
}

impl<const D: i32> Fill for StatsEntry<D> {
    #[inline(always)]
    fn fill(&mut self, v: i32) {
        self.set(v);
    }
}

impl<const D: i32> Fill for AtomicStatsEntry<D> {
    #[inline(always)]
    fn fill(&mut self, v: i32) {
        *self.0.get_mut() = v as i16;
    }
}

impl<T: Fill, const N: usize> Fill for [T; N] {
    fn fill(&mut self, v: i32) {
        for entry in self {
            entry.fill(v);
        }
    }
}

/// Sets every entry of tables that threads share, by plain stores, which
/// cost less than atomic ones, and far less in WebAssembly, where every
/// atomic store is sequentially consistent.
pub trait FillShared {
    /// # Safety
    ///
    /// No other thread may read or write the entries meanwhile.
    unsafe fn fill_shared(&self, v: i32);
}

impl<const D: i32> FillShared for AtomicStatsEntry<D> {
    #[inline(always)]
    unsafe fn fill_shared(&self, v: i32) {
        // SAFETY: no other thread accesses the entry, by the caller.
        unsafe { *self.0.as_ptr() = v as i16 }
    }
}

impl<T: FillShared, const N: usize> FillShared for [T; N] {
    unsafe fn fill_shared(&self, v: i32) {
        for entry in self {
            // SAFETY: nor the table's entries.
            unsafe { entry.fill_shared(v) }
        }
    }
}

/// ButterflyHistory records how often quiet moves have been successful or
/// unsuccessful during the current search, and is used for reduction and
/// move ordering decisions. It uses 2 tables (one for each color) indexed by
/// the move's from and to squares, see
/// https://www.chessprogramming.org/Butterfly_Boards
pub type ButterflyHistory = [[StatsEntry<7183>; UINT_16_HISTORY_SIZE]; COLOR_NB];

/// LowPlyHistory is addressed by ply and move's from and to squares, used to
/// improve move ordering near the root
pub type LowPlyHistory = [[StatsEntry<7183>; UINT_16_HISTORY_SIZE]; LOW_PLY_HISTORY_SIZE];

/// CapturePieceToHistory is addressed by a move's [piece][to][captured piece
/// type]
pub type CapturePieceToHistory = [[[StatsEntry<10692>; PIECE_TYPE_NB]; SQUARE_NB]; PIECE_NB];

/// PieceToHistory is like ButterflyHistory but is addressed by a move's
/// [piece][to]
pub type PieceToHistory = [[AtomicStatsEntry<30000>; SQUARE_NB]; PIECE_NB];

/// The continuation histories: the combined history of a given pair of
/// moves, usually the current one given a previous one, for each [in
/// check][capture][piece][to] of the previous move. The table for a
/// previous move is at the index `continuation_index` gives.
pub type ContinuationHistory = [PieceToHistory; CONTINUATION_HISTORY_NB];

pub const CONTINUATION_HISTORY_NB: usize = 2 * 2 * PIECE_NB * SQUARE_NB;

/// Where the continuation history of a move of `pc` to `to` is, after a
/// position in check or not and for a capture or not. The index of
/// NO_PIECE's first table is zero, which serves as a sentinel.
#[inline(always)]
pub fn continuation_index(in_check: bool, capture: bool, pc: Piece, to: Square) -> u16 {
    (((in_check as usize * 2 + capture as usize) * PIECE_NB + pc.idx()) * SQUARE_NB + to.idx())
        as u16
}

/// PawnHistory is addressed by the pawn structure and a move's [piece][to]
pub type PawnHistoryEntry = [[AtomicStatsEntry<8192>; SQUARE_NB]; PIECE_NB];

/// Correction histories record differences between the static evaluation of
/// positions and their search score. It is used to improve the static
/// evaluation used by some search heuristics.
/// see https://www.chessprogramming.org/Static_Evaluation_Correction_History
pub type PieceToCorrectionHistory = [[StatsEntry<CORRECTION_HISTORY_LIMIT>; SQUARE_NB]; PIECE_NB];

/// The combined correction history of move pairs, for each [piece][to] of
/// the previous move. The table for a previous move is at the index
/// `continuation_correction_index` gives.
pub type ContinuationCorrectionHistory = [PieceToCorrectionHistory; PIECE_NB * SQUARE_NB];

/// Where the continuation correction history of a move of `pc` to `to` is.
/// The index of NO_PIECE's first table is zero, which serves as a sentinel.
#[inline(always)]
pub fn continuation_correction_index(pc: Piece, to: Square) -> u16 {
    (pc.idx() * SQUARE_NB + to.idx()) as u16
}

#[derive(Default, Debug)]
pub struct CorrectionBundle {
    pub pawn: AtomicStatsEntry<CORRECTION_HISTORY_LIMIT>,
    pub minor: AtomicStatsEntry<CORRECTION_HISTORY_LIMIT>,
    pub non_pawn_white: AtomicStatsEntry<CORRECTION_HISTORY_LIMIT>,
    pub non_pawn_black: AtomicStatsEntry<CORRECTION_HISTORY_LIMIT>,
}

impl FillShared for CorrectionBundle {
    #[inline(always)]
    unsafe fn fill_shared(&self, v: i32) {
        // SAFETY: nor the bundle's entries.
        unsafe {
            self.pawn.fill_shared(v);
            self.minor.fill_shared(v);
            self.non_pawn_white.fill_shared(v);
            self.non_pawn_black.fill_shared(v);
        }
    }
}

/// The correction histories of a pawn, minor piece or non-pawn key, for each
/// side to move.
pub type CorrectionEntry = [CorrectionBundle; COLOR_NB];

pub type TTMoveHistory = StatsEntry<8192>;

/// The histories of one thread, which it alone reads and writes.
pub struct WorkerHistories {
    pub main_history: ButterflyHistory,
    pub low_ply_history: LowPlyHistory,
    pub capture_history: CapturePieceToHistory,
    pub continuation_correction_history: ContinuationCorrectionHistory,
}

impl WorkerHistories {
    pub fn new() -> LargePageBox<WorkerHistories> {
        // SAFETY: zero bytes make entries.
        unsafe { LargePageBox::zeroed() }
    }
}

/// An array of tables shared by threads, whose size scales with their
/// number, and which they clear together.
pub struct DynStats<T> {
    data: LargePageArray<T>,
}

impl<T: FillShared> DynStats<T> {
    /// `size` tables of zeros.
    fn new(size: usize) -> DynStats<T> {
        // SAFETY: the tables hold statistics, which zero bytes make.
        DynStats {
            data: unsafe { LargePageArray::zeroed(size, false) },
        }
    }

    /// Sets all values in the share of thread `thread_idx` of `num_threads`
    /// to `value`.
    ///
    /// # Safety
    ///
    /// No other thread may read or write the share meanwhile.
    pub unsafe fn clear_range(&self, value: i32, thread_idx: usize, num_threads: usize) {
        let size = self.data.len();
        let start = (thread_idx as u64 * size as u64 / num_threads as u64) as usize;
        debug_assert!(start < size);
        let end = if thread_idx + 1 == num_threads {
            size
        } else {
            ((thread_idx as u64 + 1) * size as u64 / num_threads as u64) as usize
        };

        for table in &self.data[start..end] {
            // SAFETY: nor the share's tables, by the caller.
            unsafe { table.fill_shared(value) };
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Set of histories shared between threads. Their size is the next power
/// of two of the number of threads, to make the indexing more efficient.
pub struct SharedHistories {
    pub correction_history: DynStats<CorrectionEntry>,
    pub continuation_history: LargePageBox<ContinuationHistory>,
    pub pawn_history: DynStats<PawnHistoryEntry>,
    size_minus1: usize,
    pawn_hist_size_minus1: usize,
}

impl SharedHistories {
    pub fn new(thread_count: usize) -> SharedHistories {
        assert!(thread_count.is_power_of_two());
        let correction_history = DynStats::new(thread_count * CORRHIST_BASE_SIZE);
        let pawn_history = DynStats::new(thread_count * PAWN_HISTORY_BASE_SIZE);
        let size_minus1 = correction_history.len() - 1;
        let pawn_hist_size_minus1 = pawn_history.len() - 1;
        SharedHistories {
            correction_history,
            // SAFETY: zero bytes make entries.
            continuation_history: unsafe { LargePageBox::zeroed() },
            pawn_history,
            size_minus1,
            pawn_hist_size_minus1,
        }
    }

    #[inline(always)]
    pub fn continuation(&self, index: u16) -> &PieceToHistory {
        &self.continuation_history[index as usize & (CONTINUATION_HISTORY_NB - 1)]
    }

    #[inline(always)]
    pub fn pawn_entry(&self, pos: &Position<impl Variant>) -> &PawnHistoryEntry {
        self.pawn_entry_at(pos.pawn_key())
    }

    #[inline(always)]
    pub fn pawn_entry_at(&self, pawn_key: Key) -> &PawnHistoryEntry {
        let index = pawn_key as usize & self.pawn_hist_size_minus1;
        // SAFETY: the index is masked to the length minus one, a power of
        // two minus one.
        unsafe { self.pawn_history.data.get_unchecked(index) }
    }

    #[inline(always)]
    pub fn correction_entry(&self, key: Key) -> &CorrectionEntry {
        let index = key as usize & self.size_minus1;
        // SAFETY: as in pawn_entry_at.
        unsafe { self.correction_history.data.get_unchecked(index) }
    }

    #[inline(always)]
    pub fn pawn_correction_entry(&self, pos: &Position<impl Variant>) -> &CorrectionEntry {
        self.correction_entry(pos.pawn_key())
    }

    #[inline(always)]
    pub fn minor_piece_correction_entry(&self, pos: &Position<impl Variant>) -> &CorrectionEntry {
        self.correction_entry(pos.minor_piece_key())
    }

    #[inline(always)]
    pub fn nonpawn_correction_entry(
        &self,
        pos: &Position<impl Variant>,
        c: Color,
    ) -> &CorrectionEntry {
        self.correction_entry(pos.non_pawn_key(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_approach_the_bonus_within_the_limit() {
        let mut e = StatsEntry::<100>::default();
        e.update(50);
        assert_eq!(e.get(), 50);
        // 50 + 100 - 50 * 100 / 100
        e.update(1000);
        assert_eq!(e.get(), 100);
        // 100 - 30 - 100 * 30 / 100
        e.update(-30);
        assert_eq!(e.get(), 40);
        // -7 / 100 truncates towards zero, as in C++.
        let mut e = StatsEntry::<100>::default();
        e.set(-7);
        e.update(1);
        assert_eq!(e.get(), -6);

        let shared = AtomicStatsEntry::<100>::default();
        shared.update(-1000);
        assert_eq!(shared.get(), -100);
    }

    #[test]
    fn shared_tables_are_cleared_in_shares() {
        let histories = SharedHistories::new(2);
        assert_eq!(histories.correction_history.len(), 2 * CORRHIST_BASE_SIZE);
        for i in 0..3 {
            // SAFETY: one thread.
            unsafe { histories.pawn_history.clear_range(-1338, i, 3) };
        }
        assert!(
            histories
                .pawn_history
                .data
                .iter()
                .all(|t| { t.iter().all(|row| row.iter().all(|e| e.get() == -1338)) })
        );
        // SAFETY: one thread.
        unsafe { histories.correction_history.clear_range(-5, 0, 1) };
        let e = &histories.correction_entry(0x1234_5678_9abc_def0)[BLACK.idx()];
        assert_eq!(e.non_pawn_black.get(), -5);
    }
}
