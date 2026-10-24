//! The move picker, which hands the search the moves of a position one at a
//! time, those most likely to cut off first.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `movepick.h` and `movepick.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers
//! (see AUTHORS), under the GNU General Public License, version 3 or later.

use std::mem::MaybeUninit;

use crate::bitboard::Bitboard;
use crate::history::{LOW_PLY_HISTORY_SIZE, PieceToHistory, SharedHistories, WorkerHistories};
use crate::movegen::{CAPTURES, EVASIONS, GenType, MOVE_LIST_SIZE, MoveList, QUIETS};
use crate::position::Position;
use crate::types::*;
use crate::variant::Variant;

mod atomic;

pub use atomic::AtomicMovePicker;

// The stages of the move picker
type Stage = u8;

// generate main search moves
const MAIN_TT: Stage = 0;
const CAPTURE_INIT: Stage = 1;
const GOOD_CAPTURE: Stage = 2;
const QUIET_INIT: Stage = 3;
const GOOD_QUIET: Stage = 4;
const BAD_CAPTURE: Stage = 5;
const BAD_QUIET: Stage = 6;

// generate evasion moves
const EVASION_TT: Stage = 7;
const EVASION_INIT: Stage = 8;
const EVASION: Stage = 9;

// generate probcut moves
const PROBCUT_TT: Stage = 10;
const PROBCUT_INIT: Stage = 11;
const PROBCUT: Stage = 12;

// generate qsearch moves
const QSEARCH_TT: Stage = 13;
const QCAPTURE_INIT: Stage = 14;
const QCAPTURE: Stage = 15;

/// A move and its score for ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ExtMove {
    pub mv: Move,
    pub value: i32,
}

/// Sorts up to sixteen moves with AVX-512: their moves in one vector and
/// their values in another, in descending order of value, each move inserted
/// after those of at least its value, as the insertion sort does.
#[cfg(use_avx512)]
struct MoveSorter {
    sorted_values: std::arch::x86_64::__m512i,
    sorted_moves: std::arch::x86_64::__m512i,
}

#[cfg(use_avx512)]
impl MoveSorter {
    const MAX_ELEMENTS: usize = 16;

    #[inline(always)]
    fn new(first: ExtMove) -> MoveSorter {
        use std::arch::x86_64::*;
        // SAFETY: the build targets AVX-512.
        unsafe {
            let sorted_moves = _mm512_set1_epi32(first.mv.raw() as i32);
            // The lanes not yet written sort below any move.
            let sorted_values =
                _mm512_mask_set1_epi32(_mm512_set1_epi32(first.value), !1, i32::MIN);
            MoveSorter {
                sorted_values,
                sorted_moves,
            }
        }
    }

    #[inline(always)]
    fn insert(&mut self, m: ExtMove) {
        use std::arch::x86_64::*;
        debug_assert!(m.value != i32::MIN);
        // SAFETY: the build targets AVX-512.
        unsafe {
            let mv = _mm512_set1_epi32(m.mv.raw() as i32);
            let value = _mm512_set1_epi32(m.value);
            // Every lane but the insertion point, the first of a lesser value
            let expand = _mm512_cmplt_epi32_mask(self.sorted_values, value).wrapping_sub(1);
            self.sorted_values = _mm512_mask_expand_epi32(value, expand, self.sorted_values);
            self.sorted_moves = _mm512_mask_expand_epi32(mv, expand, self.sorted_moves);
        }
    }

    /// Sorts the moves of at least `limit`, from the first, until sixteen
    /// are sorted, as the insertion sort does. Returns the index of the last
    /// sorted and of the first move not yet seen, or None if there are no
    /// moves.
    #[inline(always)]
    fn sort_head(moves: &mut [ExtMove], limit: i32) -> Option<(usize, usize)> {
        let mut sorter = MoveSorter::new(*moves.first()?);
        let mut sorted_end = 0;
        let mut p = 1;
        while p < moves.len() {
            if moves[p].value >= limit {
                if sorted_end + 1 >= Self::MAX_ELEMENTS {
                    // The sorter is full
                    break;
                }
                sorter.insert(moves[p]);
                sorted_end += 1;
                moves[p] = moves[sorted_end];
            }
            p += 1;
        }
        sorter.write_sorted(&mut moves[..sorted_end + 1]);
        Some((sorted_end, p))
    }

    /// Writes the moves inserted, the first, sorted, over `moves`.
    #[inline(always)]
    fn write_sorted(&self, moves: &mut [ExtMove]) {
        use std::arch::x86_64::*;
        const { assert!(size_of::<ExtMove>() == 8) };
        let count = moves.len();
        assert!(count <= Self::MAX_ELEMENTS);
        let out = moves.as_mut_ptr().cast::<i64>();
        // The moves and values are kept apart, so each ExtMove is made of a
        // lane of each.
        let write = |offset: usize, indices: __m512i| {
            if count > offset {
                let mask = if count - offset >= 8 {
                    0xff
                } else {
                    (1u8 << (count - offset)) - 1
                };
                // SAFETY: the build targets AVX-512, and the lanes stored are
                // the ExtMoves of `moves` from `offset`.
                unsafe {
                    let ext_moves =
                        _mm512_permutex2var_epi32(self.sorted_moves, indices, self.sorted_values);
                    _mm512_mask_storeu_epi64(out.add(offset), mask, ext_moves);
                }
            }
        };
        // SAFETY: the build targets AVX-512.
        unsafe {
            write(
                0,
                _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23),
            );
            write(
                8,
                _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31),
            );
        }
    }
}

/// Sort moves in descending order up to and including a given limit. The
/// order of moves smaller than the limit is left unspecified.
fn partial_insertion_sort(moves: &mut [ExtMove], limit: i32) {
    // With AVX-512, the first sixteen moves of the limit are sorted in
    // vectors, and the insertion sort sorts the rest.
    #[cfg(use_avx512)]
    let Some((mut sorted_end, start)) = MoveSorter::sort_head(moves, limit) else {
        return;
    };
    #[cfg(not(use_avx512))]
    let (mut sorted_end, start) = (0, 1);

    for p in start..moves.len() {
        if moves[p].value >= limit {
            let tmp = moves[p];
            sorted_end += 1;
            moves[p] = moves[sorted_end];
            let mut q = sorted_end;
            while q != 0 && moves[q - 1].value < tmp.value {
                moves[q] = moves[q - 1];
                q -= 1;
            }
            moves[q] = tmp;
        }
    }
}

/// The MovePicker picks one legal move at a time from the current position.
/// next_move() emits one new legal move on every call, until there are no
/// moves left, when Move::none() is returned. In order to improve the
/// efficiency of the alpha-beta algorithm, MovePicker attempts to return the
/// moves which are most likely to get a cut-off first.
///
/// The position and the histories are passed to each call of next_move(),
/// which reads the histories as the search has updated them so far.
pub struct MovePicker<'a> {
    continuation_history: [&'a PieceToHistory; 6],
    tt_move: Move,
    cur: usize,
    end_cur: usize,
    end_bad_captures: usize,
    end_captures: usize,
    end_generated: usize,
    stage: Stage,
    threshold: i32,
    depth: Depth,
    ply: i32,
    skip_quiets: bool,
    // Initialized from the start up to end_generated, or end_cur where the
    // quiets are not generated
    moves: [MaybeUninit<ExtMove>; MOVE_LIST_SIZE],
}

impl<'a> MovePicker<'a> {
    /// A move picker for the main search and for the quiescence search.
    #[inline(always)]
    pub fn new(
        pos: &Position<impl Variant>,
        ttm: Move,
        d: Depth,
        ch: [&'a PieceToHistory; 6],
        pl: i32,
    ) -> MovePicker<'a> {
        let has_tt_move = ttm != Move::none() && pos.legal(ttm);
        let first_stage = if pos.checkers().any() {
            EVASION_TT
        } else if d > 0 {
            MAIN_TT
        } else {
            QSEARCH_TT
        };
        let stage = first_stage + !has_tt_move as Stage;

        MovePicker::with(stage, ttm, d, 0, ch, pl)
    }

    /// A move picker for ProbCut: we generate captures with Static Exchange
    /// Evaluation (SEE) greater than or equal to the given threshold.
    #[inline(always)]
    pub fn new_probcut(
        pos: &Position<impl Variant>,
        ttm: Move,
        th: i32,
        shared: &'a SharedHistories,
    ) -> MovePicker<'a> {
        debug_assert!(pos.checkers().is_empty());

        let has_tt_move = ttm != Move::none() && pos.capture_stage(ttm) && pos.legal(ttm);
        let stage = PROBCUT_TT + !has_tt_move as Stage;
        let ch = [shared.continuation(0); 6];

        MovePicker::with(stage, ttm, 0, th, ch, 0)
    }

    #[inline(always)]
    fn with(
        stage: Stage,
        tt_move: Move,
        depth: Depth,
        threshold: i32,
        continuation_history: [&'a PieceToHistory; 6],
        ply: i32,
    ) -> MovePicker<'a> {
        MovePicker {
            continuation_history,
            tt_move,
            cur: 0,
            end_cur: 0,
            end_bad_captures: 0,
            end_captures: 0,
            end_generated: 0,
            stage,
            threshold,
            depth,
            ply,
            skip_quiets: false,
            moves: [const { MaybeUninit::uninit() }; MOVE_LIST_SIZE],
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> ExtMove {
        debug_assert!(i < self.end_cur.max(self.end_generated));
        // SAFETY: the moves are read below the end of those written.
        unsafe { self.moves[i & (MOVE_LIST_SIZE - 1)].assume_init() }
    }

    #[inline(always)]
    fn set(&mut self, i: usize, m: ExtMove) {
        self.moves[i & (MOVE_LIST_SIZE - 1)] = MaybeUninit::new(m);
    }

    /// The moves from `begin` to `end`, which were all written.
    #[inline(always)]
    fn slice_mut(&mut self, begin: usize, end: usize) -> &mut [ExtMove] {
        assert!(begin <= end && end <= MOVE_LIST_SIZE);
        // SAFETY: the range is in the array and its moves were written, and
        // MaybeUninit<ExtMove> has ExtMove's layout.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.moves.as_mut_ptr().add(begin).cast::<ExtMove>(),
                end - begin,
            )
        }
    }

    /// Assigns a numerical value to each move in a list, used for sorting,
    /// and writes them from `cur`. Captures are ordered by Most Valuable
    /// Victim (MVV), preferring captures with a good history. Quiet moves are
    /// ordered using the history tables. Returns the end of the moves
    /// written.
    #[inline(always)]
    fn score<const T: GenType>(
        &mut self,
        ml: &MoveList,
        pos: &Position<impl Variant>,
        h: &WorkerHistories,
        shared: &SharedHistories,
    ) -> usize {
        const { assert!(T == CAPTURES || T == QUIETS || T == EVASIONS, "Wrong type") };

        let us = pos.side_to_move();

        let mut threat_by_lesser = [Bitboard::EMPTY; PIECE_TYPE_NB];
        if T == QUIETS {
            threat_by_lesser[PAWN.idx()] = Bitboard::EMPTY;
            threat_by_lesser[KNIGHT.idx()] = pos.threats_by(PAWN);
            threat_by_lesser[BISHOP.idx()] = threat_by_lesser[KNIGHT.idx()];
            threat_by_lesser[ROOK.idx()] =
                pos.threats_by(KNIGHT) | pos.threats_by(BISHOP) | threat_by_lesser[KNIGHT.idx()];
            threat_by_lesser[QUEEN.idx()] = pos.threats_by(ROOK) | threat_by_lesser[ROOK.idx()];
            threat_by_lesser[KING.idx()] = Bitboard::EMPTY;
        }

        let pawn_entry = shared.pawn_entry(pos);
        let main_history = &h.main_history[us.idx()];
        let ch = self.continuation_history;

        let mut it = self.cur;
        for &m in ml {
            let from = m.from_sq();
            let to = m.to_sq();
            let pc = pos.moved_piece(m);
            let pt = pc.type_of();
            let captured_piece = pos.piece_on(to);

            let value = if T == CAPTURES {
                h.capture_history[pc.idx()][to.idx()][captured_piece.type_of().idx()].get()
                    + 7 * captured_piece.value()
            } else if T == QUIETS {
                // histories
                let mut value = 2 * main_history[m.raw() as usize].get();
                value += 2 * pawn_entry[pc.idx()][to.idx()].get();
                value += ch[0][pc.idx()][to.idx()].get();
                value += ch[1][pc.idx()][to.idx()].get();
                value += ch[2][pc.idx()][to.idx()].get();
                value += ch[3][pc.idx()][to.idx()].get();
                value += ch[5][pc.idx()][to.idx()].get();

                // bonus for checks
                value += (pos.check_squares(pt).has(to) && pos.see_ge(m, -75)) as i32 * 16384;

                // penalty for moving to a square threatened by a lesser
                // piece or bonus for escaping an attack by a lesser piece.
                let threats = threat_by_lesser[pt.idx() & 7];
                let v = 20 * (threats.has(from) as i32 - threats.has(to) as i32);
                value += PIECE_VALUE[pt.idx()] * v;

                if self.ply < LOW_PLY_HISTORY_SIZE as i32 {
                    let ply = self.ply as usize;
                    value += 8 * h.low_ply_history[ply][m.raw() as usize].get() / (1 + self.ply);
                }

                value
            } else {
                // EVASIONS
                if pos.capture_stage(m) {
                    captured_piece.value() + (1 << 28)
                } else {
                    main_history[m.raw() as usize].get() + ch[0][pc.idx()][to.idx()].get()
                }
            };

            self.set(it, ExtMove { mv: m, value });
            it += 1;
        }
        it
    }

    // The stages that generate moves are not inlined into next_move(), so
    // that the calls which only pick a move do not prepare their generation.

    #[inline(never)]
    fn init_captures(
        &mut self,
        pos: &Position<impl Variant>,
        h: &WorkerHistories,
        shared: &SharedHistories,
    ) {
        let mut ml = MoveList::empty();
        ml.generate::<CAPTURES>(pos);

        self.cur = 0;
        self.end_bad_captures = 0;
        self.end_cur = self.score::<CAPTURES>(&ml, pos, h, shared);
        self.end_captures = self.end_cur;

        let (cur, end_cur) = (self.cur, self.end_cur);
        partial_insertion_sort(self.slice_mut(cur, end_cur), i32::MIN);
    }

    #[inline(never)]
    fn init_quiets(
        &mut self,
        pos: &Position<impl Variant>,
        h: &WorkerHistories,
        shared: &SharedHistories,
    ) {
        let mut ml = MoveList::empty();
        ml.generate::<QUIETS>(pos);

        self.end_cur = self.score::<QUIETS>(&ml, pos, h, shared);
        self.end_generated = self.end_cur;

        let (cur, end_cur, limit) = (self.cur, self.end_cur, -3560 * self.depth);
        partial_insertion_sort(self.slice_mut(cur, end_cur), limit);
    }

    #[inline(never)]
    fn init_evasions(
        &mut self,
        pos: &Position<impl Variant>,
        h: &WorkerHistories,
        shared: &SharedHistories,
    ) {
        let mut ml = MoveList::empty();
        ml.generate::<EVASIONS>(pos);

        self.cur = 0;
        self.end_cur = self.score::<EVASIONS>(&ml, pos, h, shared);
        self.end_generated = self.end_cur;

        let (cur, end_cur) = (self.cur, self.end_cur);
        partial_insertion_sort(self.slice_mut(cur, end_cur), i32::MIN);
    }

    /// Returns the next move satisfying a predicate function, or
    /// Move::none(). This never returns the TT move, as it was emitted before.
    #[inline(always)]
    fn select(&mut self, filter: impl Fn(ExtMove) -> bool) -> Move {
        while self.cur < self.end_cur {
            let m = self.get(self.cur);
            self.cur += 1;
            if m.mv != self.tt_move && filter(m) {
                return m.mv;
            }
        }
        Move::none()
    }

    /// This is the most important method of the MovePicker class. We emit
    /// one new legal move on every call until there are no more moves left,
    /// picking the move with the highest score from a list of generated
    /// moves.
    pub fn next_move(
        &mut self,
        pos: &Position<impl Variant>,
        h: &WorkerHistories,
        shared: &SharedHistories,
    ) -> Move {
        const GOOD_QUIET_THRESHOLD: i32 = -14000;

        // Each stage falls through to the next, as in Stockfish's switch:
        // the blocks below end where a stage's code starts, so that only
        // the stages that generate captures dispatch on the stage again.
        'dispatch: loop {
            'bad_quiet: {
                'bad_capture: {
                    'good_quiet: {
                        'quiet_init: {
                            'good_capture: {
                                'evasion: {
                                    match self.stage {
                                        MAIN_TT | EVASION_TT | QSEARCH_TT | PROBCUT_TT => {
                                            self.stage += 1;
                                            return self.tt_move;
                                        }

                                        CAPTURE_INIT | PROBCUT_INIT | QCAPTURE_INIT => {
                                            self.init_captures(pos, h, shared);
                                            self.stage += 1;
                                            continue 'dispatch;
                                        }

                                        GOOD_CAPTURE => break 'good_capture,
                                        QUIET_INIT => break 'quiet_init,
                                        GOOD_QUIET => break 'good_quiet,
                                        BAD_CAPTURE => break 'bad_capture,
                                        BAD_QUIET => break 'bad_quiet,

                                        EVASION_INIT => {
                                            self.init_evasions(pos, h, shared);
                                            self.stage = EVASION;
                                            break 'evasion;
                                        }

                                        EVASION | QCAPTURE => break 'evasion,

                                        PROBCUT => {
                                            let threshold = self.threshold;
                                            return self.select(|m| pos.see_ge(m.mv, threshold));
                                        }

                                        _ => unreachable!("no move picker stage {}", self.stage),
                                    }
                                }

                                // EVASION and QCAPTURE
                                return self.select(|_| true);
                            }

                            // GOOD_CAPTURE
                            while self.cur < self.end_cur {
                                let m = self.get(self.cur);
                                if m.mv != self.tt_move {
                                    if pos.see_ge(m.mv, -m.value / 18) {
                                        self.cur += 1;
                                        return m.mv;
                                    }
                                    let bad = self.get(self.end_bad_captures);
                                    self.set(self.end_bad_captures, m);
                                    self.set(self.cur, bad);
                                    self.end_bad_captures += 1;
                                }
                                self.cur += 1;
                            }

                            self.stage = QUIET_INIT;
                        }

                        // QUIET_INIT
                        if !self.skip_quiets {
                            self.init_quiets(pos, h, shared);
                        }

                        self.stage = GOOD_QUIET;
                    }

                    // GOOD_QUIET
                    if !self.skip_quiets {
                        let m = self.select(|m| m.value > GOOD_QUIET_THRESHOLD);
                        if m != Move::none() {
                            return m;
                        }
                    }

                    // Prepare the pointers to loop over the bad captures
                    self.cur = 0;
                    self.end_cur = self.end_bad_captures;

                    self.stage = BAD_CAPTURE;
                }

                // BAD_CAPTURE
                let m = self.select(|_| true);
                if m != Move::none() {
                    return m;
                }

                // Prepare the pointers to loop over quiets again
                self.cur = self.end_captures;
                self.end_cur = self.end_generated;

                self.stage = BAD_QUIET;
            }

            // BAD_QUIET
            if !self.skip_quiets {
                return self.select(|m| m.value <= GOOD_QUIET_THRESHOLD);
            }

            return Move::none();
        }
    }

    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ext(values: &[i32]) -> Vec<ExtMove> {
        values
            .iter()
            .enumerate()
            .map(|(i, &value)| ExtMove {
                mv: Move(i as u16 + 1),
                value,
            })
            .collect()
    }

    #[test]
    fn partial_insertion_sort_keeps_equal_moves_in_order() {
        let mut moves = ext(&[3, 5, -9, 5, 7, -20, 3]);
        partial_insertion_sort(&mut moves, -10);
        let order: Vec<(u16, i32)> = moves.iter().map(|m| (m.mv.0, m.value)).collect();
        // The first move always counts as sorted; the move under the limit
        // stays after the sorted ones.
        assert_eq!(
            order,
            [(5, 7), (2, 5), (4, 5), (1, 3), (7, 3), (3, -9), (6, -20)]
        );

        let mut moves = ext(&[-50, 4, -60, 8]);
        partial_insertion_sort(&mut moves, 0);
        let order: Vec<u16> = moves.iter().map(|m| m.mv.0).collect();
        assert_eq!(order, [4, 2, 1, 3]);
    }

    /// Stockfish's insertion sort, which the vectors of AVX-512 must match.
    fn insertion_sort(moves: &mut [ExtMove], limit: i32) {
        let mut sorted_end = 0;
        for p in 1..moves.len() {
            if moves[p].value >= limit {
                let tmp = moves[p];
                sorted_end += 1;
                moves[p] = moves[sorted_end];
                let mut q = sorted_end;
                while q != 0 && moves[q - 1].value < tmp.value {
                    moves[q] = moves[q - 1];
                    q -= 1;
                }
                moves[q] = tmp;
            }
        }
    }

    #[test]
    fn partial_insertion_sort_orders_as_the_insertion_sort() {
        let mut rng = crate::misc::Prng::new(1070372);
        for _ in 0..20000 {
            let len = (rng.rand64() % 70) as usize;
            // Few values, so that many moves tie
            let spread = [4, 40, 40000][(rng.rand64() % 3) as usize];
            let values: Vec<i32> = (0..len)
                .map(|_| (rng.rand64() % spread) as i32 - spread as i32 / 2)
                .collect();
            let limit = match rng.rand64() % 3 {
                0 => i32::MIN,
                _ => (rng.rand64() % spread) as i32 - spread as i32 / 2,
            };
            let mut moves = ext(&values);
            let mut expected = moves.clone();
            partial_insertion_sort(&mut moves, limit);
            insertion_sort(&mut expected, limit);
            assert_eq!(moves, expected, "values {values:?}, limit {limit}");
        }
    }
}
