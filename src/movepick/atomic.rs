//! Atomic's move picker, Atomic-Stockfish's, but for the legal moves that
//! varifish generates, where Atomic-Stockfish's picks pseudo-legal ones for
//! the search to test, and for the threats it weighs quiet moves by, which
//! the position keeps, seeing through the king as chess's do. It has no
//! evasion stage: a side in check picks captures, then quiet moves, as a side
//! out of check does.
//!
//! A translation into Rust, modified for varifish in 2026, of
//! Atomic-Stockfish's `movepick.cpp` (https://github.com/Belzedar94/Atomic-Stockfish),
//! a variant of Stockfish, a UCI chess playing engine derived from Glaurung
//! 2.1, Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under
//! the GNU General Public License, version 3 or later.

use super::*;
use crate::variant::Atomic;

/// The MovePicker of Atomic, which picks one legal move at a time.
/// next_move() emits one new legal move on every call, until there
/// are no moves left, when Move::none() is returned. In order to improve the
/// efficiency of the alpha-beta algorithm, it attempts to return the moves
/// which are most likely to get a cut-off first.
///
/// The position and the histories are passed to each call of next_move(),
/// which reads the histories as the search has updated them so far.
pub struct AtomicMovePicker<'a> {
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
    quiet_checks_only: bool,
    // Initialized from the start up to end_generated, or end_cur where the
    // quiets are not generated
    moves: [MaybeUninit<ExtMove>; MOVE_LIST_SIZE],
}

impl<'a> AtomicMovePicker<'a> {
    /// A move picker for the main search and for the quiescence search.
    #[inline(always)]
    pub fn new(
        pos: &Position<Atomic>,
        ttm: Move,
        d: Depth,
        ch: [&'a PieceToHistory; 6],
        pl: i32,
    ) -> AtomicMovePicker<'a> {
        let has_tt_move = ttm != Move::none() && pos.atomic_legal(ttm);
        let first_stage = if d > 0 { MAIN_TT } else { QSEARCH_TT };
        let stage = first_stage + !has_tt_move as Stage;

        AtomicMovePicker::with(stage, ttm, d, 0, ch, pl)
    }

    /// A move picker for ProbCut: we generate captures that win at once or
    /// whose Static Exchange Evaluation (SEE) is at least the threshold.
    #[inline(always)]
    pub fn new_probcut(
        pos: &Position<Atomic>,
        ttm: Move,
        th: i32,
        shared: &'a SharedHistories,
    ) -> AtomicMovePicker<'a> {
        let has_tt_move = ttm != Move::none() && pos.capture_stage(ttm) && pos.atomic_legal(ttm);
        let stage = PROBCUT_TT + !has_tt_move as Stage;
        let ch = [shared.continuation(0); 6];

        AtomicMovePicker::with(stage, ttm, 0, th, ch, 0)
    }

    #[inline(always)]
    fn with(
        stage: Stage,
        tt_move: Move,
        depth: Depth,
        threshold: i32,
        continuation_history: [&'a PieceToHistory; 6],
        ply: i32,
    ) -> AtomicMovePicker<'a> {
        AtomicMovePicker {
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
            quiet_checks_only: false,
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
        pos: &Position<Atomic>,
        h: &WorkerHistories,
        shared: &SharedHistories,
    ) -> usize {
        const { assert!(T == CAPTURES || T == QUIETS, "Wrong type") };

        let us = pos.side_to_move();

        // The squares the opponent's lesser pieces attack, which the position
        // keeps, seeing through our king.
        let mut threat_by_lesser = [Bitboard::EMPTY; PIECE_TYPE_NB];
        if T == QUIETS {
            let by_pawns = pos.threats_by(PAWN);
            let by_minors = pos.threats_by(KNIGHT) | pos.threats_by(BISHOP) | by_pawns;
            threat_by_lesser[KNIGHT.idx()] = by_pawns;
            threat_by_lesser[BISHOP.idx()] = by_pawns;
            threat_by_lesser[ROOK.idx()] = by_minors;
            threat_by_lesser[QUEEN.idx()] = pos.threats_by(ROOK) | by_minors;
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
            } else {
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
        pos: &Position<Atomic>,
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
        pos: &Position<Atomic>,
        h: &WorkerHistories,
        shared: &SharedHistories,
    ) {
        let mut ml = MoveList::empty();
        ml.generate::<QUIETS>(pos);

        if self.skip_quiets {
            // Pruning the quiet moves keeps their checks, which Atomic-Stockfish
            // picks unscored, in the order generated.
            self.quiet_checks_only = true;
            let mut end = self.cur;
            for &m in &ml {
                if pos.gives_check(m) {
                    self.set(end, ExtMove { mv: m, value: 0 });
                    end += 1;
                }
            }
            self.end_generated = end;
            self.end_cur = end;
        } else {
            self.end_cur = self.score::<QUIETS>(&ml, pos, h, shared);
            self.end_generated = self.end_cur;

            let (cur, end_cur, limit) = (self.cur, self.end_cur, -3560 * self.depth);
            partial_insertion_sort(self.slice_mut(cur, end_cur), limit);
        }
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
    /// one new legal move on every call until there are no more moves
    /// left, picking the move with the highest score from a list of generated
    /// moves.
    pub fn next_move(
        &mut self,
        pos: &Position<Atomic>,
        h: &WorkerHistories,
        shared: &SharedHistories,
    ) -> Move {
        const GOOD_QUIET_THRESHOLD: i32 = -14000;

        // Each stage falls through to the next, as in chess's move picker.
        'dispatch: loop {
            'bad_quiet: {
                'bad_capture: {
                    'good_quiet: {
                        'quiet_init: {
                            'good_capture: {
                                match self.stage {
                                    MAIN_TT | QSEARCH_TT | PROBCUT_TT => {
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

                                    QCAPTURE => return self.select(|_| true),

                                    PROBCUT => {
                                        let threshold = self.threshold;
                                        return self.select(|m| pos.see_ge(m.mv, threshold));
                                    }

                                    _ => unreachable!("no Atomic move picker stage {}", self.stage),
                                }
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
                        self.init_quiets(pos, h, shared);
                        self.stage = GOOD_QUIET;
                    }

                    // GOOD_QUIET
                    let (checks_only, skip_quiets) = (self.quiet_checks_only, self.skip_quiets);
                    let m = self.select(|m| {
                        checks_only
                            || (m.value > GOOD_QUIET_THRESHOLD
                                && (!skip_quiets || pos.gives_check(m.mv)))
                    });
                    if m != Move::none() {
                        return m;
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
            if self.quiet_checks_only {
                return Move::none();
            }

            let skip_quiets = self.skip_quiets;
            return self.select(|m| {
                m.value <= GOOD_QUIET_THRESHOLD && (!skip_quiets || pos.gives_check(m.mv))
            });
        }
    }

    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }
}
