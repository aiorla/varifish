//! Atomic's search, Atomic-Stockfish's: the main search and the quiescence
//! search with the pruning, reductions and extensions that its SPSA
//! campaigns tuned for Atomic, over the legal moves of Atomic's move picker.
//! Atomic has no evasion stage: a side in check searches all its moves, in
//! the quiescence search too. Unlike Atomic-Stockfish's search, it need not
//! test the legality of the moves the move picker gives.
//!
//! A translation into Rust, modified for varifish in 2026, of
//! Atomic-Stockfish's `search.h` and `search.cpp`
//! (https://github.com/Belzedar94/Atomic-Stockfish), a variant of Stockfish,
//! a UCI chess playing engine derived from Glaurung 2.1, Copyright (C)
//! 2004-2026 The Stockfish developers (see AUTHORS), under the GNU General
//! Public License, version 3 or later.

use super::*;
use crate::attacks::pseudo_attacks;
use crate::bitboard::square_bb;
use crate::movepick::AtomicMovePicker;
use crate::variant::Atomic;

// Search parameters that Atomic-Stockfish's SPSA campaigns tuned

/// Fairy-Stockfish's move-count pruning for blast variants
const ATOMIC_MCP_BASE: i32 = 7;
/// A null move reduction a ply shallower than Stockfish's, since Atomic's
/// threats make a null move's cutoff less trustworthy
const ATOMIC_NMP_BASE: Depth = 6;
const ATOMIC_NMP_DEPTH_DIV: Depth = 4;
/// Atomic's futility margins out of 64 of Stockfish's: material swings by
/// whole clusters of pieces
const ATOMIC_FUTILITY_SCALE: i32 = 128;
const ATOMIC_CAPT_FUT_BASE: i32 = 227;
const ATOMIC_CAPT_FUT_LMR_MULT: i32 = 244;
const QS_FUTILITY_BASE: i32 = 345;
const SINGULAR_DEPTH_MIN: Depth = 8;
const SINGULAR_MARGIN_BASE: i32 = 65;
const SINGULAR_MARGIN_TT_PV: i32 = 83;
const SINGULAR_MARGIN_DIV: i32 = 48;
/// The scale of the reductions table, out of 128
pub(super) const LMR_LOG_SCALE: i32 = 2736;
/// Atomic-Stockfish's 1049, less the 34 by which Stockfish's 885813b0
/// lowered its offset when it dropped the reductions' window term
const LMR_BASE_OFFSET: i32 = 1015;

/// Atomic-Stockfish's time management, which forked Stockfish's before its
/// latest tuning
pub(super) const TIME: TimeConstants = TimeConstants {
    falling_eval: (11.87, 2.21, 1.0),
    falling_eval_range: (0.572, 1.708),
    time_reduction: (5.0, 18.0, 0.65, 1.55),
    time_reduction_range: (0.65, 1.55),
    reduction: (1.48, 2.157),
    best_move_instability: (1.096, 2.29),
    high_best_move_effort: (79219, 101822, 0.924, 0.71),
    high_best_move_effort_range: (0.71, 0.924),
    single_move_time: 561.7,
};

/// The move count from which quiet moves are pruned: Fairy-Stockfish's
/// formula for blast variants, since Atomic's forcing captures justify
/// pruning them earlier than chess does.
#[inline(always)]
fn move_count_pruning_threshold(improving: bool, depth: Depth) -> i32 {
    (ATOMIC_MCP_BASE + depth * depth) / (3 - improving as i32)
}

#[inline(always)]
fn null_move_reduction(depth: Depth) -> Depth {
    ATOMIC_NMP_BASE + depth / ATOMIC_NMP_DEPTH_DIV
}

/// Whether futility may prune a capture by the value of the captured piece
/// alone: a normal capture whose explosion takes no other piece but pawns,
/// of either color, since the bound does not price that bycatch.
#[inline(always)]
fn capture_futility_eligible(pos: &Position<Atomic>, m: Move) -> bool {
    if m.type_of() != NORMAL || !pos.capture_stage(m) {
        return false;
    }

    let non_pawn_bycatch = pseudo_attacks(KING, m.to_sq())
        & pos.pieces()
        & !pos.pieces_pt(PAWN)
        & !square_bb(m.from_sq());
    non_pawn_bycatch.is_empty()
}

impl Worker<Atomic> {
    /// Searches the root to `depth` in the window (alpha, beta).
    pub(crate) fn atomic_search_root(
        &mut self,
        sh: &SharedState,
        pos: &mut Position<Atomic>,
        alpha: Value,
        beta: Value,
        depth: Depth,
    ) -> Value {
        self.atomic_search::<ROOT>(sh, pos, ROOT_SS, alpha, beta, depth, false)
    }

    /// Main search function for both PV and non-PV nodes
    #[allow(clippy::too_many_arguments)]
    fn atomic_search<const NT: NodeType>(
        &mut self,
        sh: &SharedState,
        pos: &mut Position<Atomic>,
        ss: usize,
        mut alpha: Value,
        mut beta: Value,
        mut depth: Depth,
        cut_node: bool,
    ) -> Value {
        let pv_node = NT != NON_PV;
        let root_node = NT == ROOT;
        let all_node = !(pv_node || cut_node);

        // Dive into quiescence search when the depth reaches zero
        if depth <= 0 {
            return if pv_node {
                self.atomic_qsearch::<PV>(sh, pos, ss, alpha, beta)
            } else {
                self.atomic_qsearch::<NON_PV>(sh, pos, ss, alpha, beta)
            };
        }

        // Limit the depth if extensions made it too large
        depth = depth.min(MAX_PLY - 1);

        let ply = ply_of(ss);

        // Check if we have an upcoming move that draws by repetition
        if !root_node && alpha < VALUE_DRAW && pos.upcoming_repetition(ply) {
            alpha = value_draw(self.nodes());
            if alpha >= beta {
                return alpha;
            }
        }

        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || alpha == beta - 1);
        debug_assert!(0 < depth && depth < MAX_PLY);
        debug_assert!(!(pv_node && cut_node));

        let mut captures_searched = SearchedList::new();
        let mut quiets_searched = SearchedList::new();

        // Step 1. Initialize node
        let us = pos.side_to_move();
        self.stack[ss].in_check = pos.in_check();
        let prior_capture = pos.captured_piece().is_some();
        self.stack[ss].move_count = 0;
        let mut best_value = -VALUE_INFINITE;
        let mut max_value = VALUE_INFINITE;

        self.stack[ss].follow_pv = root_node
            || (self.stack[ss - 1].follow_pv
                && ((ply - 1) as usize) < self.last_iteration_idx_pv.len()
                && self.stack[ss - 1].current_move
                    == self.last_iteration_idx_pv[(ply - 1) as usize]);

        // Check for the available remaining time
        if let Some(manager) = self.manager.as_deref_mut() {
            manager.check_time(&self.limits, sh);
        }

        // Used to send selDepth info to GUI (selDepth counts from 1, ply from
        // 0)
        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }

        if !root_node {
            // A side whose king has exploded has lost, and one that has
            // exploded the other king has won.
            if !pos.has_king(us) {
                return mated_in(ply);
            }
            if !pos.has_king(!us) {
                return mate_in(ply);
            }

            // Step 2. Check for aborted search and immediate draw
            if sh.stop.load(Relaxed) || pos.is_draw(ply) || ply >= MAX_PLY {
                return if ply >= MAX_PLY && !self.stack[ss].in_check {
                    self.evaluate(sh, pos)
                } else {
                    value_draw(self.nodes())
                };
            }

            // Step 3. Mate distance pruning. Even if we mate at the next move
            // our score would be at best mate_in(ss->ply + 1), but if alpha
            // is already bigger because a shorter mate was found upward in
            // the tree then there is no need to search because we will
            // never beat the current alpha. Same logic but with reversed
            // signs apply also in the opposite condition of being mated
            // instead of giving mate. In this case, return a fail-high
            // score.
            alpha = mated_in(ply).max(alpha);
            beta = mate_in(ply + 1).min(beta);
            if alpha >= beta {
                return alpha;
            }

            // A side that can explode the other king wins at once.
            let win = pos.atomic_winning_capture();
            if win != Move::none() {
                if pv_node {
                    self.pv_table.update(ply, win, false);
                }
                return mate_in(ply + 1);
            }
        }

        debug_assert!((0..MAX_PLY).contains(&ply));

        let prev_move = self.stack[ss - 1].current_move;
        let prev_sq = if prev_move.is_ok() {
            prev_move.to_sq()
        } else {
            Square::NONE
        };
        let mut best_move = Move::none();
        let prior_reduction = self.stack[ss - 1].reduction;
        self.stack[ss - 1].reduction = 0;
        self.stack[ss].stat_score = 0;
        self.stack[ss + 2].cutoff_cnt = 0;
        self.stack[ss + 1].prior_nmp_fail_high = 0;

        let correction_value = self.atomic_correction_value(sh, pos, ss);

        // Step 4. Transposition table lookup
        let excluded_move = self.stack[ss].excluded_move;
        let pos_key = pos.key();
        let (tt_hit, mut tt_data, tt_writer) = sh.tt.probe(pos_key);

        // Need further processing of the saved data
        self.stack[ss].tt_hit = tt_hit;
        tt_data.mv = if root_node {
            self.root_moves[self.pv_idx].pv[0]
        } else if tt_hit {
            tt_data.mv
        } else {
            Move::none()
        };
        tt_data.value = if tt_hit {
            value_from_tt(tt_data.value, ply, pos.rule50_count())
        } else {
            VALUE_NONE
        };
        self.stack[ss].tt_pv = if excluded_move != Move::none() {
            self.stack[ss].tt_pv
        } else {
            pv_node || (tt_hit && tt_data.is_pv)
        };
        let tt_capture = tt_data.mv != Move::none() && pos.capture_stage(tt_data.mv);

        // Step 5. Static evaluation of the position
        let mut unadjusted_static_eval = VALUE_NONE;
        let mut eval;

        // Skip early pruning when in check
        if self.stack[ss].in_check {
            eval = self.stack[ss - 2].static_eval;
            self.stack[ss].static_eval = eval;
        } else if excluded_move != Move::none() {
            eval = self.stack[ss].static_eval;
            unadjusted_static_eval = eval;
        } else if self.stack[ss].tt_hit {
            // Never assume anything about values stored in TT
            unadjusted_static_eval = tt_data.eval;
            if !is_valid(unadjusted_static_eval) {
                unadjusted_static_eval = self.evaluate(sh, pos);
            }

            eval = to_corrected_static_eval(unadjusted_static_eval, correction_value);
            self.stack[ss].static_eval = eval;

            // ttValue can be used as a better position evaluation
            if is_valid(tt_data.value)
                && (tt_data.bound
                    & if tt_data.value > eval {
                        BOUND_LOWER
                    } else {
                        BOUND_UPPER
                    })
                    != 0
            {
                eval = tt_data.value;
            }
        } else {
            unadjusted_static_eval = self.evaluate(sh, pos);
            eval = to_corrected_static_eval(unadjusted_static_eval, correction_value);
            self.stack[ss].static_eval = eval;

            // Static evaluation is saved as it was before adjustment by
            // correction history
            tt_writer.write(
                pos_key,
                VALUE_NONE,
                self.stack[ss].tt_pv,
                BOUND_NONE,
                DEPTH_UNSEARCHED,
                Move::none(),
                unadjusted_static_eval,
                sh.tt.generation(),
            );
        }

        // Set up the improving flag, which is true if current static
        // evaluation is bigger than the previous static evaluation at our
        // turn (if we were in check at our previous move we go back until we
        // weren't in check) and is false otherwise. The improving flag is
        // used in various pruning heuristics. Similarly, opponentWorsening is
        // true if our static evaluation is better for us than at the last
        // ply.
        let mut improving = self.stack[ss].static_eval > self.stack[ss - 2].static_eval;
        let opponent_worsening = self.stack[ss].static_eval > -self.stack[ss - 1].static_eval;

        // Hindsight adjustment of reductions based on static evaluation
        // difference.
        if prior_reduction >= 3 && !opponent_worsening {
            depth += 1;
        }
        if prior_reduction >= 2
            && depth >= 2
            && self.stack[ss].static_eval + self.stack[ss - 1].static_eval > 173
        {
            depth -= 1;
        }

        // At non-PV nodes we check for an early TT cutoff
        if !pv_node
            && excluded_move == Move::none()
            && tt_data.depth > depth - (tt_data.value <= beta) as i32
            // Can happen when !ttHit or when access race in probe()
            && is_valid(tt_data.value)
            && (tt_data.bound
                & if tt_data.value >= beta {
                    BOUND_LOWER
                } else {
                    BOUND_UPPER
                })
                != 0
            && (cut_node == (tt_data.value >= beta) || depth > 4)
        {
            // If ttMove is quiet, update move sorting heuristics on TT hit
            if tt_data.mv != Move::none() && tt_data.value >= beta {
                // Bonus for a quiet ttMove that fails high
                if !tt_capture {
                    self.atomic_update_quiet_histories(sh, pos, ss, tt_data.mv, 131 * depth);
                }

                // Extra penalty for early quiet moves of the previous ply
                if prev_sq != Square::NONE && self.stack[ss - 1].move_count < 4 && !prior_capture {
                    self.atomic_update_continuation_histories(
                        sh,
                        ss - 1,
                        pos.piece_on(prev_sq),
                        prev_sq,
                        -2187,
                    );
                }
            }

            // Partial workaround for the graph history interaction problem
            // For high rule50 counts don't produce transposition table
            // cutoffs.
            if pos.rule50_count() < 96 {
                if depth >= 7
                    && tt_data.mv != Move::none()
                    && pos.atomic_legal(tt_data.mv)
                    && !is_decisive(tt_data.value)
                {
                    pos.make_move(tt_data.mv, &mut self.scratch);
                    let next_pos_key = pos.key();
                    let (_, tt_data_next, _) = sh.tt.probe(next_pos_key);
                    pos.undo_move(tt_data.mv);

                    // Check that the ttValue after the tt move would also
                    // trigger a cutoff
                    if !is_valid(tt_data_next.value) {
                        return tt_data.value;
                    }

                    if (tt_data.value >= beta) == (-tt_data_next.value >= beta) {
                        return tt_data.value;
                    }
                } else {
                    return tt_data.value;
                }
            }
        }
        // No cutoff, but why? Does the stored inexact value mismatch our
        // aspiration window?
        else if !pv_node
            && excluded_move == Move::none()
            && tt_data.depth > depth - (tt_data.value <= beta) as i32
            && is_valid(tt_data.value)
            && tt_data.bound != BOUND_EXACT
            && (tt_data.bound
                & if tt_data.value >= beta {
                    BOUND_UPPER
                } else {
                    BOUND_LOWER
                })
                != 0
            && depth > 5
        {
            // If a window-bound mismatch is the only reason cutoff failed,
            // penalize the now-useless tte
            tt_writer.penalize(1);
        }

        // Step 6. Tablebases probe
        if !root_node && excluded_move == Move::none() && self.tb_config.cardinality != 0 {
            let pieces_count = pos.count_both(ALL_PIECES);

            if pieces_count <= self.tb_config.cardinality
                && (pieces_count < self.tb_config.cardinality
                    || depth >= self.tb_config.probe_depth)
                && pos.rule50_count() == 0
                && !pos.can_castle(ANY_CASTLING)
            {
                let mut err = ProbeState::Ok;
                let wdl = Atomic::tables(&sh.tablebases).probe_wdl(pos, &mut err);

                // Force check of time on the next occasion
                if let Some(manager) = self.manager.as_deref_mut() {
                    manager.calls_cnt = 0;
                }

                if err != ProbeState::Fail {
                    increment(&self.counters.tb_hits);

                    let draw_score = i32::from(self.tb_config.use_rule50);

                    let tb_value = VALUE_TB - ply;

                    // Use the range VALUE_TB to VALUE_TB_WIN_IN_MAX_PLY to
                    // score
                    let value = if wdl < -draw_score {
                        -tb_value
                    } else if wdl > draw_score {
                        tb_value
                    } else {
                        VALUE_DRAW + 2 * wdl * draw_score
                    };

                    let b = if wdl < -draw_score {
                        BOUND_UPPER
                    } else if wdl > draw_score {
                        BOUND_LOWER
                    } else {
                        BOUND_EXACT
                    };

                    if b == BOUND_EXACT
                        || (if b == BOUND_LOWER {
                            value >= beta
                        } else {
                            value <= alpha
                        })
                    {
                        tt_writer.write(
                            pos_key,
                            value_to_tt(value, ply),
                            self.stack[ss].tt_pv,
                            b,
                            (MAX_PLY - 1).min(depth + 6),
                            Move::none(),
                            VALUE_NONE,
                            sh.tt.generation(),
                        );

                        return value;
                    }

                    if pv_node {
                        if b == BOUND_LOWER {
                            best_value = value;
                            alpha = alpha.max(best_value);
                        } else {
                            max_value = value;
                        }
                    }
                }
            }
        }

        if !self.stack[ss].in_check {
            // Use static evaluation difference to improve quiet move ordering
            if prev_move.is_ok() && !self.stack[ss - 1].in_check && !prior_capture {
                let eval_diff = (-(self.stack[ss - 1].static_eval + self.stack[ss].static_eval))
                    .clamp(-183, 180)
                    + 62;
                self.hist.main_history[(!us).idx()][prev_move.raw() as usize]
                    .update(eval_diff * 10);
                if !tt_hit
                    && pos.piece_on(prev_sq).type_of() != PAWN
                    && prev_move.type_of() != PROMOTION
                {
                    sh.histories.pawn_entry(pos)[pos.piece_on(prev_sq).idx()][prev_sq.idx()]
                        .update(eval_diff * 13);
                }
            }

            // Step 7. Razoring
            // If eval is really low, skip search entirely and return the
            // qsearch value. The margin is master's at Atomic's futility
            // scale.
            if all_node
                && eval < alpha - 342 * depth * ATOMIC_FUTILITY_SCALE / 64
                && !self.seek_mate
            {
                return self.atomic_qsearch::<NON_PV>(sh, pos, ss, alpha, beta);
            }

            // Step 8. Futility pruning: child node
            // The depth condition is important for mate finding. It should
            // NOT be tuned. A blast check is a check: the static evaluation
            // does not bound a node where every continuation can be answered
            // by exploding our king.
            if !self.stack[ss].tt_pv
                && depth < if self.seek_mate { 6 } else { 17 }
                && eval >= beta
                && (tt_data.mv == Move::none() || tt_capture)
                && !is_loss(beta)
                && !is_win(eval)
                && !pos.atomic_under_blast_threat()
            {
                let mut futility_mult = (40 + depth * 4).min(80);
                futility_mult -= 20 * (!self.stack[ss].tt_hit) as i32;
                futility_mult = futility_mult * ATOMIC_FUTILITY_SCALE / 64;

                let futility_margin = futility_mult * depth
                    - (2934 * improving as i32 + 343 * opponent_worsening as i32) * futility_mult
                        / 1024
                    + correction_value.abs() / 182069;

                if eval - futility_margin >= beta {
                    return (716 * beta + 308 * eval) / 1024;
                }
            }

            // Step 9. Null move search with verification search
            if cut_node
                && self.stack[ss].static_eval + 50 * self.stack[ss].prior_nmp_fail_high
                    >= beta - 14 * depth - 45 * improving as i32 + 374
                && excluded_move == Move::none()
                && pos.non_pawn_material(us) != 0
                && ply >= self.nmp_min_ply
                && beta >= -2000
            {
                debug_assert!(self.stack[ss - 1].current_move != Move::null());

                // Atomic-Stockfish's reduction, and a ply more for each 256
                // of static evaluation over beta
                let mut r =
                    null_move_reduction(depth) + ((self.stack[ss].static_eval - beta) / 256).max(0);
                r -= self.seek_mate as i32 * r / 2;
                self.do_null_move(pos, ss);

                let null_value = -self.atomic_search::<NON_PV>(
                    sh,
                    pos,
                    ss + 1,
                    -beta,
                    -beta + 1,
                    depth - r,
                    false,
                );

                self.undo_null_move(pos);

                // Do not return unproven mate or TB scores
                if null_value >= beta && !is_win(null_value) {
                    if self.nmp_min_ply != 0 || depth < 16 {
                        self.stack[ss].prior_nmp_fail_high += 1;
                        return null_value;
                    }

                    // Recursive verification is not allowed
                    debug_assert!(self.nmp_min_ply == 0);

                    // Do verification search at high depths, with null move
                    // pruning disabled until ply exceeds nmpMinPly.
                    self.nmp_min_ply = ply + 3 * (depth - r) / 4;

                    let v =
                        self.atomic_search::<NON_PV>(sh, pos, ss, beta - 1, beta, depth - r, false);

                    self.nmp_min_ply = 0;

                    if v >= beta {
                        self.stack[ss].prior_nmp_fail_high += 1;
                        return null_value;
                    }
                }
            }

            improving |= self.stack[ss].static_eval >= beta;

            // Step 10. Internal iterative reductions
            // At sufficient depth, reduce depth for PV/Cut nodes without a
            // TTMove. (*Scaler) Making IIR more aggressive scales poorly.
            if !self.stack[ss].follow_pv && !all_node && depth >= 6 && tt_data.mv == Move::none() {
                depth -= 1;
            }

            // Step 11. ProbCut
            // If we have a good enough capture (or queen promotion) and a
            // reduced search returns a value much above beta, we can
            // (almost) safely prune the previous move.
            let prob_cut_beta = beta + 214 - 59 * improving as i32;
            if depth >= 3
                && !is_decisive(beta)
                // If value from transposition table is lower than
                // probCutBeta, don't attempt probCut there
                && !(is_valid(tt_data.value) && tt_data.value < prob_cut_beta)
            {
                debug_assert!(prob_cut_beta < VALUE_INFINITE && prob_cut_beta > beta);

                let mut mp = AtomicMovePicker::new_probcut(
                    pos,
                    tt_data.mv,
                    prob_cut_beta - self.stack[ss].static_eval,
                    &sh.histories,
                );
                let prob_cut_depth = depth - 4 - improving as i32;

                loop {
                    let mv = mp.next_move(pos, &self.hist, &sh.histories);
                    if mv == Move::none() {
                        break;
                    }

                    debug_assert!(mv.is_ok());

                    if mv == excluded_move {
                        continue;
                    }

                    let capture = pos.capture_stage(mv);
                    debug_assert!(capture);

                    self.do_move(sh, pos, mv, pos.gives_check(mv), capture, ss);

                    // Perform a preliminary qsearch to verify that the move
                    // holds
                    let mut value = -self.atomic_qsearch::<NON_PV>(
                        sh,
                        pos,
                        ss + 1,
                        -prob_cut_beta,
                        -prob_cut_beta + 1,
                    );

                    // If the qsearch held, perform the regular search
                    if value >= prob_cut_beta && prob_cut_depth > 0 {
                        value = -self.atomic_search::<NON_PV>(
                            sh,
                            pos,
                            ss + 1,
                            -prob_cut_beta,
                            -prob_cut_beta + 1,
                            prob_cut_depth,
                            !cut_node,
                        );
                    }

                    self.undo_move(pos, mv);

                    if value >= prob_cut_beta {
                        // Save ProbCut data into transposition table
                        tt_writer.write(
                            pos_key,
                            value_to_tt(value, ply),
                            self.stack[ss].tt_pv,
                            BOUND_LOWER,
                            prob_cut_depth + 1,
                            mv,
                            unadjusted_static_eval,
                            sh.tt.generation(),
                        );

                        if !is_decisive(value) {
                            return value - (prob_cut_beta - beta);
                        }
                    }
                }
            }
        }

        // moves_loop: When in check, search starts here

        // Step 12. A small Probcut idea
        let prob_cut_beta = beta + 428;
        if (tt_data.bound & BOUND_LOWER) != 0
            && tt_data.depth >= depth - 4
            && tt_data.value >= prob_cut_beta
            && !is_decisive(beta)
            && is_valid(tt_data.value)
            && !is_decisive(tt_data.value)
            && !self.seek_mate
        {
            return prob_cut_beta;
        }

        let cont_hist: [&PieceToHistory; 6] = [
            sh.histories
                .continuation(self.stack[ss - 1].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 2].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 3].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 4].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 5].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 6].continuation_history),
        ];

        let mut mp = AtomicMovePicker::new(pos, tt_data.mv, depth, cont_hist, ply);

        let mut value = best_value;

        let mut move_count = 0;

        // Step 13. Loop through all legal moves until no moves remain
        // or a beta cutoff occurs.
        loop {
            let mv = mp.next_move(pos, &self.hist, &sh.histories);
            if mv == Move::none() {
                break;
            }

            debug_assert!(mv.is_ok());

            if mv == excluded_move {
                continue;
            }

            // At root obey the "searchmoves" option and skip moves not listed
            // in Root Move List. In MultiPV mode we also skip PV moves that
            // have been already searched and those of lower "TB rank" if we
            // are in a TB root position.
            if root_node
                && !self.root_moves[self.pv_idx..self.pv_last]
                    .iter()
                    .any(|rm| rm.pv[0] == mv)
            {
                continue;
            }

            move_count += 1;
            self.stack[ss].move_count = move_count;

            if root_node && self.manager.is_some() && self.nodes() > NODES_LIMIT_OUTPUT {
                let manager = self.manager.as_deref().expect("the main thread");
                (manager.updates.on_iter)(&InfoIteration {
                    depth,
                    currmove: &move_to_uci(mv, pos.is_chess960()),
                    currmovenumber: move_count as usize + self.pv_idx,
                });
            }

            // Whether the child's principal variation is collected, which
            // Atomic-Stockfish tells by its stack's pointer to it
            let mut child_pv = false;

            let mut extension = 0;
            let capture = pos.capture_stage(mv);
            let moved_piece = pos.moved_piece(mv);
            let gives_check = pos.gives_check(mv);

            // Calculate new depth for this move
            let mut new_depth = depth - 1;

            let mut r = self.atomic_reduction(improving, depth, move_count);

            // Decrease reduction for PvNodes
            if pv_node {
                r -= 512;
            }

            // Increase reduction for ttPv nodes (*Scaler)
            // Larger values scale well
            if self.stack[ss].tt_pv {
                r += 1006;
            }

            // Step 14. Pruning at shallow depths.
            // Depth conditions are important for mate finding.
            if !root_node && pos.non_pawn_material(us) != 0 && !is_loss(best_value) {
                // Skip quiet moves if movecount exceeds our threshold
                if move_count >= move_count_pruning_threshold(improving, depth) {
                    mp.skip_quiet_moves();
                }

                // Reduced depth of the next LMR search
                let mut lmr_depth = new_depth - r / 1024;

                if capture || gives_check {
                    let captured_piece = pos.piece_on(mv.to_sq());
                    let capt_hist = self.hist.capture_history[moved_piece.idx()][mv.to_sq().idx()]
                        [captured_piece.type_of().idx()]
                    .get();

                    // Futility pruning for captures
                    if !gives_check && lmr_depth < 7 && capture_futility_eligible(pos, mv) {
                        let futility_value = self.stack[ss].static_eval
                            + ATOMIC_CAPT_FUT_BASE
                            + ATOMIC_CAPT_FUT_LMR_MULT * lmr_depth
                            + captured_piece.value()
                            + 131 * capt_hist / 1024;

                        if futility_value <= alpha {
                            continue;
                        }
                    }

                    // SEE based pruning for captures and checks
                    // Avoid pruning sacrifices of our last piece for
                    // stalemate
                    let margin = 175 * depth + capt_hist * 34 / 1024;
                    if (alpha >= VALUE_DRAW || pos.non_pawn_material(us) != moved_piece.value())
                        && !pos.see_ge(mv, -margin)
                    {
                        continue;
                    }
                } else if !self.stack[ss].follow_pv || !pv_node {
                    let mut history = cont_hist[0][moved_piece.idx()][mv.to_sq().idx()].get()
                        + cont_hist[1][moved_piece.idx()][mv.to_sq().idx()].get()
                        + sh.histories.pawn_entry(pos)[moved_piece.idx()][mv.to_sq().idx()].get();

                    // Continuation history based pruning
                    if history < -4313 * depth {
                        continue;
                    }

                    history += 64 * self.hist.main_history[us.idx()][mv.raw() as usize].get() / 32;

                    // (*Scaler): Generally, lower divisors scale well
                    lmr_depth += history / lmr_divisor(depth);

                    let futility_value = self.stack[ss].static_eval
                        + (176
                            + 117 * lmr_depth
                            + 90 * (self.stack[ss].static_eval > alpha) as i32)
                            * ATOMIC_FUTILITY_SCALE
                            / 64;

                    // Futility pruning: parent node
                    // (*Scaler): Generally, more frequent futility pruning
                    // scales well
                    if !self.stack[ss].in_check && lmr_depth < 12 && futility_value <= alpha {
                        if best_value <= futility_value
                            && !is_decisive(best_value)
                            && !is_win(futility_value)
                        {
                            best_value = futility_value;
                        }
                        continue;
                    }

                    lmr_depth = lmr_depth.max(0);

                    // Prune moves with negative SEE
                    if !pos.see_ge(mv, -25 * lmr_depth * lmr_depth) {
                        continue;
                    }
                }
            }

            // Step 15. Extensions
            // Singular extension search. If all moves but one fail low on a
            // search of (alpha-s, beta-s), and just one fails high on
            // (alpha, beta), then that move is singular and should be
            // extended. To verify this we do a reduced search on the
            // position excluding the ttMove and if the result is lower than
            // ttValue minus a margin, then we will extend the ttMove.
            // Recursive singular search is avoided.
            //
            // (*Scaler) Generally, higher singularBeta (i.e closer to
            // ttValue) and lower extension margins scale well.
            if !root_node
                && mv == tt_data.mv
                && excluded_move == Move::none()
                && depth >= SINGULAR_DEPTH_MIN + self.stack[ss].tt_pv as i32
                && is_valid(tt_data.value)
                && !is_decisive(tt_data.value)
                && (tt_data.bound & BOUND_LOWER) != 0
                && tt_data.depth >= depth - 3
                && !self.is_shuffling(mv, ss, pos)
                && !self.seek_mate
            {
                let singular_beta = tt_data.value
                    - (SINGULAR_MARGIN_BASE
                        + SINGULAR_MARGIN_TT_PV * (self.stack[ss].tt_pv && !pv_node) as i32)
                        * depth
                        / SINGULAR_MARGIN_DIV;
                let singular_depth = new_depth / 2;

                self.stack[ss].excluded_move = mv;
                value = self.atomic_search::<NON_PV>(
                    sh,
                    pos,
                    ss,
                    singular_beta - 1,
                    singular_beta,
                    singular_depth,
                    cut_node,
                );
                self.stack[ss].excluded_move = Move::none();

                if value < singular_beta {
                    let corr_val_adj = correction_value.abs() / 194822;
                    let double_margin = -3 + 201 * pv_node as i32
                        - 157 * !tt_capture as i32
                        - corr_val_adj
                        - 1081 * self.tt_move_history.get() / 117824
                        - (ply > self.root_depth) as i32 * 41;
                    let triple_margin = 72 + 306 * pv_node as i32 - 188 * !tt_capture as i32
                        + 84 * self.stack[ss].tt_pv as i32
                        - corr_val_adj
                        - (ply > self.root_depth) as i32 * 45;

                    extension = 1
                        + (value < singular_beta - double_margin) as i32
                        + (value < singular_beta - triple_margin) as i32;

                    depth += 1;
                }
                // Multi-cut pruning
                // Our ttMove is assumed to fail high based on the bound of
                // the TT entry, and if after excluding the ttMove with a
                // reduced search we fail high over the original beta, we
                // assume this expected cut-node is not singular (multiple
                // moves fail high), and we can prune the whole subtree by
                // returning a softbound.
                else if value >= beta && !is_decisive(value) {
                    self.tt_move_history.update(-442 - 108 * depth);

                    if !self.stack[ss].in_check && value > self.stack[ss].static_eval {
                        let bonus = ((value - self.stack[ss].static_eval) * 664 / 1024)
                            .clamp(-CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
                        self.atomic_update_correction_history(sh, pos, ss, bonus);
                    }

                    return value;
                }
                // Negative extensions
                // If other moves failed high over (ttValue - margin) without
                // the ttMove on a reduced search, but we cannot do multi-cut
                // because (ttValue - margin) is lower than the original
                // beta, we do not know if the ttMove is singular or can do a
                // multi-cut, so we reduce the ttMove in favor of other moves
                // based on some conditions:

                // If the ttMove is assumed to fail high over current beta or
                // if we are on a cutNode
                else if tt_data.value >= beta || cut_node {
                    extension = -3;
                }
            }

            let node_count = if root_node { self.nodes() } else { 0 };

            // Step 16. Make the move
            self.do_move(sh, pos, mv, gives_check, capture, ss);

            // Add extension to new depth
            new_depth += extension;

            // Decrease reduction for PvNodes (*Scaler)
            if self.stack[ss].tt_pv {
                r -= 2766
                    + pv_node as i32 * 1017
                    + (tt_data.value > alpha) as i32 * 838
                    + (tt_data.depth >= depth) as i32 * (923 + cut_node as i32 * 955);
            }

            // Base reduction offset to compensate for other tweaks
            r += 714;
            r -= move_count * 62;
            r -= correction_value.abs() / 26131;

            // Increase reduction for cut nodes
            if cut_node {
                r += 3995 + 1059 * (tt_data.mv == Move::none()) as i32;
            }

            // Increase reduction if ttMove is a capture
            if tt_capture {
                r += 1039;
            }

            // Increase reduction if next ply has a lot of fail high
            if self.stack[ss + 1].cutoff_cnt > 1 {
                r += 236
                    + 1079 * (self.stack[ss + 1].cutoff_cnt > 2) as i32
                    + 1143 * all_node as i32;
            }
            // For first picked move (ttMove) reduce reduction
            else if mv == tt_data.mv {
                r -= 2016;
            }

            if capture {
                let captured = pos.captured_piece();
                self.stack[ss].stat_score = 809 * captured.value() / 128
                    + self.hist.capture_history[moved_piece.idx()][mv.to_sq().idx()]
                        [captured.type_of().idx()]
                    .get();
            } else {
                self.stack[ss].stat_score = 2 * self.hist.main_history[us.idx()][mv.raw() as usize]
                    .get()
                    + cont_hist[0][moved_piece.idx()][mv.to_sq().idx()].get()
                    + cont_hist[1][moved_piece.idx()][mv.to_sq().idx()].get();
            }

            // Decrease/increase reduction for moves with a good/bad history
            r -= self.stack[ss].stat_score * 445 / 4096;

            // The evaluation's gap to alpha at Atomic's futility scale
            if !capture && !is_decisive(alpha) {
                r += 3 * ((alpha - eval) * 64 / ATOMIC_FUTILITY_SCALE).clamp(-64, 96);
            }

            // Scale up reductions for expected ALL nodes
            if all_node {
                r += r * 272 / (256 * depth + 285);
            }

            // When seeking a mate, reduce non-checking moves more and
            // checking moves less
            if self.seek_mate && r > 0 {
                r += (if gives_check { -r } else { r }) / 4;
            }

            // Step 17. Late moves reduction / extension (LMR)
            if depth >= 2 && move_count > 1 {
                // In general we want to cap the LMR depth search at
                // newDepth, but when reduction is negative, we allow this
                // move a limited search extension beyond the first move
                // depth. To avoid search explosion, extensions are not
                // allowed deep, relative to rootDepth, in the search tree.
                let d = (new_depth
                    + (-r / 1024).min(if ply < 2 * self.root_depth { 2 } else { 0 }))
                .max(1)
                    + pv_node as i32;

                self.stack[ss].reduction = new_depth - d;
                value =
                    -self.atomic_search::<NON_PV>(sh, pos, ss + 1, -(alpha + 1), -alpha, d, true);
                self.stack[ss].reduction = 0;

                // Do a full-depth search when reduced LMR search fails high
                // (*Scaler) Shallower searches here don't scale well
                if value > alpha {
                    // Adjust full-depth search based on LMR results - if the
                    // result was good enough search deeper, if it was bad
                    // enough search shallower.
                    let do_deeper_search = d < new_depth && value > best_value + 52;
                    let do_shallower_search = value < best_value + 9;

                    new_depth += do_deeper_search as i32 - do_shallower_search as i32;

                    if new_depth > d {
                        value = -self.atomic_search::<NON_PV>(
                            sh,
                            pos,
                            ss + 1,
                            -(alpha + 1),
                            -alpha,
                            new_depth,
                            !cut_node,
                        );
                    }

                    // Post LMR continuation history updates
                    self.atomic_update_continuation_histories(
                        sh,
                        ss,
                        moved_piece,
                        mv.to_sq(),
                        1415,
                    );
                }
            }
            // Step 18. Full-depth search when LMR is skipped
            else if !pv_node || move_count > 1 {
                // Increase reduction if ttMove is not present
                if tt_data.mv == Move::none() {
                    r += 1085;
                }

                // Note that if expected reduction is high, we reduce search
                // depth here
                value = -self.atomic_search::<NON_PV>(
                    sh,
                    pos,
                    ss + 1,
                    -(alpha + 1),
                    -alpha,
                    new_depth - (r > 5039) as i32 - (r > 5223 && new_depth > 2) as i32,
                    !cut_node,
                );
            }

            // For PV nodes only, do a full PV search on the first move or
            // after a fail high, otherwise let the parent node fail low with
            // value <= alpha and try another move.
            if pv_node && (move_count == 1 || value > alpha) {
                child_pv = true;
                self.pv_table[ply].clear();

                // Extend move from transposition table if we are about to
                // dive into qsearch. Decisive score handling improves mate
                // finding and retrograde analysis.
                if mv == tt_data.mv
                    && ((is_valid(tt_data.value)
                        && is_decisive(tt_data.value)
                        && tt_data.depth > 0)
                        || tt_data.depth > 1)
                {
                    new_depth = new_depth.max(1);
                }

                value = -self.atomic_search::<PV>(sh, pos, ss + 1, -beta, -alpha, new_depth, false);
            }

            // Step 19. Undo move
            self.undo_move(pos, mv);

            debug_assert!(value > -VALUE_INFINITE && value < VALUE_INFINITE);

            // Step 20. Check for a new best move
            // Finished searching the move. If a stop occurred, the return
            // value of the search cannot be trusted, and we return
            // immediately without updating best move, principal variation
            // nor transposition table.
            if sh.stop.load(Relaxed) {
                return VALUE_ZERO;
            }

            if root_node {
                let nodes = self.nodes();
                let rm = self
                    .root_moves
                    .iter_mut()
                    .find(|rm| rm.pv[0] == mv)
                    .expect("root moves include the moves searched at the root");

                rm.effort += nodes - node_count;

                let n = nodes - node_count;
                let e_prev = (rm.effort - n).max(1);

                // Exponential moving average parameters for root move
                const SCALE: u64 = 32;
                const CHI_NUMERATOR: u64 = 3;
                const CHI_DENOMINATOR: u64 = 2; // Chi = 3/2 = 1.5
                const MIN_WEIGHT: u64 = 12; // 37.5% minimum weight
                const MAX_WEIGHT: u64 = 24; // 75% maximum weight

                let w = ((SCALE * n * CHI_DENOMINATOR)
                    / (n * CHI_DENOMINATOR + CHI_NUMERATOR * e_prev))
                    .clamp(MIN_WEIGHT, MAX_WEIGHT);
                let w_mss = w.min(16);
                let v2 = i64::from(value) * i64::from(value.abs());

                // Stockfish averages in unsigned 64-bit arithmetic, which
                // amounts to rounding the averages down.
                if rm.average_score == -VALUE_INFINITE {
                    rm.average_score = value;
                } else {
                    rm.average_score = (i64::from(value) * w as i64
                        + i64::from(rm.average_score) * (SCALE - w) as i64)
                        .div_euclid(SCALE as i64) as Value;
                }

                if rm.mean_squared_score == -VALUE_INFINITE * VALUE_INFINITE {
                    rm.mean_squared_score = value * value.abs();
                } else {
                    rm.mean_squared_score = (v2 * w_mss as i64
                        + i64::from(rm.mean_squared_score) * (SCALE - w_mss) as i64)
                        .div_euclid(SCALE as i64)
                        as Value;
                }

                // PV move or new best move?
                if move_count == 1 || value > alpha {
                    rm.score = value;
                    rm.uci_score = value;
                    rm.sel_depth = self.sel_depth;
                    rm.unset_inexact();

                    if value >= beta {
                        rm.inexact_lower = true;
                        rm.uci_score = beta;
                    } else if value <= alpha {
                        rm.inexact_upper = true;
                        rm.uci_score = alpha;
                    }

                    rm.pv.truncate(1);

                    debug_assert!(child_pv);

                    rm.pv.extend_from_slice(self.pv_table[ply].as_slice());

                    // We record how often the best move has been changed in
                    // each iteration. This information is used for time
                    // management. In MultiPV mode, we must take care to only
                    // do this for the first PV line.
                    if move_count > 1 && self.pv_idx == 0 {
                        increment(&self.counters.best_move_changes);
                    }
                } else {
                    // All other moves but the PV, are set to the lowest
                    // value: this is not a problem when sorting because the
                    // sort is stable and the move position in the list is
                    // preserved - just the PV is pushed up.
                    rm.score = -VALUE_INFINITE;
                }
            }

            // In case we have an alternative move equal in eval to the
            // current bestmove, promote it to bestmove by pretending it just
            // exceeds alpha (but not beta).
            let inc = (value == best_value
                && ply + 2 >= self.root_depth
                && (self.nodes() & 14) == 0
                && !is_win(value.abs() + 1)) as i32;

            if value + inc > best_value {
                best_value = value;

                if value + inc > alpha {
                    best_move = mv;

                    // Update pv even in fail-high case
                    if pv_node && !root_node {
                        self.pv_table.update(ply, mv, child_pv);
                    }

                    if value >= beta {
                        // (*Scaler) Infrequent and small updates scale well
                        self.stack[ss].cutoff_cnt += (extension < 2 || pv_node) as i32;
                        debug_assert!(value >= beta); // Fail high
                        break;
                    }

                    // Reduce other moves if we have found at least one score
                    // improvement
                    if depth > 2 && depth < 13 && !is_decisive(value) {
                        depth -= 2;
                    }

                    debug_assert!(depth > 0);
                    alpha = value; // Update alpha! Always alpha < beta
                }
            }

            // If the move is worse than some previously searched move,
            // remember it, to update its stats later.
            if mv != best_move && move_count <= SEARCHEDLIST_CAPACITY {
                if capture {
                    captures_searched.push(mv);
                } else {
                    quiets_searched.push(mv);
                }
            }
        }

        // Step 21. Check for mate and stalemate
        // All legal moves have been searched and if there are no legal
        // moves, it must be a mate or a stalemate. If we are in a singular
        // extension search then return a fail low score.

        debug_assert!(
            move_count != 0
                || !self.stack[ss].in_check
                || excluded_move != Move::none()
                || MoveList::new::<ALL>(pos).is_empty()
        );

        // Adjust best value for fail high cases
        if best_value >= beta && !is_decisive(best_value) && !is_decisive(alpha) {
            best_value = (best_value * depth + beta) / (depth + 1);
        }

        if move_count == 0 {
            best_value = if excluded_move != Move::none() {
                alpha
            } else if self.stack[ss].in_check {
                mated_in(ply)
            } else {
                VALUE_DRAW
            };
        }
        // If there is a move that produces search value greater than alpha,
        // we update the stats of searched moves.
        else if best_move != Move::none() {
            self.atomic_update_all_stats(
                sh,
                pos,
                ss,
                best_move,
                prev_sq,
                &quiets_searched,
                &captures_searched,
                depth,
                tt_data.mv,
                pv_node,
            );
            if !pv_node {
                self.tt_move_history
                    .update(if best_move == tt_data.mv { 792 } else { -779 });
            }
        }
        // Bonus for prior quiet countermove that caused the fail low
        else if !prior_capture && prev_sq != Square::NONE {
            let mut bonus_scale = -245;
            bonus_scale -= self.stack[ss - 1].stat_score / 98;
            bonus_scale += (59 * depth).min(430);
            bonus_scale += 191 * (self.stack[ss - 1].move_count > 8) as i32;
            bonus_scale += 143
                * (!self.stack[ss].in_check && best_value <= self.stack[ss].static_eval - 103)
                    as i32;
            bonus_scale += 151
                * (!self.stack[ss - 1].in_check
                    && best_value <= -self.stack[ss - 1].static_eval - 78) as i32;

            bonus_scale = bonus_scale.max(0);

            // scaledBonus ranges from 0 to roughly 2.3M, overflows happen for
            // multipliers larger than 900
            let scaled_bonus = (141 * depth - 82).min(1472) * bonus_scale;

            self.atomic_update_continuation_histories(
                sh,
                ss - 1,
                pos.piece_on(prev_sq),
                prev_sq,
                scaled_bonus * 236 / 16384,
            );

            self.hist.main_history[(!us).idx()][prev_move.raw() as usize]
                .update(scaled_bonus * 234 / 32768);

            if pos.piece_on(prev_sq).type_of() != PAWN && prev_move.type_of() != PROMOTION {
                sh.histories.pawn_entry(pos)[pos.piece_on(prev_sq).idx()][prev_sq.idx()]
                    .update(scaled_bonus * 322 / 8192);
            }
        }
        // Bonus for prior capture countermove that caused the fail low, for
        // the piece chess would find on its square, which the capture exploded
        else if prev_sq != Square::NONE {
            debug_assert!(prior_capture);

            let captured_piece = pos.captured_piece();
            debug_assert!(captured_piece != NO_PIECE);
            let pc = self.prior_piece(pos, ss);
            self.hist.capture_history[pc.idx()][prev_sq.idx()][captured_piece.type_of().idx()]
                .update(901);
        }

        if pv_node {
            best_value = best_value.min(max_value);
        }

        // If no good move is found and the previous position was ttPv, then
        // the previous opponent move is probably good and the new position
        // is added to the search tree.
        if best_value <= alpha {
            self.stack[ss].tt_pv = self.stack[ss].tt_pv || self.stack[ss - 1].tt_pv;
        }

        // Write gathered information in transposition table. Note that the
        // static evaluation is saved as it was before correction history.
        if excluded_move == Move::none() && !(root_node && self.pv_idx != 0) {
            tt_writer.write(
                pos_key,
                value_to_tt(best_value, ply),
                self.stack[ss].tt_pv,
                if best_value >= beta {
                    BOUND_LOWER
                } else if pv_node && best_move != Move::none() {
                    BOUND_EXACT
                } else {
                    BOUND_UPPER
                },
                if move_count != 0 {
                    depth
                } else {
                    (depth + 6).min(MAX_PLY - 1)
                },
                best_move,
                unadjusted_static_eval,
                sh.tt.generation(),
            );
        }

        // Adjust correction history if the best move is not a capture and
        // the error direction matches whether we are above/below bounds.
        if !self.stack[ss].in_check
            && !(best_move != Move::none() && pos.capture(best_move))
            && (best_value > self.stack[ss].static_eval) == (best_move != Move::none())
        {
            let bonus = ((best_value - self.stack[ss].static_eval)
                * depth
                * if best_move != Move::none() { 12 } else { 18 }
                / 128)
                .clamp(-CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
            self.atomic_update_correction_history(sh, pos, ss, 1114 * bonus / 1024);
        }

        debug_assert!(-VALUE_INFINITE < best_value && best_value < VALUE_INFINITE);
        best_value
    }

    /// Quiescence search function, which is called by the main search
    /// function with depth zero, or recursively with further decreasing
    /// depth. With depth <= 0, we "should" be using static eval only, but
    /// tactical moves may confuse the static eval. To fight this horizon
    /// effect, we implement this qsearch of tactical moves. See
    /// https://www.chessprogramming.org/Horizon_Effect and
    /// https://www.chessprogramming.org/Quiescence_Search
    fn atomic_qsearch<const NT: NodeType>(
        &mut self,
        sh: &SharedState,
        pos: &mut Position<Atomic>,
        ss: usize,
        mut alpha: Value,
        beta: Value,
    ) -> Value {
        debug_assert!(NT != ROOT);
        let pv_node = NT == PV;

        debug_assert!(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || alpha == beta - 1);

        let ply = ply_of(ss);

        // Check if we have an upcoming move that draws by repetition
        if alpha < VALUE_DRAW && pos.upcoming_repetition(ply) {
            alpha = value_draw(self.nodes());
            if alpha >= beta {
                return alpha;
            }
        }

        // Step 1. Initialize node
        if pv_node {
            // The node's own principal variation starts empty, and so does
            // its parent's, which it collects.
            self.pv_table[ply].clear();
            self.pv_table[ply - 1].clear();
        }

        let mut best_move = Move::none();
        let us = pos.side_to_move();
        self.stack[ss].in_check = pos.in_check();
        let mut move_count = 0;

        // Used to send selDepth info to GUI (selDepth counts from 1, ply from
        // 0)
        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }

        // Step 2. Check for an immediate draw or maximum ply reached
        if !pos.has_king(us) {
            return mated_in(ply);
        }
        if !pos.has_king(!us) {
            return mate_in(ply);
        }

        if pos.is_draw(ply) || ply >= MAX_PLY {
            return if ply >= MAX_PLY && !self.stack[ss].in_check {
                self.evaluate(sh, pos)
            } else {
                VALUE_DRAW
            };
        }

        // A side that can explode the other king wins at once.
        let win = pos.atomic_winning_capture();
        if win != Move::none() {
            if pv_node {
                self.pv_table.update(ply, win, false);
            }
            return mate_in(ply + 1);
        }

        debug_assert!((0..MAX_PLY).contains(&ply));

        // Step 3. Transposition table lookup
        let pos_key = pos.key();
        let (tt_hit, mut tt_data, tt_writer) = sh.tt.probe(pos_key);

        // Need further processing of the saved data
        self.stack[ss].tt_hit = tt_hit;
        tt_data.mv = if tt_hit { tt_data.mv } else { Move::none() };
        tt_data.value = if tt_hit {
            value_from_tt(tt_data.value, ply, pos.rule50_count())
        } else {
            VALUE_NONE
        };
        let pv_hit = tt_hit && tt_data.is_pv;

        // A legal TT move proves at no cost that the position is not a
        // stalemate, which a cutoff or standing pat could otherwise hide.
        let legal_tt_move = tt_data.mv != Move::none() && pos.atomic_legal(tt_data.mv);

        // At non-PV nodes we check for an early TT cutoff
        if !pv_node
            && tt_data.depth >= DEPTH_QS
            // Can happen when !ttHit or when access race in probe()
            && is_valid(tt_data.value)
            && (tt_data.bound
                & if tt_data.value >= beta {
                    BOUND_LOWER
                } else {
                    BOUND_UPPER
                })
                != 0
        {
            if self.stack[ss].in_check || legal_tt_move || pos.atomic_has_legal_move() {
                return tt_data.value;
            }
            return VALUE_DRAW;
        }

        // Step 4. Static evaluation of the position
        let mut unadjusted_static_eval = VALUE_NONE;
        let mut best_value;
        let futility_base;
        if self.stack[ss].in_check {
            best_value = -VALUE_INFINITE;
            futility_base = -VALUE_INFINITE;
        } else {
            let correction_value = self.atomic_correction_value(sh, pos, ss);

            if self.stack[ss].tt_hit {
                // Never assume anything about values stored in TT
                unadjusted_static_eval = tt_data.eval;

                if !is_valid(unadjusted_static_eval) {
                    unadjusted_static_eval = self.evaluate(sh, pos);
                }

                best_value = to_corrected_static_eval(unadjusted_static_eval, correction_value);
                self.stack[ss].static_eval = best_value;

                // ttValue can be used as a better position evaluation
                if is_valid(tt_data.value)
                    && !is_decisive(tt_data.value)
                    && (tt_data.bound
                        & if tt_data.value > best_value {
                            BOUND_LOWER
                        } else {
                            BOUND_UPPER
                        })
                        != 0
                {
                    best_value = tt_data.value;
                }
            } else {
                unadjusted_static_eval = self.evaluate(sh, pos);
                best_value = to_corrected_static_eval(unadjusted_static_eval, correction_value);
                self.stack[ss].static_eval = best_value;
            }

            // Stand pat. Return immediately if static value is at least beta
            if best_value >= beta {
                // Standing pat must not hide a stalemate.
                if !legal_tt_move && !pos.atomic_has_legal_move() {
                    return VALUE_DRAW;
                }

                if !is_decisive(best_value) {
                    best_value = (467 * best_value + 557 * beta) / 1024;
                }

                if !self.stack[ss].tt_hit {
                    tt_writer.write(
                        pos_key,
                        VALUE_NONE,
                        false,
                        BOUND_LOWER,
                        DEPTH_UNSEARCHED,
                        Move::none(),
                        unadjusted_static_eval,
                        sh.tt.generation(),
                    );
                }
                return best_value;
            }

            if best_value > alpha {
                alpha = best_value;
            }

            futility_base = self.stack[ss].static_eval + QS_FUTILITY_BASE;
        }

        let cont_hist: [&PieceToHistory; 6] = [
            sh.histories
                .continuation(self.stack[ss - 1].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 2].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 3].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 4].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 5].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 6].continuation_history),
        ];

        let prev_move = self.stack[ss - 1].current_move;
        let prev_sq = if prev_move.is_ok() {
            prev_move.to_sq()
        } else {
            Square::NONE
        };

        // Initialize a MovePicker object for the current position. The
        // quiescence search picks only captures, but in check, which no
        // evasion generator answers in Atomic, it picks every move, captures
        // first, as the main search does.
        let move_picker_depth = if self.stack[ss].in_check { 1 } else { DEPTH_QS };
        let mut mp = AtomicMovePicker::new(pos, tt_data.mv, move_picker_depth, cont_hist, ply);

        // Step 5. Loop through all legal moves until no moves remain
        // or a beta cutoff occurs.
        loop {
            let mv = mp.next_move(pos, &self.hist, &sh.histories);
            if mv == Move::none() {
                break;
            }

            debug_assert!(mv.is_ok());

            let gives_check = pos.gives_check(mv);
            let capture = pos.capture_stage(mv);

            move_count += 1;

            // Step 6. Pruning
            if !is_loss(best_value) {
                // Futility pruning and moveCount pruning
                if !gives_check
                    && mv.to_sq() != prev_sq
                    && !is_loss(futility_base)
                    && mv.type_of() != PROMOTION
                    && capture_futility_eligible(pos, mv)
                {
                    if move_count > 2 {
                        continue;
                    }

                    let futility_value = futility_base + pos.piece_on(mv.to_sq()).value();

                    // If static eval + value of piece we are going to capture
                    // is much lower than alpha, we can prune this move.
                    if futility_value <= alpha {
                        best_value = best_value.max(futility_value);
                        continue;
                    }

                    // If static exchange evaluation is low enough we can
                    // prune this move.
                    if !pos.see_ge(mv, alpha - futility_base) {
                        best_value = best_value.max(alpha.min(futility_base));
                        continue;
                    }
                }

                // Skip non-captures
                if !capture {
                    continue;
                }

                // Do not search moves with bad enough SEE values
                if !pos.see_ge(mv, -74) {
                    continue;
                }
            }

            // Step 7. Make and search the move
            self.do_move(sh, pos, mv, gives_check, capture, ss);

            let value = -self.atomic_qsearch::<NT>(sh, pos, ss + 1, -beta, -alpha);
            self.undo_move(pos, mv);

            debug_assert!(value > -VALUE_INFINITE && value < VALUE_INFINITE);

            // Step 8. Check for a new best move
            if value > best_value {
                best_value = value;

                if value > alpha {
                    best_move = mv;

                    // Update pv even in fail-high case
                    if pv_node {
                        self.pv_table.update(ply, mv, true);
                    }

                    if value < beta {
                        // Update alpha here!
                        alpha = value;
                    } else {
                        break; // Fail high
                    }
                }
            }
        }

        // Step 9. Check for mate and stalemate
        // All legal moves have been searched. A special case: if we are in
        // check and no legal moves were found, it is checkmate.
        if move_count == 0 {
            if self.stack[ss].in_check {
                // Checkmate!
                debug_assert!(MoveList::new::<ALL>(pos).is_empty());
                return mated_in(ply); // Plies to mate from the root
            }

            if !pos.atomic_has_legal_quiet() {
                best_value = VALUE_DRAW;
            }
        }

        if !is_decisive(best_value) && best_value > beta {
            best_value = (481 * best_value + 543 * beta) / 1024;
        }

        // Save gathered info in transposition table. The static evaluation
        // is saved as it was before adjustment by correction history.
        tt_writer.write(
            pos_key,
            value_to_tt(best_value, ply),
            pv_hit,
            if best_value >= beta {
                BOUND_LOWER
            } else {
                BOUND_UPPER
            },
            DEPTH_QS,
            best_move,
            unadjusted_static_eval,
            sh.tt.generation(),
        );

        debug_assert!(-VALUE_INFINITE < best_value && best_value < VALUE_INFINITE);
        best_value
    }

    #[inline(always)]
    fn atomic_reduction(&self, i: bool, d: Depth, mn: i32) -> i32 {
        let reduction_scale = self.reductions[d as usize & (MOVE_LIST_SIZE - 1)]
            * self.reductions[mn as usize & (MOVE_LIST_SIZE - 1)];
        reduction_scale + (!i) as i32 * reduction_scale * 194 / 512 + LMR_BASE_OFFSET
    }

    /// The piece chess finds on the target square of the previous move, that
    /// of stack entry `ss - 1`. An Atomic capture explodes it, so after one
    /// it is the piece that moved, by which do_move keys the entry's
    /// continuation correction index, or the piece a promotion made.
    #[inline(always)]
    fn prior_piece(&self, pos: &Position<Atomic>, ss: usize) -> Piece {
        let m = self.stack[ss - 1].current_move;
        if pos.captured_piece().is_none() {
            pos.piece_on(m.to_sq())
        } else if m.type_of() == PROMOTION {
            make_piece(!pos.side_to_move(), m.promotion_type())
        } else {
            Piece((self.stack[ss - 1].continuation_correction_history as usize / SQUARE_NB) as u8)
        }
    }

    /// The correction of the static evaluation, with the fork's weights less
    /// the 10% by which Stockfish's 7a372f3f made room for the sixth ply.
    #[inline(always)]
    fn atomic_correction_value(&self, sh: &SharedState, pos: &Position<Atomic>, ss: usize) -> i32 {
        let us = pos.side_to_move().idx();
        let m = self.stack[ss - 1].current_move;
        let shared = &sh.histories;
        let pcv = shared.pawn_correction_entry(pos)[us].pawn.get();
        let micv = shared.minor_piece_correction_entry(pos)[us].minor.get();
        let wnpcv = shared.nonpawn_correction_entry(pos, WHITE)[us]
            .non_pawn_white
            .get();
        let bnpcv = shared.nonpawn_correction_entry(pos, BLACK)[us]
            .non_pawn_black
            .get();
        let cntcv = if m.is_ok() {
            let to = m.to_sq();
            let pc = self.prior_piece(pos, ss);
            7527 * (self.continuation_correction(ss - 2, pc, to).get()
                + self.continuation_correction(ss - 4, pc, to).get())
                + 6021 * self.continuation_correction(ss - 6, pc, to).get()
        } else {
            81325
        };

        12011 * pcv + 8352 * micv + 10656 * (wnpcv + bnpcv) + cntcv
    }

    fn atomic_update_correction_history(
        &mut self,
        sh: &SharedState,
        pos: &Position<Atomic>,
        ss: usize,
        bonus: i32,
    ) {
        let m = self.stack[ss - 1].current_move;
        let us = pos.side_to_move().idx();

        const NON_PAWN_WEIGHT: i32 = 186;
        let shared = &sh.histories;

        shared.pawn_correction_entry(pos)[us].pawn.update(bonus);
        shared.minor_piece_correction_entry(pos)[us]
            .minor
            .update(bonus * 152 / 128);
        shared.nonpawn_correction_entry(pos, WHITE)[us]
            .non_pawn_white
            .update(bonus * NON_PAWN_WEIGHT / 128);
        shared.nonpawn_correction_entry(pos, BLACK)[us]
            .non_pawn_black
            .update(bonus * NON_PAWN_WEIGHT / 128);

        if m.is_ok() {
            let to = m.to_sq();
            let pc = self.prior_piece(pos, ss);
            for (i, weight) in [(2, 136), (4, 68), (6, 34)] {
                let index = self.stack[ss - i].continuation_correction_history as usize;
                self.hist.continuation_correction_history[index & (PIECE_NB * SQUARE_NB - 1)]
                    [pc.idx()][to.idx()]
                .update(bonus * weight / 128);
            }
        }
    }

    /// Updates stats at the end of search() when a bestMove is found
    #[allow(clippy::too_many_arguments)]
    fn atomic_update_all_stats(
        &mut self,
        sh: &SharedState,
        pos: &Position<Atomic>,
        ss: usize,
        best_move: Move,
        prev_sq: Square,
        quiets_searched: &SearchedList,
        captures_searched: &SearchedList,
        depth: Depth,
        tt_move: Move,
        pv_node: bool,
    ) {
        let moved_piece = pos.moved_piece(best_move);

        let mut bonus = (134 * depth - 79).min(1572)
            + 382 * (best_move == tt_move) as i32
            + self.stack[ss - 1].stat_score / 30;
        let malus = (1005 * depth - 205).min(2218);

        if !pv_node {
            // Atomic-Stockfish multiplies in unsigned 64-bit arithmetic,
            // which amounts to rounding the quotient down.
            let searched = (quiets_searched.len() + captures_searched.len()) as i64;
            bonus += (i64::from(bonus) * searched).div_euclid(256) as i32;
        }

        if !pos.capture_stage(best_move) {
            self.atomic_update_quiet_histories(sh, pos, ss, best_move, bonus * 824 / 1024);

            // Decrease stats for all non-best quiet moves
            let mut actual_malus = malus * 1136 / 1024;
            for &mv in quiets_searched {
                actual_malus = actual_malus * 956 / 1024;
                self.atomic_update_quiet_histories(sh, pos, ss, mv, -actual_malus);
            }
        } else {
            // Increase stats for the best move in case it was a capture move
            let captured_piece = pos.piece_on(best_move.to_sq()).type_of();
            self.hist.capture_history[moved_piece.idx()][best_move.to_sq().idx()]
                [captured_piece.idx()]
            .update(bonus * 1366 / 1024);
        }

        // Extra penalty for a quiet early move that was not a TT move in
        // previous ply when it gets refuted.
        if prev_sq != Square::NONE
            && self.stack[ss - 1].move_count == 1 + self.stack[ss - 1].tt_hit as i32
            && pos.captured_piece().is_none()
        {
            self.atomic_update_continuation_histories(
                sh,
                ss - 1,
                pos.piece_on(prev_sq),
                prev_sq,
                -malus * 683 / 1024,
            );
        }

        // Decrease stats for all non-best capture moves
        for &mv in captures_searched {
            let moved_piece = pos.moved_piece(mv);
            let captured_piece = pos.piece_on(mv.to_sq()).type_of();
            self.hist.capture_history[moved_piece.idx()][mv.to_sq().idx()][captured_piece.idx()]
                .update(-malus * 1518 / 1024);
        }
    }

    /// Updates the continuation histories for the move pairs formed by the
    /// current move and the moves played in previous plies.
    fn atomic_update_continuation_histories(
        &self,
        sh: &SharedState,
        ss: usize,
        pc: Piece,
        to: Square,
        bonus: i32,
    ) {
        const CONTHIST_BONUSES: [(usize, i32); 5] =
            [(1, 1040), (2, 780), (3, 300), (4, 537), (6, 423)];

        // Multipliers for positive history consistency
        const CMHC_MULTIPLIERS: [i32; 6] = [96, 113, 101, 105, 127, 121];

        // Each weight times each multiplier, so that a bonus takes one
        // multiplication, in 64 bits: in the 32 that Atomic-Stockfish
        // computes it in, it overflows in long searches.
        const WEIGHTED: [[i64; CMHC_MULTIPLIERS.len()]; CONTHIST_BONUSES.len()] = {
            let mut table = [[0; CMHC_MULTIPLIERS.len()]; CONTHIST_BONUSES.len()];
            let mut k = 0;
            while k < table.len() {
                let mut n = 0;
                while n < table[k].len() {
                    table[k][n] = CONTHIST_BONUSES[k].1 as i64 * CMHC_MULTIPLIERS[n] as i64;
                    n += 1;
                }
                k += 1;
            }
            table
        };
        let bonus = i64::from(bonus);
        let mut positive_count = 0;

        for (k, (i, _)) in CONTHIST_BONUSES.into_iter().enumerate() {
            // Only update the first 2 continuation histories if we are in
            // check
            if self.stack[ss].in_check && i > 2 {
                break;
            }

            if self.stack[ss - i].current_move.is_ok() {
                let history_entry = &sh
                    .histories
                    .continuation(self.stack[ss - i].continuation_history)[pc.idx()][to.idx()];
                if history_entry.get() > 0 {
                    positive_count += 1;
                }

                let product = bonus * WEIGHTED[k][positive_count];
                history_entry.update((product / 131072) as i32 + 71 * (i < 2) as i32);
            }
        }
    }

    /// Updates move sorting heuristics
    fn atomic_update_quiet_histories(
        &mut self,
        sh: &SharedState,
        pos: &Position<Atomic>,
        ss: usize,
        mv: Move,
        bonus: i32,
    ) {
        let us = pos.side_to_move();
        // Untuned to prevent duplicate effort
        self.hist.main_history[us.idx()][mv.raw() as usize].update(bonus);

        let ply = ply_of(ss);
        if (ply as usize) < LOW_PLY_HISTORY_SIZE {
            self.hist.low_ply_history[ply as usize][mv.raw() as usize].update(bonus * 663 / 1024);
        }

        self.atomic_update_continuation_histories(
            sh,
            ss,
            pos.moved_piece(mv),
            mv.to_sq(),
            bonus * 820 / 1024,
        );

        let weight = if bonus > -7 { 1038 } else { 525 };
        sh.histories.pawn_entry(pos)[pos.moved_piece(mv).idx()][mv.to_sq().idx()]
            .update(bonus * weight / 1024);
    }
}
