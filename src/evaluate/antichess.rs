//! The evaluation of Antichess positions: with Antichess's dedicated SFNNv20
//! network when loaded, or else a compact classical Antichess bootstrap
//! evaluation, far weaker, without it.

use std::fmt::Write;

use crate::attacks::*;
use crate::bitboard::*;
use crate::nnue::Caches;
use crate::nnue::antichess::{AntichessAccumulators, AntichessCaches};
use crate::nnue::eval_file::{Networks, UseNnue};
use crate::nnue::misc::format_trace;
use crate::position::Position;
use crate::types::*;
use crate::uci::to_cp;
use crate::variant::Antichess;

/// Classical bootstrap piece values for Antichess (negative because losing
/// pieces is the goal; heavy pieces are larger liabilities).
const PIECE_VALUES: [Value; PIECE_TYPE_NB] = [
    0,    // NO_PIECE_TYPE
    -110, // PAWN
    -180, // KNIGHT
    -160, // BISHOP
    -230, // ROOK
    -320, // QUEEN
    -140, // KING
    0,
];

/// Computes the classical Antichess bootstrap evaluation from White's point of
/// view, then flips for the side to move.
fn classical_bootstrap(pos: &Position<Antichess>) -> Value {
    let occupied = pos.pieces();
    let empty = !occupied;

    let score_side = |c: Color| -> i32 {
        let mut s = 0i32;

        // Material: fewer pieces are better.
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] {
            s += pos.count(c, pt) * PIECE_VALUES[pt.idx()];
        }

        // Blocked pawns are assets (they restrict mobility and aid stalemates).
        let pawns = pos.pieces_cpt(c, PAWN);
        let pushable_pawns = pawn_single_push_bb(c, pawns) & empty;
        let blocked = pawns.popcount() - pushable_pawns.popcount();
        s += blocked * 18;

        // Mobility penalty: more quiet targets mean more vulnerability to
        // forcing lines and fewer stalemate chances.
        let mut quiet_targets = pushable_pawns.popcount();
        for pt in [KNIGHT, BISHOP, ROOK, QUEEN, KING] {
            for sq in pos.pieces_cpt(c, pt) {
                quiet_targets += (attacks_bb(pt, sq, occupied) & empty).popcount();
            }
        }
        s -= quiet_targets * 5;

        s
    };

    let white_score = score_side(WHITE) - score_side(BLACK);
    let mut v = if pos.side_to_move() == WHITE {
        white_score
    } else {
        -white_score
    };
    v += 12; // Small tempo bonus
    v -= v * pos.rule50_count() / 189;
    v.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

/// The evaluation of a non-terminal Antichess position, from the side to move's
/// point of view.
pub fn evaluate(
    networks: &Networks,
    pos: &Position<Antichess>,
    accumulators: &mut AntichessAccumulators,
    caches: &mut AntichessCaches,
    use_nnue: UseNnue,
) -> Value {
    if let Some(network) = networks.antichess.network() {
        let mut v = network.evaluate(pos, accumulators, caches);
        if use_nnue != UseNnue::Pure {
            v -= v * pos.rule50_count() / 189;
        }
        return v.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
    }

    classical_bootstrap(pos)
}

/// Evaluation trace for `eval`, from White's point of view.
pub fn trace(networks: &Networks, pos: &Position<Antichess>, use_nnue: UseNnue) -> String {
    let mut ss = String::new();
    ss.push('\n');

    let (raw, final_v, is_nnue) = if let Some(network) = networks.antichess.network() {
        let mut accumulators = AntichessAccumulators::new_stack();
        let mut caches = AntichessCaches::new(Some(network));
        let mut trace_acc = AntichessAccumulators::new_stack();
        let t = network.trace_evaluate(pos, &mut trace_acc, &mut caches);
        ss += &format_trace(pos, &t.positional[..t.layer_stacks], t.correct_bucket);
        ss.push('\n');

        let nnue = network.evaluate(pos, &mut accumulators, &mut caches);
        let mut scaled = nnue;
        if use_nnue != UseNnue::Pure {
            scaled -= scaled * pos.rule50_count() / 189;
        }
        scaled = scaled.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
        (nnue, scaled, true)
    } else {
        let v = classical_bootstrap(pos);
        (v, v, false)
    };

    if is_nnue {
        let _ = writeln!(
            ss,
            "NNUE evaluation          {raw:+} (side to move, internal units)"
        );
    } else {
        let _ = writeln!(
            ss,
            "Classical evaluation     {raw:+} (side to move, internal units, Antichess bootstrap)"
        );
    }

    let (raw, final_v) = if pos.side_to_move() == WHITE {
        (raw, final_v)
    } else {
        (-raw, -final_v)
    };

    if is_nnue {
        let _ = writeln!(
            ss,
            "NNUE evaluation        {:+.2} (white side)",
            0.01 * f64::from(to_cp(raw, pos))
        );
    }
    let _ = writeln!(
        ss,
        "Final evaluation      {:+.2} (white side)",
        0.01 * f64::from(to_cp(final_v, pos))
    );

    ss
}
