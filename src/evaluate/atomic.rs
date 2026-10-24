//! The evaluation of Atomic positions, as Atomic-Stockfish evaluates them
//! with a Legacy Atomic V1 network: by the network, blended and scaled as
//! Fairy-Stockfish's was, corrected for bishops cornered behind pawns in
//! Chess960 and damped as the fifty-move rule nears, or by the network's
//! output alone (see `UseNnue`). Search optimism, which belongs to chess's
//! network, plays no part.
//!
//! Derived, with changes for varifish in 2026, from Atomic-Stockfish's
//! `evaluate.cpp` and `nnue/nnue_misc.cpp` (https://github.com/Belzedar94/Atomic-Stockfish),
//! a variant of Stockfish, a UCI chess playing engine derived from Glaurung
//! 2.1, Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under
//! the GNU General Public License, version 3 or later.

use std::fmt::Write;

use crate::nnue::atomic::accumulator::{AccumulatorCaches, AccumulatorStack};
use crate::nnue::atomic::network::{LAYER_STACKS, Network, value_from_scaled};
use crate::nnue::common::OUTPUT_SCALE;
use crate::nnue::eval_file::UseNnue;
use crate::position::Position;
use crate::types::*;
use crate::uci::to_cp;
use crate::variant::Atomic;

const CORNERED_BISHOP: Value = 50;

/// What Fairy-Stockfish's royal piece added to the material, which the
/// network's scaling still counts for each king.
const LEGACY_ATOMIC_ROYAL_VALUE: Value = 700;

/// The penalty for a bishop in a corner behind a pawn of its own, from the
/// side to move's point of view: Fairy-Stockfish's Chess960 correction.
fn fix_frc(pos: &Position<Atomic>) -> Value {
    let corners = Square::A1 | Square::H1 | Square::A8 | Square::H8;
    if (pos.pieces_pt(BISHOP) & corners).is_empty() {
        return VALUE_ZERO;
    }

    let mut correction = 0;
    for (corner, pawn, beyond, bishop, own_pawn, sign) in [
        (Square::A1, Square::B2, Square::B3, W_BISHOP, W_PAWN, -1),
        (Square::H1, Square::G2, Square::G3, W_BISHOP, W_PAWN, -1),
        (Square::A8, Square::B7, Square::B6, B_BISHOP, B_PAWN, 1),
        (Square::H8, Square::G7, Square::G6, B_BISHOP, B_PAWN, 1),
    ] {
        if pos.piece_on(corner) == bishop && pos.piece_on(pawn) == own_pawn {
            let blocked = if pos.empty(beyond) { 3 } else { 4 };
            correction += sign * CORNERED_BISHOP * blocked;
        }
    }

    if pos.side_to_move() == WHITE {
        correction
    } else {
        -correction
    }
}

/// Damps a value linearly to 0 as the fifty-move rule nears, where it stays
/// for any longer count a FEN may give.
fn damp_for_rule50(v: Value, pos: &Position<Atomic>) -> Value {
    let remaining = (100 - pos.rule50_count()).max(0);
    v * remaining / 100
}

/// The value of the network's raw outputs, summed in 64 bits.
fn value_from_raw(psqt: i32, positional: i32) -> Value {
    value_from_scaled((i64::from(psqt) + i64::from(positional)) / i64::from(OUTPUT_SCALE))
}

/// The evaluation of a position with both kings, from the side to move's
/// point of view, whose state the stack's latest state holds. The network
/// is loaded, as searches and `eval` check first.
pub fn evaluate(
    network: Option<&Network>,
    pos: &Position<Atomic>,
    accumulators: &mut AccumulatorStack,
    caches: &mut AccumulatorCaches,
    use_nnue: UseNnue,
) -> Value {
    debug_assert!(pos.has_king(WHITE) && pos.has_king(BLACK));

    let network = network.expect("searches check that the network is loaded");
    let (psqt, positional) = network.evaluate_raw(pos, accumulators, caches);

    let v = if use_nnue == UseNnue::Pure {
        value_from_raw(psqt, positional)
    } else {
        let delta_npm = (pos.non_pawn_material(WHITE) - pos.non_pawn_material(BLACK)).abs();
        let entertainment = if delta_npm <= BISHOP_VALUE - KNIGHT_VALUE {
            7
        } else {
            0
        };
        let blended_raw = ((128 - entertainment) * psqt + (128 + entertainment) * positional) / 128;

        let legacy_npm =
            pos.non_pawn_material_total() + LEGACY_ATOMIC_ROYAL_VALUE * pos.count_both(KING);
        let scale = 903 + 32 * pos.count_both(PAWN) + 32 * legacy_npm / 1024;
        let mut v = (blended_raw / OUTPUT_SCALE) * scale / 1024;

        if pos.is_chess960() {
            v += fix_frc(pos);
        }

        damp_for_rule50(v, pos)
    };

    // Guarantee that the evaluation does not hit the tablebase range
    v.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

/// Converts a Value into pawns, always keeping two decimals
fn format_cp_aligned_dot(v: Value, out: &mut String, pos: &Position<Atomic>) {
    let pawns = (0.01 * f64::from(to_cp(v, pos))).abs();
    let sign = match v {
        ..0 => '-',
        1.. => '+',
        0 => ' ',
    };
    let _ = write!(out, "{sign}{pawns:6.2}");
}

/// The table of what the material and the layers of each layer stack
/// contribute.
fn network_trace(pos: &Position<Atomic>, network: &Network) -> String {
    let mut accumulators = AccumulatorStack::new();
    let mut caches = AccumulatorCaches::new(network);
    let t = network.trace_evaluate(pos, &mut accumulators, &mut caches);

    let mut ss = String::new();
    ss += "NNUE network contributions (Normalized, ";
    ss += if pos.side_to_move() == WHITE {
        "White to move)\n"
    } else {
        "Black to move)\n"
    };
    ss += "+------------+------------+------------+------------+\n";
    ss += "|   Bucket   |  Material  | Positional |   Total    |\n";
    ss += "|            |   (PSQT)   |  (Layers)  |            |\n";
    ss += "+------------+------------+------------+------------+\n";

    for bucket in 0..LAYER_STACKS {
        let _ = write!(ss, "|  {bucket}         |  ");
        format_cp_aligned_dot(t.psqt[bucket], &mut ss, pos);
        ss += "   |  ";
        format_cp_aligned_dot(t.positional[bucket], &mut ss, pos);
        ss += "   |  ";
        format_cp_aligned_dot(t.total[bucket], &mut ss, pos);
        ss += "   |";
        if bucket == t.correct_bucket {
            ss += " <-- this bucket is used";
        }
        ss.push('\n');
    }

    ss += "+------------+------------+------------+------------+\n";
    ss
}

/// Like evaluate(), but instead of returning a value, it returns a string
/// (suitable for outputting to stdout) that contains the detailed
/// descriptions and values of each evaluation term. Useful for debugging.
/// Trace scores are from white's point of view. Positions in check are
/// evaluated too, as Atomic-Stockfish, whose positions are never in check,
/// evaluates them.
pub fn trace(pos: &Position<Atomic>, network: Option<&Network>, use_nnue: UseNnue) -> String {
    if !pos.has_king(WHITE) || !pos.has_king(BLACK) {
        return "Final evaluation: none (Atomic terminal)".to_string();
    }

    let network = network.expect("the network is verified");
    let mut accumulators = AccumulatorStack::new();
    let mut caches = AccumulatorCaches::new(network);

    let mut ss = String::new();
    ss.push('\n');
    ss += &network_trace(pos, network);
    ss.push('\n');

    let (psqt, positional) = network.evaluate_raw(pos, &mut accumulators, &mut caches);
    let v = value_from_raw(psqt, positional);
    let _ = writeln!(
        ss,
        "NNUE evaluation          {v:+} (side to move, internal units)"
    );
    let v = if pos.side_to_move() == WHITE { v } else { -v };
    let _ = writeln!(
        ss,
        "NNUE evaluation        {:+.2} (white side)",
        f64::from(v) / f64::from(PAWN_VALUE)
    );

    let v = evaluate(Some(network), pos, &mut accumulators, &mut caches, use_nnue);
    let v = if pos.side_to_move() == WHITE { v } else { -v };
    let _ = writeln!(
        ss,
        "Final evaluation      {:+.2} (white side) [Use NNUE={}]",
        f64::from(v) / f64::from(PAWN_VALUE),
        use_nnue.name()
    );

    ss
}
