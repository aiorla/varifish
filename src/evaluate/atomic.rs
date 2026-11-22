//! The evaluation of Atomic positions. With a Legacy Atomic V1 network, as
//! Atomic-Stockfish evaluates them: by the network, blended and scaled as
//! Fairy-Stockfish's was, corrected for bishops cornered behind pawns in
//! Chess960 and damped as the fifty-move rule nears, or by the network's
//! output alone (see `UseNnue`). With a network in Stockfish's design, by the
//! network's output scaled by the material as a Legacy network's, the scale
//! varifish's search was tuned with, and damped as the fifty-move rule nears,
//! or by the output alone, as training data is labelled. Search optimism,
//! which belongs to chess's network, plays no part.
//!
//! Derived, with changes for varifish in 2026, from Atomic-Stockfish's
//! `evaluate.cpp` and `nnue/nnue_misc.cpp` (https://github.com/Belzedar94/Atomic-Stockfish),
//! a variant of Stockfish, a UCI chess playing engine derived from Glaurung
//! 2.1, Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under
//! the GNU General Public License, version 3 or later.

use std::fmt::Write;

use crate::nnue::atomic::accumulator::{AccumulatorCaches, AccumulatorStack};
use crate::nnue::atomic::network::{LAYER_STACKS, Network, value_from_scaled};
use crate::nnue::atomic::{AtomicAccumulators, AtomicCaches, AtomicNetwork, sfnnv20};
use crate::nnue::common::OUTPUT_SCALE;
use crate::nnue::eval_file::UseNnue;
use crate::position::Position;
use crate::types::*;
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

/// Fairy-Stockfish's scale of a network's output by the material on the
/// board, in 1024ths, its base of 903 retuned by SPSA: 1998 at the
/// start, 967 with the kings alone.
fn material_scale(pos: &Position<Atomic>) -> Value {
    let legacy_npm =
        pos.non_pawn_material_total() + LEGACY_ATOMIC_ROYAL_VALUE * pos.count_both(KING);
    924 + 32 * pos.count_both(PAWN) + 32 * legacy_npm / 1024
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
/// point of view, whose state the stacks' latest states hold, those of the
/// network's format. The network is loaded, as searches and `eval` check
/// first.
#[inline(always)]
pub fn evaluate(
    network: Option<&AtomicNetwork>,
    pos: &Position<Atomic>,
    accumulators: &mut AtomicAccumulators,
    caches: &mut AtomicCaches,
    use_nnue: UseNnue,
) -> Value {
    debug_assert!(pos.has_king(WHITE) && pos.has_king(BLACK));

    let network = network.expect("searches check that the network is loaded");
    debug_assert!(accumulators.kind() == network.kind());
    match network {
        AtomicNetwork::Legacy(network) => evaluate_legacy(
            network,
            pos,
            &mut accumulators.legacy,
            &mut caches.legacy,
            use_nnue,
        ),
        AtomicNetwork::Sfnnv20(network) => evaluate_sfnnv20(
            network,
            pos,
            &mut accumulators.sfnnv20,
            &mut caches.sfnnv20,
            use_nnue,
        ),
    }
}

/// The evaluation of a network in Stockfish's design from its output: the
/// output scaled by 1.019 (tuned by SPSA) and by the material, and damped as
/// the fifty-move rule nears,
/// unless `use_nnue` is pure.
fn scaled(nnue: Value, pos: &Position<Atomic>, use_nnue: UseNnue) -> Value {
    let v = if use_nnue == UseNnue::Pure {
        nnue
    } else {
        damp_for_rule50(nnue * 1019 / 1000 * material_scale(pos) / 1024, pos)
    };

    // Guarantee that the evaluation does not hit the tablebase range
    v.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

/// The evaluation by a network in Stockfish's design.
fn evaluate_sfnnv20(
    network: &sfnnv20::Network,
    pos: &Position<Atomic>,
    accumulators: &mut sfnnv20::accumulator::AccumulatorStack,
    caches: &mut sfnnv20::accumulator::AccumulatorCaches,
    use_nnue: UseNnue,
) -> Value {
    scaled(network.evaluate(pos, accumulators, caches), pos, use_nnue)
}

/// The evaluation by a Legacy Atomic V1 network, as Atomic-Stockfish's.
fn evaluate_legacy(
    network: &Network,
    pos: &Position<Atomic>,
    accumulators: &mut AccumulatorStack,
    caches: &mut AccumulatorCaches,
    use_nnue: UseNnue,
) -> Value {
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
        let mut v = (blended_raw / OUTPUT_SCALE) * material_scale(pos) / 1024;

        if pos.is_chess960() {
            v += fix_frc(pos);
        }

        damp_for_rule50(v, pos)
    };

    // Guarantee that the evaluation does not hit the tablebase range
    v.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

/// Converts a Value into Atomic-Stockfish's pawns, of 208, always keeping
/// two decimals: the trace is Atomic-Stockfish's, which normalizes by its
/// pawn, where the search's scores go by the win rate model.
fn format_cp_aligned_dot(v: Value, out: &mut String) {
    let pawns = (0.01 * (100 * i64::from(v) / i64::from(PAWN_VALUE)) as f64).abs();
    let sign = match v {
        ..0 => '-',
        1.. => '+',
        0 => ' ',
    };
    let _ = write!(out, "{sign}{pawns:6.2}");
}

/// The table of what the material and the layers of each layer stack
/// contribute.
fn network_trace_legacy(pos: &Position<Atomic>, network: &Network) -> String {
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
        format_cp_aligned_dot(t.psqt[bucket], &mut ss);
        ss += "   |  ";
        format_cp_aligned_dot(t.positional[bucket], &mut ss);
        ss += "   |  ";
        format_cp_aligned_dot(t.total[bucket], &mut ss);
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
pub fn trace(pos: &Position<Atomic>, network: Option<&AtomicNetwork>, use_nnue: UseNnue) -> String {
    if !pos.has_king(WHITE) || !pos.has_king(BLACK) {
        return "Final evaluation: none (Atomic terminal)".to_string();
    }

    match network.expect("the network is verified") {
        AtomicNetwork::Legacy(network) => trace_legacy(pos, network, use_nnue),
        AtomicNetwork::Sfnnv20(network) => trace_sfnnv20(pos, network, use_nnue),
    }
}

/// The lines of the network's evaluation and of the final one.
fn trace_values(ss: &mut String, nnue: Value, v: Value, pos: &Position<Atomic>, use_nnue: UseNnue) {
    let _ = writeln!(
        ss,
        "NNUE evaluation          {nnue:+} (side to move, internal units)"
    );
    let white = |v: Value| if pos.side_to_move() == WHITE { v } else { -v };
    let _ = writeln!(
        ss,
        "NNUE evaluation        {:+.2} (white side)",
        f64::from(white(nnue)) / f64::from(PAWN_VALUE)
    );
    let _ = writeln!(
        ss,
        "Final evaluation      {:+.2} (white side) [Use NNUE={}]",
        f64::from(white(v)) / f64::from(PAWN_VALUE),
        use_nnue.name()
    );
}

fn trace_legacy(pos: &Position<Atomic>, network: &Network, use_nnue: UseNnue) -> String {
    let mut accumulators = AccumulatorStack::new();
    let mut caches = AccumulatorCaches::new(network);

    let mut ss = String::new();
    ss.push('\n');
    ss += &network_trace_legacy(pos, network);
    ss.push('\n');

    let (psqt, positional) = network.evaluate_raw(pos, &mut accumulators, &mut caches);
    let nnue = value_from_raw(psqt, positional);
    let v = evaluate_legacy(network, pos, &mut accumulators, &mut caches, use_nnue);
    trace_values(&mut ss, nnue, v, pos, use_nnue);
    ss
}

/// The trace of a network in Stockfish's design: the table of each stored
/// layer stack's evaluation, and the lines of the network's output and of
/// the final evaluation.
fn trace_sfnnv20(pos: &Position<Atomic>, network: &sfnnv20::Network, use_nnue: UseNnue) -> String {
    let mut accumulators = sfnnv20::accumulator::AccumulatorStack::new();
    let mut caches = sfnnv20::accumulator::AccumulatorCaches::new(network);
    let t = network.trace_evaluate(pos, &mut accumulators, &mut caches);
    let nnue = network.evaluate(pos, &mut accumulators, &mut caches);

    let mut ss = String::new();
    ss += "\nNNUE network contributions (Normalized, ";
    ss += if pos.side_to_move() == WHITE {
        "White to move)\n"
    } else {
        "Black to move)\n"
    };
    ss += "+------------+------------+\n";
    ss += "|   Bucket   | Evaluation |\n";
    ss += "+------------+------------+\n";
    for (bucket, &positional) in t.positional.iter().enumerate() {
        let _ = write!(ss, "|  {bucket:<10}|  ");
        format_cp_aligned_dot(positional, &mut ss);
        ss += "   |";
        if bucket == t.correct_bucket {
            ss += " <-- this bucket is used";
        }
        ss.push('\n');
    }
    ss += "+------------+------------+\n\n";

    trace_values(&mut ss, nnue, scaled(nnue, pos, use_nnue), pos, use_nnue);
    ss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_material_scales_as_fairy_stockfish_s() {
        let scale = |fen| material_scale(&Position::<Atomic>::from_fen(fen, false).unwrap());
        assert_eq!(
            scale("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            1998
        );
        // 2 pawns and a rook: 924 + 64 + 32 * (1276 + 1400) / 1024.
        assert_eq!(scale("4k3/1P6/8/8/8/8/1p2r3/4K3 w - - 0 1"), 1071);
        assert_eq!(scale("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), 967);
    }
}
