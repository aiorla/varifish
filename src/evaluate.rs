//! The evaluation of chess positions: the network's, scaled by the material
//! and the moves without progress, and its trace for `eval`. Atomic's is in
//! `atomic`.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `evaluate.h` and `evaluate.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

pub mod atomic;

use std::fmt::Write;

use crate::nnue::accumulator::{AccumulatorCaches, AccumulatorStack};
use crate::nnue::network::Network;
use crate::position::Position;
use crate::types::*;
use crate::uci::to_cp;

// The default net name MUST follow the format nn-[SHA256 first 12 digits].nnue
// for the build process to work: build.rs and the Makefile read it from this
// line, which must stay as it is.
pub const EVAL_FILE_DEFAULT_NAME: &str = "nn-c9e480e92425.nnue";

fn simple_eval(pos: &Position) -> i32 {
    let c = pos.side_to_move();
    PAWN_VALUE * (pos.count(c, PAWN) - pos.count(!c, PAWN)) + pos.non_pawn_material(c)
        - pos.non_pawn_material(!c)
}

/// The evaluation of a position not in check, from the side to move's
/// point of view, whose state the stack's latest state holds.
pub fn evaluate(
    network: &Network,
    pos: &Position,
    accumulators: &mut AccumulatorStack,
    caches: &mut AccumulatorCaches,
    optimism: i32,
) -> Value {
    debug_assert!(pos.checkers().is_empty());
    let nnue = network.evaluate(pos, accumulators, caches);
    scale_evaluation(nnue, optimism, pos)
}

/// Applies search-dependent scaling (optimism and rule50) to the raw NNUE
/// eval
fn scale_evaluation(nnue: Value, optimism: i32, pos: &Position) -> Value {
    let se = simple_eval(pos);

    // Normalize the raw evaluations to [-1024, 1024] to measure their
    // correlation.
    let se_norm = (se * 1024) / (se.abs() + 1024);
    let nnue_norm = (nnue * 1024) / (nnue.abs() + 1024);
    // When NNUE and material agree (positive correlation), the position is
    // straightforward; otherwise (negative correlation) it involves complex
    // compensation. In a representative sample, correl averages -1 or so,
    // i.e. it is well-centered in [-2048, 2048].
    let correl = (se_norm * nnue_norm) / 512;

    // When winning, we boost easy positions, and vice versa
    let mut v = nnue + (nnue * correl) / 65536 + (optimism * correl) / 16384;

    // Scale the combined evaluation by total material
    let material = 521 * pos.count_both(PAWN) + pos.non_pawn_material_total();
    v += (i64::from(material) * i64::from(v) / 90649) as i32;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 189;

    // Guarantee that the evaluation does not hit the tablebase range
    v.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

/// Like evaluate(), but instead of returning a value, it returns a string
/// (suitable for outputting to stdout) that contains the detailed
/// descriptions and values of each evaluation term. Useful for debugging.
/// Trace scores are from white's point of view
pub fn trace(pos: &Position, network: &Network) -> String {
    if !pos.checkers().is_empty() {
        return "Final evaluation: none (in check)".to_string();
    }

    let mut accumulators = AccumulatorStack::new();
    let mut caches = AccumulatorCaches::new(network);

    let mut ss = String::new();
    ss.push('\n');
    ss += &crate::nnue::misc::trace(pos, network, &mut caches);
    ss.push('\n');

    let nnue = network.evaluate(pos, &mut accumulators, &mut caches);
    let s_v = scale_evaluation(nnue, VALUE_ZERO, pos); // requires stm perspective

    let _ = writeln!(
        ss,
        "NNUE evaluation          {nnue:+} (side to move, internal units)"
    );

    let (nnue, s_v) = if pos.side_to_move() == WHITE {
        (nnue, s_v)
    } else {
        (-nnue, -s_v)
    };

    let _ = writeln!(
        ss,
        "NNUE evaluation        {:+.2} (white side)",
        0.01 * f64::from(to_cp(nnue, pos))
    );
    let _ = writeln!(
        ss,
        "Final evaluation      {:+.2} (white side) [with scaled NNUE, ...]",
        0.01 * f64::from(to_cp(s_v, pos))
    );

    ss
}
