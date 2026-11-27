//! The table of the evaluations by each layer stack, for `eval`.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/nnue_misc.h` and `.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

use std::fmt::Write;

use super::accumulator::{AccumulatorCaches, AccumulatorStack};
use super::network::Network;
use crate::position::Position;
use crate::types::*;
use crate::uci::to_cp;

use crate::variant::Variant;

/// Converts a Value into pawns, always keeping two decimals
fn format_cp_aligned_dot<V: Variant>(v: Value, out: &mut String, pos: &Position<V>) {
    let pawns = (0.01 * f64::from(to_cp(v, pos))).abs();
    let sign = match v {
        ..0 => '-',
        1.. => '+',
        0 => ' ',
    };
    let _ = write!(out, "{sign}{pawns:6.2}");
}

/// Formats a table of layer-stack evaluations for `pos`.
pub fn format_trace<V: Variant>(
    pos: &Position<V>,
    positional: &[Value],
    correct_bucket: usize,
) -> String {
    let mut ss = String::new();

    ss += "NNUE network contributions (Normalized, ";
    ss += if pos.side_to_move() == WHITE {
        "White to move)\n"
    } else {
        "Black to move)\n"
    };
    ss += "+------------+------------+\n";
    ss += "|   Bucket   | Evaluation |\n";
    ss += "+------------+------------+\n";

    for (bucket, &value) in positional.iter().enumerate() {
        let _ = write!(ss, "|  {bucket:<10}|  ");
        format_cp_aligned_dot(value, &mut ss, pos);
        ss += "   |";
        if bucket == correct_bucket {
            ss += " <-- this bucket is used";
        }
        ss.push('\n');
    }

    ss += "+------------+------------+\n";

    ss
}

/// Returns a string with the table for evaluation values bucket by bucket.
pub fn trace<V: Variant>(
    pos: &Position<V>,
    network: &Network,
    caches: &mut AccumulatorCaches,
) -> String {
    let mut accumulators = AccumulatorStack::new();
    accumulators.reset();

    let t = network.trace_evaluate(pos, &mut accumulators, caches);
    format_trace(pos, &t.positional[..t.layer_stacks], t.correct_bucket)
}
