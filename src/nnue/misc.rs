//! The table of the evaluations by each layer stack, for `eval`.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/nnue_misc.h` and `.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

use std::fmt::Write;

use super::accumulator::{AccumulatorCaches, AccumulatorStack};
use super::architecture::LAYER_STACKS;
use super::network::Network;
use crate::position::Position;
use crate::types::*;
use crate::uci::to_cp;

/// Converts a Value into pawns, always keeping two decimals
fn format_cp_aligned_dot(v: Value, out: &mut String, pos: &Position) {
    let pawns = (0.01 * f64::from(to_cp(v, pos))).abs();
    let sign = match v {
        ..0 => '-',
        1.. => '+',
        0 => ' ',
    };
    let _ = write!(out, "{sign}{pawns:6.2}");
}

/// Returns a string with the table for evaluation values bucket by bucket.
pub fn trace(pos: &Position, network: &Network, caches: &mut AccumulatorCaches) -> String {
    let mut ss = String::new();

    let mut accumulators = AccumulatorStack::new();
    accumulators.reset();

    let t = network.trace_evaluate(pos, &mut accumulators, caches);

    ss += "NNUE network contributions (Normalized, ";
    ss += if pos.side_to_move() == WHITE {
        "White to move)\n"
    } else {
        "Black to move)\n"
    };
    ss += "+------------+------------+\n";
    ss += "|   Bucket   | Evaluation |\n";
    ss += "+------------+------------+\n";

    for bucket in 0..LAYER_STACKS {
        let _ = write!(ss, "|  {bucket:<10}|  ");
        format_cp_aligned_dot(t.positional[bucket], &mut ss, pos);
        ss += "   |";
        if bucket == t.correct_bucket {
            ss += " <-- this bucket is used";
        }
        ss.push('\n');
    }

    ss += "+------------+------------+\n";

    ss
}
