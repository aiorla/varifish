//! The evaluation of Three-check positions: with Three-check's dedicated
//! SFNNv20 network when loaded, or else bootstrapping from chess's network
//! plus a symmetric check-count bonus.

use std::fmt::Write;

use crate::nnue::accumulator::AccumulatorStack;
use crate::nnue::eval_file::{Networks, UseNnue};
use crate::nnue::misc::{format_trace, trace as chess_trace};
use crate::nnue::threecheck::ThreeCheckCaches;
use crate::position::Position;
use crate::types::*;
use crate::uci::to_cp;
use crate::variant::ThreeCheck;

/// Bootstrap bonus in internal units for checks already given when evaluating
/// with the chess network (indexed by remaining checks `0..=3`).
const CHECK_BONUS: [Value; 4] = [0, 1400, 450, 0];

#[inline(always)]
fn check_bonus(pos: &Position<ThreeCheck>) -> Value {
    let us = pos.side_to_move();
    CHECK_BONUS[pos.checks_remaining(us) as usize] - CHECK_BONUS[pos.checks_remaining(!us) as usize]
}

/// The evaluation of a non-terminal Three-check position not in check, from
/// the side to move's point of view.
pub fn evaluate(
    networks: &Networks,
    pos: &Position<ThreeCheck>,
    accumulators: &mut AccumulatorStack,
    caches: &mut ThreeCheckCaches,
    use_nnue: UseNnue,
) -> Value {
    debug_assert!(pos.checkers().is_empty());
    debug_assert!(!pos.is_three_check_over());

    if let Some(network) = networks.threecheck.network() {
        let nnue = network.evaluate(pos, accumulators, &mut caches.0);
        if use_nnue == UseNnue::Pure {
            return nnue.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
        }
        return crate::evaluate::scale_evaluation(nnue, pos);
    }

    let chess_net = networks
        .chess
        .network()
        .expect("searches verify that a Three-check or chess network is loaded");
    let nnue = chess_net.evaluate(pos, accumulators, &mut caches.0);
    let base = if use_nnue == UseNnue::Pure {
        nnue
    } else {
        crate::evaluate::scale_evaluation(nnue, pos)
    };
    (base + check_bonus(pos)).clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

/// Evaluation trace for `eval`, from White's point of view.
pub fn trace(networks: &Networks, pos: &Position<ThreeCheck>, use_nnue: UseNnue) -> String {
    if pos.is_three_check_over() {
        return "Final evaluation: none (three checks given)".to_string();
    }
    if !pos.checkers().is_empty() {
        return "Final evaluation: none (in check)".to_string();
    }

    let mut accumulators = AccumulatorStack::new();
    let mut ss = String::new();
    ss.push('\n');

    let (raw, final_v, bootstrap) = if let Some(network) = networks.threecheck.network() {
        let mut caches = <ThreeCheckCaches as crate::nnue::Caches>::new(Some(network));
        let mut trace_acc = AccumulatorStack::new();
        let t = network.trace_evaluate(pos, &mut trace_acc, &mut caches.0);
        ss += &format_trace(pos, &t.positional[..t.layer_stacks], t.correct_bucket);
        ss.push('\n');

        let nnue = network.evaluate(pos, &mut accumulators, &mut caches.0);
        let scaled = if use_nnue == UseNnue::Pure {
            nnue.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
        } else {
            crate::evaluate::scale_evaluation(nnue, pos)
        };
        (nnue, scaled, false)
    } else {
        let chess_net = networks.chess.network().expect("the network is verified");
        let mut caches = crate::nnue::accumulator::AccumulatorCaches::new(chess_net);
        ss += &chess_trace(pos, chess_net, &mut caches);
        ss.push('\n');

        let nnue = chess_net.evaluate(pos, &mut accumulators, &mut caches);
        let scaled = if use_nnue == UseNnue::Pure {
            nnue
        } else {
            crate::evaluate::scale_evaluation(nnue, pos)
        };
        let total = (scaled + check_bonus(pos))
            .clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
        (nnue, total, true)
    };

    let _ = writeln!(
        ss,
        "NNUE evaluation          {raw:+} (side to move, internal units)"
    );
    if bootstrap {
        let bonus = check_bonus(pos);
        let _ = writeln!(
            ss,
            "Checks bonus             {bonus:+} (side to move, internal units)"
        );
    }

    let (raw, final_v) = if pos.side_to_move() == WHITE {
        (raw, final_v)
    } else {
        (-raw, -final_v)
    };

    let _ = writeln!(
        ss,
        "NNUE evaluation        {:+.2} (white side)",
        0.01 * f64::from(to_cp(raw, pos))
    );
    let _ = writeln!(
        ss,
        "Final evaluation      {:+.2} (white side)",
        0.01 * f64::from(to_cp(final_v, pos))
    );

    ss
}
