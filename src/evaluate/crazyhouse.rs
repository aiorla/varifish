//! The evaluation of Crazyhouse positions: with Crazyhouse's dedicated
//! SFNNv20 network when loaded, or else bootstrapping from chess's network
//! plus a symmetric pocket material bonus.

use std::fmt::Write;

use crate::nnue::accumulator::AccumulatorStack;
use crate::nnue::crazyhouse::CrazyhouseCaches;
use crate::nnue::eval_file::{Networks, UseNnue};
use crate::nnue::misc::{format_trace, trace as chess_trace};
use crate::position::{HAND_TYPES, Position};
use crate::types::*;
use crate::uci::to_cp;
use crate::variant::Crazyhouse;

/// Bootstrap pocket piece values in internal units (`PAWN..=QUEEN`), used only
/// when evaluating with chess's network before Crazyhouse's own network exists.
const HAND_VALUE: [Value; HAND_TYPES] = [240, 780, 820, 1100, 2100];

#[inline(always)]
fn hand_bonus(pos: &Position<Crazyhouse>) -> Value {
    let us = pos.side_to_move();
    let them = !us;
    let extra = &pos.state().extra;
    let mut bonus = 0;
    for (pt_idx, &val) in HAND_VALUE.iter().enumerate() {
        let diff =
            i32::from(extra.hand[us.idx()][pt_idx]) - i32::from(extra.hand[them.idx()][pt_idx]);
        bonus += diff * val;
    }
    bonus
}

/// The evaluation of a non-terminal Crazyhouse position not in check, from
/// the side to move's point of view.
pub fn evaluate(
    networks: &Networks,
    pos: &Position<Crazyhouse>,
    accumulators: &mut AccumulatorStack,
    caches: &mut CrazyhouseCaches,
    use_nnue: UseNnue,
) -> Value {
    debug_assert!(pos.checkers().is_empty());

    if let Some(network) = networks.crazyhouse.network() {
        let nnue = network.evaluate(pos, accumulators, &mut caches.0);
        if use_nnue == UseNnue::Pure {
            return nnue.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
        }
        return crate::evaluate::scale_evaluation(nnue, pos);
    }

    let chess_net = networks
        .chess
        .network()
        .expect("searches verify that a Crazyhouse or chess network is loaded");
    let nnue = chess_net.evaluate(pos, accumulators, &mut caches.0);
    let base = if use_nnue == UseNnue::Pure {
        nnue
    } else {
        crate::evaluate::scale_evaluation(nnue, pos)
    };
    (base + hand_bonus(pos)).clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

/// Evaluation trace for `eval`, from White's point of view.
pub fn trace(networks: &Networks, pos: &Position<Crazyhouse>, use_nnue: UseNnue) -> String {
    if !pos.checkers().is_empty() {
        return "Final evaluation: none (in check)".to_string();
    }

    let mut accumulators = AccumulatorStack::new();
    let mut ss = String::new();
    ss.push('\n');

    let (raw, final_v, bootstrap) = if let Some(network) = networks.crazyhouse.network() {
        let mut caches = <CrazyhouseCaches as crate::nnue::Caches>::new(Some(network));
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
        let total = (scaled + hand_bonus(pos))
            .clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
        (nnue, total, true)
    };

    let _ = writeln!(
        ss,
        "NNUE evaluation          {raw:+} (side to move, internal units)"
    );
    if bootstrap {
        let bonus = hand_bonus(pos);
        let _ = writeln!(
            ss,
            "Hand bonus               {bonus:+} (side to move, internal units)"
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
