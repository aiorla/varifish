//! Time management: how long to think about a move, from the clock, the
//! increment, the moves to go and the game ply.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `timeman.h` and `timeman.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers
//! (see AUTHORS), under the GNU General Public License, version 3 or later.
//! Atomic manages its time as Atomic-Stockfish
//! (https://github.com/Belzedar94/Atomic-Stockfish) does, which forked
//! Stockfish before its latest changes here.

use crate::misc::{TimePoint, max_f64, min_f64, mul_add, now};
use crate::search::{LimitsType, SearchOptions};
use crate::types::*;
use crate::variant::Variant;

const NO_BOUND: TimePoint = TimePoint::MAX / 2;

/// The TimeManagement class computes the optimal time to think depending on
/// the maximum available time, the game move number, and other parameters.
#[derive(Clone, Debug)]
pub struct TimeManagement {
    start_time: TimePoint,
    optimum_time: TimePoint,
    maximum_time: TimePoint,

    // Related to 'nodes as time' mode:
    use_nodes_time: bool,
    available_nodes: i64,
    previous_moves_to_go: i32,
    cyclic_budget: i64,
}

impl Default for TimeManagement {
    fn default() -> Self {
        TimeManagement {
            start_time: 0,
            optimum_time: NO_BOUND,
            maximum_time: NO_BOUND,
            use_nodes_time: false,
            available_nodes: -1,
            previous_moves_to_go: 0,
            cyclic_budget: 0,
        }
    }
}

impl TimeManagement {
    pub fn optimum(&self) -> TimePoint {
        self.optimum_time
    }

    pub fn maximum(&self) -> TimePoint {
        self.maximum_time
    }

    /// The time elapsed since the search started, or in 'nodes as time'
    /// mode the nodes searched, which `nodes` counts.
    #[inline]
    pub fn elapsed(&self, nodes: impl FnOnce() -> u64) -> TimePoint {
        if self.use_nodes_time {
            nodes() as TimePoint
        } else {
            self.elapsed_time()
        }
    }

    pub fn elapsed_time(&self) -> TimePoint {
        now() - self.start_time
    }

    pub fn clear(&mut self) {
        self.available_nodes = -1; // When in 'nodes as time' mode
        self.previous_moves_to_go = 0;
    }

    pub fn advance_nodes_time(&mut self, nodes: i64) {
        debug_assert!(self.use_nodes_time);
        self.available_nodes = (self.available_nodes - nodes).max(0);
    }

    /// Called at the beginning of the search and calculates the bounds of
    /// time allowed for the current game ply. We currently support:
    ///
    /// 1. x basetime (+ z increment)
    /// 2. x moves in y seconds (+ z increment)
    pub fn init<V: Variant>(
        &mut self,
        limits: &mut LimitsType,
        us: Color,
        ply: i32,
        options: &SearchOptions,
        original_time_adjust: &mut f64,
    ) {
        let npmsec = TimePoint::from(options.nodestime);
        let us = us.idx();
        let them = 1 - us;

        // If we have no time, we don't need to fully initialize TM.
        // startTime is used by movetime and useNodesTime is used in elapsed
        // calls.
        self.start_time = limits.start_time;
        self.use_nodes_time = npmsec != 0;

        // Atomic-Stockfish leaves movetime in milliseconds, which 'nodes as
        // time' then compares with the nodes searched
        if !V::ATOMIC && self.use_nodes_time {
            limits.movetime *= npmsec;
        }

        if limits.time[us] == 0 {
            self.optimum_time = NO_BOUND;
            self.maximum_time = NO_BOUND;
            return;
        }

        let mut move_overhead = TimePoint::from(options.move_overhead);

        // If we have to play in 'nodes as time' mode, then convert from time
        // to nodes, and use resulting values in time management formulas.
        // WARNING: to avoid time losses, the given npmsec (nodes per
        // millisecond) must be much lower than the real engine speed.
        if self.use_nodes_time {
            if self.available_nodes == -1 {
                // Only once at game start: the first time limit includes the
                // increment (both are in milliseconds)
                self.available_nodes = npmsec * limits.time[us];
                self.cyclic_budget = npmsec * (limits.time[us] - limits.inc[us]);
            } else if !V::ATOMIC
                && limits.movestogo > 0
                && limits.movestogo > self.previous_moves_to_go
                && self.cyclic_budget > 0
            {
                self.available_nodes += self.cyclic_budget;
            }

            self.previous_moves_to_go = limits.movestogo;

            // Convert from milliseconds to nodes
            limits.time[us] = self.available_nodes;
            limits.inc[us] *= npmsec;
            limits.npmsec = npmsec;
            move_overhead *= npmsec;
        }

        // These numbers are used where multiplications, divisions, or
        // comparisons with constants are involved.
        let scale_factor = if self.use_nodes_time { npmsec } else { 1 };
        let scaled_time = limits.time[us] / scale_factor;
        let scaled_time = if V::ATOMIC {
            scaled_time
        } else {
            scaled_time.max(1)
        };

        // Maximum move horizon
        let mut mtg = if limits.movestogo != 0 {
            limits.movestogo.min(50)
        } else {
            50
        };

        // If less than one second, gradually reduce mtg. In cyclic time
        // controls we keep the actual movestogo as horizon, which Atomic
        // does not.
        if scaled_time < 1000 && (V::ATOMIC || limits.movestogo == 0) {
            mtg = (scaled_time as f64 * 0.05) as i32;
        }

        // Make sure timeLeft is > 0 since we may use it as a divisor
        let time_left = (limits.time[us] + limits.inc[us] * TimePoint::from(mtg - 1)
            - move_overhead * TimePoint::from(2 + mtg))
        .max(1);

        // optScale is a percentage of available time to use for the current
        // move. maxScale is a multiplier applied to optimumTime.
        let mut opt_scale;
        let max_scale;

        if limits.movestogo == 0 {
            // x basetime (+ z increment): if there is a healthy increment,
            // timeLeft can exceed the actual available game time for the
            // current move, so also cap to a percentage of available game
            // time.

            // Extra time according to timeLeft
            if *original_time_adjust < 0.0 {
                *original_time_adjust = mul_add(0.3272, (time_left as f64).log10(), -0.4141);
            }

            // Calculate time constants based on current time left.
            let log_time_in_sec = (scaled_time as f64 / 1000.0).log10();
            let opt_constant = min_f64(mul_add(0.00033554, log_time_in_sec, 0.0029869), 0.004905);
            let max_constant = max_f64(mul_add(3.0608, log_time_in_sec, 3.3744), 3.1441);

            opt_scale = min_f64(
                mul_add(
                    (f64::from(ply) + 3.22713).powf(0.46866),
                    opt_constant,
                    0.012112,
                ),
                0.19404 * limits.time[us] as f64 / time_left as f64,
            ) * *original_time_adjust;

            max_scale = min_f64(6.873, max_constant + f64::from(ply) / 12.352);
        } else {
            // x moves in y seconds (+ z increment)
            opt_scale = min_f64(
                (0.88 + f64::from(ply) / 116.4) / f64::from(mtg),
                0.88 * limits.time[us] as f64 / time_left as f64,
            );
            max_scale = mul_add(0.11, f64::from(mtg), 1.3);
        }

        // Decrease time usage if behind in time. This is skipped in two
        // cases:
        // - if the nodestime option is used we can't calculate the opponent
        //   nodes budget in a deterministic way.
        // - if we use a cyclic time management (like 40/10) calculating time
        //   advantage for the last move (movestogo = 1) can be vastly off,
        //   because if the opponent had done his last move before us his
        //   time budget includes already the next cycle time increment but
        //   our not. This leads to a unnecessary big decrease in time usage
        //   which favors blunders.
        // Warning: don't remove this conditions.
        // Atomic does not decrease it.
        if !V::ATOMIC && !self.use_nodes_time && limits.movestogo != 1 {
            let time_advantage = (limits.time[us] - limits.time[them]) as f64
                / (1.0 + limits.time[us] as f64 + limits.time[them] as f64);
            opt_scale *= mul_add(0.9, min_f64(time_advantage, 0.0), 1.0);
        }

        // Limit the maximum possible time for this move
        self.optimum_time = max_f64(1.0, opt_scale * time_left as f64) as TimePoint;
        self.maximum_time = max_f64(
            self.optimum_time as f64,
            min_f64(
                mul_add(0.8097, limits.time[us] as f64, -(move_overhead as f64)),
                max_scale * self.optimum_time as f64,
            ),
        ) as TimePoint;

        if options.ponder {
            self.optimum_time += self.optimum_time / 4;
        }
    }
}
