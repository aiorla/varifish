//! A search value as UCI reports it: a mate in some plies, a tablebase win
//! or loss in some plies, or centipawns.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `score.h` and `score.cpp`: Stockfish, a UCI chess playing engine derived
//! from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! AUTHORS), under the GNU General Public License, version 3 or later.

use crate::position::Position;
use crate::types::*;
use crate::uci::to_cp;
use crate::variant::Variant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    Mate { plies: i32 },
    Tablebase { plies: i32, win: bool },
    InternalUnits { value: i32 },
}

impl Score {
    pub fn new(v: Value, pos: &Position<impl Variant>) -> Score {
        debug_assert!(-VALUE_INFINITE < v && v < VALUE_INFINITE);

        if !is_decisive(v) {
            Score::InternalUnits {
                value: to_cp(v, pos),
            }
        } else if v.abs() <= VALUE_TB {
            let distance = VALUE_TB - v.abs();
            if v > 0 {
                Score::Tablebase {
                    plies: distance,
                    win: true,
                }
            } else {
                Score::Tablebase {
                    plies: -distance,
                    win: false,
                }
            }
        } else {
            let distance = VALUE_MATE - v.abs();
            Score::Mate {
                plies: if v > 0 { distance } else { -distance },
            }
        }
    }
}
