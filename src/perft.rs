//! Perft, which counts the leaves of the tree of legal moves to a depth, to
//! verify move generation.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `perft.h`: Stockfish, a UCI chess playing engine derived from Glaurung 2.1,
//! Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under the
//! GNU General Public License, version 3 or later.

use crate::movegen::{ALL, MoveList};
use crate::position::{Position, PositionSetError};
use crate::types::*;
use crate::variant::Variant;

/// The number of leaves at `depth` plies, calling `on_root_move` with each
/// root move and its count. Depths below 1 count each root move once, as
/// depth 1 does.
pub fn perft_divide<V: Variant>(
    pos: &mut Position<V>,
    depth: Depth,
    mut on_root_move: impl FnMut(Move, u64),
) -> u64 {
    let mut dirties = V::Dirties::default();
    let mut nodes = 0;

    for &m in &MoveList::new::<ALL>(pos) {
        let cnt = if depth <= 1 {
            1
        } else {
            pos.make_move(m, &mut dirties);
            let cnt = if depth == 2 {
                MoveList::new::<ALL>(pos).len() as u64
            } else {
                perft_below(pos, depth - 1, &mut dirties)
            };
            pos.undo_move(m);
            cnt
        };
        nodes += cnt;
        on_root_move(m, cnt);
    }
    nodes
}

/// The number of leaves at `depth` plies, at least 2.
fn perft_below<V: Variant>(pos: &mut Position<V>, depth: Depth, dirties: &mut V::Dirties) -> u64 {
    let leaf = depth == 2;
    let mut nodes = 0;

    for &m in &MoveList::new::<ALL>(pos) {
        pos.make_move(m, dirties);
        nodes += if leaf {
            MoveList::new::<ALL>(pos).len() as u64
        } else {
            perft_below(pos, depth - 1, dirties)
        };
        pos.undo_move(m);
    }
    nodes
}

/// The perft of a FEN in the variant `V`, as `go perft` computes it.
pub fn perft<V: Variant>(
    fen: &str,
    depth: Depth,
    is_chess960: bool,
    on_root_move: impl FnMut(Move, u64),
) -> Result<u64, PositionSetError> {
    let mut pos = Position::<V>::from_fen(fen, is_chess960)?;
    Ok(perft_divide(&mut pos, depth, on_root_move))
}
