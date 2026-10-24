//! The input features HalfKAv2Atomic: the square of a perspective's own king
//! combined with the square and piece of each piece, Black's board flipped
//! vertically. Fairy-Stockfish played Atomic with a royal commoner, whose
//! plane both kings share; an exploded king counts as on A1.
//!
//! Derived, with changes for varifish in 2026, from Atomic-Stockfish's
//! `nnue/features/half_ka_v2_atomic.h` and `.cpp`
//! (https://github.com/Belzedar94/Atomic-Stockfish), under the GNU General
//! Public License, version 3 or later.

use crate::misc::ValueList;
use crate::position::Position;
use crate::types::*;
use crate::variant::Atomic;

/// Hash value embedded in the evaluation file
pub const HASH_VALUE: u32 = 0x5F23_4CB8;

const PS_NB: u32 = 11 * SQUARE_NB as u32;

/// Number of feature dimensions
pub const DIMENSIONS: usize = SQUARE_NB * PS_NB as usize;

/// The most features a position has: one per piece
pub const MAX_ACTIVE_DIMENSIONS: usize = 32;

/// The features a refresh changes, at most one per piece of a board.
pub type IndexList = ValueList<u32, MAX_ACTIVE_DIMENSIONS>;
/// The features a move removes: the moved piece, the captured one and up
/// to 8 the capture explodes, in a list of a power of two.
pub type RemovedList = ValueList<u32, 16>;
/// The features a move adds: the moved piece, and the rook of castling or
/// the piece a pawn promotes to.
pub type AddedList = ValueList<u32, 2>;

const PS_OWN_PAWN: u32 = 0;
const PS_OPP_PAWN: u32 = 64;
const PS_OWN_KNIGHT: u32 = 2 * 64;
const PS_OPP_KNIGHT: u32 = 3 * 64;
const PS_OWN_BISHOP: u32 = 4 * 64;
const PS_OPP_BISHOP: u32 = 5 * 64;
const PS_OWN_ROOK: u32 = 6 * 64;
const PS_OPP_ROOK: u32 = 7 * 64;
const PS_OWN_QUEEN: u32 = 8 * 64;
const PS_OPP_QUEEN: u32 = 9 * 64;
const PS_KING: u32 = 10 * 64;
const PS_NONE: u32 = 0;

#[rustfmt::skip]
const PIECE_SQUARE_INDEX: [[u32; PIECE_NB]; COLOR_NB] = [
    // Own pieces are the perspective's, opposing pieces the other side's.
    [PS_NONE, PS_OWN_PAWN, PS_OWN_KNIGHT, PS_OWN_BISHOP, PS_OWN_ROOK, PS_OWN_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_OPP_PAWN, PS_OPP_KNIGHT, PS_OPP_BISHOP, PS_OPP_ROOK, PS_OPP_QUEEN, PS_KING, PS_NONE],
    [PS_NONE, PS_OPP_PAWN, PS_OPP_KNIGHT, PS_OPP_BISHOP, PS_OPP_ROOK, PS_OPP_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_OWN_PAWN, PS_OWN_KNIGHT, PS_OWN_BISHOP, PS_OWN_ROOK, PS_OWN_QUEEN, PS_KING, PS_NONE],
];

/// The square seen from a perspective. Both see a missing king on A1.
#[inline(always)]
pub fn orient(perspective: Color, s: Square) -> Square {
    if s == Square::NONE {
        Square::A1
    } else if perspective == WHITE {
        s
    } else {
        s.flip_rank()
    }
}

/// The square of a perspective's king, or `Square::NONE` once it exploded.
#[inline(always)]
pub fn king_square(pos: &Position<Atomic>, perspective: Color) -> Square {
    let kings = pos.pieces_cpt(perspective, KING);
    if kings.any() {
        kings.lsb()
    } else {
        Square::NONE
    }
}

/// Index of a feature for a given king position and another piece on some
/// square
#[inline(always)]
pub fn make_index(perspective: Color, s: Square, pc: Piece, ksq: Square) -> u32 {
    debug_assert!(s != Square::NONE && pc != NO_PIECE);
    u32::from(orient(perspective, s).0)
        + PIECE_SQUARE_INDEX[perspective.idx()][pc.idx()]
        + u32::from(orient(perspective, ksq).0) * PS_NB
}

/// Whether a move changes the king square of `perspective`, which every
/// feature of the perspective depends on.
#[inline(always)]
pub fn requires_refresh(dirties: &AtomicDirties, perspective: Color) -> bool {
    let king = make_piece(perspective, KING);
    let dp = &dirties.dirty_piece;
    dp.pc == king
        || (dp.remove_sq != Square::NONE && dp.remove_pc == king)
        || dirties.blast.iter().any(|(pc, _)| pc == king)
}

/// The features a move removes and adds, for a perspective whose king is on
/// `ksq` before and after it.
#[inline(always)]
pub fn append_changed_indices(
    perspective: Color,
    ksq: Square,
    dirties: &AtomicDirties,
    removed: &mut RemovedList,
    added: &mut AddedList,
) {
    debug_assert!(ksq != Square::NONE);
    let dp = &dirties.dirty_piece;
    removed.push(make_index(perspective, dp.from, dp.pc, ksq));
    if dp.to != Square::NONE {
        added.push(make_index(perspective, dp.to, dp.pc, ksq));
    }
    if dp.remove_sq != Square::NONE {
        removed.push(make_index(perspective, dp.remove_sq, dp.remove_pc, ksq));
    }
    if dp.add_sq != Square::NONE {
        added.push(make_index(perspective, dp.add_sq, dp.add_pc, ksq));
    }
    for (pc, s) in dirties.blast.iter() {
        removed.push(make_index(perspective, s, pc, ksq));
    }
}
