//! The input features K32Q2: the position of a perspective's own king (32
//! buckets) and whether the opponent has a queen (2 states), combined with
//! the position of each piece, the board mirrored so that the king is always
//! on files e to h.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/features/k32q2.h` and `.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

#[cfg(use_avx512icl)]
use crate::bitboard::{Bitboard, compress_squares};
use crate::misc::ValueList;
use crate::types::*;

// Unique number for each piece type on each square
const PS_NONE: u16 = 0;
const PS_W_PAWN: u16 = 0;
const PS_B_PAWN: u16 = 64;
const PS_W_KNIGHT: u16 = 2 * 64;
const PS_B_KNIGHT: u16 = 3 * 64;
const PS_W_BISHOP: u16 = 4 * 64;
const PS_B_BISHOP: u16 = 5 * 64;
const PS_W_ROOK: u16 = 6 * 64;
const PS_B_ROOK: u16 = 7 * 64;
const PS_W_QUEEN: u16 = 8 * 64;
const PS_B_QUEEN: u16 = 9 * 64;
const PS_KING: u16 = 10 * 64;
const PS_NB: u16 = 11 * 64;

#[rustfmt::skip]
const PIECE_SQUARE_INDEX: [[u16; PIECE_NB]; COLOR_NB] = [
    // Convention: W - us, B - them
    // Viewed from other side, W and B are reversed
    [PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE],
    [PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE],
];

const fn b(v: u16) -> u16 {
    v * 2 * PS_NB
}

#[rustfmt::skip]
const KING_BUCKETS: [u16; SQUARE_NB] = [
    b(28), b(29), b(30), b(31), b(31), b(30), b(29), b(28),
    b(24), b(25), b(26), b(27), b(27), b(26), b(25), b(24),
    b(20), b(21), b(22), b(23), b(23), b(22), b(21), b(20),
    b(16), b(17), b(18), b(19), b(19), b(18), b(17), b(16),
    b(12), b(13), b(14), b(15), b(15), b(14), b(13), b(12),
    b( 8), b( 9), b(10), b(11), b(11), b(10), b( 9), b( 8),
    b( 4), b( 5), b( 6), b( 7), b( 7), b( 6), b( 5), b( 4),
    b( 0), b( 1), b( 2), b( 3), b( 3), b( 2), b( 1), b( 0),
];

// Orient a square according to perspective (rotates by 180 for black)
const ORIENT_TBL: [u16; SQUARE_NB] = {
    let mut table = [0; SQUARE_NB];
    let mut s = 0;
    while s < SQUARE_NB {
        table[s] = if s % 8 < 4 {
            Square::H1.0
        } else {
            Square::A1.0
        } as u16;
        s += 1;
    }
    table
};

pub struct K32Q2;

/// Maximum number of simultaneously active features.
pub const MAX_ACTIVE_DIMENSIONS: usize = 32;
pub type IndexList = ValueList<u16, MAX_ACTIVE_DIMENSIONS>;

impl K32Q2 {
    /// Hash value embedded in the evaluation file
    pub const HASH_VALUE: u32 = 0x32b5_e284;

    /// Number of feature dimensions (45,056 piece-squares)
    pub const DIMENSIONS: usize = SQUARE_NB * PS_NB as usize;
}

#[repr(align(64))]
struct Offsets([[[[u16; PIECE_NB]; SQUARE_NB]; COLOR_NB]; 2]);

static OFFSETS: Offsets = {
    let mut table = [[[[0; PIECE_NB]; SQUARE_NB]; COLOR_NB]; 2];
    let mut q = 0;
    while q < 2 {
        let mut c = 0;
        while c < COLOR_NB {
            let mut sq = 0;
            while sq < SQUARE_NB {
                let flip = 56 * c as u16;
                let orient = ORIENT_TBL[sq] ^ flip;
                let queen_offset = if q == 1 { PS_NB } else { 0 };
                let mut piece_index = 0;
                while piece_index < PIECE_NB {
                    table[q][c][sq][piece_index] = PIECE_SQUARE_INDEX[c][piece_index]
                        + KING_BUCKETS[sq ^ flip as usize]
                        + queen_offset
                        + orient;
                    piece_index += 1;
                }
                sq += 1;
            }
            c += 1;
        }
        q += 1;
    }
    Offsets(table)
};

/// Index of a feature for a given king position and another piece on some
/// square: always less than DIMENSIONS, as the offsets leave the low six
/// bits to the square.
#[inline(always)]
pub fn make_index(
    perspective: Color,
    s: Square,
    pc: Piece,
    ksq: Square,
    opponent_has_queen: bool,
) -> u16 {
    s.idx() as u16
        ^ OFFSETS.0[usize::from(opponent_has_queen)][perspective.idx()][ksq.idx()][pc.idx()]
}

/// Writes the indices of the pieces of `old_pieces` on the squares of
/// `removed_bb` to `removed`, and of those of `new_pieces` on the squares of
/// `added_bb` to `added`, each list in one vector store: Stockfish's
/// `write_indices` for Ice Lake's AVX-512.
#[cfg(use_avx512icl)]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn write_indices(
    old_pieces: &[Piece; SQUARE_NB],
    new_pieces: &[Piece; SQUARE_NB],
    removed_bb: Bitboard,
    added_bb: Bitboard,
    perspective: Color,
    ksq: Square,
    opponent_has_queen: bool,
    removed: &mut IndexList,
    added: &mut IndexList,
) {
    use std::arch::x86_64::*;

    let offsets = &OFFSETS.0[usize::from(opponent_has_queen)][perspective.idx()][ksq.idx()];
    // SAFETY: the build targets Ice Lake's AVX-512, the boards are 64 bytes
    // and the offsets 16 u16, and each store writes 32 indices, the first
    // those of its bitboard's squares, at most 32 as the pieces are.
    unsafe {
        // Each piece's offset, which leaves the low six bits to the square
        let offsets = _mm512_castsi256_si512(_mm256_loadu_si256(offsets.as_ptr().cast()));
        let indices = |bb: Bitboard, pieces: &[Piece; SQUARE_NB]| {
            let pieces = _mm512_loadu_si512(pieces.as_ptr().cast());
            let squares = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(compress_squares(bb)));
            let pieces = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_maskz_compress_epi8(
                bb.0, pieces,
            )));
            _mm512_xor_si512(squares, _mm512_permutexvar_epi16(pieces, offsets))
        };
        let removed_indices = indices(removed_bb, old_pieces);
        let added_indices = indices(added_bb, new_pieces);
        removed.extend_by_store(32, removed_bb.popcount() as usize, |p| {
            _mm512_storeu_si512(p.cast(), removed_indices)
        });
        added.extend_by_store(32, added_bb.popcount() as usize, |p| {
            _mm512_storeu_si512(p.cast(), added_indices)
        });
    }
}

/// Get a list of indices for recently changed features
#[inline(always)]
pub fn append_changed_indices(
    perspective: Color,
    ksq: Square,
    diff: &DirtyPiece,
    opponent_has_queen: bool,
    removed: &mut IndexList,
    added: &mut IndexList,
) {
    let q = opponent_has_queen;
    removed.push(make_index(perspective, diff.from, diff.pc, ksq, q));
    if diff.to != Square::NONE {
        added.push(make_index(perspective, diff.to, diff.pc, ksq, q));
    }

    if diff.remove_sq != Square::NONE {
        removed.push(make_index(
            perspective,
            diff.remove_sq,
            diff.remove_pc,
            ksq,
            q,
        ));
    }

    if diff.add_sq != Square::NONE {
        added.push(make_index(perspective, diff.add_sq, diff.add_pc, ksq, q));
    }
}

/// Returns whether the change stored in this DirtyPiece means that a full
/// accumulator refresh is required: a move of the perspective's king, or a
/// capture or promotion that may change whether the opponent has a queen.
#[inline(always)]
pub fn requires_refresh(diff: &DirtyPiece, perspective: Color) -> bool {
    if diff.pc == make_piece(perspective, KING) {
        return true;
    }

    let opponent = !perspective;
    (diff.remove_sq != Square::NONE && diff.remove_pc == make_piece(opponent, QUEEN))
        || (diff.add_sq != Square::NONE && diff.add_pc == make_piece(opponent, QUEEN))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_fill_the_dimensions_and_mirror_the_board() {
        let mut seen = vec![false; K32Q2::DIMENSIONS];
        for q in [false, true] {
            for c in [WHITE, BLACK] {
                for ksq in 0..64 {
                    for pc in PIECES {
                        for s in 0..64 {
                            let index = make_index(c, Square(s), pc, Square(ksq), q);
                            assert!(usize::from(index) < K32Q2::DIMENSIONS);
                            seen[usize::from(index)] = true;
                            // Mirroring the board and swapping colors for
                            // Black leaves the index unchanged.
                            let mirrored = make_index(!c, Square(s ^ 56), !pc, Square(ksq ^ 56), q);
                            assert_eq!(index, mirrored);
                            let flipped = make_index(c, Square(s ^ 7), pc, Square(ksq ^ 7), q);
                            assert_eq!(index, flipped);
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s));
    }
}
