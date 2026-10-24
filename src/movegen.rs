//! Legal move generation, in Stockfish's order.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `movegen.h` and `movegen.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers
//! (see AUTHORS), under the GNU General Public License, version 3 or later.

use crate::attacks::*;
use crate::bitboard::*;
use crate::misc::ValueList;
use crate::position::Position;
use crate::types::*;
use crate::variant::Variant;

/// Which legal moves to generate. Atomic's CAPTURES and QUIETS are legal in
/// check too, for its move picker, which has no evasion stage.
pub type GenType = u8;

/// Captures and queen promotions
pub const CAPTURES: GenType = 0;
/// Non-captures and underpromotions
pub const QUIETS: GenType = 1;
/// Check evasions
pub const EVASIONS: GenType = 2;
/// All legal moves
pub const ALL: GenType = 3;

/// The capacity of a list of moves. Each side has at most 16 pieces, which
/// `Position::set` checks and legal moves keep: at most 15 besides the king,
/// with at most 27 moves each (a queen; a promoting pawn has 12), and the king
/// with 8 moves and 2 castlings. So a position has at most
/// 15 * 27 + 10 = 415 legal moves.
pub const MOVE_LIST_SIZE: usize = 512;

pub type Moves = ValueList<Move, MOVE_LIST_SIZE>;

/// The legal moves of a kind in a position, in the order Stockfish generates
/// them in chess.
pub struct MoveList {
    moves: Moves,
}

impl MoveList {
    #[inline(always)]
    pub fn new<const T: GenType>(pos: &Position<impl Variant>) -> MoveList {
        let mut moves = Moves::new();
        pos.generate::<T>(&mut moves);
        MoveList { moves }
    }

    /// An empty list, for `generate` to fill where it lies, as the move
    /// picker does: a list returned from `new` may be copied whole.
    #[inline(always)]
    pub fn empty() -> MoveList {
        MoveList {
            moves: Moves::new(),
        }
    }

    /// Appends the legal moves of kind `T`.
    #[inline(always)]
    pub fn generate<const T: GenType>(&mut self, pos: &Position<impl Variant>) {
        pos.generate::<T>(&mut self.moves);
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    #[inline(always)]
    pub fn contains(&self, m: Move) -> bool {
        self.moves.contains(&m)
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        self.moves.as_slice()
    }

    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, Move> {
        self.moves.iter()
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Appends the legal chess moves of kind `T` to `moves`: CAPTURES (captures
/// and queen promotions), QUIETS (non-captures and underpromotions) or
/// EVASIONS, of which the side to move must be in check for EVASIONS only,
/// or ALL.
#[inline]
pub fn generate<const T: GenType>(pos: &Position, moves: &mut Moves) {
    if T == ALL {
        if pos.checkers().any() {
            generate_all::<EVASIONS>(pos, moves);
        } else {
            generate_all::<CAPTURES>(pos, moves);
            generate_all::<QUIETS>(pos, moves);
        }
    } else {
        debug_assert!((T == EVASIONS) == pos.checkers().any());
        generate_all::<T>(pos, moves);
    }
}

/// The pawn moves to the squares of `to_bb` from `offset` behind each. With
/// Ice Lake's AVX-512, as Stockfish's avx512icl builds splat them, all in one
/// vector store: the squares compressed, widened to the moves' 16 bits, and
/// each origin made from its square.
#[inline(always)]
fn splat_pawn_moves(moves: &mut Moves, to_bb: Bitboard, offset: Direction) {
    #[cfg(use_avx512icl)]
    {
        use std::arch::x86_64::*;
        // One move for each pawn of a side
        debug_assert!(to_bb.popcount() <= 8);
        let count = to_bb.popcount() as usize;
        // SAFETY: the build targets Ice Lake's AVX-512, and the store writes
        // eight moves, the first `count` those of `to_bb`.
        unsafe {
            let to = _mm_cvtepi8_epi16(_mm512_castsi512_si128(compress_squares(to_bb)));
            let from = _mm_sub_epi16(to, _mm_set1_epi16(offset as i16));
            let m = _mm_or_si128(_mm_slli_epi16::<6>(from), to);
            moves.extend_by_store(8, count, |p| _mm_storeu_si128(p.cast(), m));
        }
    }
    #[cfg(not(use_avx512icl))]
    for to in to_bb {
        moves.push(Move::new(to - offset, to));
    }
}

/// The moves from `from` to the squares of `to_bb`, with Ice Lake's AVX-512
/// in one vector store, as for pawns.
#[inline(always)]
fn splat_moves(moves: &mut Moves, from: Square, to_bb: Bitboard) {
    #[cfg(use_avx512icl)]
    {
        use std::arch::x86_64::*;
        // A queen has at most 27 moves.
        debug_assert!(to_bb.popcount() <= 32);
        let count = to_bb.popcount() as usize;
        // SAFETY: the build targets Ice Lake's AVX-512, and the store writes
        // 32 moves, the first `count` those of `to_bb`.
        unsafe {
            let to = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(compress_squares(to_bb)));
            let from = _mm512_set1_epi16(Move::new(from, Square(0)).0 as i16);
            let m = _mm512_or_si512(from, to);
            moves.extend_by_store(32, count, |p| _mm512_storeu_si512(p.cast(), m));
        }
    }
    #[cfg(not(use_avx512icl))]
    for to in to_bb {
        moves.push(Move::new(from, to));
    }
}

#[inline(always)]
fn make_promotions<const T: GenType, const ENEMY: bool>(
    moves: &mut Moves,
    to: Square,
    d: Direction,
) {
    let all = T == EVASIONS || T == ALL;
    let from = to - d;

    if T == CAPTURES || all {
        moves.push(Move::make(PROMOTION, from, to, QUEEN));
    }

    if (T == CAPTURES && ENEMY) || (T == QUIETS && !ENEMY) || all {
        moves.push(Move::make(PROMOTION, from, to, ROOK));
        moves.push(Move::make(PROMOTION, from, to, BISHOP));
        moves.push(Move::make(PROMOTION, from, to, KNIGHT));
    }
}

/// The pawn moves of the side `US` (0 for White, 1 for Black) to `target`.
#[inline(always)]
fn generate_pawn_moves<const US: u8, const T: GenType>(
    pos: &Position,
    moves: &mut Moves,
    target: Bitboard,
) {
    debug_assert!(T != ALL);

    let us = Color(US);
    let them = !us;
    let t_rank7_bb = if US == WHITE.0 { RANK_7_BB } else { RANK_2_BB };
    let t_rank3_bb = if US == WHITE.0 { RANK_3_BB } else { RANK_6_BB };
    let up = pawn_push(us);
    let up_right = if US == WHITE.0 {
        NORTH_EAST
    } else {
        SOUTH_WEST
    };
    let up_left = if US == WHITE.0 {
        NORTH_WEST
    } else {
        SOUTH_EAST
    };

    let empty_squares = !pos.pieces();
    let enemies = if T == EVASIONS {
        pos.checkers()
    } else {
        pos.pieces_c(them)
    };
    let pinned = pos.blockers_for_king(us);
    let ksq = pos.king_square(us);

    let pawns_on_7 = pos.pieces_cpt(us, PAWN) & t_rank7_bb;
    let pawns_not_on_7 = pos.pieces_cpt(us, PAWN) & !t_rank7_bb;
    let pushable = !pinned | file_bb(ksq.file());
    let can_capture_left = !pinned | antidiag_bb(ksq);
    let can_capture_right = !pinned | diag_bb(ksq);

    // Single and double pawn pushes, no promotions
    if T != CAPTURES {
        let mut b1 = shift(pawns_not_on_7 & pushable, up) & empty_squares;
        let mut b2 = shift(b1 & t_rank3_bb, up) & empty_squares;

        // Consider only blocking squares
        if T == EVASIONS {
            b1 &= target;
            b2 &= target;
        }

        splat_pawn_moves(moves, b1, up);
        splat_pawn_moves(moves, b2, up + up);
    }

    // Promotions and underpromotions
    if pawns_on_7.any() {
        let b1 = shift(pawns_on_7 & can_capture_right, up_right) & enemies;
        let b2 = shift(pawns_on_7 & can_capture_left, up_left) & enemies;
        let mut b3 = shift(pawns_on_7 & pushable, up) & empty_squares;

        if T == EVASIONS {
            b3 &= target;
        }

        for to in b1 {
            make_promotions::<T, true>(moves, to, up_right);
        }

        for to in b2 {
            make_promotions::<T, true>(moves, to, up_left);
        }

        for to in b3 {
            make_promotions::<T, false>(moves, to, up);
        }
    }

    // Standard and en passant captures
    if T == CAPTURES || T == EVASIONS {
        let b1 = shift(pawns_not_on_7 & can_capture_right, up_right) & enemies;
        let b2 = shift(pawns_not_on_7 & can_capture_left, up_left) & enemies;

        splat_pawn_moves(moves, b1, up_right);
        splat_pawn_moves(moves, b2, up_left);

        let ep_sq = pos.ep_square();

        // An en passant capture cannot resolve a discovered check.
        if ep_sq != Square::NONE && !(T == EVASIONS && target.has(ep_sq + up)) {
            debug_assert!(ep_sq.rank() == relative_rank(us, RANK_6));

            let b1 = pawns_not_on_7
                & ((can_capture_right & shift(square_bb(ep_sq), -up_right))
                    | (can_capture_left & shift(square_bb(ep_sq), -up_left)));
            debug_assert!(b1.any());

            for from in b1 {
                moves.push(Move::make(EN_PASSANT, from, ep_sq, KNIGHT));
            }
        }
    }
}

/// The moves of the pieces of type `PT`, neither pawns nor kings, to
/// `target`.
#[inline(always)]
fn generate_moves<const PT: u8>(pos: &Position, moves: &mut Moves, target: Bitboard, us: Color) {
    let pt = PieceType(PT);
    debug_assert!(pt != KING && pt != PAWN);

    let ksq = pos.king_square(us);
    let pinned = pos.blockers_for_king(us);

    for from in pos.pieces_cpt(us, pt) & !pinned {
        let b = attacks_bb(pt, from, pos.pieces()) & target;
        splat_moves(moves, from, b);
    }

    // Pinned knights cannot move.
    if pt != KNIGHT {
        for from in pos.pieces_cpt(us, pt) & pinned {
            let b = attacks_bb(pt, from, pos.pieces()) & target & line_bb(ksq, from);
            splat_moves(moves, from, b);
        }
    }
}

#[inline(always)]
fn generate_all<const T: GenType>(pos: &Position, moves: &mut Moves) {
    debug_assert!(T != ALL);

    let us = pos.side_to_move();
    let ksq = pos.king_square(us);
    let mut target = Bitboard::EMPTY;

    // Skip generating non-king moves when in double check
    if T != EVASIONS || !pos.checkers().more_than_one() {
        target = if T == EVASIONS {
            between_bb(ksq, pos.checkers().lsb())
        } else if T == CAPTURES {
            pos.pieces_c(!us)
        } else {
            !pos.pieces()
        };

        if us == WHITE {
            generate_pawn_moves::<{ WHITE.0 }, T>(pos, moves, target);
        } else {
            generate_pawn_moves::<{ BLACK.0 }, T>(pos, moves, target);
        }
        generate_moves::<{ KNIGHT.0 }>(pos, moves, target, us);
        generate_moves::<{ BISHOP.0 }>(pos, moves, target, us);
        generate_moves::<{ ROOK.0 }>(pos, moves, target, us);
        generate_moves::<{ QUEEN.0 }>(pos, moves, target, us);
    }

    if T == EVASIONS {
        target = !pos.pieces_c(us);
    }

    target &= !pos.threats_by(ALL_PIECES);

    let b = pseudo_attacks(KING, ksq) & target;
    splat_moves(moves, ksq, b);

    if T == QUIETS && pos.can_castle(castling_rights_of(us, ANY_CASTLING)) {
        for cr in [
            castling_rights_of(us, KING_SIDE),
            castling_rights_of(us, QUEEN_SIDE),
        ] {
            let rook_square = pos.castling_rook_square(cr);
            let to = (if rook_square > ksq {
                Square::G1
            } else {
                Square::C1
            })
            .relative(us);

            if !pos.castling_impeded(cr)
                && pos.can_castle(cr)
                && (between_bb(ksq, to) & pos.threats_by(ALL_PIECES)).is_empty()
                && !pos.blockers_for_king(us).has(rook_square)
            {
                moves.push(Move::make(CASTLING, ksq, rook_square, KNIGHT));
            }
        }
    }
}
