//! The input features FullThreats: each piece attacking another, by the
//! types, colors and squares of both, the board mirrored as for HalfKAv2_hm.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/features/full_threats.h` and `.cpp`: Stockfish, a UCI chess playing
//! engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

use crate::attacks::{PSEUDO_ATTACKS, attacks_bb};
use crate::bitboard::{Bitboard, shift};
use crate::misc::{ValueList, prefetch_low};
use crate::nnue::common::ORIENT_TBL;
use crate::position::Position;
use crate::types::*;
use crate::variant::Variant;

/// Pawn diagonal threats only target knights and rooks (pawn-pawn
/// relationships are handled by the PP_3Wide feature set), so pawns have 4
/// valid targets.
const NUM_VALID_TARGETS: [i32; PIECE_NB] = [0, 4, 10, 8, 8, 10, 0, 0, 0, 4, 10, 8, 8, 10, 0, 0];

pub struct FullThreats;

impl FullThreats {
    /// Hash value embedded in the evaluation file
    pub const HASH_VALUE: u32 = 0x2c41_eaea;

    /// Number of feature dimensions
    pub const DIMENSIONS: u32 = 59808;
}

/// The place of each attacked piece type among an attacker type's targets,
/// or -1 where no feature exists.
#[rustfmt::skip]
const MAP: [[i32; PIECE_TYPE_NB - 2]; PIECE_TYPE_NB - 2] = [
    [-1,  0, -1,  1, -1, -1],
    [ 0,  1,  2,  3,  4, -1],
    [ 0,  1,  2,  3, -1, -1],
    [ 0,  1,  2,  3, -1, -1],
    [ 0,  1,  2,  3,  4, -1],
    [-1, -1, -1, -1, -1, -1],
];

/// Maximum number of simultaneously active features.
pub const MAX_ACTIVE_DIMENSIONS: usize = 256;
pub type IndexList = ValueList<u16, MAX_ACTIVE_DIMENSIONS>;

/// What the features read of a board: its pieces, by square and by type and
/// color, and the kings' squares. Positions of every variant are such
/// boards.
pub trait ThreatBoard {
    fn king_square(&self, c: Color) -> Square;
    fn pieces(&self) -> Bitboard;
    fn pieces_pt(&self, pt: PieceType) -> Bitboard;
    fn pieces_pt2(&self, pt1: PieceType, pt2: PieceType) -> Bitboard;
    fn pieces_cpt(&self, c: Color, pt: PieceType) -> Bitboard;
    fn piece_on(&self, s: Square) -> Piece;
}

impl<V: Variant> ThreatBoard for Position<V> {
    #[inline(always)]
    fn king_square(&self, c: Color) -> Square {
        Position::<V>::king_square(self, c)
    }

    #[inline(always)]
    fn pieces(&self) -> Bitboard {
        Position::<V>::pieces(self)
    }

    #[inline(always)]
    fn pieces_pt(&self, pt: PieceType) -> Bitboard {
        Position::<V>::pieces_pt(self, pt)
    }

    #[inline(always)]
    fn pieces_pt2(&self, pt1: PieceType, pt2: PieceType) -> Bitboard {
        Position::<V>::pieces_pt2(self, pt1, pt2)
    }

    #[inline(always)]
    fn pieces_cpt(&self, c: Color, pt: PieceType) -> Bitboard {
        Position::<V>::pieces_cpt(self, c, pt)
    }

    #[inline(always)]
    fn piece_on(&self, s: Square) -> Piece {
        Position::<V>::piece_on(self, s)
    }
}

/// The attacks a piece makes from a square on an empty board, pawns by
/// color.
#[cfg(test)]
const fn piece_attacks(piece: Piece, from: usize) -> u64 {
    if piece.type_of().0 == PAWN.0 {
        PSEUDO_ATTACKS[piece.color().idx()][from].0
    } else {
        PSEUDO_ATTACKS[piece.type_of().idx()][from].0
    }
}

/// The final index is calculated from summing data found in these two LUTs
#[repr(C, align(64))]
struct ThreatLuts {
    /// [attacker][attacked][from < to]
    lut1: [[[u32; 2]; PIECE_NB]; PIECE_NB],
    /// [pt][from][to]: the number of attacks a piece of type `pt` makes from
    /// the squares before `from`, then from `from` on squares below `to`. A
    /// pawn's attacks are Black's where `to` is below `from`, else White's.
    lut2: [[[u16; SQUARE_NB]; SQUARE_NB]; PIECE_TYPE_NB],
}

const fn init_threat_luts() -> ThreatLuts {
    let mut tables = ThreatLuts {
        lut1: [[[0; 2]; PIECE_NB]; PIECE_NB],
        lut2: [[[0; SQUARE_NB]; SQUARE_NB]; PIECE_TYPE_NB],
    };
    let mut offsets = [0; PIECE_TYPE_NB];

    let mut pt = PAWN.idx();
    while pt <= KING.idx() {
        let mut offset = 0;
        let mut from = 0;
        while from < SQUARE_NB {
            let mut to = 0;
            while to < SQUARE_NB {
                let attacks = if pt != PAWN.idx() {
                    PSEUDO_ATTACKS[pt][from].0
                } else if to < from {
                    PSEUDO_ATTACKS[BLACK.idx()][from].0
                } else {
                    PSEUDO_ATTACKS[WHITE.idx()][from].0
                };
                tables.lut2[pt][from][to] =
                    offset + (((1u64 << to) - 1) & attacks).count_ones() as u16;
                to += 1;
            }

            if pt != PAWN.idx() {
                offset += PSEUDO_ATTACKS[pt][from].0.count_ones() as u16;
            } else if from >= Square::A2.0 as usize && from <= Square::H7.0 as usize {
                offset += PSEUDO_ATTACKS[WHITE.idx()][from].0.count_ones() as u16;
            }
            from += 1;
        }

        offsets[pt] = offset;
        pt += 1;
    }

    let mut cumulative_offset = 0;
    let mut i = 0;
    while i < PIECES.len() {
        let attacker = PIECES[i];
        let attacker_type = attacker.type_of().0 as usize;
        let cumulative_piece_offset = offsets[attacker_type] as i32;
        let valid_targets = NUM_VALID_TARGETS[attacker.idx()];

        let mut j = 0;
        while j < PIECES.len() {
            let attacked = PIECES[j];
            let enemy = (attacker.0 ^ attacked.0) == 8;
            let attacked_type = attacked.type_of().0 as usize;

            let map = MAP[attacker_type - 1][attacked_type - 1];
            let semi_excluded =
                attacker_type == attacked_type && (enemy || attacker_type != PAWN.0 as usize);
            let feature = cumulative_offset
                + (attacked.color().0 as i32 * (valid_targets / 2) + map) * cumulative_piece_offset;

            let excluded = map < 0;
            tables.lut1[attacker.idx()][attacked.idx()][0] = if excluded {
                FullThreats::DIMENSIONS
            } else {
                feature as u32
            };
            tables.lut1[attacker.idx()][attacked.idx()][1] = if excluded || semi_excluded {
                FullThreats::DIMENSIONS
            } else {
                feature as u32
            };
            j += 1;
        }

        cumulative_offset += valid_targets * cumulative_piece_offset;
        i += 1;
    }

    tables
}

static THREAT_LUTS: ThreatLuts = init_threat_luts();

/// Index of a feature for a given king position: DIMENSIONS or more for a
/// threat that has no feature.
#[inline(always)]
pub fn make_index(
    perspective: Color,
    attacker: Piece,
    from: Square,
    to: Square,
    attacked: Piece,
    ksq: Square,
) -> u32 {
    let orientation = usize::from(ORIENT_TBL[ksq.idx()]) ^ (56 * perspective.idx());
    let from_oriented = from.idx() ^ orientation;
    let to_oriented = to.idx() ^ orientation;

    let swap = 8 * perspective.idx();
    let attacker_oriented = attacker.idx() ^ swap;
    let attacked_oriented = attacked.idx() ^ swap;

    THREAT_LUTS.lut1[attacker_oriented][attacked_oriented][usize::from(from_oriented < to_oriented)]
        + u32::from(THREAT_LUTS.lut2[attacker.type_of().idx()][from_oriented][to_oriented])
}

/// Get a list of indices for active features
pub fn append_active_indices(perspective: Color, pos: &Position, active: &mut IndexList) {
    let ksq = pos.king_square(perspective);
    let occupied = pos.pieces();
    let pawn_targets = pos.pieces_pt2(KNIGHT, ROOK);
    let minor_slider_targets = pos.pieces_pt2(PAWN, KNIGHT) | pos.pieces_pt2(BISHOP, ROOK);
    let queen_targets = minor_slider_targets | pos.pieces_pt(QUEEN);

    let mut process_pawn_attacks = |c: Color, attk_dir: Direction| {
        let c_pawns = pos.pieces_cpt(c, PAWN);
        let mut attacks = shift(c_pawns, attk_dir) & pawn_targets;
        while attacks.any() {
            let to = attacks.pop_lsb();
            let from = to - attk_dir;
            let attacked = pos.piece_on(to);
            let attacker = make_piece(c, PAWN);
            let index = make_index(perspective, attacker, from, to, attacked, ksq);
            active.push_back_if_lt(index as u16, FullThreats::DIMENSIONS as u16);
        }
    };

    process_pawn_attacks(WHITE, NORTH_EAST);
    process_pawn_attacks(WHITE, NORTH_WEST);

    process_pawn_attacks(BLACK, SOUTH_WEST);
    process_pawn_attacks(BLACK, SOUTH_EAST);

    // Spelled out, so that each piece type gets its own loop.
    let targets = [
        queen_targets,
        minor_slider_targets,
        minor_slider_targets,
        queen_targets,
    ];
    for c in [WHITE, BLACK] {
        let mut append = |pt: PieceType| {
            let targets = targets[pt.idx() - KNIGHT.idx()];
            append_piece_threats(
                perspective,
                pos,
                make_piece(c, pt),
                targets,
                occupied,
                ksq,
                active,
            )
        };
        append(KNIGHT);
        append(BISHOP);
        append(ROOK);
        append(QUEEN);
    }
}

#[inline(always)]
fn append_piece_threats(
    perspective: Color,
    pos: &Position,
    attacker: Piece,
    targets: Bitboard,
    occupied: Bitboard,
    ksq: Square,
    active: &mut IndexList,
) {
    let mut bb = pos.pieces_cpt(attacker.color(), attacker.type_of());
    while bb.any() {
        let from = bb.pop_lsb();
        let mut attacks = attacks_bb(attacker.type_of(), from, occupied) & targets;
        while attacks.any() {
            let to = attacks.pop_lsb();
            let attacked = pos.piece_on(to);
            let index = make_index(perspective, attacker, from, to, attacked, ksq);
            active.push_back_if_lt(index as u16, FullThreats::DIMENSIONS as u16);
        }
    }
}

/// `append_active_indices` on any board, as Atomic's positions. Chess keeps
/// its own copy, whose machine code a generic function would change. Inlined
/// into the refresh of each of Atomic's networks with threats, as it was when
/// one network alone called it.
#[inline(always)]
pub fn append_active_indices_of<B: ThreatBoard>(
    perspective: Color,
    pos: &B,
    active: &mut IndexList,
) {
    let ksq = pos.king_square(perspective);
    let occupied = pos.pieces();
    let pawn_targets = pos.pieces_pt2(KNIGHT, ROOK);
    let minor_slider_targets = pos.pieces_pt2(PAWN, KNIGHT) | pos.pieces_pt2(BISHOP, ROOK);
    let queen_targets = minor_slider_targets | pos.pieces_pt(QUEEN);

    let mut process_pawn_attacks = |c: Color, attk_dir: Direction| {
        let c_pawns = pos.pieces_cpt(c, PAWN);
        let mut attacks = shift(c_pawns, attk_dir) & pawn_targets;
        while attacks.any() {
            let to = attacks.pop_lsb();
            let from = to - attk_dir;
            let attacked = pos.piece_on(to);
            let attacker = make_piece(c, PAWN);
            let index = make_index(perspective, attacker, from, to, attacked, ksq);
            active.push_back_if_lt(index as u16, FullThreats::DIMENSIONS as u16);
        }
    };

    process_pawn_attacks(WHITE, NORTH_EAST);
    process_pawn_attacks(WHITE, NORTH_WEST);

    process_pawn_attacks(BLACK, SOUTH_WEST);
    process_pawn_attacks(BLACK, SOUTH_EAST);

    // Spelled out, so that each piece type gets its own loop.
    let targets = [
        queen_targets,
        minor_slider_targets,
        minor_slider_targets,
        queen_targets,
    ];
    for c in [WHITE, BLACK] {
        let mut append = |pt: PieceType| {
            let targets = targets[pt.idx() - KNIGHT.idx()];
            append_piece_threats_of(
                perspective,
                pos,
                make_piece(c, pt),
                targets,
                occupied,
                ksq,
                active,
            )
        };
        append(KNIGHT);
        append(BISHOP);
        append(ROOK);
        append(QUEEN);
    }
}

#[inline(always)]
fn append_piece_threats_of<B: ThreatBoard>(
    perspective: Color,
    pos: &B,
    attacker: Piece,
    targets: Bitboard,
    occupied: Bitboard,
    ksq: Square,
    active: &mut IndexList,
) {
    let mut bb = pos.pieces_cpt(attacker.color(), attacker.type_of());
    while bb.any() {
        let from = bb.pop_lsb();
        let mut attacks = attacks_bb(attacker.type_of(), from, occupied) & targets;
        while attacks.any() {
            let to = attacks.pop_lsb();
            let attacked = pos.piece_on(to);
            let index = make_index(perspective, attacker, from, to, attacked, ksq);
            active.push_back_if_lt(index as u16, FullThreats::DIMENSIONS as u16);
        }
    }
}

/// Get a list of indices for recently changed features, prefetching the
/// row of `prefetch_stride` weights of each. Atomic's captures, whose blasts
/// change more threats than a chess move, record them in longer lists.
#[inline(always)]
pub fn append_changed_indices<const N: usize, const M: usize>(
    perspective: Color,
    ksq: Square,
    diff: &ValueList<DirtyThreat, N>,
    removed: &mut ValueList<u16, M>,
    added: &mut ValueList<u16, M>,
    prefetch_base: *const i8,
    prefetch_stride: usize,
) {
    let orient = usize::from(ORIENT_TBL[ksq.idx()]) ^ (56 * perspective.idx());
    let pair_swap = 0x88 * perspective.idx();
    let lut1_flat = THREAT_LUTS.lut1.as_flattened();

    for dirty in diff {
        let from = dirty.pc_sq().idx() ^ orient;
        let to = dirty.threatened_sq().idx() ^ orient;
        // The attacker and the attacked piece in one byte, 16 * attacker +
        // attacked, so that XOR 0x88 swaps both colors
        let pair = usize::from((dirty.raw() >> 16) as u8);
        let plut2 = &THREAT_LUTS.lut2[(pair >> 4) & 7];

        let index =
            lut1_flat[pair ^ pair_swap][usize::from(from < to)] + u32::from(plut2[from][to]);

        prefetch_low(prefetch_base.wrapping_add(index as usize * prefetch_stride));

        if dirty.add() {
            added.push_back_if_lt(index as u16, FullThreats::DIMENSIONS as u16);
        } else {
            removed.push_back_if_lt(index as u16, FullThreats::DIMENSIONS as u16);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn append_changed_indices_both<const N: usize, const M: usize>(
    white_ksq: Square,
    black_ksq: Square,
    diff: &ValueList<DirtyThreat, N>,
    white_removed: &mut ValueList<u16, M>,
    white_added: &mut ValueList<u16, M>,
    black_removed: &mut ValueList<u16, M>,
    black_added: &mut ValueList<u16, M>,
    prefetch_base: *const i8,
    prefetch_stride: usize,
) {
    let w_orient = usize::from(ORIENT_TBL[white_ksq.idx()]);
    let b_orient = usize::from(ORIENT_TBL[black_ksq.idx()]) ^ 56;
    let lut1_flat = THREAT_LUTS.lut1.as_flattened();

    for dirty in diff {
        let from = dirty.pc_sq().idx();
        let to = dirty.threatened_sq().idx();
        // As in append_changed_indices
        let pair = usize::from((dirty.raw() >> 16) as u8);
        let plut2 = &THREAT_LUTS.lut2[(pair >> 4) & 7];

        let w_from = from ^ w_orient;
        let w_to = to ^ w_orient;
        let white_index =
            lut1_flat[pair][usize::from(w_from < w_to)] + u32::from(plut2[w_from][w_to]);

        let b_from = from ^ b_orient;
        let b_to = to ^ b_orient;
        let black_index =
            lut1_flat[pair ^ 0x88][usize::from(b_from < b_to)] + u32::from(plut2[b_from][b_to]);

        prefetch_low(prefetch_base.wrapping_add(white_index as usize * prefetch_stride));
        prefetch_low(prefetch_base.wrapping_add(black_index as usize * prefetch_stride));

        if dirty.add() {
            white_added.push_back_if_lt(white_index as u16, FullThreats::DIMENSIONS as u16);
            black_added.push_back_if_lt(black_index as u16, FullThreats::DIMENSIONS as u16);
        } else {
            white_removed.push_back_if_lt(white_index as u16, FullThreats::DIMENSIONS as u16);
            black_removed.push_back_if_lt(black_index as u16, FullThreats::DIMENSIONS as u16);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_feature_is_one_attack() {
        // Every attack of every pair of pieces on an empty board, from
        // White's side with its king on e1.
        let mut seen = vec![0u8; FullThreats::DIMENSIONS as usize];
        for attacker in PIECES {
            for attacked in PIECES {
                for from in 0..64 {
                    let from = Square(from);
                    if attacker.type_of() == PAWN && !(8..56).contains(&from.0) {
                        continue;
                    }
                    for to in Bitboard(piece_attacks(attacker, from.idx())) {
                        let index = make_index(WHITE, attacker, from, to, attacked, Square::E1);
                        if index < FullThreats::DIMENSIONS {
                            seen[index as usize] += 1;
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&n| n <= 1));

        // Pieces of the same type attack each other alike, so only attacks
        // from the higher square have features: half of the attacks of
        // knights, bishops, rooks and queens, on pieces of either color.
        let attacks = |pt: PieceType| -> usize {
            (0..64)
                .map(|s| PSEUDO_ATTACKS[pt.idx()][s].popcount() as usize)
                .sum()
        };
        let unused = 2 * [KNIGHT, BISHOP, ROOK, QUEEN]
            .map(attacks)
            .iter()
            .sum::<usize>();
        let used = seen.iter().filter(|&&n| n == 1).count();
        assert_eq!(used + unused, FullThreats::DIMENSIONS as usize);
    }

    /// The indices that each way of listing changed features gives for one
    /// changed threat: removed and added from White's side, then from
    /// Black's.
    fn changed_indices(
        threat: DirtyThreat,
        white_ksq: Square,
        black_ksq: Square,
    ) -> [[IndexList; 4]; 2] {
        let mut diff = DirtyThreats::new();
        diff.push(threat);
        let row = [0i8; 1];
        let mut each = [IndexList::new(); 4];
        let [white_removed, white_added, black_removed, black_added] = &mut each;
        append_changed_indices(
            WHITE,
            white_ksq,
            &diff,
            white_removed,
            white_added,
            row.as_ptr(),
            0,
        );
        append_changed_indices(
            BLACK,
            black_ksq,
            &diff,
            black_removed,
            black_added,
            row.as_ptr(),
            0,
        );
        let mut both = [IndexList::new(); 4];
        let [white_removed, white_added, black_removed, black_added] = &mut both;
        append_changed_indices_both(
            white_ksq,
            black_ksq,
            &diff,
            white_removed,
            white_added,
            black_removed,
            black_added,
            row.as_ptr(),
            0,
        );
        [each, both]
    }

    #[test]
    fn changed_indices_are_make_index() {
        // Every attack of every pair of pieces, removed and added, with the
        // kings on either half of the board.
        let mut threats = Vec::new();
        for attacker in PIECES {
            for attacked in PIECES {
                for from in (0..64).map(Square) {
                    for to in Bitboard(piece_attacks(attacker, from.idx())) {
                        for add in [false, true] {
                            threats.push(DirtyThreat::new(attacker, attacked, from, to, add));
                        }
                    }
                }
            }
        }
        for threat in threats {
            for (white_ksq, black_ksq) in [(Square::E1, Square::D8), (Square::B2, Square::G7)] {
                let index = |perspective, ksq| {
                    let mut list = IndexList::new();
                    let (attacker, attacked) = (threat.pc(), threat.threatened_pc());
                    let (from, to) = (threat.pc_sq(), threat.threatened_sq());
                    let index = make_index(perspective, attacker, from, to, attacked, ksq);
                    list.push_back_if_lt(index as u16, FullThreats::DIMENSIONS as u16);
                    list
                };
                let (white, black, none) = (
                    index(WHITE, white_ksq),
                    index(BLACK, black_ksq),
                    IndexList::new(),
                );
                let expected = if threat.add() {
                    [none, white, none, black]
                } else {
                    [white, none, black, none]
                };
                for found in changed_indices(threat, white_ksq, black_ksq) {
                    let found = found.each_ref().map(IndexList::as_slice);
                    assert_eq!(found, expected.each_ref().map(IndexList::as_slice));
                }
            }
        }
    }
}
