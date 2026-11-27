//! Attack tables and sliding attacks.
//!
//! Every table is built during compilation. Sliding attacks are Stockfish's,
//! by hyperbola quintessence on every target, with AVX2 or WebAssembly's
//! SIMD128 where the build has them.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `attacks.h` and `attacks.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers
//! (see AUTHORS), under the GNU General Public License, version 3 or later.

use crate::bitboard::*;
use crate::types::*;

/// The target square `step` away from `s`, or no square if the step leaves the
/// board.
const fn safe_destination(s: u8, step: i8) -> u64 {
    let to = s.wrapping_add(step as u8);
    let file_distance = (s & 7).abs_diff(to & 7);
    if to < 64 && file_distance <= 2 {
        1 << to
    } else {
        0
    }
}

const ROOK_DIRECTIONS: [Direction; 4] = [NORTH, SOUTH, EAST, WEST];
const BISHOP_DIRECTIONS: [Direction; 4] = [NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST];

/// The squares a rook or bishop on `sq` attacks with the board occupied as in
/// `occupied`: the slow reference the fast methods are built from.
pub const fn sliding_attack(pt: PieceType, sq: Square, occupied: u64) -> u64 {
    let directions = if pt.0 == ROOK.0 {
        ROOK_DIRECTIONS
    } else {
        BISHOP_DIRECTIONS
    };
    let mut attacks = 0;
    let mut i = 0;
    while i < 4 {
        let d = directions[i];
        let mut s = sq.0;
        loop {
            let dest = safe_destination(s, d);
            if dest == 0 {
                break;
            }
            attacks |= dest;
            if occupied & dest != 0 {
                break;
            }
            s = s.wrapping_add(d as u8);
        }
        i += 1;
    }
    attacks
}

const fn steps_attack(sq: u8, steps: [i8; 8]) -> u64 {
    let mut b = 0;
    let mut i = 0;
    while i < 8 {
        b |= safe_destination(sq, steps[i]);
        i += 1;
    }
    b
}

/// The squares from `sq` along directions `d1` and `d2` to the edges, without
/// `sq`.
const fn line_mask(sq: u8, d1: Direction, d2: Direction) -> u64 {
    let mut mask = 0;
    let directions = [d1, d2];
    let mut i = 0;
    while i < 2 {
        let d = directions[i];
        let mut s = sq;
        loop {
            let dest = safe_destination(s, d);
            if dest == 0 {
                break;
            }
            mask |= dest;
            s = s.wrapping_add(d as u8);
        }
        i += 1;
    }
    mask
}

/// The attacks of each piece type from each square on an empty board. Pawns
/// use the first two rows, indexed by color.
pub static PSEUDO_ATTACKS: [[Bitboard; SQUARE_NB]; PIECE_TYPE_NB] = {
    let mut attacks = [[Bitboard::EMPTY; SQUARE_NB]; PIECE_TYPE_NB];
    let mut s = 0;
    while s < SQUARE_NB {
        let sq = Square(s as u8);
        attacks[WHITE.0 as usize][s] = pawn_attacks_bb(WHITE, square_bb(sq));
        attacks[BLACK.0 as usize][s] = pawn_attacks_bb(BLACK, square_bb(sq));
        attacks[KING.idx()][s] = Bitboard(steps_attack(s as u8, [-9, -8, -7, -1, 1, 7, 8, 9]));
        attacks[KNIGHT.idx()][s] =
            Bitboard(steps_attack(s as u8, [-17, -15, -10, -6, 6, 10, 15, 17]));
        let bishop = sliding_attack(BISHOP, sq, 0);
        let rook = sliding_attack(ROOK, sq, 0);
        attacks[BISHOP.idx()][s] = Bitboard(bishop);
        attacks[ROOK.idx()][s] = Bitboard(rook);
        attacks[QUEEN.idx()][s] = Bitboard(bishop | rook);
        s += 1;
    }
    attacks
};

struct Lines {
    line: [[Bitboard; SQUARE_NB]; SQUARE_NB],
    between: [[Bitboard; SQUARE_NB]; SQUARE_NB],
    inverse_ray_pass: [[Bitboard; SQUARE_NB]; SQUARE_NB],
    diag: [Bitboard; SQUARE_NB],
    anti_diag: [Bitboard; SQUARE_NB],
}

static LINES: Lines = {
    let mut lines = Lines {
        line: [[Bitboard::EMPTY; SQUARE_NB]; SQUARE_NB],
        between: [[Bitboard::EMPTY; SQUARE_NB]; SQUARE_NB],
        inverse_ray_pass: [[Bitboard::EMPTY; SQUARE_NB]; SQUARE_NB],
        diag: [Bitboard::EMPTY; SQUARE_NB],
        anti_diag: [Bitboard::EMPTY; SQUARE_NB],
    };
    let mut s1 = 0;
    while s1 < SQUARE_NB {
        let sq1 = Square(s1 as u8);
        let pts = [BISHOP, ROOK];
        let mut i = 0;
        while i < 2 {
            let pt = pts[i];
            let empty1 = sliding_attack(pt, sq1, 0);
            let mut s2 = 0;
            while s2 < SQUARE_NB {
                let sq2 = Square(s2 as u8);
                if empty1 & (1 << s2) != 0 {
                    lines.line[s1][s2] =
                        Bitboard((empty1 & sliding_attack(pt, sq2, 0)) | (1 << s1) | (1 << s2));
                    lines.between[s1][s2] = Bitboard(
                        sliding_attack(pt, sq1, 1 << s2) & sliding_attack(pt, sq2, 1 << s1),
                    );
                    lines.inverse_ray_pass[s2][s1] =
                        Bitboard(empty1 & (sliding_attack(pt, sq2, 1 << s1) | (1 << s2)));
                }
                lines.between[s1][s2].0 |= 1 << s2;
                s2 += 1;
            }
            i += 1;
        }
        lines.diag[s1] = Bitboard(line_mask(s1 as u8, NORTH_EAST, SOUTH_WEST) | (1 << s1));
        lines.anti_diag[s1] = Bitboard(line_mask(s1 as u8, NORTH_WEST, SOUTH_EAST) | (1 << s1));
        s1 += 1;
    }
    lines
};

/// The whole line through `s1` and `s2`, a rank, file or diagonal, or no
/// squares when they are not aligned.
#[inline(always)]
pub fn line_bb(s1: Square, s2: Square) -> Bitboard {
    LINES.line[s1.idx()][s2.idx()]
}

/// The segment from `s1` to `s2` without `s1` but with `s2`; just `s2` when
/// they are not aligned.
#[inline(always)]
pub fn between_bb(s1: Square, s2: Square) -> Bitboard {
    LINES.between[s1.idx()][s2.idx()]
}

/// The ray from `s1` through `s2` and beyond, without `s1`.
#[inline(always)]
pub fn ray_pass_bb(s1: Square, s2: Square) -> Bitboard {
    LINES.inverse_ray_pass[s2.idx()][s1.idx()]
}

#[inline(always)]
pub fn diag_bb(s: Square) -> Bitboard {
    LINES.diag[s.idx()]
}

#[inline(always)]
pub fn antidiag_bb(s: Square) -> Bitboard {
    LINES.anti_diag[s.idx()]
}

/// The attacks of piece type `pt` (not a pawn) from `s` on an empty board.
#[inline(always)]
pub fn pseudo_attacks(pt: PieceType, s: Square) -> Bitboard {
    PSEUDO_ATTACKS[pt.idx()][s.idx()]
}

/// The squares a pawn of color `c` on `s` attacks.
#[inline(always)]
pub fn pawn_attacks(c: Color, s: Square) -> Bitboard {
    PSEUDO_ATTACKS[c.idx()][s.idx()]
}

/// Bishop and rook attacks from `s` with the board occupied as in `occupied`,
/// both at once.
#[inline(always)]
pub fn both_attacks_bb(s: Square, occupied: Bitboard) -> (Bitboard, Bitboard) {
    let (bishop, rook) = sliders::both_attacks(s, occupied.0);
    (Bitboard(bishop), Bitboard(rook))
}

#[inline(always)]
pub fn bishop_attacks(s: Square, occupied: Bitboard) -> Bitboard {
    Bitboard(sliders::bishop_attacks(s, occupied.0))
}

#[inline(always)]
pub fn rook_attacks(s: Square, occupied: Bitboard) -> Bitboard {
    Bitboard(sliders::rook_attacks(s, occupied.0))
}

/// The attacks of piece type `pt` (not a pawn) from `s` with the board
/// occupied as in `occupied`. Sliding attacks stop at the first occupied
/// square.
#[inline(always)]
pub fn attacks_bb(pt: PieceType, s: Square, occupied: Bitboard) -> Bitboard {
    debug_assert!(pt != PAWN);
    match pt {
        BISHOP => bishop_attacks(s, occupied),
        ROOK => rook_attacks(s, occupied),
        QUEEN => {
            let (bishop, rook) = both_attacks_bb(s, occupied);
            bishop | rook
        }
        _ => pseudo_attacks(pt, s),
    }
}

/// The attacks of piece `pc` from `s`, pawns included.
#[inline(always)]
pub fn piece_attacks_bb(pc: Piece, s: Square, occupied: Bitboard) -> Bitboard {
    if pc.type_of() == PAWN {
        pawn_attacks(pc.color(), s)
    } else {
        attacks_bb(pc.type_of(), s, occupied)
    }
}

/// Sliding attacks by hyperbola quintessence, see
/// <https://www.chessprogramming.org/Hyperbola_Quintessence>. Files and both
/// diagonals reverse by a byte swap, since each of those lines has one square
/// per byte. Ranks, which a byte swap cannot reverse, come from a table
/// indexed by the slider's file and the rank's six inner squares (the edge
/// squares never change the attacks). With AVX2 the three lines go in one
/// vector, and with WebAssembly's SIMD128 the diagonals in one and the file
/// in another; elsewhere they are computed one by one.
mod sliders {
    #[cfg(use_simd128)]
    use std::arch::wasm32::*;
    #[cfg(use_avx2)]
    use std::arch::x86_64::*;

    use super::{BISHOP_DIRECTIONS, ROOK_DIRECTIONS, line_mask, sliding_attack};
    use crate::types::*;

    static RANK_ATTACKS: [[u8; 64]; FILE_NB] = {
        let mut table = [[0; 64]; FILE_NB];
        let mut file = 0;
        while file < FILE_NB {
            let mut occ6 = 0;
            while occ6 < 64 {
                table[file][occ6] =
                    sliding_attack(ROOK, Square(file as u8), (occ6 as u64) << 1) as u8;
                occ6 += 1;
            }
            file += 1;
        }
        table
    };

    #[repr(C, align(32))]
    struct DualMagic {
        // File, diagonal, anti-diagonal, unused.
        masks: [u64; 4],
        // 2 * square_bb(sq) and 2 * square_bb(63 - sq).
        r: u64,
        rr: u64,
        file: u32,
        // 8 * rank_of(sq)
        shift: u32,
    }

    static DUAL_MAGICS: [DualMagic; SQUARE_NB] = {
        let mut magics = [const {
            DualMagic {
                masks: [0; 4],
                r: 0,
                rr: 0,
                file: 0,
                shift: 0,
            }
        }; SQUARE_NB];
        let mut s = 0;
        while s < SQUARE_NB {
            let sq = s as u8;
            magics[s] = DualMagic {
                masks: [
                    line_mask(sq, ROOK_DIRECTIONS[0], ROOK_DIRECTIONS[1]),
                    line_mask(sq, BISHOP_DIRECTIONS[0], BISHOP_DIRECTIONS[2]),
                    line_mask(sq, BISHOP_DIRECTIONS[3], BISHOP_DIRECTIONS[1]),
                    0,
                ],
                r: (1u64 << s) << 1,
                rr: (1u64 << (63 - s)) << 1,
                file: (s & 7) as u32,
                shift: 8 * (s >> 3) as u32,
            };
            s += 1;
        }
        magics
    };

    #[inline(always)]
    fn rank_attacks(m: &DualMagic, occupied: u64) -> u64 {
        let row =
            RANK_ATTACKS[(m.file & 7) as usize][((occupied >> (m.shift + 1)) & 0x3f) as usize];
        (row as u64) << m.shift
    }

    /// Bishop and rook attacks at once. Callers that need one of them leave
    /// the other for the compiler to drop.
    #[cfg(use_avx2)]
    #[inline(always)]
    pub fn both_attacks(s: Square, occupied: u64) -> (u64, u64) {
        let m = &DUAL_MAGICS[s.idx()];
        let rank_attacks = rank_attacks(m, occupied);
        // SAFETY: AVX2 is enabled for the whole build (this function exists
        // only then), and `masks` is 32-byte aligned for the aligned load.
        unsafe {
            // Byte swap within each 128-bit lane.
            let bswap = |v: __m256i| {
                _mm256_shuffle_epi8(
                    v,
                    _mm256_set_epi8(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6,
                        7, 8, 9, 10, 11, 12, 13, 14, 15,
                    ),
                )
            };
            // Each lane holds a mask, and follows the scalar algorithm below.
            let mask = _mm256_load_si256(m.masks.as_ptr().cast::<__m256i>());
            let rs = _mm256_set1_epi64x(m.r as i64);
            let rrs = _mm256_set1_epi64x(m.rr as i64);

            let o = _mm256_and_si256(mask, _mm256_set1_epi64x(occupied as i64));
            let fwd = _mm256_sub_epi64(o, rs);
            let rev = bswap(_mm256_sub_epi64(bswap(o), rrs));
            let result = _mm256_and_si256(_mm256_xor_si256(fwd, rev), mask);

            // Lane 0: rook attacks along the file; lanes 1 and 2: bishop
            // attacks.
            let rook_bishop =
                _mm256_or_si256(result, _mm256_permute4x64_epi64::<0b11_01_10_00>(result));

            (
                _mm_extract_epi64::<1>(_mm256_castsi256_si128(rook_bishop)) as u64,
                (_mm_cvtsi128_si64(_mm256_castsi256_si128(result)) as u64)
                    .wrapping_add(rank_attacks),
            )
        }
    }

    /// Bishop and rook attacks at once, by WebAssembly's SIMD128, which has
    /// no scalar byte swap: the diagonals in one vector and the file in
    /// another, so that a bishop's or a rook's query computes only one.
    #[cfg(all(use_simd128, not(use_avx2)))]
    #[inline(always)]
    pub fn both_attacks(s: Square, occupied: u64) -> (u64, u64) {
        let m = &DUAL_MAGICS[s.idx()];
        let rank_attacks = rank_attacks(m, occupied);

        // Byte swap within each 64-bit lane.
        let bswap =
            |v: v128| i8x16_shuffle::<7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8>(v, v);

        let occ = i64x2_splat(occupied as i64);
        let rs = i64x2_splat(m.r as i64);
        let rrs = i64x2_splat(m.rr as i64);

        // Two lanes of the scalar algorithm below
        let hyperbola2 = |mask: v128| {
            let o = v128_and(mask, occ);
            let fwd = i64x2_sub(o, rs);
            let rev = bswap(i64x2_sub(bswap(o), rrs));
            v128_and(v128_xor(fwd, rev), mask)
        };

        // Bishop: [diagonal, anti-diagonal]; rook: [file, none].
        let diags = hyperbola2(u64x2(m.masks[1], m.masks[2]));
        let file = hyperbola2(u64x2(m.masks[0], 0));

        (
            u64x2_extract_lane::<0>(diags) | u64x2_extract_lane::<1>(diags),
            u64x2_extract_lane::<0>(file).wrapping_add(rank_attacks),
        )
    }

    /// Bishop and rook attacks at once, one line at a time.
    #[cfg(any(test, not(any(use_avx2, use_simd128))))]
    #[cfg_attr(test, allow(dead_code))]
    #[inline(always)]
    pub fn scalar_both_attacks(s: Square, occupied: u64) -> (u64, u64) {
        let m = &DUAL_MAGICS[s.idx()];
        let hyperbola = |mask: u64| {
            let o = occupied & mask;
            (o.wrapping_sub(m.r) ^ o.swap_bytes().wrapping_sub(m.rr).swap_bytes()) & mask
        };
        (
            hyperbola(m.masks[1]) | hyperbola(m.masks[2]),
            hyperbola(m.masks[0]).wrapping_add(rank_attacks(m, occupied)),
        )
    }

    #[cfg(not(any(use_avx2, use_simd128)))]
    pub use scalar_both_attacks as both_attacks;

    #[inline(always)]
    pub fn bishop_attacks(s: Square, occupied: u64) -> u64 {
        both_attacks(s, occupied).0
    }

    #[inline(always)]
    pub fn rook_attacks(s: Square, occupied: u64) -> u64 {
        both_attacks(s, occupied).1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::Prng;

    fn check_sliders(
        name: &str,
        bishop: impl Fn(Square, u64) -> u64,
        rook: impl Fn(Square, u64) -> u64,
    ) {
        let mut rng = Prng::new(1070372);
        for s in 0..64 {
            let sq = Square(s);
            for i in 0..1000 {
                // Sparse and dense occupancies, and the empty and full board.
                let occupied = match i {
                    0 => 0,
                    1 => !0,
                    _ if i % 2 == 0 => rng.rand64() & rng.rand64() & rng.rand64(),
                    _ => rng.rand64(),
                };
                assert_eq!(
                    bishop(sq, occupied),
                    sliding_attack(BISHOP, sq, occupied),
                    "{name}"
                );
                assert_eq!(
                    rook(sq, occupied),
                    sliding_attack(ROOK, sq, occupied),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn every_slider_method_matches_the_reference() {
        check_sliders(
            "scalar",
            |s, o| sliders::scalar_both_attacks(s, o).0,
            |s, o| sliders::scalar_both_attacks(s, o).1,
        );
        check_sliders(
            "active",
            |s, o| bishop_attacks(s, Bitboard(o)).0,
            |s, o| rook_attacks(s, Bitboard(o)).0,
        );
    }

    #[test]
    fn lines_between_and_rays() {
        assert_eq!(line_bb(Square::A1, Square::C3), diag_bb(Square::A1));
        assert_eq!(line_bb(Square::A1, Square::B3), Bitboard::EMPTY);
        assert_eq!(
            between_bb(Square::A1, Square::D4),
            Square::B2 | Square::C3 | Square::D4
        );
        assert_eq!(between_bb(Square::A1, Square::B3), square_bb(Square::B3));
        assert_eq!(between_bb(Square::E4, Square::E4), square_bb(Square::E4));
        assert_eq!(
            ray_pass_bb(Square::A1, Square::C1),
            pseudo_attacks(ROOK, Square::A1) & RANK_1_BB
        );
        assert_eq!(ray_pass_bb(Square::C3, Square::B2), Square::B2 | Square::A1);
        assert_eq!(
            antidiag_bb(Square::H1),
            between_bb(Square::A8, Square::H1) | Square::A8
        );
    }

    #[test]
    fn pseudo_attacks_count() {
        let total = |pt: PieceType| -> i32 {
            (0..64)
                .map(|s| pseudo_attacks(pt, Square(s)).popcount())
                .sum()
        };
        assert_eq!(total(KNIGHT), 336);
        assert_eq!(total(BISHOP), 560);
        assert_eq!(total(ROOK), 896);
        assert_eq!(total(KING), 420);
        assert_eq!(pawn_attacks(WHITE, Square::E4), Square::D5 | Square::F5);
        assert_eq!(pawn_attacks(BLACK, Square::A5), square_bb(Square::B4));
    }
}
