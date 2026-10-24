//! Attack tables and sliding attacks.
//!
//! Every table is built during compilation. Sliding attacks follow Stockfish's
//! choice per target: dual hyperbola quintessence on x86-64 with AVX2,
//! hyperbola quintessence where a bit-reversal instruction exists (aarch64 and
//! 64-bit LoongArch), and magic bitboards elsewhere. All three give the same
//! attacks, so the choice only changes speed.
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use dual_hyperbola as sliders;
#[cfg(all(
    not(all(target_arch = "x86_64", target_feature = "avx2")),
    any(target_arch = "aarch64", target_arch = "loongarch64")
))]
use hyperbola as sliders;
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
    target_arch = "loongarch64"
)))]
use magic as sliders;

/// Hyperbola quintessence, for targets with an efficient bit reversal. See
/// <https://www.chessprogramming.org/Hyperbola_Quintessence>.
#[cfg(any(test, target_arch = "aarch64", target_arch = "loongarch64"))]
#[cfg_attr(test, allow(dead_code))]
mod hyperbola {
    use super::{BISHOP_DIRECTIONS, ROOK_DIRECTIONS, line_mask};
    use crate::types::*;

    // For each square: file, rank, diagonal and anti-diagonal masks.
    static MASKS: [[u64; 4]; SQUARE_NB] = {
        let mut masks = [[0; 4]; SQUARE_NB];
        let mut s = 0;
        while s < SQUARE_NB {
            let sq = s as u8;
            masks[s] = [
                line_mask(sq, ROOK_DIRECTIONS[0], ROOK_DIRECTIONS[1]),
                line_mask(sq, ROOK_DIRECTIONS[2], ROOK_DIRECTIONS[3]),
                line_mask(sq, BISHOP_DIRECTIONS[0], BISHOP_DIRECTIONS[2]),
                line_mask(sq, BISHOP_DIRECTIONS[3], BISHOP_DIRECTIONS[1]),
            ];
            s += 1;
        }
        masks
    };

    #[inline(always)]
    fn hyperbola(s: u8, occupied: u64, mask: u64) -> u64 {
        let o = occupied & mask;
        let fwd = o.wrapping_sub(1 << s);
        let rev = o.reverse_bits().wrapping_sub(1 << (63 - s));
        (fwd ^ rev.reverse_bits()) & mask
    }

    #[inline(always)]
    pub fn bishop_attacks(s: Square, occupied: u64) -> u64 {
        let m = &MASKS[s.idx()];
        let s = s.0 & 63;
        hyperbola(s, occupied, m[2]) | hyperbola(s, occupied, m[3])
    }

    #[inline(always)]
    pub fn rook_attacks(s: Square, occupied: u64) -> u64 {
        let m = &MASKS[s.idx()];
        let s = s.0 & 63;
        hyperbola(s, occupied, m[0]) | hyperbola(s, occupied, m[1])
    }

    #[inline(always)]
    pub fn both_attacks(s: Square, occupied: u64) -> (u64, u64) {
        (bishop_attacks(s, occupied), rook_attacks(s, occupied))
    }
}

/// Dual hyperbola quintessence for AVX2: file and both diagonals in one
/// vector, with a byte swap for the reversal, since each of those lines has
/// one square per byte. Ranks, which a byte swap cannot reverse, come from a
/// table indexed by the slider's file and the rank's six inner squares (the
/// edge squares never change the attacks).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod dual_hyperbola {
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

    /// Bishop and rook attacks at once. Callers that need one of them leave
    /// the other for the compiler to drop.
    #[inline(always)]
    pub fn both_attacks(s: Square, occupied: u64) -> (u64, u64) {
        let m = &DUAL_MAGICS[s.idx()];
        // SAFETY: AVX2 is enabled for the whole build (this module exists
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

            let row =
                RANK_ATTACKS[(m.file & 7) as usize][((occupied >> (m.shift + 1)) & 0x3f) as usize];
            let rank_attacks = (row as u64) << m.shift;

            (
                _mm_extract_epi64::<1>(_mm256_castsi256_si128(rook_bishop)) as u64,
                (_mm_cvtsi128_si64(_mm256_castsi256_si128(result)) as u64)
                    .wrapping_add(rank_attacks),
            )
        }
    }

    #[inline(always)]
    pub fn bishop_attacks(s: Square, occupied: u64) -> u64 {
        both_attacks(s, occupied).0
    }

    #[inline(always)]
    pub fn rook_attacks(s: Square, occupied: u64) -> u64 {
        both_attacks(s, occupied).1
    }
}

/// Magic bitboards, with the magics Stockfish's seeded search finds.
#[cfg(any(
    test,
    not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        target_arch = "aarch64",
        target_arch = "loongarch64"
    ))
))]
#[cfg_attr(test, allow(dead_code))]
mod magic {
    use super::sliding_attack;
    use crate::types::*;

    #[derive(Clone, Copy)]
    struct Magic {
        mask: u64,
        magic: u64,
        offset: usize,
        shift: u32,
    }

    impl Magic {
        #[inline(always)]
        const fn index(&self, occupied: u64) -> usize {
            ((occupied & self.mask).wrapping_mul(self.magic) >> self.shift) as usize
        }
    }

    struct Magics<const N: usize> {
        magics: [Magic; SQUARE_NB],
        table: [u64; N],
    }

    const fn edges(s: usize) -> u64 {
        const RANKS_1_8: u64 = 0xFF | (0xFF << 56);
        const FILES_A_H: u64 = 0x0101010101010101 | (0x0101010101010101 << 7);
        let rank = 0xFFu64 << (8 * (s >> 3));
        let file = 0x0101010101010101u64 << (s & 7);
        (RANKS_1_8 & !rank) | (FILES_A_H & !file)
    }

    /// The squares whose occupancy matters to a slider on `s`.
    pub(super) const fn mask(pt: PieceType, s: usize) -> u64 {
        sliding_attack(pt, Square(s as u8), 0) & !edges(s)
    }

    /// The two lines through `s` that a slider of type `pt` moves along,
    /// without `s`.
    const fn lines(pt: PieceType, s: usize) -> [u64; 2] {
        let (f, r) = ((s & 7) as i32, (s >> 3) as i32);
        let mut lines = [0u64; 2];
        let mut t = 0;
        while t < SQUARE_NB {
            let (tf, tr) = ((t & 7) as i32, (t >> 3) as i32);
            let on = if pt.0 == ROOK.0 {
                [tf == f, tr == r]
            } else {
                [tf - tr == f - r, tf + tr == f + r]
            };
            if t != s && on[0] {
                lines[0] |= 1 << t;
            }
            if t != s && on[1] {
                lines[1] |= 1 << t;
            }
            t += 1;
        }
        lines
    }

    /// The attacks along `line` from `s` by hyperbola quintessence, which
    /// constant evaluation computes much faster than by stepping.
    const fn hyperbola(s: usize, occupied: u64, line: u64) -> u64 {
        let o = occupied & line;
        let fwd = o.wrapping_sub(1 << s);
        let rev = o.reverse_bits().wrapping_sub(1 << (63 - s));
        (fwd ^ rev.reverse_bits()) & line
    }

    /// Fills the attack tables, checking that each magic maps every
    /// occupancy of its mask to an entry of its own attacks.
    const fn init<const N: usize>(pt: PieceType, numbers: &[u64; SQUARE_NB]) -> Magics<N> {
        let empty = Magic {
            mask: 0,
            magic: 0,
            offset: 0,
            shift: 0,
        };
        let mut result = Magics {
            magics: [empty; SQUARE_NB],
            table: [0; N],
        };
        let mut filled = [false; N];
        let mut offset = 0;

        let mut s = 0;
        while s < SQUARE_NB {
            let mask = mask(pt, s);
            let [line1, line2] = lines(pt, s);
            let m = Magic {
                mask,
                magic: numbers[s],
                offset,
                shift: 64 - mask.count_ones(),
            };

            // Every subset of the mask, by the Carry-Rippler trick
            let mut b = 0u64;
            loop {
                let slot = offset + m.index(b);
                let attacks = hyperbola(s, b, line1) | hyperbola(s, b, line2);
                assert!(
                    !filled[slot] || result.table[slot] == attacks,
                    "not a magic"
                );
                filled[slot] = true;
                result.table[slot] = attacks;
                b = b.wrapping_sub(mask) & mask;
                if b == 0 {
                    break;
                }
            }

            result.magics[s] = m;
            offset += 1 << mask.count_ones();
            s += 1;
        }
        assert!(offset == N);
        result
    }

    // Found by Stockfish's search on 64-bit targets; see the test that
    // repeats it.
    pub(super) const ROOK_NUMBERS: [u64; SQUARE_NB] = [
        0x0a80004000801220,
        0x8040004010002008,
        0x2080200010008008,
        0x1100100008210004,
        0xc200209084020008,
        0x2100010004000208,
        0x0400081000822421,
        0x0200010422048844,
        0x0800800080400024,
        0x0001402000401000,
        0x3000801000802001,
        0x4400800800100083,
        0x0904802402480080,
        0x4040800400020080,
        0x0018808042000100,
        0x4040800080004100,
        0x0040048001458024,
        0x00a0004000205000,
        0x3100808010002000,
        0x4825010010000820,
        0x5004808008000401,
        0x2024818004000a00,
        0x0005808002000100,
        0x2100060004806104,
        0x0080400880008421,
        0x4062220600410280,
        0x010a004a00108022,
        0x0000100080080080,
        0x0021000500080010,
        0x0044000202001008,
        0x0000100400080102,
        0xc020128200040545,
        0x0080002000400040,
        0x0000804000802004,
        0x0000120022004080,
        0x010a386103001001,
        0x9010080080800400,
        0x8440020080800400,
        0x0004228824001001,
        0x000000490a000084,
        0x0080002000504000,
        0x200020005000c000,
        0x0012088020420010,
        0x0010010080080800,
        0x0085001008010004,
        0x0002000204008080,
        0x0040413002040008,
        0x0000304081020004,
        0x0080204000800080,
        0x3008804000290100,
        0x1010100080200080,
        0x2008100208028080,
        0x5000850800910100,
        0x8402019004680200,
        0x0120911028020400,
        0x0000008044010200,
        0x0020850200244012,
        0x0020850200244012,
        0x0000102001040841,
        0x140900040a100021,
        0x000200282410a102,
        0x000200282410a102,
        0x000200282410a102,
        0x4048240043802106,
    ];
    pub(super) const BISHOP_NUMBERS: [u64; SQUARE_NB] = [
        0x40106000a1160020,
        0x0020010250810120,
        0x2010010220280081,
        0x002806004050c040,
        0x0002021018000000,
        0x2001112010000400,
        0x0881010120218080,
        0x1030820110010500,
        0x0000120222042400,
        0x2000020404040044,
        0x8000480094208000,
        0x0003422a02000001,
        0x000a220210100040,
        0x8004820202226000,
        0x0018234854100800,
        0x0100004042101040,
        0x0004001004082820,
        0x0010000810010048,
        0x1014004208081300,
        0x2080818802044202,
        0x0040880c00a00100,
        0x0080400200522010,
        0x0001000188180b04,
        0x0080249202020204,
        0x1004400004100410,
        0x00013100a0022206,
        0x2148500001040080,
        0x4241080011004300,
        0x4020848004002000,
        0x10101380d1004100,
        0x0008004422020284,
        0x01010a1041008080,
        0x0808080400082121,
        0x0808080400082121,
        0x0091128200100c00,
        0x0202200802010104,
        0x8c0a020200440085,
        0x01a0008080b10040,
        0x0889520080122800,
        0x100902022202010a,
        0x04081a0816002000,
        0x0000681208005000,
        0x8170840041008802,
        0x0a00004200810805,
        0x0830404408210100,
        0x2602208106006102,
        0x1048300680802628,
        0x2602208106006102,
        0x0602010120110040,
        0x0941010801043000,
        0x000040440a210428,
        0x0008240020880021,
        0x0400002012048200,
        0x00ac102001210220,
        0x0220021002009900,
        0x84440c080a013080,
        0x0001008044200440,
        0x0004c04410841000,
        0x2000500104011130,
        0x1a0c010011c20229,
        0x0044800112202200,
        0x0434804908100424,
        0x0300404822c08200,
        0x48081010008a2a80,
    ];

    static ROOK_MAGICS: Magics<0x19000> = init(ROOK, &ROOK_NUMBERS);
    static BISHOP_MAGICS: Magics<0x1480> = init(BISHOP, &BISHOP_NUMBERS);

    #[inline(always)]
    pub fn bishop_attacks(s: Square, occupied: u64) -> u64 {
        let m = &BISHOP_MAGICS.magics[s.idx()];
        BISHOP_MAGICS.table[m.offset + m.index(occupied)]
    }

    #[inline(always)]
    pub fn rook_attacks(s: Square, occupied: u64) -> u64 {
        let m = &ROOK_MAGICS.magics[s.idx()];
        ROOK_MAGICS.table[m.offset + m.index(occupied)]
    }

    #[inline(always)]
    pub fn both_attacks(s: Square, occupied: u64) -> (u64, u64) {
        (bishop_attacks(s, occupied), rook_attacks(s, occupied))
    }

    /// Stockfish's search for magics on 64-bit targets: from a seed for each
    /// rank, random sparse numbers until one maps every occupancy of the
    /// mask to an entry that holds its attacks.
    #[cfg(test)]
    pub(super) fn search(pt: PieceType) -> [u64; SQUARE_NB] {
        use crate::misc::Prng;

        const SEEDS: [u64; 8] = [728, 10316, 55013, 32803, 12281, 15100, 16645, 255];
        let mut numbers = [0; SQUARE_NB];

        for (s, number) in numbers.iter_mut().enumerate() {
            let mask = mask(pt, s);
            let shift = 64 - mask.count_ones();
            let mut occupancy = Vec::new();
            let mut reference = Vec::new();
            let mut b = 0u64;
            loop {
                occupancy.push(b);
                reference.push(sliding_attack(pt, Square(s as u8), b));
                b = b.wrapping_sub(mask) & mask;
                if b == 0 {
                    break;
                }
            }

            // `epoch` marks the entries the current attempt set.
            let mut table = vec![0u64; occupancy.len()];
            let mut epoch = vec![0u32; occupancy.len()];
            let mut cnt = 0;
            let mut rng = Prng::new(SEEDS[s >> 3]);
            let mut i = 0;
            while i < occupancy.len() {
                let mut magic = 0u64;
                while (magic.wrapping_mul(mask) >> 56).count_ones() < 6 {
                    magic = rng.sparse_rand();
                }
                cnt += 1;
                i = 0;
                while i < occupancy.len() {
                    let slot = ((occupancy[i] & mask).wrapping_mul(magic) >> shift) as usize;
                    if epoch[slot] < cnt {
                        epoch[slot] = cnt;
                        table[slot] = reference[i];
                    } else if table[slot] != reference[i] {
                        break;
                    }
                    i += 1;
                }
                *number = magic;
            }
        }
        numbers
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
                    _ if i % 2 == 0 => rng.sparse_rand(),
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
        check_sliders("magic", magic::bishop_attacks, magic::rook_attacks);
        check_sliders(
            "hyperbola",
            hyperbola::bishop_attacks,
            hyperbola::rook_attacks,
        );
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        check_sliders(
            "dual hyperbola",
            dual_hyperbola::bishop_attacks,
            dual_hyperbola::rook_attacks,
        );
        check_sliders(
            "active",
            |s, o| bishop_attacks(s, Bitboard(o)).0,
            |s, o| rook_attacks(s, Bitboard(o)).0,
        );
    }

    #[test]
    fn the_magics_are_those_stockfish_finds() {
        assert_eq!(magic::search(ROOK), magic::ROOK_NUMBERS);
        assert_eq!(magic::search(BISHOP), magic::BISHOP_NUMBERS);

        // Every entry of the tables
        for s in 0..64 {
            for (pt, attacks) in [
                (ROOK, magic::rook_attacks as fn(Square, u64) -> u64),
                (BISHOP, magic::bishop_attacks),
            ] {
                let mask = magic::mask(pt, s);
                let mut b = 0u64;
                loop {
                    assert_eq!(
                        attacks(Square(s as u8), b),
                        sliding_attack(pt, Square(s as u8), b)
                    );
                    b = b.wrapping_sub(mask) & mask;
                    if b == 0 {
                        break;
                    }
                }
            }
        }
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
