//! Bitboards: sets of squares, one bit per square from A1 (bit 0) to H8.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `bitboard.h` and `bitboard.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers
//! (see AUTHORS), under the GNU General Public License, version 3 or later.

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

pub const FILE_A_BB: Bitboard = Bitboard(0x0101010101010101);
pub const FILE_B_BB: Bitboard = Bitboard(FILE_A_BB.0 << 1);
pub const FILE_C_BB: Bitboard = Bitboard(FILE_A_BB.0 << 2);
pub const FILE_D_BB: Bitboard = Bitboard(FILE_A_BB.0 << 3);
pub const FILE_E_BB: Bitboard = Bitboard(FILE_A_BB.0 << 4);
pub const FILE_F_BB: Bitboard = Bitboard(FILE_A_BB.0 << 5);
pub const FILE_G_BB: Bitboard = Bitboard(FILE_A_BB.0 << 6);
pub const FILE_H_BB: Bitboard = Bitboard(FILE_A_BB.0 << 7);

pub const RANK_1_BB: Bitboard = Bitboard(0xFF);
pub const RANK_2_BB: Bitboard = Bitboard(RANK_1_BB.0 << 8);
pub const RANK_3_BB: Bitboard = Bitboard(RANK_1_BB.0 << (8 * 2));
pub const RANK_4_BB: Bitboard = Bitboard(RANK_1_BB.0 << (8 * 3));
pub const RANK_5_BB: Bitboard = Bitboard(RANK_1_BB.0 << (8 * 4));
pub const RANK_6_BB: Bitboard = Bitboard(RANK_1_BB.0 << (8 * 5));
pub const RANK_7_BB: Bitboard = Bitboard(RANK_1_BB.0 << (8 * 6));
pub const RANK_8_BB: Bitboard = Bitboard(RANK_1_BB.0 << (8 * 7));

/// The squares in order, a byte each, from which the byte compression of
/// Ice Lake's AVX-512 takes a bitboard's squares, as Stockfish's
/// `AllSquares`.
#[cfg(use_avx512icl)]
#[repr(C, align(64))]
struct AllSquares([u8; SQUARE_NB]);

#[cfg(use_avx512icl)]
static ALL_SQUARES: AllSquares = AllSquares({
    let mut squares = [0; SQUARE_NB];
    let mut i = 0;
    while i < SQUARE_NB {
        squares[i] = i as u8;
        i += 1;
    }
    squares
});

/// The squares in order, a byte each.
#[cfg(use_avx512icl)]
#[inline(always)]
pub fn all_squares() -> std::arch::x86_64::__m512i {
    // SAFETY: the build targets Ice Lake's AVX-512, and ALL_SQUARES is 64
    // aligned bytes.
    unsafe { std::arch::x86_64::_mm512_load_si512(ALL_SQUARES.0.as_ptr().cast()) }
}

/// The squares of `b`, lowest first, a byte each from the vector's first,
/// and zeros after them.
#[cfg(use_avx512icl)]
#[inline(always)]
pub fn compress_squares(b: Bitboard) -> std::arch::x86_64::__m512i {
    // SAFETY: the build targets Ice Lake's AVX-512.
    unsafe { std::arch::x86_64::_mm512_maskz_compress_epi8(b.0, all_squares()) }
}

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(!0);

    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn has(self, s: Square) -> bool {
        self.0 & square_bb(s).0 != 0
    }

    #[inline(always)]
    pub const fn more_than_one(self) -> bool {
        self.0 & self.0.wrapping_sub(1) != 0
    }

    #[inline(always)]
    pub const fn popcount(self) -> i32 {
        self.0.count_ones() as i32
    }

    /// The least significant square of a non-empty bitboard.
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(lowest_bit(self.0) as u8)
    }

    /// The most significant square of a non-empty bitboard.
    #[inline(always)]
    pub const fn msb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(63 ^ self.0.leading_zeros() as u8)
    }

    /// The least significant square of a non-empty bitboard, as a bitboard.
    #[inline(always)]
    pub const fn lsb_bb(self) -> Bitboard {
        debug_assert!(self.0 != 0);
        Bitboard(self.0 & self.0.wrapping_neg())
    }

    /// Removes and returns the least significant square of a non-empty
    /// bitboard.
    #[inline(always)]
    pub const fn pop_lsb(&mut self) -> Square {
        let s = self.lsb();
        self.0 &= self.0.wrapping_sub(1);
        s
    }
}

impl std::fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bitboard({:#018x})", self.0)
    }
}

/// The squares of a bitboard, from the least significant.
pub struct Squares(u64);

impl Iterator for Squares {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            return None;
        }
        let s = Square(lowest_bit(self.0) as u8);
        self.0 &= self.0 - 1;
        Some(s)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = Squares;

    #[inline(always)]
    fn into_iter(self) -> Squares {
        Squares(self.0)
    }
}

/// The index of the lowest set bit of a non-zero word, its trailing zeros.
/// In WebAssembly it is the leading zeros of that bit alone, from the top:
/// V8 compiles WebAssembly's count of trailing zeros for arm64 processors
/// without its instruction into a call to a C function, but its count of
/// leading zeros into one instruction.
#[inline(always)]
pub const fn lowest_bit(x: u64) -> u32 {
    debug_assert!(x != 0);
    #[cfg(target_arch = "wasm32")]
    {
        63 ^ x.isolate_lowest_one().leading_zeros()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        x.trailing_zeros()
    }
}

#[inline(always)]
pub const fn square_bb(s: Square) -> Bitboard {
    debug_assert!(s.is_ok());
    Bitboard(1 << (s.0 & 63))
}

macro_rules! bit_ops {
    ($rhs:ty, $to_bits:expr) => {
        impl BitAnd<$rhs> for Bitboard {
            type Output = Bitboard;

            #[inline(always)]
            fn bitand(self, rhs: $rhs) -> Bitboard {
                Bitboard(self.0 & $to_bits(rhs))
            }
        }

        impl BitOr<$rhs> for Bitboard {
            type Output = Bitboard;

            #[inline(always)]
            fn bitor(self, rhs: $rhs) -> Bitboard {
                Bitboard(self.0 | $to_bits(rhs))
            }
        }

        impl BitXor<$rhs> for Bitboard {
            type Output = Bitboard;

            #[inline(always)]
            fn bitxor(self, rhs: $rhs) -> Bitboard {
                Bitboard(self.0 ^ $to_bits(rhs))
            }
        }

        impl BitAndAssign<$rhs> for Bitboard {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: $rhs) {
                self.0 &= $to_bits(rhs);
            }
        }

        impl BitOrAssign<$rhs> for Bitboard {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: $rhs) {
                self.0 |= $to_bits(rhs);
            }
        }

        impl BitXorAssign<$rhs> for Bitboard {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: $rhs) {
                self.0 ^= $to_bits(rhs);
            }
        }
    };
}

bit_ops!(Bitboard, |b: Bitboard| b.0);
bit_ops!(Square, |s: Square| square_bb(s).0);

impl Not for Bitboard {
    type Output = Bitboard;

    #[inline(always)]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl BitAnd<Bitboard> for Square {
    type Output = Bitboard;

    #[inline(always)]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        rhs & self
    }
}

impl BitOr<Bitboard> for Square {
    type Output = Bitboard;

    #[inline(always)]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        rhs | self
    }
}

impl BitXor<Bitboard> for Square {
    type Output = Bitboard;

    #[inline(always)]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        rhs ^ self
    }
}

impl BitOr<Square> for Square {
    type Output = Bitboard;

    #[inline(always)]
    fn bitor(self, rhs: Square) -> Bitboard {
        square_bb(self) | rhs
    }
}

#[inline(always)]
pub const fn rank_bb(r: Rank) -> Bitboard {
    Bitboard(RANK_1_BB.0 << (8 * (r & 7)))
}

#[inline(always)]
pub const fn file_bb(f: File) -> Bitboard {
    Bitboard(FILE_A_BB.0 << (f & 7))
}

/// Moves a bitboard one or two steps in the direction `d`.
#[inline(always)]
pub const fn shift(b: Bitboard, d: Direction) -> Bitboard {
    let b = b.0;
    Bitboard(match d {
        NORTH => b << 8,
        SOUTH => b >> 8,
        16 => b << 16,
        -16 => b >> 16,
        EAST => (b & !FILE_H_BB.0) << 1,
        WEST => (b & !FILE_A_BB.0) >> 1,
        NORTH_EAST => (b & !FILE_H_BB.0) << 9,
        NORTH_WEST => (b & !FILE_A_BB.0) << 7,
        SOUTH_EAST => (b & !FILE_H_BB.0) >> 7,
        SOUTH_WEST => (b & !FILE_A_BB.0) >> 9,
        _ => 0,
    })
}

/// The squares attacked by pawns of color `c` from the squares in `b`.
#[inline(always)]
pub const fn pawn_attacks_bb(c: Color, b: Bitboard) -> Bitboard {
    if c.0 == WHITE.0 {
        Bitboard(shift(b, NORTH_WEST).0 | shift(b, NORTH_EAST).0)
    } else {
        Bitboard(shift(b, SOUTH_WEST).0 | shift(b, SOUTH_EAST).0)
    }
}

#[inline(always)]
pub const fn pawn_single_push_bb(c: Color, b: Bitboard) -> Bitboard {
    shift(b, pawn_push(c))
}

const PAWN_PAIR_BB: [Bitboard; SQUARE_NB] = {
    let mut result = [Bitboard::EMPTY; SQUARE_NB];
    let mut s = 0;
    while s < SQUARE_NB {
        let file = file_bb(Square(s as u8).file());
        let files = file.0 | shift(file, EAST).0 | shift(file, WEST).0;
        result[s] = Bitboard(files & !(RANK_1_BB.0 | RANK_8_BB.0) & !(1 << s));
        s += 1;
    }
    result
};

/// The squares that can hold a pawn forming a "pawn pair" with a pawn on `s`:
/// its own file and the adjacent ones, on ranks 2 to 7, without `s`. The
/// geometry is the same for both colors.
#[inline(always)]
pub const fn pawn_pair_bb(s: Square) -> Bitboard {
    PAWN_PAIR_BB[s.idx()]
}

#[inline(always)]
pub const fn edge_distance(f: File) -> i32 {
    let f = f as i32;
    if f < FILE_H as i32 - f {
        f
    } else {
        FILE_H as i32 - f
    }
}

/// An ASCII drawing of a bitboard, for debugging.
pub fn pretty(b: Bitboard) -> String {
    let mut s = String::from("+---+---+---+---+---+---+---+---+\n");
    for r in (RANK_1..=RANK_8).rev() {
        for f in FILE_A..=FILE_H {
            s += if b.has(Square::new(f, r)) {
                "| X "
            } else {
                "|   "
            };
        }
        s += &format!("| {}\n+---+---+---+---+---+---+---+---+\n", 1 + r);
    }
    s += "  a   b   c   d   e   f   g   h\n";
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifts_stay_on_the_board() {
        assert_eq!(shift(FILE_H_BB, EAST), Bitboard::EMPTY);
        assert_eq!(shift(FILE_A_BB, WEST), Bitboard::EMPTY);
        assert_eq!(shift(RANK_8_BB, NORTH), Bitboard::EMPTY);
        assert_eq!(
            shift(square_bb(Square::E4), NORTH_WEST),
            square_bb(Square::D5)
        );
        assert_eq!(
            shift(square_bb(Square::E2), NORTH + NORTH),
            square_bb(Square::E4)
        );
    }

    #[test]
    fn pawn_pairs_exclude_the_pawn_and_the_back_ranks() {
        let b = pawn_pair_bb(Square::A2);
        assert!(!b.has(Square::A2) && !b.has(Square::A1) && !b.has(Square::B8));
        assert!(b.has(Square::B2) && b.has(Square::A7));
        assert_eq!(b.popcount(), 11);
    }

    #[test]
    fn squares_come_out_from_the_least_significant() {
        let b = Square::H8 | Square::A1;
        assert_eq!(
            b.into_iter().collect::<Vec<_>>(),
            vec![Square::A1, Square::H8]
        );
        assert_eq!(b.lsb(), Square::A1);
        assert_eq!(b.msb(), Square::H8);
        assert!(b.more_than_one());
    }
}
