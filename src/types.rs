//! Colors, pieces, squares, moves, values and the records of what a move
//! changes.
//!
//! Colors, piece types, pieces and squares are thin wrappers around `u8` with
//! Stockfish's numbering. Their `idx` masks the value to the size of the
//! tables they index, so lookups need no bounds checks and an out-of-range
//! value reads a wrong entry rather than memory outside the table.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `types.h`: Stockfish, a UCI chess playing engine derived from Glaurung 2.1,
//! Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under the
//! GNU General Public License, version 3 or later.

use std::ops::{Add, AddAssign, Not, Sub, SubAssign};

use crate::misc::ValueList;

pub type Key = u64;
pub type Value = i32;
pub type Depth = i32;

pub const MAX_MOVES: usize = 256;
pub const MAX_PLY: i32 = 246;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct Color(pub u8);

pub const WHITE: Color = Color(0);
pub const BLACK: Color = Color(1);
pub const COLOR_NB: usize = 2;

impl Color {
    #[inline(always)]
    pub const fn idx(self) -> usize {
        (self.0 & 1) as usize
    }
}

impl Not for Color {
    type Output = Color;

    #[inline(always)]
    fn not(self) -> Color {
        Color(self.0 ^ 1)
    }
}

/// A set of castling rights, one bit per right.
pub type CastlingRights = u8;

pub const NO_CASTLING: CastlingRights = 0;
pub const WHITE_OO: CastlingRights = 1;
pub const WHITE_OOO: CastlingRights = WHITE_OO << 1;
pub const BLACK_OO: CastlingRights = WHITE_OO << 2;
pub const BLACK_OOO: CastlingRights = WHITE_OO << 3;
pub const KING_SIDE: CastlingRights = WHITE_OO | BLACK_OO;
pub const QUEEN_SIDE: CastlingRights = WHITE_OOO | BLACK_OOO;
pub const WHITE_CASTLING: CastlingRights = WHITE_OO | WHITE_OOO;
pub const BLACK_CASTLING: CastlingRights = BLACK_OO | BLACK_OOO;
pub const ANY_CASTLING: CastlingRights = WHITE_CASTLING | BLACK_CASTLING;
pub const CASTLING_RIGHT_NB: usize = 16;

/// The rights in `cr` that belong to `c`, Stockfish's `c & cr`.
#[inline(always)]
pub const fn castling_rights_of(c: Color, cr: CastlingRights) -> CastlingRights {
    (if c.0 == 0 {
        WHITE_CASTLING
    } else {
        BLACK_CASTLING
    }) & cr
}

// The following DEPTH_ constants are used for transposition table entries
// and quiescence search move generation stages. In regular search, the
// depth stored in the transposition table is literal: the search depth
// (effort) used to make the corresponding transposition table value. In
// quiescence search, however, the transposition table entries only store
// the current quiescence move generation stage (which should thus compare
// lower than any regular search depth).
pub const DEPTH_QS: Depth = 0;
// For transposition table entries where no searching at all was done
// (whether regular or qsearch) we use DEPTH_UNSEARCHED, which should thus
// compare lower than any quiescence or regular depth. DEPTH_NONE is used
// for the transposition table entry occupancy check (see tt.rs), and
// should thus be lower than DEPTH_UNSEARCHED.
pub const DEPTH_UNSEARCHED: Depth = -2;
pub const DEPTH_NONE: Depth = -3;

pub type Bound = u8;

pub const BOUND_NONE: Bound = 0;
pub const BOUND_UPPER: Bound = 1;
pub const BOUND_LOWER: Bound = 2;
pub const BOUND_EXACT: Bound = BOUND_UPPER | BOUND_LOWER;

// Search values are always in the range (-VALUE_NONE, VALUE_NONE].
pub const VALUE_ZERO: Value = 0;
pub const VALUE_DRAW: Value = 0;
pub const VALUE_NONE: Value = 32002;
pub const VALUE_INFINITE: Value = 32001;

pub const VALUE_MATE: Value = 32000;
pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY;
pub const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;

pub const VALUE_TB: Value = VALUE_MATE_IN_MAX_PLY - 1;
pub const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_TB - MAX_PLY;
pub const VALUE_TB_LOSS_IN_MAX_PLY: Value = -VALUE_TB_WIN_IN_MAX_PLY;

pub const fn is_valid(value: Value) -> bool {
    value != VALUE_NONE
}

pub const fn is_win(value: Value) -> bool {
    debug_assert!(is_valid(value));
    value >= VALUE_TB_WIN_IN_MAX_PLY
}

pub const fn is_loss(value: Value) -> bool {
    debug_assert!(is_valid(value));
    value <= VALUE_TB_LOSS_IN_MAX_PLY
}

pub const fn is_decisive(value: Value) -> bool {
    is_win(value) || is_loss(value)
}

pub const fn is_mate(value: Value) -> bool {
    debug_assert!(is_valid(value));
    value >= VALUE_MATE_IN_MAX_PLY
}

pub const fn is_mated(value: Value) -> bool {
    debug_assert!(is_valid(value));
    value <= VALUE_MATED_IN_MAX_PLY
}

pub const fn is_mate_or_mated(value: Value) -> bool {
    is_mate(value) || is_mated(value)
}

pub const fn mate_in(ply: i32) -> Value {
    VALUE_MATE - ply
}

pub const fn mated_in(ply: i32) -> Value {
    -VALUE_MATE + ply
}

// These values identify the material uniquely through non_pawn_material().
pub const PAWN_VALUE: Value = 208;
pub const KNIGHT_VALUE: Value = 781;
pub const BISHOP_VALUE: Value = 825;
pub const ROOK_VALUE: Value = 1276;
pub const QUEEN_VALUE: Value = 2538;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct PieceType(pub u8);

pub const NO_PIECE_TYPE: PieceType = PieceType(0);
pub const PAWN: PieceType = PieceType(1);
pub const KNIGHT: PieceType = PieceType(2);
pub const BISHOP: PieceType = PieceType(3);
pub const ROOK: PieceType = PieceType(4);
pub const QUEEN: PieceType = PieceType(5);
pub const KING: PieceType = PieceType(6);
pub const ALL_PIECES: PieceType = PieceType(0);
pub const PIECE_TYPE_NB: usize = 8;

impl PieceType {
    #[inline(always)]
    pub const fn idx(self) -> usize {
        (self.0 & 7) as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct Piece(pub u8);

pub const NO_PIECE: Piece = Piece(0);
pub const W_PAWN: Piece = Piece(1);
pub const W_KNIGHT: Piece = Piece(2);
pub const W_BISHOP: Piece = Piece(3);
pub const W_ROOK: Piece = Piece(4);
pub const W_QUEEN: Piece = Piece(5);
pub const W_KING: Piece = Piece(6);
pub const B_PAWN: Piece = Piece(9);
pub const B_KNIGHT: Piece = Piece(10);
pub const B_BISHOP: Piece = Piece(11);
pub const B_ROOK: Piece = Piece(12);
pub const B_QUEEN: Piece = Piece(13);
pub const B_KING: Piece = Piece(14);
pub const PIECE_NB: usize = 16;

/// The twelve pieces in Stockfish's order, which the Zobrist keys follow.
pub const PIECES: [Piece; 12] = [
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK,
    B_QUEEN, B_KING,
];

pub const PIECE_VALUE: [Value; PIECE_NB] = [
    VALUE_ZERO,
    PAWN_VALUE,
    KNIGHT_VALUE,
    BISHOP_VALUE,
    ROOK_VALUE,
    QUEEN_VALUE,
    VALUE_ZERO,
    VALUE_ZERO,
    VALUE_ZERO,
    PAWN_VALUE,
    KNIGHT_VALUE,
    BISHOP_VALUE,
    ROOK_VALUE,
    QUEEN_VALUE,
    VALUE_ZERO,
    VALUE_ZERO,
];

impl Piece {
    #[inline(always)]
    pub const fn idx(self) -> usize {
        (self.0 & 15) as usize
    }

    #[inline(always)]
    pub const fn type_of(self) -> PieceType {
        PieceType(self.0 & 7)
    }

    #[inline(always)]
    pub const fn color(self) -> Color {
        debug_assert!(self.0 != NO_PIECE.0);
        Color(self.0 >> 3)
    }

    #[inline(always)]
    pub const fn is_some(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn value(self) -> Value {
        PIECE_VALUE[self.idx()]
    }
}

/// Swaps the color of a piece: B_KNIGHT <-> W_KNIGHT.
impl Not for Piece {
    type Output = Piece;

    #[inline(always)]
    fn not(self) -> Piece {
        Piece(self.0 ^ 8)
    }
}

#[inline(always)]
pub const fn make_piece(c: Color, pt: PieceType) -> Piece {
    Piece((c.0 << 3) + pt.0)
}

/// The index of the count of `c`'s pieces of type `pt` (all of them for
/// ALL_PIECES), as Stockfish's `pieceCount[make_piece(c, pt)]`.
#[inline(always)]
pub const fn piece_count_index(c: Color, pt: PieceType) -> usize {
    (((c.0 & 1) << 3) | (pt.0 & 7)) as usize
}

pub type File = u8;

pub const FILE_A: File = 0;
pub const FILE_B: File = 1;
pub const FILE_C: File = 2;
pub const FILE_D: File = 3;
pub const FILE_E: File = 4;
pub const FILE_F: File = 5;
pub const FILE_G: File = 6;
pub const FILE_H: File = 7;
pub const FILE_NB: usize = 8;

pub type Rank = u8;

pub const RANK_1: Rank = 0;
pub const RANK_2: Rank = 1;
pub const RANK_3: Rank = 2;
pub const RANK_4: Rank = 3;
pub const RANK_5: Rank = 4;
pub const RANK_6: Rank = 5;
pub const RANK_7: Rank = 6;
pub const RANK_8: Rank = 7;
pub const RANK_NB: usize = 8;

pub type Direction = i8;

pub const NORTH: Direction = 8;
pub const EAST: Direction = 1;
pub const SOUTH: Direction = -NORTH;
pub const WEST: Direction = -EAST;
pub const NORTH_EAST: Direction = NORTH + EAST;
pub const SOUTH_EAST: Direction = SOUTH + EAST;
pub const SOUTH_WEST: Direction = SOUTH + WEST;
pub const NORTH_WEST: Direction = NORTH + WEST;

#[inline(always)]
pub const fn pawn_push(c: Color) -> Direction {
    if c.0 == 0 { NORTH } else { SOUTH }
}

#[inline(always)]
pub const fn relative_rank(c: Color, r: Rank) -> Rank {
    r ^ (c.0 * 7)
}

/// A square from A1 = 0 to H8 = 63, or `Square::NONE`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Square(pub u8);

pub const SQUARE_NB: usize = 64;

macro_rules! squares {
    ($($name:ident = $value:expr),* $(,)?) => {
        impl Square {
            $(pub const $name: Square = Square($value);)*
        }
    };
}

squares! {
    A1 = 0, B1 = 1, C1 = 2, D1 = 3, E1 = 4, F1 = 5, G1 = 6, H1 = 7,
    A2 = 8, B2 = 9, C2 = 10, D2 = 11, E2 = 12, F2 = 13, G2 = 14, H2 = 15,
    A3 = 16, B3 = 17, C3 = 18, D3 = 19, E3 = 20, F3 = 21, G3 = 22, H3 = 23,
    A4 = 24, B4 = 25, C4 = 26, D4 = 27, E4 = 28, F4 = 29, G4 = 30, H4 = 31,
    A5 = 32, B5 = 33, C5 = 34, D5 = 35, E5 = 36, F5 = 37, G5 = 38, H5 = 39,
    A6 = 40, B6 = 41, C6 = 42, D6 = 43, E6 = 44, F6 = 45, G6 = 46, H6 = 47,
    A7 = 48, B7 = 49, C7 = 50, D7 = 51, E7 = 52, F7 = 53, G7 = 54, H7 = 55,
    A8 = 56, B8 = 57, C8 = 58, D8 = 59, E8 = 60, F8 = 61, G8 = 62, H8 = 63,
    NONE = 64,
}

impl Square {
    #[inline(always)]
    pub const fn new(f: File, r: Rank) -> Square {
        Square((r << 3) + f)
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        debug_assert!(self.0 < 64);
        (self.0 & 63) as usize
    }

    #[inline(always)]
    pub const fn is_ok(self) -> bool {
        self.0 < 64
    }

    #[inline(always)]
    pub const fn file(self) -> File {
        self.0 & 7
    }

    #[inline(always)]
    pub const fn rank(self) -> Rank {
        self.0 >> 3
    }

    /// The square as seen by `c`: flipped vertically for Black.
    #[inline(always)]
    pub const fn relative(self, c: Color) -> Square {
        Square(self.0 ^ (c.0 * 56))
    }

    #[inline(always)]
    pub const fn relative_rank(self, c: Color) -> Rank {
        relative_rank(c, self.rank())
    }

    /// Swaps A1 <-> A8.
    #[inline(always)]
    pub const fn flip_rank(self) -> Square {
        Square(self.0 ^ Square::A8.0)
    }

    /// Swaps A1 <-> H1.
    #[inline(always)]
    pub const fn flip_file(self) -> Square {
        Square(self.0 ^ Square::H1.0)
    }

    /// The square `d` away, wrapping around as Stockfish's arithmetic on its
    /// 8-bit squares does; `is_ok` tells whether it is on the board.
    #[inline(always)]
    pub const fn offset(self, d: Direction) -> Square {
        Square(self.0.wrapping_add(d as u8))
    }
}

impl std::fmt::Debug for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_ok() {
            write!(
                f,
                "{}{}",
                (b'a' + self.file()) as char,
                (b'1' + self.rank()) as char
            )
        } else {
            write!(f, "Square({})", self.0)
        }
    }
}

impl Add<Direction> for Square {
    type Output = Square;

    #[inline(always)]
    fn add(self, d: Direction) -> Square {
        self.offset(d)
    }
}

impl Sub<Direction> for Square {
    type Output = Square;

    #[inline(always)]
    fn sub(self, d: Direction) -> Square {
        self.offset(d.wrapping_neg())
    }
}

impl AddAssign<Direction> for Square {
    #[inline(always)]
    fn add_assign(&mut self, d: Direction) {
        *self = self.offset(d);
    }
}

impl SubAssign<Direction> for Square {
    #[inline(always)]
    fn sub_assign(&mut self, d: Direction) {
        *self = self.offset(d.wrapping_neg());
    }
}

/// Based on a congruential pseudo-random number generator.
#[inline(always)]
pub const fn make_key(seed: u64) -> Key {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

pub type MoveType = u16;

pub const NORMAL: MoveType = 0;
pub const PROMOTION: MoveType = 1 << 14;
pub const EN_PASSANT: MoveType = 2 << 14;
pub const CASTLING: MoveType = 3 << 14;

/// A move in 16 bits:
///
/// - bits 0-5: destination square (from 0 to 63)
/// - bits 6-11: origin square (from 0 to 63)
/// - bits 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
/// - bits 14-15: special move flag: promotion (1), en passant (2), castling (3)
///
/// Castling is encoded as the king capturing its own rook. `Move::none()`
/// and `Move::null()` have the same origin and destination, which no real
/// move has.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(transparent)]
pub struct Move(pub u16);

impl Move {
    pub const FROM_SQ_SHIFT: u32 = 6;
    pub const TO_SQ_SHIFT: u32 = 0;

    #[inline(always)]
    pub const fn new(from: Square, to: Square) -> Move {
        Move(((from.0 as u16) << 6) + to.0 as u16)
    }

    /// A move of type `mt`; `pt` is the promotion piece, and KNIGHT for
    /// moves that do not promote.
    #[inline(always)]
    pub const fn make(mt: MoveType, from: Square, to: Square, pt: PieceType) -> Move {
        Move(mt + (((pt.0 - KNIGHT.0) as u16) << 12) + ((from.0 as u16) << 6) + to.0 as u16)
    }

    #[inline(always)]
    pub const fn from_sq(self) -> Square {
        debug_assert!(self.is_ok());
        Square(((self.0 >> 6) & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn to_sq(self) -> Square {
        debug_assert!(self.is_ok());
        Square((self.0 & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn type_of(self) -> MoveType {
        self.0 & (3 << 14)
    }

    #[inline(always)]
    pub const fn promotion_type(self) -> PieceType {
        PieceType((((self.0 >> 12) & 3) as u8) + KNIGHT.0)
    }

    #[inline(always)]
    pub const fn is_ok(self) -> bool {
        Move::none().0 != self.0 && Move::null().0 != self.0
    }

    #[inline(always)]
    pub const fn null() -> Move {
        Move(65)
    }

    #[inline(always)]
    pub const fn none() -> Move {
        Move(0)
    }

    #[inline(always)]
    pub const fn raw(self) -> u16 {
        self.0
    }
}

/// What a move changes on the board, for the network.
///
/// `to` is `Square::NONE` for promotions. Castling removes and adds the rook
/// through `remove_sq` and `add_sq`; when either is `Square::NONE`, its piece
/// means nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DirtyPiece {
    pub pc: Piece,
    pub from: Square,
    pub to: Square,
    pub remove_sq: Square,
    pub add_sq: Square,
    pub remove_pc: Piece,
    pub add_pc: Piece,
}

/// A threat that a move adds or removes, for the network: a piece on a square
/// attacking another piece on another square.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(transparent)]
pub struct DirtyThreat(pub u32);

impl DirtyThreat {
    pub const PC_SQ_OFFSET: u32 = 0;
    pub const THREATENED_SQ_OFFSET: u32 = 8;
    pub const THREATENED_PC_OFFSET: u32 = 16;
    pub const PC_OFFSET: u32 = 20;

    #[inline(always)]
    pub const fn new(
        pc: Piece,
        threatened_pc: Piece,
        pc_sq: Square,
        threatened_sq: Square,
        add: bool,
    ) -> DirtyThreat {
        DirtyThreat(
            ((add as u32) << 31)
                | ((pc.0 as u32) << Self::PC_OFFSET)
                | ((threatened_pc.0 as u32) << Self::THREATENED_PC_OFFSET)
                | ((threatened_sq.0 as u32) << Self::THREATENED_SQ_OFFSET)
                | ((pc_sq.0 as u32) << Self::PC_SQ_OFFSET),
        )
    }

    #[inline(always)]
    pub const fn pc(self) -> Piece {
        Piece((self.0 >> Self::PC_OFFSET & 0xf) as u8)
    }

    #[inline(always)]
    pub const fn threatened_pc(self) -> Piece {
        Piece((self.0 >> Self::THREATENED_PC_OFFSET & 0xf) as u8)
    }

    #[inline(always)]
    pub const fn threatened_sq(self) -> Square {
        Square((self.0 >> Self::THREATENED_SQ_OFFSET & 0xff) as u8)
    }

    #[inline(always)]
    pub const fn pc_sq(self) -> Square {
        Square((self.0 >> Self::PC_SQ_OFFSET & 0xff) as u8)
    }

    #[inline(always)]
    pub const fn add(self) -> bool {
        self.0 >> 31 != 0
    }

    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

// A piece takes part in at most 8 outgoing and 16 incoming attacks, and moving
// it reveals at most 8 attacks, so a move other than castling changes at most
// (8 + 16) * 3 + 8 = 80 threats, and castling (5 + 1 + 3 + 9) * 2 = 36. The
// list holds the next power of two.
pub type DirtyThreats = ValueList<DirtyThreat, 128>;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DirtyPawnPairs {
    pub before: [crate::bitboard::Bitboard; COLOR_NB],
    pub after: [crate::bitboard::Bitboard; COLOR_NB],
}

/// Everything a move changes that the network follows.
#[derive(Clone, Default)]
pub struct Dirties {
    pub dirty_piece: DirtyPiece,
    pub dirty_threats: DirtyThreats,
    pub dirty_pawn_pairs: DirtyPawnPairs,
}

/// What a move changes that a variant's network follows.
pub trait MoveDirties: Default + Send + 'static {
    /// Readies the record for a move made outside a search, which fills it
    /// again.
    fn reset(&mut self);
}

impl MoveDirties for Dirties {
    #[inline(always)]
    fn reset(&mut self) {
        // do_move appends threats to those already recorded.
        self.dirty_threats.clear();
    }
}

/// The pieces an Atomic capture explodes besides the capturing and the
/// captured ones: those next to the capture but pawns. Packed to twelve
/// bytes, which keeps an Atomic state to 256, a size compilers copy inline.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C, packed(4))]
pub struct Blast {
    squares: crate::bitboard::Bitboard,
    // The pieces on `squares`, four bits each, the lowest square's lowest
    pieces: u32,
}

impl Blast {
    /// The pieces `board` holds on `squares`, at most 8.
    #[inline(always)]
    pub fn new(squares: crate::bitboard::Bitboard, board: &[Piece; SQUARE_NB]) -> Blast {
        debug_assert!(squares.popcount() <= 8);
        let mut pieces = 0;
        for (i, s) in squares.into_iter().enumerate() {
            pieces |= (board[s.idx()].0 as u32) << (4 * i);
        }
        Blast { squares, pieces }
    }

    #[inline(always)]
    pub fn squares(self) -> crate::bitboard::Bitboard {
        self.squares
    }

    /// The exploded pieces and their squares, the lowest square first.
    #[inline(always)]
    pub fn iter(self) -> impl Iterator<Item = (Piece, Square)> {
        self.squares
            .into_iter()
            .enumerate()
            .map(move |(i, s)| (Piece((self.pieces >> (4 * i) & 15) as u8), s))
    }
}

/// Everything an Atomic move changes that the network follows: the move as
/// `DirtyPiece` records a chess move, but where a capture explodes the
/// capturing piece `to` is `Square::NONE` and nothing is added, and the other
/// pieces the capture explodes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct AtomicDirties {
    pub dirty_piece: DirtyPiece,
    pub blast: Blast,
}

impl MoveDirties for AtomicDirties {
    #[inline(always)]
    fn reset(&mut self) {}
}
