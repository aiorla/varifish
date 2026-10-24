//! The position: pieces, side to move, castling, the state kept for each ply,
//! Zobrist keys, making and unmaking moves, legality, checks, static exchange
//! evaluation and repetitions.
//!
//! The states of the plies played since the position was set form a stack,
//! where Stockfish links them through `StateInfo::previous`. Doing a move
//! pushes the new state and undoing it pops it.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `position.h` and `position.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers
//! (see AUTHORS), under the GNU General Public License, version 3 or later.

use std::cell::Cell;
use std::fmt;
use std::ops::{Index, IndexMut};

use crate::attacks::*;
use crate::bitboard::*;
use crate::misc::{IStream, Prng, is_space};
use crate::movegen::{ALL, GenType, MoveList, Moves};
use crate::types::*;
use crate::variant::{Chess, Variant};

mod atomic;

pub use atomic::AtomicState;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const PIECE_TO_CHAR: &[u8; 16] = b" PNBRQK  pnbrqk ";

pub struct Zobrist {
    pub psq: [[Key; SQUARE_NB]; PIECE_NB],
    pub enpassant: [Key; FILE_NB],
    pub castling: [Key; CASTLING_RIGHT_NB],
    pub side: Key,
    pub no_pawns: Key,
}

/// Stockfish's Zobrist keys, from its generator and seed, so that hash keys
/// and everything that depends on them (the transposition table, the history
/// tables) behave identically.
pub static ZOBRIST: Zobrist = {
    let mut rng = Prng::new(1070372);
    let mut z = Zobrist {
        psq: [[0; SQUARE_NB]; PIECE_NB],
        enpassant: [0; FILE_NB],
        castling: [0; CASTLING_RIGHT_NB],
        side: 0,
        no_pawns: 0,
    };
    let mut i = 0;
    while i < PIECES.len() {
        let pc = PIECES[i].0 as usize;
        let mut s = 0;
        while s < SQUARE_NB {
            z.psq[pc][s] = rng.rand64();
            s += 1;
        }
        i += 1;
    }
    // Pawns on these squares will promote.
    let mut f = 0;
    while f < FILE_NB {
        z.psq[W_PAWN.0 as usize][Square::A8.0 as usize + f] = 0;
        z.psq[B_PAWN.0 as usize][f] = 0;
        f += 1;
    }
    f = 0;
    while f < FILE_NB {
        z.enpassant[f] = rng.rand64();
        f += 1;
    }
    let mut cr = 0;
    while cr < CASTLING_RIGHT_NB {
        z.castling[cr] = rng.rand64();
        cr += 1;
    }
    z.side = rng.rand64();
    z.no_pawns = rng.rand64();
    z
};

// Marcel van Kervinck's cuckoo tables, which detect that a reversible move
// could repeat a position: the keys of the reversible moves of each piece
// between two squares, and the moves themselves, hashed two ways. See
// http://web.archive.org/web/20201107002606/https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf

const fn h1(h: Key) -> usize {
    (h & 0x1fff) as usize
}

const fn h2(h: Key) -> usize {
    ((h >> 16) & 0x1fff) as usize
}

struct Cuckoo {
    keys: [Key; 8192],
    moves: [Move; 8192],
}

static CUCKOO: Cuckoo = {
    let mut c = Cuckoo {
        keys: [0; 8192],
        moves: [Move::none(); 8192],
    };
    let mut count = 0;
    let mut i = 0;
    while i < PIECES.len() {
        let pc = PIECES[i];
        let pt = pc.type_of();
        let mut s1 = 0;
        while pt.0 != PAWN.0 && s1 < SQUARE_NB {
            let mut s2 = s1 + 1;
            while s2 < SQUARE_NB {
                if PSEUDO_ATTACKS[pt.0 as usize][s1].0 & (1 << s2) != 0 {
                    let mut mv = Move::new(Square(s1 as u8), Square(s2 as u8));
                    let mut key = ZOBRIST.psq[pc.0 as usize][s1]
                        ^ ZOBRIST.psq[pc.0 as usize][s2]
                        ^ ZOBRIST.side;
                    let mut j = h1(key);
                    loop {
                        let evicted_key = c.keys[j];
                        let evicted_move = c.moves[j];
                        c.keys[j] = key;
                        c.moves[j] = mv;
                        key = evicted_key;
                        mv = evicted_move;
                        // Arrived at an empty slot?
                        if mv.0 == Move::none().0 {
                            break;
                        }
                        // Push the victim to its alternative slot.
                        j = if j == h1(key) { h2(key) } else { h1(key) };
                    }
                    count += 1;
                }
                s2 += 1;
            }
            s1 += 1;
        }
        i += 1;
    }
    assert!(count == 3668);
    c
};

/// What a search prefetches while a move is made, once the new position's
/// keys are known: Stockfish passes `do_move` its transposition table and
/// shared histories for this. `key` is the new position's key as `key()`
/// will return it, `st` its state so far, and `pc` the piece moved to `to`,
/// the king's destination when castling.
pub trait Prefetcher {
    fn prefetch<V: Variant>(&self, key: Key, st: &StateInfo<V>, pc: Piece, to: Square);
}

/// A prefetcher that prefetches nothing.
pub struct NoPrefetch;

impl Prefetcher for NoPrefetch {
    #[inline(always)]
    fn prefetch<V: Variant>(&self, _: Key, _: &StateInfo<V>, _: Piece, _: Square) {}
}

/// What a move needs to be undone, and what is worked out once per position.
/// Laid out in order, so that a move copies from the state before it only
/// the bytes before `key`, as Stockfish's do_move copies the start of its
/// StateInfo, and sets the rest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct StateInfo<V: Variant = Chess> {
    // Copied when making a move
    pub material_key: Key,
    pub pawn_key: Key,
    pub minor_piece_key: Key,
    pub non_pawn_key: [Key; COLOR_NB],
    pub non_pawn_material: [Value; COLOR_NB],
    pub rule50: i32,
    pub plies_from_null: i32,
    pub castling_rights: CastlingRights,
    pub ep_square: Square,

    // Recomputed when making a move, the first two in the bytes before `key`
    pub captured_piece: Piece,
    pub repetition: i32,
    pub key: Key,
    pub checkers_bb: Bitboard,
    pub blockers_for_king: [Bitboard; COLOR_NB],
    pub pinners: [Bitboard; COLOR_NB],
    pub check_squares: [Bitboard; PIECE_TYPE_NB],
    // Computed when first asked for, by threats_by: ALL_PIECES's is empty
    // until then
    threats: [Cell<Bitboard>; PIECE_TYPE_NB],

    // What the variant keeps besides
    pub extra: V::StateExtra,
}

impl<V: Variant> StateInfo<V> {
    /// The bytes a move copies from the state before it.
    const CARRIED: usize = std::mem::offset_of!(StateInfo<V>, key);
}

const _: () = assert!(StateInfo::<Chess>::CARRIED == 64);

/// The states of the plies since the position was set, the current one last,
/// and after them the slots of the states undone since, which stay states, so
/// that a move copies into the next slot only the fields it carries over.
#[derive(Clone)]
struct States<V: Variant> {
    slots: Vec<StateInfo<V>>,
    len: usize,
}

impl<V: Variant> States<V> {
    fn new() -> States<V> {
        States {
            slots: vec![StateInfo::default()],
            len: 1,
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    /// The states, the current one last: not the slots after it, of which a
    /// move may be writing the next.
    #[inline(always)]
    fn as_slice(&self) -> &[StateInfo<V>] {
        // SAFETY: the first `len` slots hold the states, and `as_ptr` takes
        // no reference to the others.
        unsafe { std::slice::from_raw_parts(self.slots.as_ptr(), self.len) }
    }

    #[inline(always)]
    fn last(&self) -> &StateInfo<V> {
        debug_assert!(self.len > 0);
        // SAFETY: `len` is never 0, as `pop` never removes the last state.
        unsafe { &*self.slots.as_ptr().add(self.len - 1) }
    }

    /// Removes the current state, which must not be the only one.
    #[inline(always)]
    fn pop(&mut self) {
        debug_assert!(self.len > 1);
        self.len -= 1;
    }

    /// The slot after the current state, with the current state's first
    /// `bytes` bytes copied into it.
    ///
    /// # Safety
    ///
    /// `bytes` is at most a state's size. The reference is not tied to
    /// `self`: the caller must call `push` before the states change in any
    /// other way, and not use the reference after.
    #[inline(always)]
    unsafe fn next<'a>(&mut self, bytes: usize) -> &'a mut StateInfo<V> {
        if self.len == self.slots.len() {
            self.grow();
        }
        // SAFETY: slots `len - 1` and `len` both hold states, whose fields
        // are all plain data but the threats' cells, of `Copy` values, so
        // that the slot stays a state whatever of the other's fields are
        // copied into it. No slice of the states covers slot `len`.
        unsafe {
            let base = self.slots.as_mut_ptr();
            let next = base.add(self.len);
            std::ptr::copy_nonoverlapping(
                base.add(self.len - 1).cast::<u8>(),
                next.cast::<u8>(),
                bytes,
            );
            &mut *next
        }
    }

    /// Adds a slot after the current state, set to a copy of it.
    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        let st = self.last().clone();
        self.slots.push(st);
    }

    /// Makes the slot that `next` returned the current state.
    #[inline(always)]
    fn push(&mut self) {
        debug_assert!(self.len < self.slots.len());
        self.len += 1;
    }
}

impl<V: Variant> Index<usize> for States<V> {
    type Output = StateInfo<V>;

    #[inline(always)]
    fn index(&self, i: usize) -> &StateInfo<V> {
        &self.as_slice()[i]
    }
}

impl<V: Variant> IndexMut<usize> for States<V> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut StateInfo<V> {
        let len = self.len;
        &mut self.slots[..len][i]
    }
}

/// Why a FEN was refused: invalid, or a position varifish does not support.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionSetError(pub String);

impl fmt::Display for PositionSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PositionSetError {}

fn error<T>(message: impl Into<String>) -> Result<T, PositionSetError> {
    Err(PositionSetError(message.into()))
}

/// A position of the variant `V`.
#[derive(Clone)]
pub struct Position<V: Variant = Chess> {
    board: [Piece; SQUARE_NB],
    by_type_bb: [Bitboard; PIECE_TYPE_NB],
    by_color_bb: [Bitboard; COLOR_NB],
    piece_count: [i32; PIECE_NB],
    castling_rights_mask: [CastlingRights; SQUARE_NB],
    castling_rook_square: [Square; CASTLING_RIGHT_NB],
    castling_path: [Bitboard; CASTLING_RIGHT_NB],
    // Never empty: the last state is the current one.
    states: States<V>,
    game_ply: i32,
    side_to_move: Color,
    chess960: bool,
}

impl<V: Variant> Default for Position<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Variant> Position<V> {
    /// A position without pieces, to be set from a FEN.
    fn blank() -> Position<V> {
        Position {
            board: [NO_PIECE; SQUARE_NB],
            by_type_bb: [Bitboard::EMPTY; PIECE_TYPE_NB],
            by_color_bb: [Bitboard::EMPTY; COLOR_NB],
            piece_count: [0; PIECE_NB],
            castling_rights_mask: [0; SQUARE_NB],
            castling_rook_square: [Square(0); CASTLING_RIGHT_NB],
            castling_path: [Bitboard::EMPTY; CASTLING_RIGHT_NB],
            states: States::new(),
            game_ply: 0,
            side_to_move: WHITE,
            chess960: false,
        }
    }

    /// The starting position.
    pub fn new() -> Position<V> {
        Position::from_fen(START_FEN, false).expect("the starting position is valid")
    }

    pub fn from_fen(fen: &str, is_chess960: bool) -> Result<Position<V>, PositionSetError> {
        let mut pos = Position::blank();
        pos.parse(fen, is_chess960)?;
        Ok(pos)
    }

    /// Sets the position from a FEN, validated as strictly as Stockfish does,
    /// and forgets the moves played before. On an error the position is left
    /// unchanged.
    ///
    /// A FEN has six fields separated by spaces:
    ///
    /// 1. Piece placement, from rank 8 to rank 1, each from file A to H.
    ///    Pieces are "PNBRQK" for White and "pnbrqk" for Black, digits 1 to 8
    ///    count empty squares, and "/" separates ranks.
    /// 2. The side to move, "w" or "b".
    /// 3. Castling rights: "-", or "KQkq" style letters, or the files of the
    ///    castling rooks as in Shredder-FEN and X-FEN.
    /// 4. The en passant square, recorded only if a legal en passant capture
    ///    exists.
    /// 5. The halfmove clock for the fifty-move rule.
    /// 6. The fullmove number, starting at 1.
    pub fn set(&mut self, fen: &str, is_chess960: bool) -> Result<(), PositionSetError> {
        *self = Position::from_fen(fen, is_chess960)?;
        Ok(())
    }

    /// Stockfish's `Position::set`, on an empty position.
    fn parse(&mut self, fen: &str, is_chess960: bool) -> Result<(), PositionSetError> {
        let mut st = StateInfo::default();
        let mut ss = IStream::new(fen);

        let mut num_pieces = 0;
        let mut file = FILE_A as i32;
        let mut rank = RANK_8 as i32;

        // 1. Piece placement
        loop {
            let Some(token) = ss.get() else {
                return error("Invalid FEN. Unexpected end of stream.");
            };
            if is_space(token) {
                break;
            }
            if token.is_ascii_digit() {
                let diff = (token - b'0') as i32;
                if !(1..=8).contains(&diff) {
                    return error("Invalid FEN. Invalid number of squares to skip.");
                }
                file += diff;
                if file > FILE_NB as i32 {
                    return error("Invalid FEN. Invalid file reached.");
                }
            } else if token == b'/' {
                if file != FILE_NB as i32 {
                    return error("Invalid FEN. Trying to end rank when not at the end of it.");
                }
                rank -= 1;
                file = FILE_A as i32;
                if rank < RANK_1 as i32 {
                    return error("Invalid FEN. Invalid rank reached.");
                }
            } else {
                if file >= FILE_NB as i32 {
                    return error("Invalid FEN. Invalid file reached.");
                }
                let Some(idx) = PIECE_TO_CHAR[..15].iter().position(|&c| c == token) else {
                    return error(format!("Invalid FEN. Invalid piece: {}", token as char));
                };
                num_pieces += 1;
                if num_pieces > 32 {
                    return error("Invalid FEN. More than 32 pieces on the board.");
                }
                self.put_piece(Piece(idx as u8), Square::new(file as u8, rank as u8), None);
                file += 1;
            }
        }

        if rank != RANK_1 as i32 || file != FILE_NB as i32 {
            return error("Invalid FEN. Board state encoding ended but cursor not at end.");
        }

        if (self.pieces_pt(PAWN) & (RANK_1_BB | RANK_8_BB)).any() {
            return error("Unsupported position. Pawns on the first or eighth rank.");
        }

        // In Atomic, the side to move has no king once it exploded; the side
        // to move is checked below.
        if if V::ATOMIC {
            self.count(WHITE, KING) > 1 || self.count(BLACK, KING) > 1 || self.count_both(KING) == 0
        } else {
            self.count(WHITE, KING) != 1 || self.count(BLACK, KING) != 1
        } {
            return error("Unsupported position. Incorrect number of kings.");
        }

        for c in [WHITE, BLACK] {
            let name = if c == WHITE { "WHITE" } else { "BLACK" };
            if self.count(c, PAWN) > 8 {
                return error(format!(
                    "Unsupported position. {name} has more than 8 pawns."
                ));
            }

            let additional = (self.count(c, KNIGHT) - 2).max(0)
                + (self.count(c, BISHOP) - 2).max(0)
                + (self.count(c, ROOK) - 2).max(0)
                + (self.count(c, QUEEN) - 1).max(0);
            if additional > 8 - self.count(c, PAWN) {
                return error(format!("Unsupported position. Too many pieces for {name}."));
            }
        }

        // 2. Active color
        let Some(token) = ss.get() else {
            return error("Invalid FEN. Unexpected end of stream.");
        };
        if token != b'w' && token != b'b' {
            return error(format!(
                "Invalid FEN. Invalid side to move: {}",
                token as char
            ));
        }
        self.side_to_move = if token == b'w' { WHITE } else { BLACK };
        if !ss.get().is_some_and(is_space) || ss.eof() {
            return error("Invalid FEN. Expected whitespace after side to move.");
        }

        // Only the side to move can have lost its king, to the last move.
        if V::ATOMIC && !self.has_king(!self.side_to_move) {
            return error("Unsupported position. Incorrect number of kings.");
        }

        // 3. Castling availability, as KQkq, Shredder-FEN or X-FEN. Since
        // wrong or missing castling rights are common, they are not validated
        // strictly, but rights that cannot apply are dropped.
        let mut num_castling_rights = 0;
        while let Some(mut token) = ss.get() {
            if is_space(token) {
                break;
            }
            if num_castling_rights == 0 && token == b'-' {
                ss.skip_ws();
                break;
            }

            num_castling_rights += 1;
            if num_castling_rights > 4 {
                return error("Invalid FEN. Maximum of 4 castling rights can be specified.");
            }

            let mut rsq = Square::NONE;
            let mut ksq = Square::NONE;
            let c = if token.is_ascii_lowercase() {
                BLACK
            } else {
                WHITE
            };
            let rook = make_piece(c, ROOK);
            let king = make_piece(c, KING);

            token = token.to_ascii_uppercase();

            if token == b'K' || token == b'Q' {
                // The first rook from the corner, then the king, which castling
                // rights keep between files B and G.
                let dir: Direction = if token == b'K' { -1 } else { 1 };
                let mut sq = (if token == b'K' {
                    Square::H1
                } else {
                    Square::A1
                })
                .relative(c);
                for _ in 0..7 {
                    let pc = self.piece_on(sq);
                    if pc == king {
                        ksq = sq;
                        break;
                    } else if pc == rook && rsq == Square::NONE {
                        rsq = sq;
                    }
                    sq += dir;
                }
            } else if token.is_ascii_uppercase() && token <= b'H' {
                let rsq_candidate = Square::new(token - b'A', relative_rank(c, RANK_1));
                if self.piece_on(rsq_candidate) == rook {
                    rsq = rsq_candidate;
                }
                let mut sq = Square::B1.relative(c);
                for _ in 0..6 {
                    if self.piece_on(sq) == king {
                        ksq = sq;
                    }
                    sq += EAST;
                }
            } else {
                return error(format!(
                    "Invalid FEN. Expected castling rights. Got: {}",
                    token as char
                ));
            }

            if ksq != Square::NONE && rsq != Square::NONE {
                self.set_castling_right(c, rsq, &mut st);
            }
        }

        // 4. En passant square, kept only if a pawn of the side to move
        // attacks it, an enemy pawn stands in front of it, it and the square
        // behind it are empty, and some en passant capture is legal.
        let mut enpassant = false;
        let mut legal_ep = false;
        let col = ss.get().unwrap_or(b'-');
        if col != b'-' {
            let Some(row) = ss.get() else {
                return error("Invalid FEN. Unexpected end of stream.");
            };
            let stm = self.side_to_move;
            if (b'a'..=b'h').contains(&col) && row == (if stm == WHITE { b'6' } else { b'3' }) {
                st.ep_square = Square::new(col - b'a', row - b'1');

                let mut pawns = pawn_attacks(!stm, st.ep_square) & self.pieces_cpt(stm, PAWN);
                let target = self.pieces_cpt(!stm, PAWN) & (st.ep_square + pawn_push(!stm));
                let occ = self.pieces() ^ target ^ st.ep_square;

                enpassant = pawns.any()
                    && target.any()
                    && (self.pieces() & (st.ep_square | (st.ep_square + pawn_push(stm))))
                        .is_empty();

                if V::ATOMIC {
                    legal_ep = enpassant && self.atomic_ep_is_legal(st.ep_square);
                } else {
                    while pawns.any() {
                        let from = pawns.pop_lsb();
                        legal_ep |= (self.attackers_to_occ(self.king_square(stm), occ ^ from)
                            & self.pieces_c(!stm)
                            & !target)
                            .is_empty();
                    }
                }
            } else {
                return error("Invalid FEN. Invalid en-passant square.");
            }
        }

        if !enpassant || !legal_ep {
            st.ep_square = Square::NONE;
        }

        // 5-6. Halfmove clock and fullmove number
        if let Some(rule50) = ss.read_i32() {
            st.rule50 = rule50;
        }
        if let Some(fullmove) = ss.read_i32() {
            self.game_ply = fullmove;
        }

        // Values above 99 are pointless except to ignore the 50-move rule for
        // tablebases; the limit is 2^15, as search multiplies evaluations by
        // the counter.
        if st.rule50 < 0 || st.rule50 > 32767 {
            return error("Unsupported position. Rule50 counter out of range.");
        }

        if self.game_ply < 0 || self.game_ply > 100000 {
            return error("Unsupported position. Game ply out of range.");
        }

        // From the fullmove number starting at 1 to game plies starting at 0,
        // also handling the common fullmove number 0.
        self.game_ply = (2 * (self.game_ply - 1)).max(0) + (self.side_to_move == BLACK) as i32;

        self.chess960 = is_chess960;
        self.set_state(&mut st);
        self.states[0] = st;

        if if V::ATOMIC {
            self.atomic_checkers_of(!self.side_to_move).any()
        } else {
            (self.attackers_to(self.king_square(!self.side_to_move))
                & self.pieces_c(self.side_to_move))
            .any()
        } {
            return error("Unsupported position. King can be captured.");
        }

        debug_assert_eq!(self.check(), Ok(()));
        Ok(())
    }

    /// Sets the castling right of `c` with the rook on `rfrom`.
    fn set_castling_right(&mut self, c: Color, rfrom: Square, st: &mut StateInfo<V>) {
        let kfrom = self.king_square(c);
        let cr = castling_rights_of(c, if kfrom < rfrom { KING_SIDE } else { QUEEN_SIDE });

        st.castling_rights |= cr;
        self.castling_rights_mask[kfrom.idx()] |= cr;
        self.castling_rights_mask[rfrom.idx()] |= cr;
        self.castling_rook_square[cr as usize & 15] = rfrom;

        let kto = (if cr & KING_SIDE != 0 {
            Square::G1
        } else {
            Square::C1
        })
        .relative(c);
        let rto = (if cr & KING_SIDE != 0 {
            Square::F1
        } else {
            Square::D1
        })
        .relative(c);

        self.castling_path[cr as usize & 15] =
            (between_bb(rfrom, rto) | between_bb(kfrom, kto)) & !(kfrom | rfrom);
    }

    /// Sets the pinners and the squares from which the side to move would
    /// give check. The squares the other side attacks are computed lazily by
    /// compute_threats(). Inlined, as Stockfish's set_check_info is in its
    /// do_move.
    #[inline(always)]
    fn set_extra_bitboards(&self, st: &mut StateInfo<V>) {
        if V::ATOMIC {
            self.atomic_set_extra_bitboards(st);
            return;
        }

        self.update_slider_blockers(WHITE, st);
        self.update_slider_blockers(BLACK, st);

        let them = !self.side_to_move;
        let ksq = self.king_square(them);
        let (bishop_attacks, rook_attacks) = both_attacks_bb(ksq, self.pieces());

        st.check_squares[PAWN.idx()] = pawn_attacks(them, ksq);
        st.check_squares[KNIGHT.idx()] = pseudo_attacks(KNIGHT, ksq);
        st.check_squares[BISHOP.idx()] = bishop_attacks;
        st.check_squares[ROOK.idx()] = rook_attacks;
        st.check_squares[QUEEN.idx()] = bishop_attacks | rook_attacks;
        st.check_squares[KING.idx()] = Bitboard::EMPTY;

        st.threats[ALL_PIECES.idx()].set(Bitboard::EMPTY);
    }

    /// Sets the current state's squares that the side not to move attacks,
    /// seeing through the king of the side to move, which has none in
    /// Atomic once exploded.
    #[inline(never)]
    fn compute_threats(&self) {
        let us = self.side_to_move;
        let them = !us;
        let occ = if V::ATOMIC && !self.has_king(us) {
            self.pieces()
        } else {
            self.pieces() ^ self.king_square(us)
        };

        // Spelled out rather than looped over, which compilers may leave
        // rolled, branching on each piece type.
        let pawns = self.xattacks_by(PAWN, them, occ);
        let knights = self.xattacks_by(KNIGHT, them, occ);
        let bishops = self.xattacks_by(BISHOP, them, occ);
        let rooks = self.xattacks_by(ROOK, them, occ);
        let queens = self.xattacks_by(QUEEN, them, occ);
        let king = self.xattacks_by(KING, them, occ);
        let threats = &self.st().threats;
        threats[PAWN.idx()].set(pawns);
        threats[KNIGHT.idx()].set(knights);
        threats[BISHOP.idx()].set(bishops);
        threats[ROOK.idx()].set(rooks);
        threats[QUEEN.idx()].set(queens);
        threats[KING.idx()].set(king);
        threats[ALL_PIECES.idx()].set(pawns | knights | bishops | rooks | queens | king);
    }

    /// Computes the hash keys and the other state that is then updated
    /// incrementally as moves are made, when a position is set.
    fn set_state(&self, st: &mut StateInfo<V>) {
        st.key = 0;
        st.minor_piece_key = 0;
        st.non_pawn_key = [0; COLOR_NB];
        st.pawn_key = ZOBRIST.no_pawns;
        st.non_pawn_material = [VALUE_ZERO; COLOR_NB];
        st.checkers_bb = if V::ATOMIC {
            self.atomic_checkers_of(self.side_to_move)
        } else {
            self.attackers_to(self.king_square(self.side_to_move))
                & self.pieces_c(!self.side_to_move)
        };

        self.set_extra_bitboards(st);

        for s in self.pieces() {
            let pc = self.piece_on(s);
            let psq = ZOBRIST.psq[pc.idx()][s.idx()];
            st.key ^= psq;

            if pc.type_of() == PAWN {
                st.pawn_key ^= psq;
            } else {
                st.non_pawn_key[pc.color().idx()] ^= psq;

                if pc.type_of() != KING {
                    st.non_pawn_material[pc.color().idx()] += pc.value();

                    if pc.type_of() <= BISHOP {
                        st.minor_piece_key ^= psq;
                    }
                }
            }
        }

        if st.ep_square != Square::NONE {
            st.key ^= ZOBRIST.enpassant[st.ep_square.file() as usize];
        }

        if self.side_to_move == BLACK {
            st.key ^= ZOBRIST.side;
        }

        st.key ^= ZOBRIST.castling[st.castling_rights as usize & 15];
        st.material_key = self.compute_material_key();
    }

    fn compute_material_key(&self) -> Key {
        material_key_of(&self.piece_count)
    }

    /// The FEN of the position, with Shredder-FEN castling in Chess960.
    pub fn fen(&self) -> String {
        let mut ss = String::new();

        for r in (RANK_1..=RANK_8).rev() {
            let mut f = FILE_A;
            while f <= FILE_H {
                let mut empty_cnt = 0;
                while f <= FILE_H && self.empty(Square::new(f, r)) {
                    empty_cnt += 1;
                    f += 1;
                }
                if empty_cnt > 0 {
                    ss += &empty_cnt.to_string();
                }
                if f <= FILE_H {
                    ss.push(PIECE_TO_CHAR[self.piece_on(Square::new(f, r)).idx()] as char);
                    f += 1;
                }
            }
            if r != RANK_1 {
                ss.push('/');
            }
        }

        ss += if self.side_to_move == WHITE {
            " w "
        } else {
            " b "
        };

        let rights = [
            (WHITE_OO, b'A', 'K'),
            (WHITE_OOO, b'A', 'Q'),
            (BLACK_OO, b'a', 'k'),
            (BLACK_OOO, b'a', 'q'),
        ];
        for (cr, file_char, letter) in rights {
            if self.can_castle(cr) {
                ss.push(if self.chess960 {
                    (file_char + self.castling_rook_square(cr).file()) as char
                } else {
                    letter
                });
            }
        }

        if !self.can_castle(ANY_CASTLING) {
            ss.push('-');
        }

        if self.ep_square() == Square::NONE {
            ss += " - ";
        } else {
            ss += &format!(" {} ", square_to_uci(self.ep_square()));
        }
        ss += &format!(
            "{} {}",
            self.st().rule50,
            1 + (self.game_ply - (self.side_to_move == BLACK) as i32) / 2
        );
        ss
    }

    /// Sets the pieces that keep the king of `c` out of check
    /// (`blockers_for_king[c]`) and the sliders of `!c` pinning pieces of `c`
    /// to it (`pinners[!c]`).
    fn update_slider_blockers(&self, c: Color, st: &mut StateInfo<V>) {
        let ksq = self.king_square(c);

        st.blockers_for_king[c.idx()] = Bitboard::EMPTY;
        st.pinners[(!c).idx()] = Bitboard::EMPTY;

        // Snipers are sliders that would attack the king if the pieces between
        // were removed.
        let snipers = ((pseudo_attacks(ROOK, ksq) & self.pieces_pt2(QUEEN, ROOK))
            | (pseudo_attacks(BISHOP, ksq) & self.pieces_pt2(QUEEN, BISHOP)))
            & self.pieces_c(!c);
        let occupancy = self.pieces() ^ snipers;

        for sniper_sq in snipers {
            let b = between_bb(ksq, sniper_sq) & occupancy;

            if b.any() && !b.more_than_one() {
                st.blockers_for_king[c.idx()] |= b;
                if (b & self.pieces_c(c)).any() {
                    st.pinners[(!c).idx()] |= sniper_sq;
                }
            }
        }
    }

    /// All the pieces that attack `s`, sliders seeing through the board as
    /// occupied in `occupied`.
    #[inline(always)]
    pub fn attackers_to_occ(&self, s: Square, occupied: Bitboard) -> Bitboard {
        let (bishop_attacks, rook_attacks) = both_attacks_bb(s, occupied);

        (rook_attacks & self.pieces_pt2(ROOK, QUEEN))
            | (bishop_attacks & self.pieces_pt2(BISHOP, QUEEN))
            | (pawn_attacks(BLACK, s) & self.pieces_cpt(WHITE, PAWN))
            | (pawn_attacks(WHITE, s) & self.pieces_cpt(BLACK, PAWN))
            | (pseudo_attacks(KNIGHT, s) & self.pieces_pt(KNIGHT))
            | (pseudo_attacks(KING, s) & self.pieces_pt(KING))
    }

    #[inline(always)]
    pub fn attackers_to(&self, s: Square) -> Bitboard {
        self.attackers_to_occ(s, self.pieces())
    }

    /// The attacks of the pieces of type `pt` and color `c` with the board
    /// occupied as in `occupied`, so x-raying through a king left out of it.
    #[inline(always)]
    pub fn xattacks_by(&self, pt: PieceType, c: Color, occupied: Bitboard) -> Bitboard {
        if pt == PAWN {
            pawn_attacks_bb(c, self.pieces_cpt(c, PAWN))
        } else {
            let mut threats = Bitboard::EMPTY;
            for s in self.pieces_cpt(c, pt) {
                threats |= attacks_bb(pt, s, occupied);
            }
            threats
        }
    }

    /// Tests whether a move, possibly corrupt as moves from the transposition
    /// table can be, is legal.
    #[inline(always)]
    pub fn legal(&self, m: Move) -> bool {
        V::legal(self, m)
    }

    /// Tests whether a legal move gives check.
    #[inline(always)]
    pub fn gives_check(&self, m: Move) -> bool {
        V::gives_check(self, m)
    }

    /// Plays a legal move as Stockfish's three-argument `do_move` does,
    /// working out whether it gives check; `scratch` receives what the move
    /// changes for the network, which callers without one ignore.
    pub fn make_move(&mut self, m: Move, scratch: &mut V::Dirties) {
        scratch.reset();
        let gives_check = self.gives_check(m);
        self.do_move(m, gives_check, scratch);
    }

    /// Plays a legal move, pushing a new state, and records in `dirties` what
    /// changed for the network. `dirties.dirty_threats` receives the threats
    /// after what it already holds.
    pub fn do_move(&mut self, m: Move, gives_check: bool, dirties: &mut V::Dirties) {
        self.do_move_prefetching(m, gives_check, dirties, &NoPrefetch);
    }

    /// `do_move`, calling `prefetcher` once the new keys are known.
    #[inline(always)]
    pub fn do_move_prefetching<P: Prefetcher>(
        &mut self,
        m: Move,
        gives_check: bool,
        dirties: &mut V::Dirties,
        prefetcher: &P,
    ) {
        V::do_move(self, m, gives_check, dirties, prefetcher);
    }

    /// Unmakes a move, restoring the position exactly as it was before it.
    #[inline(always)]
    pub fn undo_move(&mut self, m: Move) {
        V::undo_move(self, m);
    }

    /// Does or undoes castling, which in Chess960 can move the king and rook
    /// to each other's squares. Returns the rook's origin and destination and
    /// turns `to` from the rook's square into the king's destination.
    fn do_castling<const DO: bool>(
        &mut self,
        us: Color,
        from: Square,
        to: &mut Square,
        mut dts: Option<&mut DirtyThreats>,
        dp: Option<&mut DirtyPiece>,
    ) -> (Square, Square) {
        let king_side = *to > from;
        // Castling is encoded as the king capturing its own rook.
        let rfrom = *to;
        let rto = (if king_side { Square::F1 } else { Square::D1 }).relative(us);
        *to = (if king_side { Square::G1 } else { Square::C1 }).relative(us);

        if DO {
            let dp = dp.expect("castling records its changes");
            dp.to = *to;
            dp.remove_pc = make_piece(us, ROOK);
            dp.add_pc = make_piece(us, ROOK);
            dp.remove_sq = rfrom;
            dp.add_sq = rto;
        }

        // Remove both pieces first, since the squares can overlap in Chess960.
        self.remove_piece(if DO { from } else { *to }, dts.as_deref_mut());
        self.remove_piece(if DO { rfrom } else { rto }, dts.as_deref_mut());
        self.put_piece(
            make_piece(us, KING),
            if DO { *to } else { from },
            dts.as_deref_mut(),
        );
        self.put_piece(make_piece(us, ROOK), if DO { rto } else { rfrom }, dts);

        (rfrom, rto)
    }

    /// Passes the move to the opponent without moving a piece.
    pub fn do_null_move(&mut self) {
        debug_assert!(self.checkers().is_empty());

        // SAFETY: the state is pushed below, before `states` changes.
        let st = unsafe { self.next_state_copy() };

        if st.ep_square != Square::NONE {
            st.key ^= ZOBRIST.enpassant[st.ep_square.file() as usize];
            st.ep_square = Square::NONE;
        }

        st.key ^= ZOBRIST.side;
        st.plies_from_null = 0;
        st.captured_piece = NO_PIECE;

        self.side_to_move = !self.side_to_move;

        self.set_extra_bitboards(st);

        st.repetition = 0;
        // SAFETY: `next_state` was called above.
        unsafe { self.push_next_state() };

        debug_assert_eq!(self.check(), Ok(()));
    }

    pub fn undo_null_move(&mut self) {
        debug_assert!(self.checkers().is_empty());
        assert!(
            self.states.len() > 1,
            "undo_null_move without a null move to undo"
        );

        self.states.pop();
        self.side_to_move = !self.side_to_move;
    }

    /// The key after a move that is not a capture or a pawn move, close
    /// enough to prefetch the transposition table entry.
    pub fn prefetch_key(&self, m: Move) -> Key {
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.piece_on(from);
        let captured = self.piece_on(to);
        let mut k = self.st().key ^ ZOBRIST.side;

        k ^= ZOBRIST.psq[captured.idx()][to.idx()]
            ^ ZOBRIST.psq[pc.idx()][to.idx()]
            ^ ZOBRIST.psq[pc.idx()][from.idx()];

        if captured.is_some() || pc.type_of() == PAWN {
            return k;
        }

        self.adjust_key50::<true>(k)
    }

    /// Tests whether the static exchange evaluation of a move is at least
    /// `threshold`.
    #[inline(always)]
    pub fn see_ge(&self, m: Move, threshold: i32) -> bool {
        V::see_ge(self, m, threshold)
    }

    /// Tests whether the position is drawn by the 50-move rule or by
    /// repetition. Stalemates are not detected.
    pub fn is_draw(&self, ply: i32) -> bool {
        if self.st().rule50 > 99
            && (!self.in_check()
                || if V::ATOMIC {
                    self.atomic_has_legal_move()
                } else {
                    !MoveList::new::<ALL>(self).is_empty()
                })
        {
            return true;
        }

        self.is_repetition(ply)
    }

    /// A draw score if the position repeats once after the root, or twice at
    /// or before it.
    pub fn is_repetition(&self, ply: i32) -> bool {
        let repetition = self.st().repetition;
        repetition != 0 && repetition < ply
    }

    /// Tests whether some position repeated since the last capture or pawn
    /// move.
    pub fn has_repeated(&self) -> bool {
        let mut idx = self.states.len() - 1;
        let mut end = self.st().rule50.min(self.st().plies_from_null);
        while end >= 4 {
            if self.states[idx].repetition != 0 {
                return true;
            }
            idx -= 1;
            end -= 1;
        }
        false
    }

    /// Tests whether the side to move has a move that draws by repetition,
    /// matching `is_draw` over all legal moves.
    pub fn upcoming_repetition(&self, ply: i32) -> bool {
        let st = self.st();
        let end = st.rule50.min(st.plies_from_null);

        if end < 3 {
            return false;
        }

        let n = self.states.len() - 1;
        let original_key = st.key;
        let mut other = original_key ^ self.states[n - 1].key ^ ZOBRIST.side;

        let mut i = 3;
        while i <= end {
            let idx = n - i as usize;
            other ^= self.states[idx + 1].key ^ self.states[idx].key ^ ZOBRIST.side;

            if other == 0 {
                let stp = &self.states[idx];
                let move_key = original_key ^ stp.key;

                let mut j = h1(move_key);
                if CUCKOO.keys[j] != move_key {
                    j = h2(move_key);
                }
                if CUCKOO.keys[j] == move_key {
                    let mv = CUCKOO.moves[j];
                    let s1 = mv.from_sq();
                    let s2 = mv.to_sq();

                    if ((between_bb(s1, s2) ^ s2) & self.pieces()).is_empty() {
                        if ply > i {
                            return true;
                        }

                        // At or before the root, the move must repeat a
                        // position rather than reach the current one.
                        if stp.repetition != 0 {
                            return true;
                        }
                    }
                }
            }
            i += 2;
        }
        false
    }

    /// Swaps the colors: the same position with White and Black exchanged,
    /// for debugging, such as evaluation symmetry.
    pub fn flip(&mut self) -> Result<(), PositionSetError> {
        let fen = self.fen();
        let mut fields = fen.split(' ');
        let placement = fields.next().unwrap_or("");
        let side = fields.next().unwrap_or("w");
        let castling = fields.next().unwrap_or("-");
        let ep = fields.next().unwrap_or("-");
        let rest: Vec<&str> = fields.collect();

        let swap_case = |s: &str| -> String {
            s.chars()
                .map(|c| {
                    if c.is_ascii_lowercase() {
                        c.to_ascii_uppercase()
                    } else {
                        c.to_ascii_lowercase()
                    }
                })
                .collect()
        };

        let mut f = placement.split('/').rev().collect::<Vec<_>>().join("/");
        f += if side == "w" { " B " } else { " W " };
        f += castling;
        f += " ";
        f = swap_case(&f);

        if ep == "-" {
            f += ep;
        } else {
            let mut ep = ep.to_string();
            let row = if ep.as_bytes()[1] == b'3' { "6" } else { "3" };
            ep.replace_range(1..2, row);
            f += &ep;
        }
        for field in rest {
            f += " ";
            f += field;
        }

        let chess960 = self.chess960;
        self.set(&f, chess960)
    }

    pub fn material_key_is_ok(&self) -> bool {
        self.compute_material_key() == self.st().material_key
    }

    /// Consistency checks for debugging, Stockfish's `pos_is_ok`, naming the
    /// first check that fails.
    pub fn check(&self) -> Result<(), &'static str> {
        if V::ATOMIC {
            return self.atomic_check();
        }

        let stm = self.side_to_move;
        if (stm != WHITE && stm != BLACK)
            || self.piece_on(self.king_square(WHITE)) != W_KING
            || self.piece_on(self.king_square(BLACK)) != B_KING
            || (self.ep_square() != Square::NONE && self.ep_square().relative_rank(stm) != RANK_6)
        {
            return Err("pos_is_ok: Default");
        }

        if self.count(WHITE, KING) != 1
            || self.count(BLACK, KING) != 1
            || (self.attackers_to(self.king_square(!stm)) & self.pieces_c(stm)).any()
        {
            return Err("pos_is_ok: Kings");
        }

        if (self.pieces_pt(PAWN) & (RANK_1_BB | RANK_8_BB)).any()
            || self.count(WHITE, PAWN) > 8
            || self.count(BLACK, PAWN) > 8
        {
            return Err("pos_is_ok: Pawns");
        }

        if self.ep_square() != Square::NONE {
            let ksq = self.king_square(stm);
            let ep = self.ep_square();

            let captured = (ep + pawn_push(!stm)) & self.pieces_cpt(!stm, PAWN);
            let pawns = pawn_attacks(!stm, ep) & self.pieces_cpt(stm, PAWN);
            let potential_checkers = self.pieces_c(!stm) ^ captured;

            if captured.is_empty()
                || pawns.is_empty()
                || ((self.attackers_to_occ(ksq, self.pieces() ^ captured ^ ep ^ pawns.lsb())
                    & potential_checkers)
                    .any()
                    && (self.attackers_to_occ(ksq, self.pieces() ^ captured ^ ep ^ pawns.msb())
                        & potential_checkers)
                        .any())
            {
                return Err("pos_is_ok: En passant square");
            }
        }

        if (self.pieces_c(WHITE) & self.pieces_c(BLACK)).any()
            || (self.pieces_c(WHITE) | self.pieces_c(BLACK)) != self.pieces()
            || self.pieces_c(WHITE).popcount() > 16
            || self.pieces_c(BLACK).popcount() > 16
        {
            return Err("pos_is_ok: Bitboards");
        }

        for p1 in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] {
            for p2 in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] {
                if p1 != p2 && (self.pieces_pt(p1) & self.pieces_pt(p2)).any() {
                    return Err("pos_is_ok: Bitboards");
                }
            }
        }

        for pc in PIECES {
            let count = self.piece_count[pc.idx()];
            if count != self.pieces_cpt(pc.color(), pc.type_of()).popcount()
                || count != self.board.iter().filter(|&&p| p == pc).count() as i32
            {
                return Err("pos_is_ok: Pieces");
            }
        }

        for c in [WHITE, BLACK] {
            for cr in [
                castling_rights_of(c, KING_SIDE),
                castling_rights_of(c, QUEEN_SIDE),
            ] {
                if !self.can_castle(cr) {
                    continue;
                }

                if self.piece_on(self.castling_rook_square(cr)) != make_piece(c, ROOK)
                    || self.castling_rights_mask[self.castling_rook_square(cr).idx()] != cr
                    || (self.castling_rights_mask[self.king_square(c).idx()] & cr) != cr
                {
                    return Err("pos_is_ok: Castling");
                }
            }
        }

        if !self.material_key_is_ok() {
            return Err("pos_is_ok: materialKey");
        }

        Ok(())
    }

    #[inline(always)]
    fn st(&self) -> &StateInfo<V> {
        // `states` is never empty: it starts with one state, set() leaves
        // one, and the undo functions refuse to remove the last.
        self.states.last()
    }

    /// The slot after the current state, with the fields a move carries over
    /// copied from it, for the move to update them in place and to set the
    /// others: a new state built apart and then pushed is copied a second
    /// time, its wide loads right after the move's narrow stores to it,
    /// which processors cannot forward. The slot becomes the current state at
    /// `push_next_state`; until then `st()` is still the state before the
    /// move, as the move's code expects.
    ///
    /// # Safety
    ///
    /// The reference is not tied to `self`: the caller must call
    /// `push_next_state` before `states` changes in any other way, and not
    /// use the reference after.
    #[inline(always)]
    unsafe fn next_state<'a>(&mut self) -> &'a mut StateInfo<V> {
        // SAFETY: the bytes before `key`, within a state; the caller keeps
        // the rest of the contract.
        unsafe { self.states.next(StateInfo::<V>::CARRIED) }
    }

    /// `next_state` with the whole current state copied, for a null move,
    /// which keeps most of it.
    ///
    /// # Safety
    ///
    /// As for `next_state`.
    #[inline(always)]
    unsafe fn next_state_copy<'a>(&mut self) -> &'a mut StateInfo<V> {
        // SAFETY: a state's bytes; the caller keeps the rest of the contract.
        unsafe { self.states.next(std::mem::size_of::<StateInfo<V>>()) }
    }

    /// Makes the state that `next_state` returned the current one.
    ///
    /// # Safety
    ///
    /// `next_state` must have been called since `states` last changed.
    #[inline(always)]
    unsafe fn push_next_state(&mut self) {
        self.states.push();
    }

    /// The current state.
    #[inline(always)]
    pub fn state(&self) -> &StateInfo<V> {
        self.st()
    }

    /// The states since the position was set, the current one last.
    #[inline(always)]
    pub fn states(&self) -> &[StateInfo<V>] {
        self.states.as_slice()
    }

    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    #[inline(always)]
    pub fn piece_on(&self, s: Square) -> Piece {
        self.board[s.idx()]
    }

    #[inline(always)]
    pub fn piece_array(&self) -> &[Piece; SQUARE_NB] {
        &self.board
    }

    #[inline(always)]
    pub fn empty(&self, s: Square) -> bool {
        self.piece_on(s) == NO_PIECE
    }

    #[inline(always)]
    pub fn moved_piece(&self, m: Move) -> Piece {
        self.piece_on(m.from_sq())
    }

    #[inline(always)]
    pub fn pieces(&self) -> Bitboard {
        self.by_type_bb[ALL_PIECES.idx()]
    }

    #[inline(always)]
    pub fn pieces_pt(&self, pt: PieceType) -> Bitboard {
        self.by_type_bb[pt.idx()]
    }

    #[inline(always)]
    pub fn pieces_pt2(&self, pt1: PieceType, pt2: PieceType) -> Bitboard {
        self.by_type_bb[pt1.idx()] | self.by_type_bb[pt2.idx()]
    }

    #[inline(always)]
    pub fn pieces_c(&self, c: Color) -> Bitboard {
        self.by_color_bb[c.idx()]
    }

    #[inline(always)]
    pub fn pieces_cpt(&self, c: Color, pt: PieceType) -> Bitboard {
        self.pieces_c(c) & self.pieces_pt(pt)
    }

    #[inline(always)]
    pub fn pieces_cpt2(&self, c: Color, pt1: PieceType, pt2: PieceType) -> Bitboard {
        self.pieces_c(c) & self.pieces_pt2(pt1, pt2)
    }

    /// The number of pieces of type `pt` and color `c`, or all of `c`'s pieces
    /// for ALL_PIECES.
    #[inline(always)]
    pub fn count(&self, c: Color, pt: PieceType) -> i32 {
        self.piece_count[piece_count_index(c, pt)]
    }

    #[inline(always)]
    pub fn count_both(&self, pt: PieceType) -> i32 {
        self.count(WHITE, pt) + self.count(BLACK, pt)
    }

    /// The square of the one piece of type `pt` and color `c`.
    #[inline(always)]
    pub fn square(&self, pt: PieceType, c: Color) -> Square {
        debug_assert!(self.count(c, pt) == 1);
        self.pieces_cpt(c, pt).lsb()
    }

    #[inline(always)]
    pub fn king_square(&self, c: Color) -> Square {
        self.square(KING, c)
    }

    /// Whether `c` has a king, which in Atomic it loses when it explodes.
    #[inline(always)]
    pub fn has_king(&self, c: Color) -> bool {
        self.pieces_cpt(c, KING).any()
    }

    /// Appends the legal moves of kind `T` to `moves`.
    #[inline(always)]
    pub fn generate<const T: GenType>(&self, moves: &mut Moves) {
        V::generate::<T>(self, moves);
    }

    #[inline(always)]
    pub fn ep_square(&self) -> Square {
        self.st().ep_square
    }

    #[inline(always)]
    pub fn can_castle(&self, cr: CastlingRights) -> bool {
        self.st().castling_rights & cr != 0
    }

    #[inline(always)]
    pub fn castling_impeded(&self, cr: CastlingRights) -> bool {
        debug_assert!(matches!(cr, WHITE_OO | WHITE_OOO | BLACK_OO | BLACK_OOO));
        (self.pieces() & self.castling_path[cr as usize & 15]).any()
    }

    #[inline(always)]
    pub fn castling_rook_square(&self, cr: CastlingRights) -> Square {
        debug_assert!(matches!(cr, WHITE_OO | WHITE_OOO | BLACK_OO | BLACK_OOO));
        self.castling_rook_square[cr as usize & 15]
    }

    /// The pieces giving check to the side to move: in Atomic, none when
    /// either king is missing or the kings are adjacent.
    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        self.st().checkers_bb
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers().any()
    }

    #[inline(always)]
    pub fn blockers_for_king(&self, c: Color) -> Bitboard {
        self.st().blockers_for_king[c.idx()]
    }

    #[inline(always)]
    pub fn pinners(&self, c: Color) -> Bitboard {
        self.st().pinners[c.idx()]
    }

    #[inline(always)]
    pub fn check_squares(&self, pt: PieceType) -> Bitboard {
        self.st().check_squares[pt.idx()]
    }

    /// The squares the side not to move attacks with pieces of type `pt`,
    /// seeing through the king of the side to move.
    #[inline(always)]
    pub fn threats_by(&self, pt: PieceType) -> Bitboard {
        let threats = &self.st().threats;
        if threats[ALL_PIECES.idx()].get().is_empty() {
            self.compute_threats();
        }
        threats[pt.idx()].get()
    }

    /// The hash key, which changes as the 50-move counter grows so that
    /// positions near the 50-move rule do not share entries with fresh ones.
    #[inline(always)]
    pub fn key(&self) -> Key {
        self.adjust_key50::<false>(self.st().key)
    }

    #[inline(always)]
    fn adjust_key50<const AFTER_MOVE: bool>(&self, k: Key) -> Key {
        let threshold = 14 - AFTER_MOVE as i32;
        let rule50 = self.st().rule50;
        if rule50 < threshold {
            k
        } else {
            k ^ make_key(((rule50 - threshold) / 8) as u64)
        }
    }

    #[inline(always)]
    pub fn pawn_key(&self) -> Key {
        self.st().pawn_key
    }

    #[inline(always)]
    pub fn material_key(&self) -> Key {
        self.st().material_key
    }

    #[inline(always)]
    pub fn minor_piece_key(&self) -> Key {
        self.st().minor_piece_key
    }

    #[inline(always)]
    pub fn non_pawn_key(&self, c: Color) -> Key {
        self.st().non_pawn_key[c.idx()]
    }

    #[inline(always)]
    pub fn non_pawn_material(&self, c: Color) -> Value {
        self.st().non_pawn_material[c.idx()]
    }

    #[inline(always)]
    pub fn non_pawn_material_total(&self) -> Value {
        self.non_pawn_material(WHITE) + self.non_pawn_material(BLACK)
    }

    #[inline(always)]
    pub fn game_ply(&self) -> i32 {
        self.game_ply
    }

    #[inline(always)]
    pub fn rule50_count(&self) -> i32 {
        self.st().rule50
    }

    #[inline(always)]
    pub fn is_chess960(&self) -> bool {
        self.chess960
    }

    /// Pawnless with three pieces, or four with neither queens nor rooks.
    #[inline(always)]
    pub fn dtz_is_dtm(&self) -> bool {
        let all = self.count_both(ALL_PIECES);
        self.count_both(PAWN) == 0
            && (all == 3 || (all == 4 && self.pieces_pt2(QUEEN, ROOK).is_empty()))
    }

    #[inline(always)]
    pub fn capture(&self, m: Move) -> bool {
        debug_assert!(m.is_ok());

        let mt = m.type_of();
        if mt == NORMAL || mt == PROMOTION {
            return !self.empty(m.to_sq());
        }

        mt == EN_PASSANT
    }

    /// Whether the capture stage of move generation generates the move: the
    /// captures and the queen promotions.
    #[inline(always)]
    pub fn capture_stage(&self, m: Move) -> bool {
        debug_assert!(m.is_ok());

        let mt = m.type_of();
        if mt == NORMAL {
            return !self.empty(m.to_sq());
        }

        if mt == PROMOTION {
            return !self.empty(m.to_sq()) || m.promotion_type() == QUEEN;
        }

        mt == EN_PASSANT
    }

    #[inline(always)]
    pub fn captured_piece(&self) -> Piece {
        self.st().captured_piece
    }

    #[inline(always)]
    fn put_piece(&mut self, pc: Piece, s: Square, dts: Option<&mut DirtyThreats>) {
        self.board[s.idx()] = pc;
        self.by_type_bb[ALL_PIECES.idx()] |= s;
        self.by_type_bb[pc.type_of().idx()] |= s;
        self.by_color_bb[pc.color().idx()] |= s;
        self.piece_count[pc.idx()] += 1;
        self.piece_count[piece_count_index(pc.color(), ALL_PIECES)] += 1;

        if let Some(dts) = dts {
            self.update_piece_threats::<true>(pc, true, s, dts, Bitboard::ALL);
        }
    }

    #[inline(always)]
    fn remove_piece(&mut self, s: Square, dts: Option<&mut DirtyThreats>) {
        let pc = self.board[s.idx()];

        if let Some(dts) = dts {
            self.update_piece_threats::<true>(pc, false, s, dts, Bitboard::ALL);
        }

        self.by_type_bb[ALL_PIECES.idx()] ^= s;
        self.by_type_bb[pc.type_of().idx()] ^= s;
        self.by_color_bb[pc.color().idx()] ^= s;
        self.board[s.idx()] = NO_PIECE;
        self.piece_count[pc.idx()] -= 1;
        self.piece_count[piece_count_index(pc.color(), ALL_PIECES)] -= 1;
    }

    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square, mut dts: Option<&mut DirtyThreats>) {
        let pc = self.board[from.idx()];
        let from_to = from | to;

        if let Some(dts) = dts.as_deref_mut() {
            self.update_piece_threats::<true>(pc, false, from, dts, from_to);
        }

        self.by_type_bb[ALL_PIECES.idx()] ^= from_to;
        self.by_type_bb[pc.type_of().idx()] ^= from_to;
        self.by_color_bb[pc.color().idx()] ^= from_to;
        self.board[from.idx()] = NO_PIECE;
        self.board[to.idx()] = pc;

        if let Some(dts) = dts {
            self.update_piece_threats::<true>(pc, true, to, dts, from_to);
        }
    }

    #[inline(always)]
    fn swap_piece(&mut self, s: Square, pc: Piece, mut dts: Option<&mut DirtyThreats>) {
        let old = self.board[s.idx()];

        self.remove_piece(s, None);

        if let Some(dts) = dts.as_deref_mut() {
            self.update_piece_threats::<false>(old, false, s, dts, Bitboard::ALL);
        }

        self.put_piece(pc, s, None);

        if let Some(dts) = dts {
            self.update_piece_threats::<false>(pc, true, s, dts, Bitboard::ALL);
        }
    }

    /// Records the threats that putting (or removing) `pc` on `s` adds (or
    /// removes): those it makes, those made on it, and, with COMPUTE_RAY,
    /// those of sliders whose line through `s` it opens or closes, except
    /// along a line that holds both squares of `no_rays_containing`. Out of
    /// line, as Stockfish's is, since each move calls it several times.
    #[inline(never)]
    fn update_piece_threats<const COMPUTE_RAY: bool>(
        &self,
        pc: Piece,
        put_piece: bool,
        s: Square,
        dts: &mut DirtyThreats,
        no_rays_containing: Bitboard,
    ) {
        let occupied = self.pieces();
        let (b_attacks, r_attacks) = both_attacks_bb(s, occupied);
        let slider_attacks = b_attacks | r_attacks;
        let occupied_no_k = occupied ^ self.pieces_pt(KING);
        let pt = pc.type_of();
        let sliders = (self.pieces_pt2(BISHOP, QUEEN) & b_attacks)
            | (self.pieces_pt2(ROOK, QUEEN) & r_attacks);

        // Kings make no threats.
        if pt == KING {
            if COMPUTE_RAY {
                self.process_sliders(
                    pc,
                    put_piece,
                    s,
                    sliders,
                    slider_attacks,
                    occupied_no_k,
                    no_rays_containing,
                    false,
                    dts,
                );
            }
            return;
        }

        let threat_targets = if pt == PAWN {
            self.pieces_pt2(KNIGHT, ROOK)
        } else if pt == BISHOP || pt == ROOK {
            self.pieces_pt2(PAWN, KNIGHT) | self.pieces_pt2(BISHOP, ROOK)
        } else {
            occupied_no_k
        };
        let threatened = (if pt == BISHOP {
            b_attacks
        } else if pt == ROOK {
            r_attacks
        } else if pt == QUEEN {
            slider_attacks
        } else if pt == PAWN {
            pawn_attacks(pc.color(), s)
        } else {
            pseudo_attacks(pt, s)
        }) & threat_targets;
        let mut incoming_threats = pseudo_attacks(KNIGHT, s) & self.pieces_pt(KNIGHT);

        if pt == KNIGHT || pt == ROOK {
            incoming_threats |= (pawn_attacks(WHITE, s) & self.pieces_cpt(BLACK, PAWN))
                | (pawn_attacks(BLACK, s) & self.pieces_cpt(WHITE, PAWN));
        }

        // With Ice Lake's AVX-512, as in Stockfish's avx512icl builds, the
        // threats of `pc` in one vector store, and those on it, the sliders'
        // among them, in another.
        #[cfg(use_avx512icl)]
        {
            self.write_threats::<true>(pc, put_piece, s, threatened, dts);

            let direct_sliders = if pt == QUEEN {
                sliders & self.pieces_pt(QUEEN)
            } else {
                sliders
            };
            self.write_threats::<false>(pc, put_piece, s, direct_sliders | incoming_threats, dts);

            if COMPUTE_RAY {
                self.process_sliders(
                    pc,
                    put_piece,
                    s,
                    sliders,
                    slider_attacks,
                    occupied_no_k,
                    no_rays_containing,
                    false,
                    dts,
                );
            }
        }

        #[cfg(not(use_avx512icl))]
        {
            let mut threatened = threatened;
            while threatened.any() {
                let threatened_sq = threatened.pop_lsb();
                let threatened_pc = self.piece_on(threatened_sq);
                debug_assert!(threatened_sq != s);
                debug_assert!(threatened_pc != NO_PIECE);

                dts.push(DirtyThreat::new(
                    pc,
                    threatened_pc,
                    s,
                    threatened_sq,
                    put_piece,
                ));
            }

            if COMPUTE_RAY {
                self.process_sliders(
                    pc,
                    put_piece,
                    s,
                    sliders,
                    slider_attacks,
                    occupied_no_k,
                    no_rays_containing,
                    true,
                    dts,
                );
            } else {
                incoming_threats |= if pt == QUEEN {
                    sliders & self.pieces_pt(QUEEN)
                } else {
                    sliders
                };
            }

            while incoming_threats.any() {
                let src_sq = incoming_threats.pop_lsb();
                let src_pc = self.piece_on(src_sq);
                debug_assert!(src_sq != s);
                debug_assert!(src_pc != NO_PIECE);

                dts.push(DirtyThreat::new(src_pc, pc, src_sq, s, put_piece));
            }
        }
    }

    /// Records in one vector store the threats between `pc`, put on `s` or
    /// removed from it, and the pieces on the squares of `mask`: those that
    /// `pc` makes on them if OUTGOING, else those they make on `pc`.
    /// Stockfish's `write_multiple_dirties`.
    #[cfg(use_avx512icl)]
    #[inline(always)]
    fn write_threats<const OUTGOING: bool>(
        &self,
        pc: Piece,
        put_piece: bool,
        s: Square,
        mask: Bitboard,
        dts: &mut DirtyThreats,
    ) {
        use std::arch::x86_64::*;
        // A piece attacks at most 8 pieces, and at most 16 attack it: 8
        // knights and 8 sliders, or pawns in their place.
        debug_assert!(mask.popcount() <= 16);
        debug_assert!(!mask.has(s) && (mask & !self.pieces()).is_empty());
        let count = mask.popcount() as usize;
        let template = if OUTGOING {
            DirtyThreat::new(pc, NO_PIECE, s, Square(0), put_piece)
        } else {
            DirtyThreat::new(NO_PIECE, pc, Square(0), s, put_piece)
        };
        // SAFETY: the build targets Ice Lake's AVX-512, and the store writes
        // 16 threats, the first `count` those of `mask`.
        unsafe {
            let board = _mm512_loadu_si512(self.board.as_ptr().cast());
            // The squares, then their pieces, in the low byte of each 32 bits
            let squares = _mm512_cvtepi8_epi32(_mm512_castsi512_si128(compress_squares(mask)));
            let pieces = _mm512_maskz_permutexvar_epi8(0x1111_1111_1111_1111, squares, board);
            // Each in its place in the threat
            let (squares, pieces) = if OUTGOING {
                (
                    _mm512_slli_epi32::<{ DirtyThreat::THREATENED_SQ_OFFSET }>(squares),
                    _mm512_slli_epi32::<{ DirtyThreat::THREATENED_PC_OFFSET }>(pieces),
                )
            } else {
                (
                    _mm512_slli_epi32::<{ DirtyThreat::PC_SQ_OFFSET }>(squares),
                    _mm512_slli_epi32::<{ DirtyThreat::PC_OFFSET }>(pieces),
                )
            };
            let threats = _mm512_ternarylogic_epi32::<0xfe>(
                _mm512_set1_epi32(template.raw() as i32),
                squares,
                pieces,
            );
            dts.extend_by_store(16, count, |p| _mm512_storeu_si512(p.cast(), threats));
        }
    }

    /// The sliders' part of `update_piece_threats`: the threat each slider
    /// aiming at `s` makes on `pc` (with `add_direct_attacks`) and the one it
    /// makes on the next piece beyond `s`, which `pc` hides or reveals.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn process_sliders(
        &self,
        pc: Piece,
        put_piece: bool,
        s: Square,
        sliders: Bitboard,
        slider_attacks: Bitboard,
        occupied_no_k: Bitboard,
        no_rays_containing: Bitboard,
        add_direct_attacks: bool,
        dts: &mut DirtyThreats,
    ) {
        for slider_sq in sliders {
            let slider = self.piece_on(slider_sq);

            let ray = ray_pass_bb(slider_sq, s);
            let discovered = ray & slider_attacks & occupied_no_k;

            debug_assert!(!discovered.more_than_one());
            if discovered.any() && (ray & no_rays_containing) != no_rays_containing {
                let threatened_sq = discovered.lsb();
                let threatened_pc = self.piece_on(threatened_sq);
                if can_slider_threat(threatened_pc, slider) {
                    dts.push(DirtyThreat::new(
                        slider,
                        threatened_pc,
                        slider_sq,
                        threatened_sq,
                        !put_piece,
                    ));
                }
            }

            if add_direct_attacks && can_slider_threat(pc, slider) {
                dts.push(DirtyThreat::new(slider, pc, slider_sq, s, put_piece));
            }
        }
    }
}

/// The rules of chess, which the `Variant` implementation of `Chess` calls.
impl Position<Chess> {
    /// Tests whether a move, possibly corrupt as moves from the transposition
    /// table can be, is legal.
    pub(crate) fn chess_legal(&self, m: Move) -> bool {
        let us = self.side_to_move;
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.moved_piece(m);

        // Uncommon cases use the slower but simpler test.
        if m.type_of() != NORMAL {
            return MoveList::new::<ALL>(self).contains(m);
        }

        // Not a promotion, so the promotion bits must be empty.
        debug_assert!(m.promotion_type() == KNIGHT);

        // The moved piece must belong to the side to move...
        if pc == NO_PIECE || pc.color() != us {
            return false;
        }

        // ...and the destination cannot hold a piece of its own.
        if self.pieces_c(us).has(to) {
            return false;
        }

        if pc.type_of() == PAWN {
            // Promotions were handled above, so the pawn cannot reach the last
            // rank.
            if (RANK_8_BB | RANK_1_BB).has(to) {
                return false;
            }

            let is_capture = (pawn_attacks(us, from) & self.pieces_c(!us)).has(to);
            let is_single_push = from + pawn_push(us) == to && self.empty(to);
            let is_double_push = from + 2 * pawn_push(us) == to
                && from.relative_rank(us) == RANK_2
                && self.empty(to)
                && self.empty(to - pawn_push(us));

            if !(is_capture || is_single_push || is_double_push) {
                return false;
            }
        } else if !attacks_bb(pc.type_of(), from, self.pieces()).has(to) {
            return false;
        }

        if pc.type_of() != KING {
            if self.checkers().any() {
                // In double check, only a king move can evade.
                if self.checkers().more_than_one() {
                    return false;
                }

                // The move must block the check or capture the checker.
                if !between_bb(self.king_square(us), self.checkers().lsb()).has(to) {
                    return false;
                }
            }
            // Out of check, a move other than the king's is legal if and only
            // if the piece is not pinned or moves along the line of its king.
            !self.blockers_for_king(us).has(from)
                || (line_bb(from, to) & self.pieces_cpt(us, KING)).any()
        } else {
            // The king must not move to a square the opponent attacks.
            !self.threats_by(ALL_PIECES).has(to)
        }
    }

    /// Tests whether a legal move gives check.
    pub(crate) fn chess_gives_check(&self, m: Move) -> bool {
        debug_assert!(m.is_ok());
        debug_assert!(self.moved_piece(m).color() == self.side_to_move);

        let from = m.from_sq();
        let to = m.to_sq();
        let us = self.side_to_move;

        // Direct check?
        if self.check_squares(self.piece_on(from).type_of()).has(to) {
            return true;
        }

        // Discovered check?
        if self.blockers_for_king(!us).has(from) {
            return (line_bb(from, to) & self.pieces_cpt(!us, KING)).is_empty()
                || m.type_of() == CASTLING;
        }

        // The other kinds of move apart, so that this, the common path,
        // saves no registers.
        m.type_of() != NORMAL && self.special_gives_check(m)
    }

    /// The rest of `chess_gives_check`, for a promotion, an en passant
    /// capture or castling that gives neither a direct nor an ordinary
    /// discovered check.
    #[cold]
    #[inline(never)]
    fn special_gives_check(&self, m: Move) -> bool {
        let from = m.from_sq();
        let to = m.to_sq();
        let us = self.side_to_move;

        match m.type_of() {
            PROMOTION => (attacks_bb(m.promotion_type(), to, self.pieces() ^ from)
                & self.pieces_cpt(!us, KING))
            .any(),
            // Direct and ordinary discovered checks were handled above, which
            // leaves a discovered check through the captured pawn.
            EN_PASSANT => {
                let capsq = Square::new(to.file(), from.rank());
                let b = (self.pieces() ^ from ^ capsq) | to;
                let (bishop_attacks, rook_attacks) = both_attacks_bb(self.king_square(!us), b);

                ((rook_attacks & self.pieces_cpt2(us, QUEEN, ROOK))
                    | (bishop_attacks & self.pieces_cpt2(us, QUEEN, BISHOP)))
                .any()
            }
            // Castling, encoded as the king capturing the rook
            _ => {
                let rto = (if to > from { Square::F1 } else { Square::D1 }).relative(us);
                self.check_squares(ROOK).has(rto)
            }
        }
    }

    /// Plays a legal move, calling `prefetcher` once the new keys are known.
    pub(crate) fn chess_do_move<P: Prefetcher>(
        &mut self,
        m: Move,
        gives_check: bool,
        dirties: &mut Dirties,
        prefetcher: &P,
    ) {
        debug_assert!(m.is_ok());

        let mut k = self.st().key ^ ZOBRIST.side;
        let previous_blockers = self.st().blockers_for_king;
        // SAFETY: the state is pushed below, before `states` changes.
        let st = unsafe { self.next_state() };

        // The ply counters; rule50 is reset below for captures and pawn moves.
        self.game_ply += 1;
        st.rule50 += 1;
        st.plies_from_null += 1;

        let Dirties {
            dirty_piece: dp,
            dirty_threats: dts,
            dirty_pawn_pairs: dpps,
        } = dirties;

        dpps.before = [self.pieces_cpt(WHITE, PAWN), self.pieces_cpt(BLACK, PAWN)];

        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let mut to = m.to_sq();
        let pc = self.piece_on(from);
        let mut captured = if m.type_of() == EN_PASSANT {
            make_piece(them, PAWN)
        } else {
            self.piece_on(to)
        };

        dp.pc = pc;
        dp.from = from;
        dp.to = to;
        dp.add_sq = Square::NONE;

        debug_assert!(pc.color() == us);
        debug_assert!(
            captured == NO_PIECE
                || captured.color() == (if m.type_of() != CASTLING { them } else { us })
        );
        debug_assert!(captured.type_of() != KING);

        if m.type_of() == CASTLING {
            debug_assert!(pc == make_piece(us, KING));
            debug_assert!(captured == make_piece(us, ROOK));

            let (rfrom, rto) = self.do_castling::<true>(us, from, &mut to, Some(dts), Some(dp));

            let rook_keys =
                ZOBRIST.psq[captured.idx()][rfrom.idx()] ^ ZOBRIST.psq[captured.idx()][rto.idx()];
            k ^= rook_keys;
            st.non_pawn_key[us.idx()] ^= rook_keys;
            captured = NO_PIECE;
        } else if captured.is_some() {
            let mut capsq = to;

            // A captured pawn changes the pawn key, any other piece the
            // non-pawn material.
            if captured.type_of() == PAWN {
                if m.type_of() == EN_PASSANT {
                    capsq -= pawn_push(us);

                    debug_assert!(pc == make_piece(us, PAWN));
                    debug_assert!(to == st.ep_square);
                    debug_assert!(to.relative_rank(us) == RANK_6);
                    debug_assert!(self.piece_on(to) == NO_PIECE);
                    debug_assert!(self.piece_on(capsq) == make_piece(them, PAWN));

                    // En passant updates the board here, other captures below.
                    self.remove_piece(capsq, Some(dts));
                }

                st.pawn_key ^= ZOBRIST.psq[captured.idx()][capsq.idx()];
            } else {
                st.non_pawn_material[them.idx()] -= captured.value();
                st.non_pawn_key[them.idx()] ^= ZOBRIST.psq[captured.idx()][capsq.idx()];

                if captured.type_of() <= BISHOP {
                    st.minor_piece_key ^= ZOBRIST.psq[captured.idx()][capsq.idx()];
                }
            }

            dp.remove_pc = captured;
            dp.remove_sq = capsq;

            k ^= ZOBRIST.psq[captured.idx()][capsq.idx()];
            let remaining = self.piece_count[captured.idx()] - (m.type_of() != EN_PASSANT) as i32;
            st.material_key ^= ZOBRIST.psq[captured.idx()][(8 + remaining) as usize & 63];

            st.rule50 = 0;
        } else {
            dp.remove_sq = Square::NONE;
        }

        k ^= ZOBRIST.psq[pc.idx()][from.idx()] ^ ZOBRIST.psq[pc.idx()][to.idx()];

        if st.ep_square != Square::NONE {
            k ^= ZOBRIST.enpassant[st.ep_square.file() as usize];
            st.ep_square = Square::NONE;
        }

        k ^= ZOBRIST.castling[st.castling_rights as usize & 15];
        st.castling_rights &=
            !(self.castling_rights_mask[from.idx()] | self.castling_rights_mask[to.idx()]);
        k ^= ZOBRIST.castling[st.castling_rights as usize & 15];

        if pc.type_of() == PAWN {
            // A double push sets the en passant square only when a legal en
            // passant capture follows, which the key and repetitions need.
            if (to.0 ^ from.0) == 16 {
                let ep_square = to - pawn_push(us);
                let pawns = pawn_attacks(us, ep_square) & self.pieces_cpt(them, PAWN);

                if pawns.any() {
                    let ksq = self.king_square(them);
                    let not_blockers = !previous_blockers[them.idx()];
                    let no_discovery = (from & not_blockers).any() || from.file() == ksq.file();

                    // A pawn that discovers check never allows en passant.
                    // Otherwise a pawn that was not blocking a check on its
                    // king, or that stays on the line through the king, can
                    // take.
                    if no_discovery && (pawns & (not_blockers | line_bb(ep_square, ksq))).any() {
                        st.ep_square = ep_square;
                        k ^= ZOBRIST.enpassant[ep_square.file() as usize];
                    }
                }
            } else if m.type_of() == PROMOTION {
                let pt = m.promotion_type();
                let promotion = make_piece(us, pt);

                debug_assert!(to.relative_rank(us) == RANK_8);
                debug_assert!(pt >= KNIGHT && pt <= QUEEN);

                dp.add_pc = promotion;
                dp.add_sq = to;
                dp.to = Square::NONE;

                // The pawn's key on the last rank is zero, so there is no need
                // to clear it.
                k ^= ZOBRIST.psq[promotion.idx()][to.idx()];
                st.material_key ^= ZOBRIST.psq[promotion.idx()]
                    [(8 + self.piece_count[promotion.idx()]) as usize & 63]
                    ^ ZOBRIST.psq[pc.idx()][(8 + self.piece_count[pc.idx()] - 1) as usize & 63];
                st.non_pawn_key[us.idx()] ^= ZOBRIST.psq[promotion.idx()][to.idx()];

                if pt <= BISHOP {
                    st.minor_piece_key ^= ZOBRIST.psq[promotion.idx()][to.idx()];
                }

                st.non_pawn_material[us.idx()] += promotion.value();
            }

            st.pawn_key ^= ZOBRIST.psq[pc.idx()][from.idx()] ^ ZOBRIST.psq[pc.idx()][to.idx()];

            st.rule50 = 0;
        } else {
            let keys = ZOBRIST.psq[pc.idx()][from.idx()] ^ ZOBRIST.psq[pc.idx()][to.idx()];
            st.non_pawn_key[us.idx()] ^= keys;

            if pc.type_of() <= BISHOP {
                st.minor_piece_key ^= keys;
            }
        }

        let key = if st.rule50 < 14 {
            k
        } else {
            k ^ make_key(((st.rule50 - 14) / 8) as u64)
        };
        prefetcher.prefetch(key, st, pc, to);

        st.key = k;

        // Move the piece; castling was handled above.
        if m.type_of() != CASTLING {
            let to_pc = if m.type_of() == PROMOTION {
                make_piece(us, m.promotion_type())
            } else {
                pc
            };

            if captured.is_some() && m.type_of() != EN_PASSANT {
                self.remove_piece(from, Some(dts));
                self.swap_piece(to, to_pc, Some(dts));
            } else if pc == to_pc {
                self.move_piece(from, to, Some(dts));
            } else {
                self.remove_piece(from, Some(dts));
                self.put_piece(to_pc, to, Some(dts));
            }
        }

        st.captured_piece = captured;

        st.checkers_bb = if gives_check {
            self.attackers_to(self.king_square(them)) & self.pieces_c(us)
        } else {
            Bitboard::EMPTY
        };

        self.side_to_move = !self.side_to_move;

        self.set_extra_bitboards(st);

        // The distance in plies to the previous occurrence of the same
        // position, negative for a third occurrence, or zero if it did not
        // occur since the last irreversible move.
        st.repetition = 0;
        let end = st.rule50.min(st.plies_from_null);
        if end >= 4 {
            let n = self.states.len();
            let mut i = 4;
            while i <= end {
                let stp = &self.states[n - i as usize];
                if stp.key == st.key {
                    st.repetition = if stp.repetition != 0 { -i } else { i };
                    break;
                }
                i += 2;
            }
        }

        // SAFETY: `next_state` was called above.
        unsafe { self.push_next_state() };

        debug_assert_eq!(self.check(), Ok(()));

        dpps.after = [self.pieces_cpt(WHITE, PAWN), self.pieces_cpt(BLACK, PAWN)];

        debug_assert!(dp.pc != NO_PIECE);
        debug_assert!(
            (captured.is_some() || m.type_of() == CASTLING) == (dp.remove_sq != Square::NONE)
        );
        debug_assert!(dp.from != Square::NONE);
        debug_assert!(
            (dp.add_sq != Square::NONE) == (m.type_of() == PROMOTION || m.type_of() == CASTLING)
        );
    }

    /// Unmakes a move, restoring the position exactly as it was before it.
    pub(crate) fn chess_undo_move(&mut self, m: Move) {
        debug_assert!(m.is_ok());
        assert!(self.states.len() > 1, "undo_move without a move to undo");

        self.side_to_move = !self.side_to_move;

        let us = self.side_to_move;
        let from = m.from_sq();
        let mut to = m.to_sq();
        let mut pc = self.piece_on(to);

        debug_assert!(self.empty(from) || m.type_of() == CASTLING);
        debug_assert!(self.st().captured_piece.type_of() != KING);

        if m.type_of() == PROMOTION {
            debug_assert!(to.relative_rank(us) == RANK_8);
            debug_assert!(pc.type_of() == m.promotion_type());
            debug_assert!(pc.type_of() >= KNIGHT && pc.type_of() <= QUEEN);

            pc = make_piece(us, PAWN);
            self.swap_piece(to, pc, None);
        }

        if m.type_of() == CASTLING {
            self.do_castling::<false>(us, from, &mut to, None, None);
        } else {
            // Put the piece back on its square.
            self.move_piece(to, from, None);

            let captured = self.st().captured_piece;
            if captured.is_some() {
                let mut capsq = to;

                if m.type_of() == EN_PASSANT {
                    capsq -= pawn_push(us);

                    debug_assert!(pc.type_of() == PAWN);
                    debug_assert!(to == self.states[self.states.len() - 2].ep_square);
                    debug_assert!(to.relative_rank(us) == RANK_6);
                    debug_assert!(self.piece_on(capsq) == NO_PIECE);
                    debug_assert!(captured == make_piece(!us, PAWN));
                }

                // Restore the captured piece.
                self.put_piece(captured, capsq, None);
            }
        }

        // Back to the previous state.
        self.states.pop();
        self.game_ply -= 1;

        debug_assert_eq!(self.check(), Ok(()));
    }

    /// Tests whether the static exchange evaluation of a move is at least
    /// `threshold`, by an algorithm like alpha-beta with a null window.
    pub(crate) fn chess_see_ge(&self, m: Move, threshold: i32) -> bool {
        debug_assert!(m.is_ok());

        // Only normal moves are evaluated; the others pass a simple SEE.
        if m.type_of() != NORMAL {
            return VALUE_ZERO >= threshold;
        }

        let from = m.from_sq();
        let to = m.to_sq();

        debug_assert!(self.piece_on(from) != NO_PIECE);

        let mut swap = self.piece_on(to).value() - threshold;
        if swap < 0 {
            return false;
        }

        swap = self.piece_on(from).value() - swap;
        if swap <= 0 {
            return true;
        }

        debug_assert!(self.piece_on(from).color() == self.side_to_move);
        // Removing `to` matters for the pinned-piece logic.
        let mut occupied = self.pieces() ^ from ^ to;
        let mut stm = self.side_to_move;
        let mut attackers = self.attackers_to_occ(to, occupied);
        let mut res = 1;

        loop {
            stm = !stm;
            attackers &= occupied;

            // The side to move loses when it has no more attackers.
            let mut stm_attackers = attackers & self.pieces_c(stm);
            if stm_attackers.is_empty() {
                break;
            }

            // Pinned pieces cannot attack while their pinners stand.
            if (self.pinners(!stm) & occupied).any() {
                stm_attackers &= !self.blockers_for_king(stm);

                if stm_attackers.is_empty() {
                    break;
                }
            }

            res ^= 1;

            // Remove the least valuable attacker, and add the x-ray attackers
            // behind it.
            let bb = stm_attackers & self.pieces_pt(PAWN);
            if bb.any() {
                swap = PAWN_VALUE - swap;
                if swap < res {
                    break;
                }
                occupied ^= bb.lsb_bb();

                attackers |= bishop_attacks(to, occupied) & self.pieces_pt2(BISHOP, QUEEN);
                continue;
            }

            let bb = stm_attackers & self.pieces_pt(KNIGHT);
            if bb.any() {
                swap = KNIGHT_VALUE - swap;
                if swap < res {
                    break;
                }
                occupied ^= bb.lsb_bb();
                continue;
            }

            let bb = stm_attackers & self.pieces_pt(BISHOP);
            if bb.any() {
                swap = BISHOP_VALUE - swap;
                if swap < res {
                    break;
                }
                occupied ^= bb.lsb_bb();

                attackers |= bishop_attacks(to, occupied) & self.pieces_pt2(BISHOP, QUEEN);
                continue;
            }

            let bb = stm_attackers & self.pieces_pt(ROOK);
            if bb.any() {
                swap = ROOK_VALUE - swap;
                if swap < res {
                    break;
                }
                occupied ^= bb.lsb_bb();

                attackers |= rook_attacks(to, occupied) & self.pieces_pt2(ROOK, QUEEN);
                continue;
            }

            let bb = stm_attackers & self.pieces_pt(QUEEN);
            if bb.any() {
                swap = QUEEN_VALUE - swap;
                // The previous capture was by a piece worth more than a queen,
                // the king aside.
                debug_assert!(swap >= res);
                occupied ^= bb.lsb_bb();

                let (bishop_attacks, rook_attacks) = both_attacks_bb(to, occupied);
                attackers |= (bishop_attacks & self.pieces_pt2(BISHOP, QUEEN))
                    | (rook_attacks & self.pieces_pt2(ROOK, QUEEN));
                continue;
            }

            // The king: capturing with it while the opponent still has
            // attackers reverses the result.
            return if (attackers & !self.pieces_c(stm)).any() {
                res ^ 1 != 0
            } else {
                res != 0
            };
        }

        res != 0
    }
}

/// Bishops and rooks are not counted as threatening queens.
#[inline(always)]
const fn can_slider_threat(pc: Piece, slider: Piece) -> bool {
    pc.type_of().0 != QUEEN.0 || slider.type_of().0 == QUEEN.0
}

/// The material key of the positions with `piece_count[pc]` pieces `pc`, by
/// which the tablebases find their tables.
pub fn material_key_of(piece_count: &[i32; PIECE_NB]) -> Key {
    let mut k = 0;
    for pc in PIECES {
        for cnt in 0..piece_count[pc.idx()] as usize {
            k ^= ZOBRIST.psq[pc.idx()][8 + cnt];
        }
    }
    k
}

/// A square in UCI notation, such as "e4".
pub fn square_to_uci(s: Square) -> String {
    format!("{}{}", (b'a' + s.file()) as char, (b'1' + s.rank()) as char)
}

/// The board as Stockfish's `d` command prints it, with the FEN, the key and
/// the checkers.
impl<V: Variant> fmt::Display for Position<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const LINE: &str = " +---+---+---+---+---+---+---+---+";
        write!(f, "\n{LINE}\n")?;

        for r in (RANK_1..=RANK_8).rev() {
            for file in FILE_A..=FILE_H {
                write!(
                    f,
                    " | {}",
                    PIECE_TO_CHAR[self.piece_on(Square::new(file, r)).idx()] as char
                )?;
            }
            write!(f, " | {}\n{LINE}\n", 1 + r)?;
        }

        writeln!(f, "   a   b   c   d   e   f   g   h")?;
        write!(
            f,
            "\nFen: {}\nKey: {:016X}\nCheckers: ",
            self.fen(),
            self.key()
        )?;

        for s in self.checkers() {
            write!(f, "{} ", square_to_uci(s))?;
        }
        Ok(())
    }
}
