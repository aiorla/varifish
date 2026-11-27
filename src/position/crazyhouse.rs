//! Crazyhouse's state and rules (Lichess rules): each side's pocket of
//! captured pieces, the bitboard of promoted pieces (which demote back to
//! pawns when captured), Zobrist keys for pockets and promoted squares, and
//! Crazyhouse's drop generation, legality, check detection and move
//! execution on top of chess's.

use crate::attacks::between_bb;
use crate::bitboard::*;
use crate::misc::Prng;
use crate::movegen::{ALL, CAPTURES, EVASIONS, GenType, Moves, QUIETS};
use crate::position::{PIECE_TO_CHAR, Position, Prefetcher, ZOBRIST, pack_pieces};
use crate::types::*;
use crate::variant::{Crazyhouse, Variant};

/// Droppable piece types (`PAWN..=QUEEN`), indexed by `pt.idx() - 1`.
pub const HAND_TYPES: usize = 5;

/// Piece types that can sit in a Crazyhouse pocket, in index order.
pub const HAND_PIECE_TYPES: [PieceType; HAND_TYPES] = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN];

/// Pocket display order in FENs (queen down to pawn, as Lichess writes them).
pub const FEN_HAND_ORDER: [PieceType; HAND_TYPES] = [QUEEN, ROOK, BISHOP, KNIGHT, PAWN];

/// Crazyhouse extra state carried across plies in `StateInfo`: each side's
/// pocket counts and the bitboard of promoted pieces on the board.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct CrazyhouseState {
    /// Pieces in hand for each color and droppable piece type (`pt.idx() - 1`).
    pub hand: [[u8; HAND_TYPES]; COLOR_NB],
    /// Squares on the board occupied by promoted pieces (which become pawns
    /// when captured).
    pub promoted: Bitboard,
}

impl CrazyhouseState {
    #[inline(always)]
    pub const fn count(&self, c: Color, pt: PieceType) -> u8 {
        if pt.0 < PAWN.0 || pt.0 > QUEEN.0 {
            return 0;
        }
        self.hand[c.idx()][(pt.0 - PAWN.0) as usize]
    }

    #[inline(always)]
    pub fn set_count(&mut self, c: Color, pt: PieceType, n: u8) {
        debug_assert!(pt.0 >= PAWN.0 && pt.0 <= QUEEN.0);
        self.hand[c.idx()][(pt.0 - PAWN.0) as usize] = n;
    }

    #[inline(always)]
    pub fn add(&mut self, c: Color, pt: PieceType) -> (u8, u8) {
        debug_assert!(pt.0 >= PAWN.0 && pt.0 <= QUEEN.0);
        let slot = &mut self.hand[c.idx()][(pt.0 - PAWN.0) as usize];
        let old = *slot;
        let new = old.saturating_add(1);
        *slot = new;
        (old, new)
    }

    #[inline(always)]
    pub fn remove(&mut self, c: Color, pt: PieceType) -> (u8, u8) {
        debug_assert!(pt.0 >= PAWN.0 && pt.0 <= QUEEN.0);
        let slot = &mut self.hand[c.idx()][(pt.0 - PAWN.0) as usize];
        let old = *slot;
        debug_assert!(old > 0);
        let new = old - 1;
        *slot = new;
        (old, new)
    }

    #[inline(always)]
    pub const fn has_any(&self, c: Color) -> bool {
        let h = self.hand[c.idx()];
        (h[0] | h[1] | h[2] | h[3] | h[4]) != 0
    }

    #[inline(always)]
    pub const fn total(&self, c: Color) -> u8 {
        let h = self.hand[c.idx()];
        h[0].saturating_add(h[1])
            .saturating_add(h[2])
            .saturating_add(h[3])
            .saturating_add(h[4])
    }

    #[inline(always)]
    pub const fn is_promoted(&self, sq: Square) -> bool {
        (self.promoted.0 & (1u64 << sq.0)) != 0
    }

    #[inline(always)]
    pub fn set_promoted(&mut self, sq: Square) {
        self.promoted |= sq;
    }

    #[inline(always)]
    pub fn clear_promoted(&mut self, sq: Square) {
        self.promoted &= !square_bb(sq);
    }
}

/// Zobrist keys for each color, droppable piece type, and pocket count
/// (`0..64`), with count `0` zero so empty pockets add nothing and LSB clear.
pub static ZOBRIST_HAND: [[[Key; 64]; HAND_TYPES]; COLOR_NB] = {
    let mut rng = Prng::new(0x5A68_4841_4E44_71C5);
    let mut keys = [[[0; 64]; HAND_TYPES]; COLOR_NB];
    let mut c = 0;
    while c < COLOR_NB {
        let mut pt = 0;
        while pt < HAND_TYPES {
            let mut cnt = 1;
            while cnt < 64 {
                keys[c][pt][cnt] = rng.rand64() << 1;
                cnt += 1;
            }
            pt += 1;
        }
        c += 1;
    }
    keys
};

/// Zobrist keys for promoted-piece squares, with LSB clear.
pub static ZOBRIST_PROMOTED: [Key; SQUARE_NB] = {
    let mut rng = Prng::new(0x5A68_5052_4F4D_9E37);
    let mut keys = [0; SQUARE_NB];
    let mut sq = 0;
    while sq < SQUARE_NB {
        keys[sq] = rng.rand64() << 1;
        sq += 1;
    }
    keys
};

const _: () = {
    let mut any = 0;
    let mut c = 0;
    while c < COLOR_NB {
        let mut pt = 0;
        while pt < HAND_TYPES {
            assert!(ZOBRIST_HAND[c][pt][0] == 0);
            let mut cnt = 1;
            while cnt < 64 {
                assert!(ZOBRIST_HAND[c][pt][cnt] != 0);
                any |= ZOBRIST_HAND[c][pt][cnt];
                cnt += 1;
            }
            pt += 1;
        }
        c += 1;
    }
    let mut sq = 0;
    while sq < SQUARE_NB {
        assert!(ZOBRIST_PROMOTED[sq] != 0);
        any |= ZOBRIST_PROMOTED[sq];
        sq += 1;
    }
    assert!(
        any & 1 == 0,
        "Least significant bit of Crazyhouse Zobrist keys must be zero"
    );
};

#[inline(always)]
pub(crate) fn hand_key(c: Color, pt: PieceType, count: u8) -> Key {
    ZOBRIST_HAND[c.idx()][(pt.0 - PAWN.0) as usize][(count as usize) & 63]
}

/// Computes the full Zobrist contribution of a `CrazyhouseState`.
pub(crate) fn crazyhouse_extra_key(extra: &CrazyhouseState) -> Key {
    let mut k = 0;
    for c in [WHITE, BLACK] {
        for pt in HAND_PIECE_TYPES {
            let cnt = extra.count(c, pt);
            if cnt > 0 {
                k ^= hand_key(c, pt, cnt);
            }
        }
    }
    let mut bb = extra.promoted;
    while bb.any() {
        let sq = bb.pop_lsb();
        k ^= ZOBRIST_PROMOTED[sq.idx()];
    }
    k
}

/// Updates `extra` for a capture on `capsq` by `captor` and returns the
/// Zobrist delta.
#[inline(always)]
pub(crate) fn on_capture(
    extra: &mut CrazyhouseState,
    captor: Color,
    capsq: Square,
    captured_pt: PieceType,
) -> Key {
    let mut delta = 0;
    let hand_pt = if extra.is_promoted(capsq) {
        extra.clear_promoted(capsq);
        delta ^= ZOBRIST_PROMOTED[capsq.idx()];
        PAWN
    } else {
        captured_pt
    };
    let (old, new) = extra.add(captor, hand_pt);
    delta ^ hand_key(captor, hand_pt, old) ^ hand_key(captor, hand_pt, new)
}

/// Updates `extra` for a non-promotion, non-castling board move `from -> to`
/// and returns the Zobrist delta.
#[inline(always)]
pub(crate) fn on_board_move(extra: &mut CrazyhouseState, from: Square, to: Square) -> Key {
    if extra.is_promoted(from) {
        extra.clear_promoted(from);
        extra.set_promoted(to);
        ZOBRIST_PROMOTED[from.idx()] ^ ZOBRIST_PROMOTED[to.idx()]
    } else {
        0
    }
}

/// Updates `extra` for a pawn promotion onto `to` and returns the Zobrist delta.
#[inline(always)]
pub(crate) fn on_promotion(extra: &mut CrazyhouseState, to: Square) -> Key {
    extra.set_promoted(to);
    ZOBRIST_PROMOTED[to.idx()]
}

/// Parses a single pocket character (`P,N,B,R,Q,p,n,b,r,q`).
pub(super) fn parse_hand_char(token: u8) -> Option<(Color, PieceType)> {
    match token {
        b'P' => Some((WHITE, PAWN)),
        b'N' => Some((WHITE, KNIGHT)),
        b'B' => Some((WHITE, BISHOP)),
        b'R' => Some((WHITE, ROOK)),
        b'Q' => Some((WHITE, QUEEN)),
        b'p' => Some((BLACK, PAWN)),
        b'n' => Some((BLACK, KNIGHT)),
        b'b' => Some((BLACK, BISHOP)),
        b'r' => Some((BLACK, ROOK)),
        b'q' => Some((BLACK, QUEEN)),
        _ => None,
    }
}

/// Appends `[QRBNPqrbnp]` to `out` in Lichess/XBoard pocket order.
pub(crate) fn write_hand_fen(extra: &CrazyhouseState, out: &mut String) {
    out.push('[');
    for c in [WHITE, BLACK] {
        for pt in FEN_HAND_ORDER {
            let ch = PIECE_TO_CHAR[make_piece(c, pt).idx()] as char;
            for _ in 0..extra.count(c, pt) {
                out.push(ch);
            }
        }
    }
    out.push(']');
}

/// Appends drop moves for `us` onto `target` squares (`!occupied` when not in
/// check, or interposition squares when in single check).
#[inline(always)]
fn append_drops(extra: &CrazyhouseState, us: Color, target: Bitboard, moves: &mut Moves) {
    if !extra.has_any(us) || target.is_empty() {
        return;
    }
    for pt in HAND_PIECE_TYPES {
        if extra.count(us, pt) == 0 {
            continue;
        }
        let mut squares = if pt == PAWN {
            target & !(RANK_1_BB | RANK_8_BB)
        } else {
            target
        };
        while squares.any() {
            let to = squares.pop_lsb();
            moves.push(Move::make_drop(to, pt));
        }
    }
}

impl<V: Variant> Position<V> {
    /// The pawns a Crazyhouse game holds with `extra`'s pockets and promoted
    /// squares: on the board, in both hands, and as promoted pieces, which a
    /// capture turns back into pawns. At most 16 in any game.
    pub(crate) fn crazyhouse_pawns(&self, extra: &V::StateExtra) -> i32 {
        self.count_both(PAWN)
            + [WHITE, BLACK]
                .into_iter()
                .map(|c| V::hand_count(extra, c, PAWN) as i32)
                .sum::<i32>()
            + V::promoted_bb(extra).popcount()
    }

    /// Count of `pt` in `c`'s pocket (meaningful when `V::CRAZYHOUSE`).
    #[inline(always)]
    pub fn hand_count(&self, c: Color, pt: PieceType) -> u8 {
        V::hand_count(&self.st().extra, c, pt)
    }

    /// Whether `c` has at least one piece in hand (meaningful when `V::CRAZYHOUSE`).
    #[inline(always)]
    pub fn has_hand(&self, c: Color) -> bool {
        V::has_hand(&self.st().extra, c)
    }

    /// Whether `sq` holds a promoted piece (meaningful when `V::CRAZYHOUSE`).
    #[inline(always)]
    pub fn is_promoted(&self, sq: Square) -> bool {
        V::is_promoted(&self.st().extra, sq)
    }
}

impl Position<Crazyhouse> {
    /// Appends the legal Crazyhouse moves of kind `T` to `moves`.
    #[inline]
    pub(crate) fn crazyhouse_generate<const T: GenType>(&self, moves: &mut Moves) {
        if T == CAPTURES {
            crate::movegen::generate::<CAPTURES, Crazyhouse>(self, moves);
        } else if T == QUIETS {
            crate::movegen::generate::<QUIETS, Crazyhouse>(self, moves);
            append_drops(&self.st().extra, self.side_to_move, !self.pieces(), moves);
        } else if T == EVASIONS {
            crate::movegen::generate::<EVASIONS, Crazyhouse>(self, moves);
            if !self.checkers().more_than_one() {
                let ksq = self.king_square(self.side_to_move);
                let checker = self.checkers().lsb();
                let interpose = between_bb(ksq, checker) & !self.pieces();
                append_drops(&self.st().extra, self.side_to_move, interpose, moves);
            }
        } else {
            debug_assert!(T == ALL);
            if self.in_check() {
                self.crazyhouse_generate::<EVASIONS>(moves);
            } else {
                self.crazyhouse_generate::<CAPTURES>(moves);
                self.crazyhouse_generate::<QUIETS>(moves);
            }
        }
    }

    /// Tests whether `m`, possibly from the transposition table, is a legal
    /// Crazyhouse move.
    #[inline]
    pub(crate) fn crazyhouse_legal(&self, m: Move) -> bool {
        if !m.is_drop() {
            return self.chess_legal(m);
        }
        let pt = m.dropped_piece_type();
        if pt.0 < PAWN.0 || pt.0 > QUEEN.0 {
            return false;
        }
        if self.st().extra.count(self.side_to_move, pt) == 0 {
            return false;
        }
        let to = m.to_sq();
        if !self.empty(to) {
            return false;
        }
        if pt == PAWN && (square_bb(to) & (RANK_1_BB | RANK_8_BB)).any() {
            return false;
        }
        if self.in_check() {
            if self.checkers().more_than_one() {
                return false;
            }
            let ksq = self.king_square(self.side_to_move);
            let checker = self.checkers().lsb();
            return (between_bb(ksq, checker) & to).any();
        }
        true
    }

    /// Tests whether legal move `m` gives check in Crazyhouse.
    #[inline]
    pub(crate) fn crazyhouse_gives_check(&self, m: Move) -> bool {
        if m.is_drop() {
            (self.st().check_squares[m.dropped_piece_type().idx()] & m.to_sq()).any()
        } else {
            self.chess_gives_check(m)
        }
    }

    /// Plays a legal Crazyhouse move (either a drop or a board move).
    #[inline]
    pub(crate) fn crazyhouse_do_move<P: Prefetcher>(
        &mut self,
        m: Move,
        gives_check: bool,
        dirties: &mut Dirties,
        prefetcher: &P,
    ) {
        if m.is_drop() {
            self.crazyhouse_do_drop(m, gives_check, dirties, prefetcher);
        } else {
            self.chess_do_move(m, gives_check, dirties, prefetcher);
        }
    }

    fn crazyhouse_do_drop<P: Prefetcher>(
        &mut self,
        m: Move,
        gives_check: bool,
        dirties: &mut Dirties,
        prefetcher: &P,
    ) {
        debug_assert!(m.is_ok() && m.is_drop());

        let mut k = self.st().key ^ ZOBRIST.side;
        let prev_extra = self.st().extra;
        // SAFETY: the state is pushed below, before `states` changes.
        let st = unsafe { self.next_state() };
        st.extra = prev_extra;

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
        let to = m.to_sq();
        let pt = m.dropped_piece_type();
        let pc = make_piece(us, pt);

        debug_assert!(self.empty(to));
        debug_assert!(pt >= PAWN && pt <= QUEEN);
        debug_assert!(pt != PAWN || (square_bb(to) & (RANK_1_BB | RANK_8_BB)).is_empty());

        let (old_cnt, new_cnt) = st.extra.remove(us, pt);
        k ^= hand_key(us, pt, old_cnt) ^ hand_key(us, pt, new_cnt);

        if st.ep_square != Square::NONE {
            k ^= ZOBRIST.enpassant[st.ep_square.file() as usize];
            st.ep_square = Square::NONE;
        }

        k ^= ZOBRIST.psq[pc.idx()][to.idx()];
        st.material_key ^= ZOBRIST.psq[pc.idx()][(8 + self.piece_count[pc.idx()]) as usize & 63];

        if pt == PAWN {
            st.pawn_key ^= ZOBRIST.psq[pc.idx()][to.idx()];
            st.rule50 = 0;
        } else {
            st.non_pawn_key[us.idx()] ^= ZOBRIST.psq[pc.idx()][to.idx()];
            st.non_pawn_material[us.idx()] += pc.value();
            if pt <= BISHOP {
                st.minor_piece_key ^= ZOBRIST.psq[pc.idx()][to.idx()];
            }
        }

        dp.pc = pc;
        dp.from = Square::NONE;
        dp.to = to;
        dp.remove_sq = Square::NONE;
        dp.add_sq = Square::NONE;

        st.key = k;
        prefetcher.prefetch(st, pc, to);

        self.put_piece(pc, to, Some(dts));
        st.captured_piece = NO_PIECE;

        st.checkers_bb = if gives_check {
            square_bb(to)
        } else {
            Bitboard::EMPTY
        };
        debug_assert_eq!(
            st.checkers_bb,
            self.attackers_to(self.king_square(them)) & self.pieces_c(us)
        );

        self.side_to_move = them;
        self.set_extra_bitboards(st);
        st.piece_placement = pack_pieces(&self.board);

        // In Crazyhouse captures and drops are reversible via pocket cycles,
        // so scan repetitions back through `plies_from_null`.
        st.repetition = 0;
        let end = st.plies_from_null;
        if end >= 4 {
            let n = self.states.len();
            let mut i = 4;
            while i <= end {
                let stp = &self.states[n - i as usize];
                if stp.key == st.key
                    && stp.piece_placement == st.piece_placement
                    && stp.extra == st.extra
                {
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
    }

    /// Unmakes a Crazyhouse move.
    #[inline]
    pub(crate) fn crazyhouse_undo_move(&mut self, m: Move) {
        if m.is_drop() {
            debug_assert!(m.is_ok());
            assert!(self.states.len() > 1, "undo_move without a move to undo");
            self.side_to_move = !self.side_to_move;
            self.remove_piece(m.to_sq(), None);
            self.states.pop();
            self.game_ply -= 1;
            debug_assert_eq!(self.check(), Ok(()));
        } else {
            self.chess_undo_move(m);
        }
    }
}
