//! The rules of Atomic chess, as Lichess plays it: a capture explodes the
//! capturing piece, the captured one and every piece but pawns on the squares
//! next to the capture; kings cannot capture, a capture next to one's own king
//! is illegal, and exploding the opponent's king wins at once, even from
//! check. Adjacent kings cannot be captured, so neither is ever in check.
//!
//! Derived, with changes for varifish in 2026, from Atomic-Stockfish's
//! `position.cpp` and `movegen.cpp` (https://github.com/Belzedar94/Atomic-Stockfish),
//! a variant of Stockfish, a UCI chess playing engine derived from Glaurung
//! 2.1, Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under
//! the GNU General Public License, version 3 or later.
//!
//! Unlike Atomic-Stockfish:
//!
//! - Moves are generated legal, as Stockfish's master generates chess moves,
//!   rather than filtered, captures and quiet moves in check too, for the
//!   move picker, which has no evasion stage.
//! - The side to move's checkers are kept with the position, as in chess,
//!   rather than worked out whenever the search asks whether it is in check.
//! - The en passant square is recorded only when a legal en passant capture
//!   follows, as in chess, so that repetitions are found whichever way a
//!   position is reached, where Atomic-Stockfish records it whenever an
//!   enemy pawn attacks it.
//! - A FEN where the side not to move is in check, or has no king, is
//!   refused, as Lichess and Stockfish's chess refuse such positions.

use super::*;
use crate::movegen::{CAPTURES, EVASIONS, QUIETS};
use crate::variant::Atomic;

/// The atomic capture values of Atomic-Stockfish's static exchange
/// evaluation: Fairy-Stockfish's middlegame values for blast variants, with
/// its 3/2 bonus for pawns.
const ATOMIC_CAPTURE_PIECE_VALUE: [Value; PIECE_NB] = [
    0, 301, 702, 738, 1079, 1862, 0, 0, 0, 301, 702, 738, 1079, 1862, 0, 0,
];

/// What an Atomic state keeps to undo a capture: the capturing piece, and the
/// pieces the capture exploded besides it and the captured one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtomicState {
    pub capturer: Piece,
    pub blast: Blast,
}

/// What generating moves works out once for all of them.
struct Generation {
    ksq: Square,
    their_ksq: Square,
    // The kings are next to each other, so no move of ours but a king move
    // can put ours in check.
    adjacent: bool,
    in_check: bool,
    // Our pieces that alone keep an enemy slider off our king, unless the
    // kings are adjacent.
    pinned: Bitboard,
    // The squares between our king and the enemy sliders aligned with it,
    // which a capture must empty for the king to come under attack.
    danger: Bitboard,
    // Captures may land on enemy pieces not next to our king.
    capture_targets: Bitboard,
    // Moves other than captures and the king's may land on these empty
    // squares: in check, those that block it.
    quiet_targets: Bitboard,
}

/// Whether the kings on `s1` and `s2` are next to each other.
#[inline(always)]
fn adjacent(s1: Square, s2: Square) -> bool {
    pseudo_attacks(KING, s1).has(s2)
}

/// Rules that positions of any variant can compute, which the code shared by
/// all variants calls for Atomic.
impl<V: Variant> Position<V> {
    /// The pieces of `!c` attacking the king of `c`: none when either king is
    /// missing or the kings are adjacent.
    #[inline(always)]
    pub(crate) fn atomic_checkers_of(&self, c: Color) -> Bitboard {
        let ours = self.pieces_cpt(c, KING);
        let theirs = self.pieces_cpt(!c, KING);
        if ours.is_empty() || theirs.is_empty() {
            return Bitboard::EMPTY;
        }

        let ksq = ours.lsb();
        if (pseudo_attacks(KING, ksq) & theirs).any() {
            return Bitboard::EMPTY;
        }
        self.attackers_to(ksq) & self.pieces_c(!c)
    }

    /// Sets the pinners and the squares from which the side to move would
    /// give check, as for chess but none while the kings are adjacent or
    /// either is missing. The squares the other side attacks are computed
    /// lazily, as for chess.
    pub(super) fn atomic_set_extra_bitboards(&self, st: &mut StateInfo<V>) {
        for c in [WHITE, BLACK] {
            if self.has_king(c) {
                self.update_slider_blockers(c, st);
            } else {
                st.blockers_for_king[c.idx()] = Bitboard::EMPTY;
                st.pinners[(!c).idx()] = Bitboard::EMPTY;
            }
        }

        let us = self.side_to_move;
        let them = !us;
        st.check_squares = [Bitboard::EMPTY; PIECE_TYPE_NB];
        if self.has_king(us) && self.has_king(them) {
            let ksq = self.king_square(them);
            if !adjacent(ksq, self.king_square(us)) {
                let (bishop_attacks, rook_attacks) = both_attacks_bb(ksq, self.pieces());
                st.check_squares[PAWN.idx()] = pawn_attacks(them, ksq);
                st.check_squares[KNIGHT.idx()] = pseudo_attacks(KNIGHT, ksq);
                st.check_squares[BISHOP.idx()] = bishop_attacks;
                st.check_squares[ROOK.idx()] = rook_attacks;
                st.check_squares[QUEEN.idx()] = bishop_attacks | rook_attacks;
            }
        }

        st.threats[ALL_PIECES.idx()].set(Bitboard::EMPTY);
    }

    /// Tests whether a pseudo-legal move of the side to move, which has a
    /// king, is legal: whether it leaves that king on the board and out of
    /// check, or explodes the other king. It looks at the board only, so it
    /// also serves while a position is set.
    pub(crate) fn atomic_is_legal(&self, m: Move) -> bool {
        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq();
        let ksq = self.king_square(us);
        let their_ksq = self.king_square(them);

        if m.type_of() == CASTLING {
            let king_side = to > from;
            let kto = (if king_side { Square::G1 } else { Square::C1 }).relative(us);
            let rto = (if king_side { Square::F1 } else { Square::D1 }).relative(us);

            // Not out of check, nor through attacked squares, with the rook
            // still in place...
            let path_occupied = self.pieces() ^ from;
            for s in between_bb(from, kto) ^ kto | from {
                if !adjacent(s, their_ksq)
                    && (self.attackers_to_occ(s, path_occupied) & self.pieces_c(them)).any()
                {
                    return false;
                }
            }

            // ...nor into check, on the board after castling, where in
            // Chess960 the rook can shield the king.
            let occupied = (self.pieces() ^ from ^ to) | kto | rto;
            return adjacent(kto, their_ksq)
                || (self.attackers_to_occ(kto, occupied) & self.pieces_c(them) & occupied)
                    .is_empty();
        }

        let mut occupied = (self.pieces() ^ from) | to;
        if m.type_of() == EN_PASSANT {
            occupied ^= to - pawn_push(us);
        }

        if self.capture(m) {
            occupied &= !(square_bb(to)
                | (pseudo_attacks(KING, to) & self.pieces() & !self.pieces_pt(PAWN)));
        }

        let our_ksq = if self.piece_on(from).type_of() == KING {
            to
        } else {
            ksq
        };

        // A king never captures, nor may a capture explode one's own king...
        if !occupied.has(our_ksq) {
            return false;
        }

        // ...but exploding the other king wins at once.
        if !occupied.has(their_ksq) {
            return true;
        }

        adjacent(our_ksq, their_ksq)
            || (self.attackers_to_occ(our_ksq, occupied) & self.pieces_c(them) & occupied)
                .is_empty()
    }

    /// Whether the side to move can take en passant on `ep`, which the other
    /// side's last move, a double push, passed.
    pub(crate) fn atomic_ep_is_legal(&self, ep: Square) -> bool {
        let us = self.side_to_move;
        if !self.has_king(us) {
            return false;
        }
        (pawn_attacks(!us, ep) & self.pieces_cpt(us, PAWN))
            .into_iter()
            .any(|from| self.atomic_is_legal(Move::make(EN_PASSANT, from, ep, KNIGHT)))
    }

    /// Atomic-Stockfish's `has_legal_quiet`: whether the side to move has a
    /// legal move other than a capture, while both kings are on the board.
    pub(crate) fn atomic_has_legal_quiet(&self) -> bool {
        if !self.has_king(WHITE) || !self.has_king(BLACK) {
            return false;
        }

        let us = self.side_to_move;
        let up = pawn_push(us);
        let empty_squares = !self.pieces();
        let seventh_rank = if us == WHITE { RANK_7_BB } else { RANK_2_BB };
        let third_rank = if us == WHITE { RANK_3_BB } else { RANK_6_BB };
        let pawns_on_7 = self.pieces_cpt(us, PAWN) & seventh_rank;
        let pawns_not_on_7 = self.pieces_cpt(us, PAWN) & !seventh_rank;
        let blockers = self.blockers_for_king(us);
        let in_check = self.in_check();

        // Out of check, a piece other than the king that does not alone keep
        // a slider off it cannot expose the king by moving without capture.
        let legal_non_king =
            |m: Move| (!in_check && !blockers.has(m.from_sq())) || self.atomic_is_legal(m);

        let single_pushes = shift(pawns_not_on_7, up) & empty_squares;
        let double_pushes = shift(single_pushes & third_rank, up) & empty_squares;
        if single_pushes
            .into_iter()
            .any(|to| legal_non_king(Move::new(to - up, to)))
            || double_pushes
                .into_iter()
                .any(|to| legal_non_king(Move::new(to - up - up, to)))
        {
            return true;
        }

        // Whatever it promotes to, a pawn leaves and occupies the same squares.
        if (shift(pawns_on_7, up) & empty_squares)
            .into_iter()
            .any(|to| legal_non_king(Move::make(PROMOTION, to - up, to, QUEEN)))
        {
            return true;
        }

        for pt in [KNIGHT, BISHOP, ROOK, QUEEN, KING] {
            for from in self.pieces_cpt(us, pt) {
                for to in attacks_bb(pt, from, self.pieces()) & empty_squares {
                    let m = Move::new(from, to);
                    if if pt == KING {
                        self.atomic_is_legal(m)
                    } else {
                        legal_non_king(m)
                    } {
                        return true;
                    }
                }
            }
        }

        self.can_castle(castling_rights_of(us, ANY_CASTLING))
            && [
                castling_rights_of(us, KING_SIDE),
                castling_rights_of(us, QUEEN_SIDE),
            ]
            .into_iter()
            .any(|cr| {
                !self.castling_impeded(cr)
                    && self.can_castle(cr)
                    && self.atomic_is_legal(Move::make(
                        CASTLING,
                        self.king_square(us),
                        self.castling_rook_square(cr),
                        KNIGHT,
                    ))
            })
    }

    /// Atomic-Stockfish's `has_legal_move`: whether the side to move has a
    /// legal move, while both kings are on the board.
    pub(crate) fn atomic_has_legal_move(&self) -> bool {
        self.atomic_has_legal_quiet()
            || (self.has_king(WHITE)
                && self.has_king(BLACK)
                && MoveList::new::<CAPTURES>(self)
                    .iter()
                    .any(|&m| self.atomic_is_legal(m)))
    }

    /// Consistency checks for debugging, Stockfish's `pos_is_ok` with Atomic's
    /// kings, checks and en passant.
    pub(super) fn atomic_check(&self) -> Result<(), &'static str> {
        let stm = self.side_to_move;
        if (stm != WHITE && stm != BLACK)
            || self.count(WHITE, KING) > 1
            || self.count(BLACK, KING) > 1
            || !self.has_king(!stm)
            || (self.has_king(WHITE) && self.piece_on(self.king_square(WHITE)) != W_KING)
            || (self.has_king(BLACK) && self.piece_on(self.king_square(BLACK)) != B_KING)
            || (self.ep_square() != Square::NONE && self.ep_square().relative_rank(stm) != RANK_6)
        {
            return Err("pos_is_ok: Default");
        }

        if self.atomic_checkers_of(!stm).any() {
            return Err("pos_is_ok: Kings");
        }

        if self.checkers() != self.atomic_checkers_of(stm) {
            return Err("pos_is_ok: Checkers");
        }

        if (self.pieces_pt(PAWN) & (RANK_1_BB | RANK_8_BB)).any()
            || self.count(WHITE, PAWN) > 8
            || self.count(BLACK, PAWN) > 8
        {
            return Err("pos_is_ok: Pawns");
        }

        if self.ep_square() != Square::NONE {
            let ep = self.ep_square();
            if !(ep + pawn_push(!stm)).is_ok()
                || self.piece_on(ep + pawn_push(!stm)) != make_piece(!stm, PAWN)
                || !self.empty(ep)
                || !self.atomic_ep_is_legal(ep)
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

                if !self.has_king(c)
                    || self.piece_on(self.castling_rook_square(cr)) != make_piece(c, ROOK)
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
}

/// The rules of Atomic, which the `Variant` implementation of `Atomic` calls.
impl Position<Atomic> {
    /// Tests whether a move, possibly corrupt as moves from the transposition
    /// table can be, is legal.
    pub(crate) fn atomic_legal(&self, m: Move) -> bool {
        let us = self.side_to_move;
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.moved_piece(m);

        if !self.has_king(WHITE) || !self.has_king(BLACK) {
            return false;
        }

        // Uncommon cases use the slower but simpler test.
        if m.type_of() != NORMAL {
            return MoveList::new::<ALL>(self).contains(m);
        }

        // Not a promotion, so the promotion bits must be empty.
        debug_assert!(m.promotion_type() == KNIGHT);

        if pc == NO_PIECE || pc.color() != us || self.pieces_c(us).has(to) {
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

        self.atomic_is_legal(m)
    }

    /// A capture by which the side to move, which has a king, explodes the
    /// other king and wins at once, or `Move::none()`: one onto a piece next
    /// to the other king and not next to its own, which its king, attacking
    /// only squares next to itself, never makes, or en passant onto such a
    /// square.
    #[inline]
    pub fn atomic_winning_capture(&self) -> Move {
        let us = self.side_to_move;
        let them = !us;
        debug_assert!(self.has_king(us) && self.has_king(them));

        let blast = pseudo_attacks(KING, self.king_square(them))
            & !pseudo_attacks(KING, self.king_square(us));

        for to in blast & self.pieces_c(them) {
            let from = self.attackers_to(to) & self.pieces_c(us);
            if from.any() {
                let from = from.lsb();
                return if self.piece_on(from).type_of() == PAWN && (RANK_1_BB | RANK_8_BB).has(to) {
                    Move::make(PROMOTION, from, to, QUEEN)
                } else {
                    Move::new(from, to)
                };
            }
        }

        let ep = self.ep_square();
        if ep != Square::NONE && blast.has(ep) {
            let from = pawn_attacks(them, ep) & self.pieces_cpt(us, PAWN);
            if from.any() {
                return Move::make(EN_PASSANT, from.lsb(), ep, KNIGHT);
            }
        }

        Move::none()
    }

    /// Whether the side to move is under a blast check: a piece of the other
    /// side but its king attacks one of the side to move's pieces next to its
    /// king, so that capturing it explodes that king, unless the blast would
    /// take the other king too. Atomic-Stockfish's `under_blast_threat`, but
    /// testing the other king's distance before the attackers.
    pub(crate) fn atomic_under_blast_threat(&self) -> bool {
        let us = self.side_to_move;
        if !self.has_king(us) || !self.has_king(!us) {
            return false;
        }

        let their_ksq = self.king_square(!us);
        let capturers = self.pieces_c(!us) ^ their_ksq;
        let ring = pseudo_attacks(KING, self.king_square(us)) & self.pieces_c(us);

        ring.into_iter()
            .any(|s| !adjacent(s, their_ksq) && (self.attackers_to(s) & capturers).any())
    }

    /// Tests whether a legal move gives check, which the explosion of a
    /// capture can give by removing a piece between a slider and the king.
    /// Inlined, as most moves are quiet moves that the bitboards set for
    /// each position settle: those can give check only directly or by
    /// discovery, or by the king leaving the other.
    #[inline(always)]
    pub(crate) fn atomic_gives_check(&self, m: Move) -> bool {
        let from = m.from_sq();
        let to = m.to_sq();

        if m.type_of() == NORMAL && self.empty(to) {
            let them = !self.side_to_move;
            let pt = self.piece_on(from).type_of();
            if pt != KING {
                // While the kings are adjacent, which makes both immune, the
                // check squares are empty, and a discovered check is none.
                if self.check_squares(pt).has(to) {
                    return true;
                }
                if !self.blockers_for_king(them).has(from) {
                    return false;
                }
                return !adjacent(self.king_square(self.side_to_move), self.king_square(them))
                    && (line_bb(from, to) & self.pieces_cpt(them, KING)).is_empty();
            }

            // The king itself gives check only by discovery: by leaving a
            // line from one of our sliders to the other king, or the other
            // king's side, where our sliders' checks did not count.
            let their_ksq = self.king_square(them);
            if adjacent(from, their_ksq) {
                if adjacent(to, their_ksq) {
                    return false;
                }
            } else if !self.blockers_for_king(them).has(from) {
                return false;
            }
        }

        self.atomic_gives_check_on_board(m)
    }

    /// The rest of `atomic_gives_check`, out of line: whether the move gives
    /// check on the board after it.
    #[inline(never)]
    fn atomic_gives_check_on_board(&self, m: Move) -> bool {
        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.piece_on(from);
        let ksq = self.king_square(us);
        let their_ksq = self.king_square(them);

        // Find the board after the move.
        let mut occupied;
        let mut vacated = square_bb(from);
        let our_ksq;
        let mut checker = NO_PIECE;
        let mut checker_sq = to;

        if m.type_of() == CASTLING {
            let king_side = to > from;
            let kto = (if king_side { Square::G1 } else { Square::C1 }).relative(us);
            let rto = (if king_side { Square::F1 } else { Square::D1 }).relative(us);

            occupied = (self.pieces() ^ from ^ to) | kto | rto;
            vacated |= to;
            our_ksq = kto;
            checker = make_piece(us, ROOK);
            checker_sq = rto;
        } else {
            occupied = (self.pieces() ^ from) | to;
            if m.type_of() == EN_PASSANT {
                occupied ^= to - pawn_push(us);
            }

            our_ksq = if pc.type_of() == KING { to } else { ksq };

            if self.capture(m) {
                // The capturing piece explodes too.
                occupied &= !(square_bb(to)
                    | (pseudo_attacks(KING, to) & self.pieces() & !self.pieces_pt(PAWN)));
            } else {
                checker = if m.type_of() == PROMOTION {
                    make_piece(us, m.promotion_type())
                } else {
                    pc
                };
            }
        }

        // An exploded king ends the game rather than being in check, and
        // adjacent kings are immune.
        if !occupied.has(their_ksq) || !occupied.has(our_ksq) || adjacent(our_ksq, their_ksq) {
            return false;
        }

        // The pieces that stayed, the moved ones excepted since they may
        // have traded squares in Chess960 castling...
        if (self.attackers_to_occ(their_ksq, occupied) & self.pieces_c(us) & occupied & !vacated)
            .any()
        {
            return true;
        }

        // ...and the moved piece, unless it exploded.
        checker != NO_PIECE
            && checker.type_of() != KING
            && piece_attacks_bb(checker, checker_sq, occupied).has(their_ksq)
    }

    /// Plays a legal move, pushing a new state, and records in `dirties` what
    /// changed for the network. `prefetcher` is called once the new keys are
    /// known.
    pub(crate) fn atomic_do_move<P: Prefetcher>(
        &mut self,
        m: Move,
        gives_check: bool,
        dirties: &mut AtomicDirties,
        prefetcher: &P,
    ) {
        debug_assert!(m.is_ok());

        let mut k = self.st().key ^ ZOBRIST.side;
        // SAFETY: the state is pushed below, before `states` changes.
        let st = unsafe { self.next_state() };

        // The ply counters; rule50 is reset below for captures and pawn moves.
        self.game_ply += 1;
        st.rule50 += 1;
        st.plies_from_null += 1;
        st.extra = AtomicState::default();

        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let mut to = m.to_sq();
        let pc = self.piece_on(from);
        let captured = if m.type_of() == EN_PASSANT {
            make_piece(them, PAWN)
        } else if m.type_of() == CASTLING {
            NO_PIECE
        } else {
            self.piece_on(to)
        };

        debug_assert!(m.type_of() != EN_PASSANT || to == st.ep_square);

        let dp = &mut dirties.dirty_piece;
        dp.pc = pc;
        dp.from = from;
        dp.to = to;
        dp.remove_sq = Square::NONE;
        dp.add_sq = Square::NONE;
        dirties.blast = Blast::default();

        debug_assert!(pc.color() == us);
        debug_assert!(captured.is_none() || captured.color() == them);
        debug_assert!(captured.type_of() != KING);

        if st.ep_square != Square::NONE {
            k ^= ZOBRIST.enpassant[st.ep_square.file() as usize];
            st.ep_square = Square::NONE;
        }

        let mut castling_rights = st.castling_rights
            & !(self.castling_rights_mask[from.idx()] | self.castling_rights_mask[to.idx()]);

        if captured.is_some() {
            // The capturing piece, the captured one and the pieces but pawns
            // next to the capture all leave the board.
            let capsq = if m.type_of() == EN_PASSANT {
                to - pawn_push(us)
            } else {
                to
            };

            self.remove_exploded(st, &mut k, from);
            self.remove_exploded(st, &mut k, capsq);

            let blast = Blast::new(
                pseudo_attacks(KING, to) & self.pieces() & !self.pieces_pt(PAWN),
                &self.board,
            );
            for (_, s) in blast.iter() {
                castling_rights &= !self.castling_rights_mask[s.idx()];
                self.remove_exploded(st, &mut k, s);
            }
            debug_assert!(self.has_king(us));

            dp.to = Square::NONE;
            dp.remove_pc = captured;
            dp.remove_sq = capsq;
            dirties.blast = blast;
            st.extra = AtomicState {
                capturer: pc,
                blast,
            };
            st.rule50 = 0;
        } else if m.type_of() == CASTLING {
            debug_assert!(pc == make_piece(us, KING));

            let rook = make_piece(us, ROOK);
            let (rfrom, rto) = self.do_castling::<true>(us, from, &mut to, None, Some(dp));
            let keys = ZOBRIST.psq[rook.idx()][rfrom.idx()]
                ^ ZOBRIST.psq[rook.idx()][rto.idx()]
                ^ ZOBRIST.psq[pc.idx()][from.idx()]
                ^ ZOBRIST.psq[pc.idx()][to.idx()];
            k ^= keys;
            st.non_pawn_key[us.idx()] ^= keys;
        } else {
            let keys = ZOBRIST.psq[pc.idx()][from.idx()] ^ ZOBRIST.psq[pc.idx()][to.idx()];
            k ^= keys;

            if pc.type_of() == PAWN {
                st.pawn_key ^= keys;
                st.rule50 = 0;

                if m.type_of() == PROMOTION {
                    let promotion = make_piece(us, m.promotion_type());

                    debug_assert!(to.relative_rank(us) == RANK_8);

                    // The pawn's key on the last rank is zero, so there is no
                    // need to clear it.
                    k ^= ZOBRIST.psq[promotion.idx()][to.idx()];
                    st.material_key ^= ZOBRIST.psq[promotion.idx()]
                        [(8 + self.piece_count[promotion.idx()]) as usize & 63]
                        ^ ZOBRIST.psq[pc.idx()][(8 + self.piece_count[pc.idx()] - 1) as usize & 63];
                    st.non_pawn_key[us.idx()] ^= ZOBRIST.psq[promotion.idx()][to.idx()];
                    if promotion.type_of() <= BISHOP {
                        st.minor_piece_key ^= ZOBRIST.psq[promotion.idx()][to.idx()];
                    }
                    st.non_pawn_material[us.idx()] += promotion.value();

                    self.remove_piece(from, None);
                    self.put_piece(promotion, to, None);
                    dp.to = Square::NONE;
                    dp.add_pc = promotion;
                    dp.add_sq = to;
                } else {
                    self.move_piece(from, to, None);
                }
            } else {
                st.non_pawn_key[us.idx()] ^= keys;
                if pc.type_of() <= BISHOP {
                    st.minor_piece_key ^= keys;
                }
                self.move_piece(from, to, None);
            }
        }

        k ^= ZOBRIST.castling[st.castling_rights as usize & 15]
            ^ ZOBRIST.castling[castling_rights as usize & 15];
        st.castling_rights = castling_rights;
        st.captured_piece = captured;

        self.side_to_move = them;

        // A double push sets the en passant square only when a legal en
        // passant capture follows, which the key and repetitions need.
        if pc.type_of() == PAWN && (to.0 ^ from.0) == 16 {
            let ep_square = to - pawn_push(us);
            if (pawn_attacks(us, ep_square) & self.pieces_cpt(them, PAWN)).any()
                && self.atomic_ep_is_legal(ep_square)
            {
                st.ep_square = ep_square;
                k ^= ZOBRIST.enpassant[ep_square.file() as usize];
            }
        }

        let key = if st.rule50 < 14 {
            k
        } else {
            k ^ make_key(((st.rule50 - 14) / 8) as u64)
        };
        prefetcher.prefetch(key, st, pc, to);

        st.key = k;
        st.checkers_bb = if gives_check {
            self.atomic_checkers_of(them)
        } else {
            Bitboard::EMPTY
        };

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
    }

    /// Removes the piece on `s`, which a capture explodes, from the board and
    /// from the keys and material of `st`; the key is `k`.
    #[inline(always)]
    fn remove_exploded(&mut self, st: &mut StateInfo<Atomic>, k: &mut Key, s: Square) {
        let pc = self.piece_on(s);
        let c = pc.color();
        let psq = ZOBRIST.psq[pc.idx()][s.idx()];

        *k ^= psq;
        st.material_key ^=
            ZOBRIST.psq[pc.idx()][(8 + self.piece_count[pc.idx()] - 1) as usize & 63];

        if pc.type_of() == PAWN {
            st.pawn_key ^= psq;
        } else {
            st.non_pawn_key[c.idx()] ^= psq;
            st.non_pawn_material[c.idx()] -= pc.value();
            if pc.type_of() <= BISHOP {
                st.minor_piece_key ^= psq;
            }
        }

        self.remove_piece(s, None);
    }

    /// Unmakes a move, restoring the position exactly as it was before it.
    pub(crate) fn atomic_undo_move(&mut self, m: Move) {
        debug_assert!(m.is_ok());
        assert!(self.states.len() > 1, "undo_move without a move to undo");

        self.side_to_move = !self.side_to_move;

        let us = self.side_to_move;
        let from = m.from_sq();
        let mut to = m.to_sq();
        let st = self.st().clone();

        if m.type_of() == CASTLING {
            self.do_castling::<false>(us, from, &mut to, None, None);
        } else if st.captured_piece.is_some() {
            for (pc, s) in st.extra.blast.iter() {
                self.put_piece(pc, s, None);
            }
            let capsq = if m.type_of() == EN_PASSANT {
                to - pawn_push(us)
            } else {
                to
            };
            self.put_piece(st.captured_piece, capsq, None);
            self.put_piece(st.extra.capturer, from, None);
        } else if m.type_of() == PROMOTION {
            self.remove_piece(to, None);
            self.put_piece(make_piece(us, PAWN), from, None);
        } else {
            self.move_piece(to, from, None);
        }

        self.states.pop();
        self.game_ply -= 1;

        debug_assert_eq!(self.check(), Ok(()));
    }

    /// Tests whether Atomic-Stockfish's static exchange evaluation of a move
    /// is at least `threshold`. Since a capture ends the exchange, it counts
    /// what the capture explodes, or for another move what the cheapest
    /// capture of the moved piece would explode, which the opponent may also
    /// decline.
    pub(crate) fn atomic_see_ge(&self, m: Move, threshold: i32) -> bool {
        debug_assert!(m.is_ok());

        // Only normal moves are evaluated; the others pass a simple SEE.
        if m.type_of() != NORMAL {
            return VALUE_ZERO >= threshold;
        }

        let from = m.from_sq();
        let to = m.to_sq();
        let us = self.piece_on(from).color();
        let from_to = from | to;
        let mut blast =
            ((pseudo_attacks(KING, to) & !self.pieces_pt(PAWN)) | from_to) & self.pieces();
        let mut result = 0;
        let is_capture = self.capture(m);

        // The opponent's cheapest capture of the moved piece, at no cost when
        // the capturing piece is in the blast anyway.
        if !is_capture {
            let attackers = self.attackers_to_occ(to, self.pieces() ^ from_to)
                & self.pieces_c(!us)
                & !self.pieces_pt(KING);
            let Some(min_attacker) = attackers
                .into_iter()
                .map(|s| {
                    if blast.has(s) {
                        0
                    } else {
                        ATOMIC_CAPTURE_PIECE_VALUE[self.piece_on(s).idx()]
                    }
                })
                .min()
            else {
                return VALUE_ZERO >= threshold;
            };
            result += min_attacker;
        }

        let mut explodes_our_king = false;
        let mut explodes_their_king = false;
        while blast.any() {
            let pc = self.piece_on(blast.pop_lsb());
            if pc.type_of() == KING {
                if pc.color() == us {
                    explodes_our_king = true;
                } else {
                    explodes_their_king = true;
                }
            }
            let value = ATOMIC_CAPTURE_PIECE_VALUE[pc.idx()];
            result += if pc.color() == us { -value } else { value };
        }

        if is_capture {
            if explodes_our_king {
                result = -VALUE_MATE;
            } else if explodes_their_king {
                result = VALUE_MATE;
            } else {
                result -= 1;
            }
        } else if explodes_our_king && !explodes_their_king {
            result = -VALUE_MATE;
        } else if explodes_their_king {
            // The opponent cannot capture next to its own king.
            result = 0;
        } else {
            result = result.min(0);
        }

        result >= threshold
    }

    /// Appends the legal moves of kind `T` to `moves`: CAPTURES (captures
    /// and queen promotions) and QUIETS (the other moves), in check or not,
    /// EVASIONS in check, or ALL; none once a king has exploded. Kings never
    /// capture.
    pub(crate) fn atomic_generate<const T: GenType>(&self, moves: &mut Moves) {
        if !self.has_king(WHITE) || !self.has_king(BLACK) {
            return;
        }

        if T == ALL {
            if self.checkers().any() {
                self.atomic_generate_all::<EVASIONS>(moves);
            } else {
                self.atomic_generate_all::<CAPTURES>(moves);
                self.atomic_generate_all::<QUIETS>(moves);
            }
        } else {
            debug_assert!(T != EVASIONS || self.checkers().any());
            self.atomic_generate_all::<T>(moves);
        }
    }

    #[inline(always)]
    fn atomic_generate_all<const T: GenType>(&self, moves: &mut Moves) {
        debug_assert!(T != ALL);

        let us = self.side_to_move;
        let them = !us;
        let ksq = self.king_square(us);
        let their_ksq = self.king_square(them);
        let adjacent = adjacent(ksq, their_ksq);
        let checkers = self.checkers();

        let mut danger = Bitboard::EMPTY;
        if !adjacent && T != QUIETS {
            let snipers = ((pseudo_attacks(ROOK, ksq) & self.pieces_pt2(QUEEN, ROOK))
                | (pseudo_attacks(BISHOP, ksq) & self.pieces_pt2(QUEEN, BISHOP)))
                & self.pieces_c(them);
            for sniper in snipers {
                danger |= between_bb(ksq, sniper) ^ sniper;
            }
        }

        let g = Generation {
            ksq,
            their_ksq,
            adjacent,
            in_check: checkers.any(),
            pinned: if adjacent {
                Bitboard::EMPTY
            } else {
                self.blockers_for_king(us) & self.pieces_c(us)
            },
            danger,
            capture_targets: self.pieces_c(them) & !pseudo_attacks(KING, ksq),
            quiet_targets: if checkers.is_empty() {
                !self.pieces()
            } else if checkers.more_than_one() {
                Bitboard::EMPTY
            } else {
                between_bb(ksq, checkers.lsb()) & !self.pieces()
            },
        };

        if us == WHITE {
            self.atomic_pawn_moves::<{ WHITE.0 }, T>(&g, moves);
        } else {
            self.atomic_pawn_moves::<{ BLACK.0 }, T>(&g, moves);
        }
        self.atomic_piece_moves::<{ KNIGHT.0 }, T>(&g, moves);
        self.atomic_piece_moves::<{ BISHOP.0 }, T>(&g, moves);
        self.atomic_piece_moves::<{ ROOK.0 }, T>(&g, moves);
        self.atomic_piece_moves::<{ QUEEN.0 }, T>(&g, moves);

        // The king moves to empty squares that no enemy piece but the king
        // attacks, or next to the other king, where it is immune.
        if T != CAPTURES {
            let safe = !self.threats_by(ALL_PIECES) | self.threats_by(KING);
            for to in pseudo_attacks(KING, ksq) & !self.pieces() & safe {
                moves.push(Move::new(ksq, to));
            }
        }

        if T == QUIETS && !g.in_check && self.can_castle(castling_rights_of(us, ANY_CASTLING)) {
            for cr in [
                castling_rights_of(us, KING_SIDE),
                castling_rights_of(us, QUEEN_SIDE),
            ] {
                if self.can_castle(cr) && !self.castling_impeded(cr) {
                    let rfrom = self.castling_rook_square(cr);
                    if self.atomic_castling_is_legal(&g, rfrom) {
                        moves.push(Move::make(CASTLING, ksq, rfrom, KNIGHT));
                    }
                }
            }
        }
    }

    /// Whether castling with the rook on `rfrom`, out of check and with an
    /// empty path, is legal: the king must not pass through attacked squares
    /// nor end on one, unless next to the other king.
    #[inline(always)]
    fn atomic_castling_is_legal(&self, g: &Generation, rfrom: Square) -> bool {
        let us = self.side_to_move;
        let king_side = rfrom > g.ksq;
        let kto = (if king_side { Square::G1 } else { Square::C1 }).relative(us);
        let rto = (if king_side { Square::F1 } else { Square::D1 }).relative(us);

        // The threats are computed without the king, which these squares
        // need, and with the rook, which they keep.
        if ((between_bb(g.ksq, kto) ^ kto) & self.threats_by(ALL_PIECES) & !self.threats_by(KING))
            .any()
        {
            return false;
        }

        let occupied = (self.pieces() ^ g.ksq ^ rfrom) | kto | rto;
        adjacent(kto, g.their_ksq)
            || (self.attackers_to_occ(kto, occupied) & self.pieces_c(!us) & occupied).is_empty()
    }

    /// Whether a capture from `from` to `to`, of the piece on `capsq`, is
    /// legal, given that `to` is not next to our king.
    #[inline(always)]
    fn atomic_capture_is_legal(
        &self,
        g: &Generation,
        from: Square,
        to: Square,
        capsq: Square,
    ) -> bool {
        debug_assert!(!adjacent(to, g.ksq));

        // Exploding the other king wins, and next to it our king is immune.
        if (pseudo_attacks(KING, to) | to).has(g.their_ksq) || g.adjacent {
            return true;
        }

        let removed =
            from | to | capsq | (pseudo_attacks(KING, to) & self.pieces() & !self.pieces_pt(PAWN));

        // Out of check, the king comes under attack only if the explosion
        // opens the line of an enemy slider.
        if !g.in_check && (removed & g.danger).is_empty() {
            return true;
        }

        let occupied = self.pieces() & !removed;
        (self.attackers_to_occ(g.ksq, occupied) & self.pieces_c(!self.side_to_move) & occupied)
            .is_empty()
    }

    /// The pawn moves of the side `US` (0 for White, 1 for Black).
    #[inline(always)]
    fn atomic_pawn_moves<const US: u8, const T: GenType>(&self, g: &Generation, moves: &mut Moves) {
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

        let empty_squares = !self.pieces();
        let pawns_on_7 = self.pieces_cpt(us, PAWN) & t_rank7_bb;
        let pawns_not_on_7 = self.pieces_cpt(us, PAWN) & !t_rank7_bb;
        // A pinned pawn can only push along the king's file.
        let pushable = !g.pinned | file_bb(g.ksq.file());

        // Single and double pawn pushes, no promotions
        if T != CAPTURES {
            let b1 = shift(pawns_not_on_7 & pushable, up) & empty_squares;
            let b2 = shift(b1 & t_rank3_bb, up) & empty_squares;

            for to in b1 & g.quiet_targets {
                moves.push(Move::new(to - up, to));
            }
            for to in b2 & g.quiet_targets {
                moves.push(Move::new(to - up - up, to));
            }
        }

        // Promotions and underpromotions
        if pawns_on_7.any() {
            if T != QUIETS {
                for d in [up_right, up_left] {
                    for to in shift(pawns_on_7, d) & g.capture_targets {
                        if self.atomic_capture_is_legal(g, to - d, to, to) {
                            make_promotions::<T, true>(moves, to - d, to);
                        }
                    }
                }
            }

            for to in shift(pawns_on_7 & pushable, up) & empty_squares & g.quiet_targets {
                make_promotions::<T, false>(moves, to - up, to);
            }
        }

        // Standard and en passant captures
        if T != QUIETS {
            for d in [up_right, up_left] {
                for to in shift(pawns_not_on_7, d) & g.capture_targets {
                    if self.atomic_capture_is_legal(g, to - d, to, to) {
                        moves.push(Move::new(to - d, to));
                    }
                }
            }

            let ep = self.ep_square();
            if ep != Square::NONE {
                debug_assert!(ep.rank() == relative_rank(us, RANK_6));

                for from in pawns_not_on_7 & pawn_attacks(them, ep) {
                    let m = Move::make(EN_PASSANT, from, ep, KNIGHT);
                    if self.atomic_is_legal(m) {
                        moves.push(m);
                    }
                }
            }
        }
    }

    /// The moves of the pieces of type `PT`, neither pawns nor kings.
    #[inline(always)]
    fn atomic_piece_moves<const PT: u8, const T: GenType>(
        &self,
        g: &Generation,
        moves: &mut Moves,
    ) {
        let pt = PieceType(PT);
        let us = self.side_to_move;

        for from in self.pieces_cpt(us, pt) {
            let attacks = attacks_bb(pt, from, self.pieces());

            if T != QUIETS {
                for to in attacks & g.capture_targets {
                    if self.atomic_capture_is_legal(g, from, to, to) {
                        moves.push(Move::new(from, to));
                    }
                }
            }

            if T != CAPTURES {
                let mut b = attacks & g.quiet_targets;
                if g.pinned.has(from) {
                    b &= line_bb(g.ksq, from);
                }
                for to in b {
                    moves.push(Move::new(from, to));
                }
            }
        }
    }
}

/// The promotions of a pawn from `from` to `to` that a kind of generation
/// generates: CAPTURES takes queen promotions and captures that promote,
/// QUIETS underpromotions without capture, and the others all.
#[inline(always)]
fn make_promotions<const T: GenType, const ENEMY: bool>(
    moves: &mut Moves,
    from: Square,
    to: Square,
) {
    let all = T == EVASIONS || T == ALL;

    if T == CAPTURES || all {
        moves.push(Move::make(PROMOTION, from, to, QUEEN));
    }

    if (T == CAPTURES && ENEMY) || (T == QUIETS && !ENEMY) || all {
        moves.push(Move::make(PROMOTION, from, to, ROOK));
        moves.push(Move::make(PROMOTION, from, to, BISHOP));
        moves.push(Move::make(PROMOTION, from, to, KNIGHT));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blast_checks() {
        let blast_check = |fen: &str| {
            Position::<Atomic>::from_fen(fen, false)
                .unwrap()
                .atomic_under_blast_threat()
        };

        // Taking the knight next to White's king explodes the king.
        assert!(blast_check("1r5k/8/8/8/8/8/1N6/K7 w - - 0 1"));
        assert!(blast_check("7k/8/8/8/8/p7/1N6/K7 w - - 0 1"));
        // Not when the blast would take Black's king too, nor when only that
        // king attacks the knight, as kings cannot capture.
        assert!(!blast_check("1r6/8/8/8/8/2k5/1N6/K7 w - - 0 1"));
        assert!(!blast_check("8/8/8/8/8/2k5/1N6/K7 w - - 0 1"));
        // Nor when no piece next to the king is attacked.
        assert!(!blast_check("2r4k/8/8/8/8/8/1N6/K7 w - - 0 1"));
    }
}
