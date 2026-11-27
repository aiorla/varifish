//! Antichess's rules (Lichess rules): captures are mandatory, there is no
//! check or castling, kings are ordinary pieces that may be captured or
//! promoted to, and a side wins immediately by losing all its pieces or by
//! having no legal moves (stalemate).

use crate::attacks::*;
use crate::bitboard::*;
use crate::movegen::{ALL, CAPTURES, EVASIONS, GenType, Moves, QUIETS};
use crate::position::{Position, Prefetcher, ZOBRIST, pack_pieces};
use crate::types::*;
use crate::variant::{Antichess, Variant};

/// Returns all attacks made by `pc` from `s` given board occupancy `occupied`.
#[inline(always)]
fn piece_attacks(pc: Piece, s: Square, occupied: Bitboard) -> Bitboard {
    let pt = pc.type_of();
    if pt == PAWN {
        pawn_attacks(pc.color(), s)
    } else {
        attacks_bb(pt, s, occupied)
    }
}

/// Emits all five Antichess promotions (Q, R, B, N, K) from `from` to `to`.
#[inline(always)]
fn push_promotions(moves: &mut Moves, from: Square, to: Square) {
    moves.push(Move::make(PROMOTION, from, to, QUEEN));
    moves.push(Move::make(PROMOTION, from, to, ROOK));
    moves.push(Move::make(PROMOTION, from, to, BISHOP));
    moves.push(Move::make(PROMOTION, from, to, KNIGHT));
    moves.push(Move::make_king_promotion(from, to));
}

/// Returns the promoted piece type of an Antichess move, or `None` if it is
/// not a promotion.
#[inline(always)]
pub(crate) const fn promotion_piece_type(m: Move) -> Option<PieceType> {
    if m.type_of() == PROMOTION {
        Some(m.promotion_type())
    } else if m.is_king_promotion() {
        Some(KING)
    } else {
        None
    }
}

impl<V: Variant> Position<V> {
    /// Whether the side to move has already won under Antichess rules by
    /// having no pieces left on the board.
    #[inline(always)]
    pub fn is_antichess_win(&self) -> bool {
        debug_assert!(V::ANTICHESS);
        self.pieces_c(self.side_to_move).is_empty()
    }

    /// Whether the side to move has already lost because the opponent has no
    /// pieces left (only reachable from a hand-crafted terminal FEN).
    #[inline(always)]
    pub fn is_antichess_loss(&self) -> bool {
        debug_assert!(V::ANTICHESS);
        self.pieces_c(!self.side_to_move).is_empty()
    }

    /// Tests whether the side to move has at least one legal capture (including
    /// en passant).
    #[inline]
    pub fn antichess_has_captures(&self) -> bool {
        debug_assert!(V::ANTICHESS);
        let us = self.side_to_move;
        let them = !us;
        let enemies = self.pieces_c(them);
        if enemies.is_empty() || self.pieces_c(us).is_empty() {
            return false;
        }

        let pawns = self.pieces_cpt(us, PAWN);
        if (pawn_attacks_bb(us, pawns) & enemies).any() {
            return true;
        }
        if self.ep_square() != Square::NONE && (pawn_attacks(them, self.ep_square()) & pawns).any()
        {
            return true;
        }

        for sq in self.pieces_cpt(us, KNIGHT) {
            if (pseudo_attacks(KNIGHT, sq) & enemies).any() {
                return true;
            }
        }
        for sq in self.pieces_cpt(us, KING) {
            if (pseudo_attacks(KING, sq) & enemies).any() {
                return true;
            }
        }

        let occupied = self.pieces();
        for sq in self.pieces_c(us) & self.pieces_pt2(BISHOP, QUEEN) {
            if (attacks_bb(BISHOP, sq, occupied) & enemies).any() {
                return true;
            }
        }
        for sq in self.pieces_c(us) & self.pieces_pt2(ROOK, QUEEN) {
            if (attacks_bb(ROOK, sq, occupied) & enemies).any() {
                return true;
            }
        }

        false
    }

    /// Tests whether the side to move has at least one quiet move (assuming no
    /// captures exist, or independently of captures).
    #[inline]
    pub fn antichess_has_quiet_move(&self) -> bool {
        debug_assert!(V::ANTICHESS);
        let us = self.side_to_move;
        if self.pieces_c(us).is_empty() || self.pieces_c(!us).is_empty() {
            return false;
        }
        let empty = !self.pieces();

        if (pawn_single_push_bb(us, self.pieces_cpt(us, PAWN)) & empty).any() {
            return true;
        }
        for sq in self.pieces_cpt(us, KNIGHT) {
            if (pseudo_attacks(KNIGHT, sq) & empty).any() {
                return true;
            }
        }
        for sq in self.pieces_cpt(us, KING) {
            if (pseudo_attacks(KING, sq) & empty).any() {
                return true;
            }
        }
        let occupied = self.pieces();
        for sq in self.pieces_c(us) & self.pieces_pt2(BISHOP, QUEEN) {
            if (attacks_bb(BISHOP, sq, occupied) & empty).any() {
                return true;
            }
        }
        for sq in self.pieces_c(us) & self.pieces_pt2(ROOK, QUEEN) {
            if (attacks_bb(ROOK, sq, occupied) & empty).any() {
                return true;
            }
        }
        false
    }

    /// Tests whether the side to move has any legal Antichess move.
    #[inline]
    pub fn antichess_has_legal_move(&self) -> bool {
        debug_assert!(V::ANTICHESS);
        self.antichess_has_captures() || self.antichess_has_quiet_move()
    }

    /// Consistency checks for Antichess positions (`pos_is_ok`).
    pub(crate) fn antichess_check(&self) -> Result<(), &'static str> {
        let stm = self.side_to_move;
        if (stm != WHITE && stm != BLACK)
            || (self.ep_square() != Square::NONE && self.ep_square().relative_rank(stm) != RANK_6)
        {
            return Err("pos_is_ok: Default");
        }

        if self.st().castling_rights != NO_CASTLING {
            return Err("pos_is_ok: Castling");
        }

        if (self.pieces_pt(PAWN) & (RANK_1_BB | RANK_8_BB)).any()
            || self.count(WHITE, PAWN) > 8
            || self.count(BLACK, PAWN) > 8
        {
            return Err("pos_is_ok: Pawns");
        }

        if self.ep_square() != Square::NONE {
            let ep = self.ep_square();
            let captured = (ep + pawn_push(!stm)) & self.pieces_cpt(!stm, PAWN);
            let pawns = pawn_attacks(!stm, ep) & self.pieces_cpt(stm, PAWN);
            if captured.is_empty()
                || pawns.is_empty()
                || (self.pieces() & (ep | (ep + pawn_push(stm)))).any()
                || self.pieces_c(WHITE).is_empty()
                || self.pieces_c(BLACK).is_empty()
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

        let types = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING];
        for (i, &p1) in types.iter().enumerate() {
            for &p2 in &types[i + 1..] {
                if (self.pieces_pt(p1) & self.pieces_pt(p2)).any() {
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

        if !self.material_key_is_ok() {
            return Err("pos_is_ok: materialKey");
        }

        Ok(())
    }
}

impl Position<Antichess> {
    /// Generates all legal captures (including en passant and 5-way
    /// capture-promotions).
    fn generate_captures(&self, moves: &mut Moves) {
        let us = self.side_to_move;
        let them = !us;
        let enemies = self.pieces_c(them);
        if enemies.is_empty() || self.pieces_c(us).is_empty() {
            return;
        }

        let t_rank7_bb = if us == WHITE { RANK_7_BB } else { RANK_2_BB };
        let up_right = if us == WHITE { NORTH_EAST } else { SOUTH_WEST };
        let up_left = if us == WHITE { NORTH_WEST } else { SOUTH_EAST };

        let pawns_on_7 = self.pieces_cpt(us, PAWN) & t_rank7_bb;
        let pawns_not_on_7 = self.pieces_cpt(us, PAWN) & !t_rank7_bb;

        // Pawn capture-promotions
        if pawns_on_7.any() {
            for to in shift(pawns_on_7, up_right) & enemies {
                push_promotions(moves, to - up_right, to);
            }
            for to in shift(pawns_on_7, up_left) & enemies {
                push_promotions(moves, to - up_left, to);
            }
        }

        // Standard pawn captures
        for to in shift(pawns_not_on_7, up_right) & enemies {
            moves.push(Move::new(to - up_right, to));
        }
        for to in shift(pawns_not_on_7, up_left) & enemies {
            moves.push(Move::new(to - up_left, to));
        }

        // En passant captures
        if self.ep_square() != Square::NONE {
            let ep = self.ep_square();
            for from in pawns_not_on_7 & pawn_attacks(them, ep) {
                moves.push(Move::make(EN_PASSANT, from, ep, KNIGHT));
            }
        }

        // Non-pawn captures (including kings)
        let occupied = self.pieces();
        for pt in [KNIGHT, BISHOP, ROOK, QUEEN, KING] {
            for from in self.pieces_cpt(us, pt) {
                for to in attacks_bb(pt, from, occupied) & enemies {
                    moves.push(Move::new(from, to));
                }
            }
        }
    }

    /// Generates all quiet moves (including 5-way quiet promotions). Caller
    /// ensures no captures exist.
    fn generate_quiets(&self, moves: &mut Moves) {
        let us = self.side_to_move;
        if self.pieces_c(us).is_empty() || self.pieces_c(!us).is_empty() {
            return;
        }

        let empty = !self.pieces();
        let t_rank7_bb = if us == WHITE { RANK_7_BB } else { RANK_2_BB };
        let t_rank3_bb = if us == WHITE { RANK_3_BB } else { RANK_6_BB };
        let up = pawn_push(us);

        let pawns_on_7 = self.pieces_cpt(us, PAWN) & t_rank7_bb;
        let pawns_not_on_7 = self.pieces_cpt(us, PAWN) & !t_rank7_bb;

        // Quiet promotions
        if pawns_on_7.any() {
            for to in shift(pawns_on_7, up) & empty {
                push_promotions(moves, to - up, to);
            }
        }

        // Single and double pawn pushes
        let b1 = shift(pawns_not_on_7, up) & empty;
        let b2 = shift(b1 & t_rank3_bb, up) & empty;
        for to in b1 {
            moves.push(Move::new(to - up, to));
        }
        for to in b2 {
            moves.push(Move::new(to - up - up, to));
        }

        // Non-pawn quiet moves (including kings)
        let occupied = self.pieces();
        for pt in [KNIGHT, BISHOP, ROOK, QUEEN, KING] {
            for from in self.pieces_cpt(us, pt) {
                for to in attacks_bb(pt, from, occupied) & empty {
                    moves.push(Move::new(from, to));
                }
            }
        }
    }

    /// Generates legal Antichess moves of kind `T`:
    /// - `CAPTURES`: all captures (mandatory if non-empty);
    /// - `QUIETS`: all quiet moves (including quiet promotions) only when no
    ///   capture exists;
    /// - `ALL`: captures if any exist, otherwise quiet moves;
    /// - `EVASIONS`: never called (there is no check in Antichess).
    #[inline]
    pub(crate) fn antichess_generate<const T: GenType>(&self, moves: &mut Moves) {
        debug_assert!(T != EVASIONS);
        if self.pieces_c(WHITE).is_empty() || self.pieces_c(BLACK).is_empty() {
            return;
        }
        if T == CAPTURES {
            self.generate_captures(moves);
        } else if T == QUIETS {
            if !self.antichess_has_captures() {
                self.generate_quiets(moves);
            }
        } else if T == ALL {
            let start = moves.len();
            self.generate_captures(moves);
            if moves.len() == start {
                self.generate_quiets(moves);
            }
        }
    }

    /// Tests whether a possibly corrupt move (e.g. from the transposition
    /// table) is legal in Antichess.
    pub(crate) fn antichess_legal(&self, m: Move) -> bool {
        if !m.is_ok() || m.is_drop() || m.type_of() == CASTLING {
            return false;
        }
        if self.pieces_c(WHITE).is_empty() || self.pieces_c(BLACK).is_empty() {
            return false;
        }

        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.piece_on(from);

        if pc == NO_PIECE || pc.color() != us {
            return false;
        }

        let target = self.piece_on(to);
        if target != NO_PIECE && target.color() == us {
            return false;
        }

        let mt = m.type_of();
        let is_promo = mt == PROMOTION || m.is_king_promotion();
        if mt == EN_PASSANT {
            // No generated en passant capture sets the promotion bits.
            if m != Move::make(EN_PASSANT, from, to, KNIGHT)
                || pc.type_of() != PAWN
                || to != self.ep_square()
                || target != NO_PIECE
            {
                return false;
            }
            return (pawn_attacks(them, to) & from).any();
        }
        // Reject unused tag bits (only NORMAL, PROMOTION, EN_PASSANT, KING_PROMO_TAG).
        if mt == NORMAL && !m.is_king_promotion() && (m.raw() & 0xF000) != 0 {
            return false;
        }

        let is_capture = target != NO_PIECE;
        if pc.type_of() == PAWN {
            let promo_rank = to.relative_rank(us) == RANK_8;
            if is_promo != promo_rank {
                return false;
            }
            if is_capture {
                if !(pawn_attacks(us, from) & to).any() {
                    return false;
                }
            } else {
                let up = pawn_push(us);
                let single = from + up == to;
                let double = !single
                    && from.relative_rank(us) == RANK_2
                    && from + up + up == to
                    && self.empty(from + up);
                if !single && !double {
                    return false;
                }
            }
        } else {
            if is_promo {
                return false;
            }
            if !(attacks_bb(pc.type_of(), from, self.pieces()) & to).any() {
                return false;
            }
        }

        // Captures are mandatory: a quiet move is legal only when no capture exists.
        is_capture || !self.antichess_has_captures()
    }

    /// Records the threats that putting (`put_piece = true`) or removing
    /// (`put_piece = false`) `pc` on `s` adds or removes under Antichess's
    /// all-pieces threat model (all six piece types, including kings).
    #[inline(never)]
    fn antichess_update_piece_threats<const COMPUTE_RAY: bool>(
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
        let sliders = (self.pieces_pt2(BISHOP, QUEEN) & b_attacks)
            | (self.pieces_pt2(ROOK, QUEEN) & r_attacks);

        // Outgoing threats from `pc` on `s` to every piece it attacks.
        let mut threatened = piece_attacks(pc, s, occupied) & occupied;
        while threatened.any() {
            let sq = threatened.pop_lsb();
            dts.push(DirtyThreat::new(pc, self.piece_on(sq), s, sq, put_piece));
        }

        // Incoming direct threats onto `pc` on `s` from all piece types.
        let mut incoming = (pseudo_attacks(KNIGHT, s) & self.pieces_pt(KNIGHT))
            | (pseudo_attacks(KING, s) & self.pieces_pt(KING))
            | (pawn_attacks(WHITE, s) & self.pieces_cpt(BLACK, PAWN))
            | (pawn_attacks(BLACK, s) & self.pieces_cpt(WHITE, PAWN))
            | sliders;
        while incoming.any() {
            let src_sq = incoming.pop_lsb();
            dts.push(DirtyThreat::new(
                self.piece_on(src_sq),
                pc,
                src_sq,
                s,
                put_piece,
            ));
        }

        // Ray threats discovered (when removing) or blocked (when placing) through `s`.
        if COMPUTE_RAY {
            for slider_sq in sliders {
                let slider = self.piece_on(slider_sq);
                let ray = ray_pass_bb(slider_sq, s);
                let discovered = ray & slider_attacks & occupied;
                debug_assert!(!discovered.more_than_one());
                if discovered.any() && (ray & no_rays_containing) != no_rays_containing {
                    let target_sq = discovered.lsb();
                    let target_pc = self.piece_on(target_sq);
                    dts.push(DirtyThreat::new(
                        slider, target_pc, slider_sq, target_sq, !put_piece,
                    ));
                }
            }
        }
    }

    #[inline(always)]
    fn antichess_put_piece(&mut self, pc: Piece, s: Square, dts: Option<&mut DirtyThreats>) {
        self.put_piece(pc, s, None);
        if let Some(dts) = dts {
            self.antichess_update_piece_threats::<true>(pc, true, s, dts, Bitboard::ALL);
        }
    }

    #[inline(always)]
    fn antichess_remove_piece(&mut self, s: Square, dts: Option<&mut DirtyThreats>) {
        let pc = self.board[s.idx()];
        if let Some(dts) = dts {
            self.antichess_update_piece_threats::<true>(pc, false, s, dts, Bitboard::ALL);
        }
        self.remove_piece(s, None);
    }

    #[inline(always)]
    fn antichess_move_piece(
        &mut self,
        from: Square,
        to: Square,
        mut dts: Option<&mut DirtyThreats>,
    ) {
        let pc = self.board[from.idx()];
        let from_to = from | to;
        if let Some(dts) = dts.as_deref_mut() {
            self.antichess_update_piece_threats::<true>(pc, false, from, dts, from_to);
        }
        self.move_piece(from, to, None);
        if let Some(dts) = dts {
            self.antichess_update_piece_threats::<true>(pc, true, to, dts, from_to);
        }
    }

    #[inline(always)]
    fn antichess_swap_piece(&mut self, s: Square, pc: Piece, mut dts: Option<&mut DirtyThreats>) {
        let old = self.board[s.idx()];
        self.remove_piece(s, None);
        if let Some(dts) = dts.as_deref_mut() {
            self.antichess_update_piece_threats::<false>(old, false, s, dts, Bitboard::ALL);
        }
        self.put_piece(pc, s, None);
        if let Some(dts) = dts {
            self.antichess_update_piece_threats::<false>(pc, true, s, dts, Bitboard::ALL);
        }
    }

    /// Plays a legal Antichess move, updating board, keys, repetition info, and
    /// NNUE dirties (piece, threats including kings, and pawn pairs).
    pub(crate) fn antichess_do_move<P: Prefetcher>(
        &mut self,
        m: Move,
        dirties: &mut Dirties,
        prefetcher: &P,
    ) {
        debug_assert!(m.is_ok());
        debug_assert!(!m.is_drop() && m.type_of() != CASTLING);

        // The key is read from the current state: the next one carries only
        // the fields before it, and its own key is what its slot last held.
        let mut k = self.st().key ^ ZOBRIST.side;
        // SAFETY: `push_next_state` is called before returning.
        let st = unsafe { self.next_state() };

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
        let to = m.to_sq();
        let pc = self.piece_on(from);
        let captured = if m.type_of() == EN_PASSANT {
            make_piece(them, PAWN)
        } else {
            self.piece_on(to)
        };
        let promo_pt = promotion_piece_type(m);

        dp.pc = pc;
        dp.from = from;
        dp.to = to;
        dp.add_sq = Square::NONE;

        debug_assert!(pc.color() == us);
        debug_assert!(captured == NO_PIECE || captured.color() == them);

        if captured.is_some() {
            let mut capsq = to;
            if captured.type_of() == PAWN {
                if m.type_of() == EN_PASSANT {
                    capsq -= pawn_push(us);
                    debug_assert!(pc == make_piece(us, PAWN));
                    debug_assert!(to == st.ep_square);
                    debug_assert!(self.piece_on(to) == NO_PIECE);
                    debug_assert!(self.piece_on(capsq) == make_piece(them, PAWN));
                    self.antichess_remove_piece(capsq, Some(dts));
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

        if pc.type_of() == PAWN {
            if (to.0 ^ from.0) == 16 {
                let ep_square = to - pawn_push(us);
                let pawns = pawn_attacks(us, ep_square) & self.pieces_cpt(them, PAWN);
                if pawns.any() {
                    st.ep_square = ep_square;
                    k ^= ZOBRIST.enpassant[ep_square.file() as usize];
                }
            } else if let Some(pt) = promo_pt {
                let promotion = make_piece(us, pt);
                debug_assert!(to.relative_rank(us) == RANK_8);

                dp.add_pc = promotion;
                dp.add_sq = to;
                dp.to = Square::NONE;

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

        prefetcher.prefetch(st, pc, to);

        st.key = k;

        let to_pc = match promo_pt {
            Some(pt) => make_piece(us, pt),
            None => pc,
        };

        if captured.is_some() && m.type_of() != EN_PASSANT {
            self.antichess_remove_piece(from, Some(dts));
            self.antichess_swap_piece(to, to_pc, Some(dts));
        } else if pc == to_pc {
            self.antichess_move_piece(from, to, Some(dts));
        } else {
            self.antichess_remove_piece(from, Some(dts));
            self.antichess_put_piece(to_pc, to, Some(dts));
        }

        st.captured_piece = captured;
        st.checkers_bb = Bitboard::EMPTY;
        self.side_to_move = !self.side_to_move;

        self.set_extra_bitboards(st);
        st.piece_placement = pack_pieces(&self.board);

        st.repetition = 0;
        let end = st.rule50.min(st.plies_from_null);
        if end >= 4 {
            let n = self.states.len();
            let mut i = 4;
            while i <= end {
                let stp = &self.states[n - i as usize];
                if stp.key == st.key && stp.piece_placement == st.piece_placement {
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

    /// Unmakes an Antichess move, restoring the position exactly.
    pub(crate) fn antichess_undo_move(&mut self, m: Move) {
        debug_assert!(m.is_ok());
        assert!(self.states.len() > 1, "undo_move without a move to undo");

        self.side_to_move = !self.side_to_move;

        let us = self.side_to_move;
        let from = m.from_sq();
        let to = m.to_sq();
        let mut pc = self.piece_on(to);

        debug_assert!(self.empty(from));

        if let Some(pt) = promotion_piece_type(m) {
            debug_assert!(to.relative_rank(us) == RANK_8);
            debug_assert!(pc.type_of() == pt);
            pc = make_piece(us, PAWN);
            self.swap_piece(to, pc, None);
        }

        self.move_piece(to, from, None);

        let captured = self.st().captured_piece;
        if captured.is_some() {
            let mut capsq = to;
            if m.type_of() == EN_PASSANT {
                capsq -= pawn_push(us);
                debug_assert!(pc.type_of() == PAWN);
                debug_assert!(to == self.states[self.states.len() - 2].ep_square);
                debug_assert!(self.piece_on(capsq) == NO_PIECE);
                debug_assert!(captured == make_piece(!us, PAWN));
            }
            self.put_piece(captured, capsq, None);
        }

        self.states.pop();
        self.game_ply -= 1;

        debug_assert_eq!(self.check(), Ok(()));
    }
}
