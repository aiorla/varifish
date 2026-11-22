//! The accumulators of Atomic's networks in Stockfish's design: for each
//! position of the line being searched, the sums of the feature
//! transformer's weights over the active threat, pawn-pair and K32H3
//! features of each perspective, each brought up to date from another rather
//! than summed anew, and the caches that full refreshes start from: of the
//! piece-square sums by block and king square, and of the pairs' sums by the
//! half of the board the king is on.
//!
//! Derived, with changes for varifish in 2026, from Stockfish's
//! `nnue/nnue_accumulator.h` and `.cpp`: Stockfish, a UCI chess playing
//! engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later. Chess's accumulators, as varifish's `nnue::accumulator` has
//! them, with Atomic's captures, which remove the capturing piece, the
//! captured one and the pieces the capture explodes: their piece-square
//! changes are those pieces', their threats are those the move records, as
//! chess's moves record theirs, in a longer list, and their pawn pairs come
//! from the pawns before the move, which the stack records, and the move's
//! dirty piece, since blasts spare pawns. The sums wrap, as Stockfish's do,
//! so the order of the changes does not matter.

// The tile functions are unsafe for their pointers; the operations they
// call are unsafe only on some platforms.
#![allow(unused_unsafe)]

use super::{FeatureTransformer, Network, THREAT_AND_PAIR_DIMENSIONS};
use crate::bitboard::{Bitboard, square_bb};
use crate::memory::LargePageBox;
use crate::misc::ValueList;
use crate::nnue::accumulator::{
    DIMENSIONS, Tile, append_changed_psq_indices, apply, apply_changed_features, apply_features,
    apply_features_but_odd, apply_odd_features, load_tile, store_tile,
};
use crate::nnue::architecture::L1;
use crate::nnue::features::k32h3::{self, K32H3};
use crate::nnue::features::{full_threats, pp_3wide};
use crate::nnue::simd::TILE_HEIGHT;
use crate::position::Position;
use crate::types::*;
use crate::variant::Atomic;

const _: () = assert!(DIMENSIONS == L1);

type PsqIndexList = k32h3::IndexList;

/// The threat features of a position, or the pawn pairs that a refresh
/// changes in a cache entry.
type ActiveThreatList = full_threats::IndexList;

/// The threat features and pawn pairs a move removes or adds: at most one
/// for each threat it records, and the pairs of at most two pawns, 29.
type ChangedThreatList = ValueList<u16, MAX_ATOMIC_DIRTY_THREATS>;

// How many times each way of updating ran on the thread, for the tests to
// see that they cover each: the paths of `UPDATE_PATHS`.
#[cfg(test)]
thread_local! {
    static UPDATES: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
}

#[cfg(test)]
const UPDATE_PATHS: [&str; 4] = ["one side", "both sides", "king move", "refresh"];

#[cfg(test)]
fn count_update(path: usize) {
    UPDATES.with(|updates| {
        let mut counts = updates.get();
        counts[path] += 1;
        updates.set(counts);
    });
}

/// The result of the affine transformation of a position's input features
/// for both perspectives, which of them are up to date, what the move to the
/// position changed, its threats among them, and the state of the
/// piece-square blocks and the pawns before it.
#[repr(C, align(64))]
pub struct AccumulatorState {
    pub accumulation: [[i16; L1]; COLOR_NB],
    pub computed: [bool; COLOR_NB],
    pub dirties: AtomicDirties,
    pub block_before: k32h3::BlockState,
    /// The threats the move changed, which `dirties` has the move record
    pub dirty_threats: AtomicDirtyThreats,
    /// The pawns of each color before the move, for the pawn pairs it
    /// changed
    pub pawns_before: [Bitboard; COLOR_NB],
}

impl AccumulatorState {
    /// The pawns before the move and after it, which differ only where the
    /// dirty piece says: the pawn that moved, or that a capture exploded or
    /// a promotion took off the board, and the pawn captured. A blast
    /// spares the pawns next to it, kings never take pawns' squares, and
    /// castling moves no pawn.
    #[inline(always)]
    fn pawn_pairs(&self) -> DirtyPawnPairs {
        let dp = &self.dirties.dirty_piece;
        let before = self.pawns_before;
        let mut after = before;
        if dp.pc.type_of() == PAWN {
            let pawns = &mut after[dp.pc.color().idx()];
            *pawns ^= square_bb(dp.from);
            if dp.to != Square::NONE {
                *pawns |= square_bb(dp.to);
            }
        }
        if dp.remove_sq != Square::NONE && dp.remove_pc.type_of() == PAWN {
            after[dp.remove_pc.color().idx()] ^= square_bb(dp.remove_sq);
        }
        DirtyPawnPairs { before, after }
    }
}

/// For each block of the perspective's features, king square and
/// perspective, the piece-square sums of the board last refreshed there,
/// which a refresh brings up to date with the pieces that differ ("Finny
/// tables", first described by Luecx, the author of Koivisto); and for each
/// perspective and half of the board its king is on, which orients the pawn
/// pairs, the pairs' sums of the pawns last refreshed there, which a refresh
/// brings up to date with the pairs of the pawns that differ.
pub struct AccumulatorCaches {
    entries: [[[Entry; COLOR_NB]; SQUARE_NB]; K32H3::NUM_BLOCKS],
    pairs: [[PairEntry; 2]; COLOR_NB],
}

#[repr(C, align(64))]
struct Entry {
    accumulation: [i16; L1],
    pieces: [Piece; SQUARE_NB],
    piece_bb: Bitboard,
}

impl Entry {
    /// An entry of an empty board: the biases, without any weights on top.
    fn clear(&mut self, biases: &[i16; L1]) {
        self.accumulation = *biases;
        self.pieces = [NO_PIECE; SQUARE_NB];
        self.piece_bb = Bitboard(0);
    }
}

/// The sums of the pawn pairs' rows over the pairs of the pawns it holds of
/// each color: no biases, which the piece-square entries hold.
#[repr(C, align(64))]
struct PairEntry {
    accumulation: [i16; L1],
    pawns: [Bitboard; COLOR_NB],
}

impl PairEntry {
    /// An entry of a board without pawns, which make no pairs.
    fn clear(&mut self) {
        self.accumulation = [0; L1];
        self.pawns = [Bitboard(0); COLOR_NB];
    }
}

/// The half of the board, files a to d or e to h, that a king on `ksq` is
/// on, which orients its side's pawn pairs as ORIENT_TBL does.
#[inline(always)]
fn king_half(ksq: Square) -> usize {
    usize::from(ksq.0 & 0b100 != 0)
}

impl AccumulatorCaches {
    pub fn new(network: &Network) -> LargePageBox<AccumulatorCaches> {
        let mut caches = AccumulatorCaches::empty();
        caches.clear(network);
        caches
    }

    /// Caches of zeros, to be cleared with a network before they are used.
    pub fn empty() -> LargePageBox<AccumulatorCaches> {
        // SAFETY: zero bytes make entries.
        unsafe { LargePageBox::<AccumulatorCaches>::zeroed() }
    }

    pub fn clear(&mut self, network: &Network) {
        for entries in self.entries.as_flattened_mut() {
            for entry in entries {
                entry.clear(&network.feature_transformer.biases.0);
            }
        }
        for entry in self.pairs.as_flattened_mut() {
            entry.clear();
        }
    }
}

/// The states of the positions along the line being searched, the root
/// first.
pub struct AccumulatorStack {
    accumulators: [AccumulatorState; AccumulatorStack::MAX_SIZE],
    size: usize,
}

impl AccumulatorStack {
    pub const MAX_SIZE: usize = MAX_PLY as usize + 1;

    /// A stack holding only the root.
    pub fn new() -> LargePageBox<AccumulatorStack> {
        // SAFETY: zero bytes make a stack, which reset() then empties.
        let mut stack = unsafe { LargePageBox::<AccumulatorStack>::zeroed() };
        stack.reset();
        stack
    }

    #[inline(always)]
    pub fn latest(&self) -> &AccumulatorState {
        &self.accumulators[self.size - 1]
    }

    #[inline(always)]
    fn mut_latest(&mut self) -> &mut AccumulatorState {
        &mut self.accumulators[self.size - 1]
    }

    pub fn reset(&mut self) {
        let root = &mut self.accumulators[0];
        root.dirties = AtomicDirties::default();
        root.dirty_threats.clear();
        root.computed = [false; COLOR_NB];
        self.size = 1;
    }

    /// Adds the state of the position after a move from `pos`, whose
    /// changes the move records in the dirties returned, its threats in the
    /// state's list, and which records the pawns of `pos`, for the pawn
    /// pairs.
    #[inline(always)]
    pub fn push(&mut self, pos: &Position<Atomic>) -> &mut AtomicDirties {
        debug_assert!(self.size < Self::MAX_SIZE);
        let st = &mut self.accumulators[self.size];
        st.computed = [false; COLOR_NB];
        st.block_before = k32h3::block_state(pos);
        st.pawns_before = [pos.pieces_cpt(WHITE, PAWN), pos.pieces_cpt(BLACK, PAWN)];
        st.dirty_threats.clear();
        st.dirties.threats = ThreatRecord::new(&mut st.dirty_threats);
        self.size += 1;
        &mut st.dirties
    }

    #[inline(always)]
    pub fn pop(&mut self) {
        debug_assert!(self.size > 1);
        self.size -= 1;
    }

    /// Brings the latest state up to date for both perspectives, in a
    /// position with both kings.
    pub fn evaluate(
        &mut self,
        pos: &Position<Atomic>,
        feature_transformer: &FeatureTransformer,
        cache: &mut AccumulatorCaches,
    ) {
        debug_assert!(pos.has_king(WHITE) && pos.has_king(BLACK));
        let last_white = self.find_last_usable_accumulator(WHITE, pos);
        let last_black = self.find_last_usable_accumulator(BLACK, pos);

        if self.accumulators[last_white].computed[WHITE.idx()]
            && self.accumulators[last_black].computed[BLACK.idx()]
        {
            self.forward_update_incremental_both(pos, feature_transformer, last_white, last_black);
        } else {
            self.evaluate_side(WHITE, pos, feature_transformer, cache, last_white);
            self.evaluate_side(BLACK, pos, feature_transformer, cache, last_black);
        }
    }

    fn evaluate_side(
        &mut self,
        perspective: Color,
        pos: &Position<Atomic>,
        feature_transformer: &FeatureTransformer,
        cache: &mut AccumulatorCaches,
        last_usable_accum: usize,
    ) {
        const MIN_PC_COUNT_HYBRID: i32 = 15;

        if self.accumulators[last_usable_accum].computed[perspective.idx()] {
            self.forward_update_incremental(
                perspective,
                pos,
                feature_transformer,
                last_usable_accum,
            );
        } else {
            let dirty_piece = self.latest().dirties.dirty_piece;

            // The root's dirty piece is none, so a king move has a state
            // before it. A king never captures, which would explode it.
            if dirty_piece.pc == make_piece(perspective, KING)
                && usize::from(self.latest().block_before[perspective.idx()])
                    == k32h3::block_of(pos, perspective)
                && self.size >= 2
                && self.accumulators[self.size - 2].computed[perspective.idx()]
                && pos.count_both(ALL_PIECES) >= MIN_PC_COUNT_HYBRID
                && (dirty_piece.from.0 & 0b100) == (dirty_piece.to.0 & 0b100)
                && dirty_piece.add_sq == Square::NONE
            // excludes castling
            {
                let (target, computed) =
                    target_and_source(&mut self.accumulators, self.size - 1, self.size - 2);
                update_accumulator_hybrid(
                    perspective,
                    pos,
                    feature_transformer,
                    target,
                    computed,
                    cache,
                );
                return;
            }

            update_accumulator_refresh_cache(
                perspective,
                feature_transformer,
                pos,
                self.mut_latest(),
                cache,
            );
            self.backward_update_incremental(
                perspective,
                pos,
                feature_transformer,
                last_usable_accum,
            );
        }
    }

    /// Find the earliest usable accumulator, this can either be a computed
    /// accumulator or the accumulator state just after a change that
    /// requires full refresh: a move of the perspective's king, which also
    /// covers the threats, whose features change only when the king crosses
    /// the center, or a change of its piece-square block.
    fn find_last_usable_accumulator(&self, perspective: Color, pos: &Position<Atomic>) -> usize {
        // All traversed states must use the current block.
        let psq_block = k32h3::block_of(pos, perspective);

        for curr_idx in (1..self.size).rev() {
            let state = &self.accumulators[curr_idx];
            if state.computed[perspective.idx()] {
                return curr_idx;
            }

            let before = usize::from(state.block_before[perspective.idx()]);
            if requires_refresh(&state.dirties, perspective, before, psq_block) {
                return curr_idx;
            }
        }

        0
    }

    fn forward_update_incremental(
        &mut self,
        perspective: Color,
        pos: &Position<Atomic>,
        feature_transformer: &FeatureTransformer,
        begin: usize,
    ) {
        debug_assert!(begin < self.size);
        debug_assert!(self.accumulators[begin].computed[perspective.idx()]);

        let ksq = pos.king_square(perspective);
        let psq_block = k32h3::block_of(pos, perspective);

        for next in begin + 1..self.size {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next - 1);
            update_accumulator_incremental::<true>(
                perspective,
                feature_transformer,
                ksq,
                psq_block,
                target,
                computed,
            );
        }

        debug_assert!(self.latest().computed[perspective.idx()]);
    }

    fn backward_update_incremental(
        &mut self,
        perspective: Color,
        pos: &Position<Atomic>,
        feature_transformer: &FeatureTransformer,
        end: usize,
    ) {
        debug_assert!(end < self.size);
        debug_assert!(self.latest().computed[perspective.idx()]);

        let ksq = pos.king_square(perspective);
        let psq_block = k32h3::block_of(pos, perspective);

        for next in (end..self.size - 1).rev() {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next + 1);
            update_accumulator_incremental::<false>(
                perspective,
                feature_transformer,
                ksq,
                psq_block,
                target,
                computed,
            );
        }

        debug_assert!(self.accumulators[end].computed[perspective.idx()]);
    }

    fn forward_update_incremental_both(
        &mut self,
        pos: &Position<Atomic>,
        feature_transformer: &FeatureTransformer,
        white_begin: usize,
        black_begin: usize,
    ) {
        debug_assert!(white_begin < self.size);
        debug_assert!(black_begin < self.size);
        debug_assert!(self.accumulators[white_begin].computed[WHITE.idx()]);
        debug_assert!(self.accumulators[black_begin].computed[BLACK.idx()]);

        let white_ksq = pos.king_square(WHITE);
        let black_ksq = pos.king_square(BLACK);
        let white_block = k32h3::block_of(pos, WHITE);
        let black_block = k32h3::block_of(pos, BLACK);
        let shared_begin = white_begin.max(black_begin);

        // Catch up the lagging perspective, then traverse the common suffix
        // once.
        for next in white_begin + 1..=shared_begin {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next - 1);
            update_accumulator_incremental::<true>(
                WHITE,
                feature_transformer,
                white_ksq,
                white_block,
                target,
                computed,
            );
        }
        for next in black_begin + 1..=shared_begin {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next - 1);
            update_accumulator_incremental::<true>(
                BLACK,
                feature_transformer,
                black_ksq,
                black_block,
                target,
                computed,
            );
        }

        for next in shared_begin + 1..self.size {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next - 1);
            update_accumulator_incremental_both(
                feature_transformer,
                white_ksq,
                black_ksq,
                white_block,
                black_block,
                target,
                computed,
            );
        }

        debug_assert!(self.latest().computed[WHITE.idx()]);
        debug_assert!(self.latest().computed[BLACK.idx()]);
    }
}

/// Whether a move changes what every feature of `perspective` depends on:
/// the square of its king, which moved or exploded, or the block of its
/// features, from `block_before` to `block_after`.
#[inline(always)]
fn requires_refresh(
    dirties: &AtomicDirties,
    perspective: Color,
    block_before: usize,
    block_after: usize,
) -> bool {
    let king = make_piece(perspective, KING);
    let dp = &dirties.dirty_piece;
    dp.pc == king
        || (dp.remove_sq != Square::NONE && dp.remove_pc == king)
        || dirties.blast.iter().any(|(pc, _)| pc == king)
        || block_before != block_after
}

/// The features a move removes and adds, for a perspective whose king is on
/// `ksq` before and after it: the moved piece, from its square to its
/// destination unless a capture explodes it, the captured piece, the pieces
/// the capture explodes, and the rook of castling or the piece a pawn
/// promotes to.
#[inline(always)]
fn append_changed_indices(
    perspective: Color,
    ksq: Square,
    dirties: &AtomicDirties,
    removed: &mut PsqIndexList,
    added: &mut PsqIndexList,
) {
    let dp = &dirties.dirty_piece;
    removed.push(k32h3::make_index(perspective, dp.from, dp.pc, ksq));
    if dp.to != Square::NONE {
        added.push(k32h3::make_index(perspective, dp.to, dp.pc, ksq));
    }
    if dp.remove_sq != Square::NONE {
        removed.push(k32h3::make_index(
            perspective,
            dp.remove_sq,
            dp.remove_pc,
            ksq,
        ));
    }
    if dp.add_sq != Square::NONE {
        added.push(k32h3::make_index(perspective, dp.add_sq, dp.add_pc, ksq));
    }
    for (pc, s) in dirties.blast.iter() {
        removed.push(k32h3::make_index(perspective, s, pc, ksq));
    }
}

/// The state at `target`, to update, and the one at `source`, to update it
/// from.
#[inline(always)]
fn target_and_source(
    states: &mut [AccumulatorState],
    target: usize,
    source: usize,
) -> (&mut AccumulatorState, &AccumulatorState) {
    let [target, source] = states
        .get_disjoint_mut([target, source])
        .expect("two states of the stack");
    (target, source)
}

/// Removes and adds the rows of piece-square features, in the weights that
/// `psq_base` points at, and of threat features, in those that `thr_base`
/// points at, each at its weights' first row, as chess's
/// `apply_combined_features` does: with paired widening, the last row of an
/// odd number of piece-square rows goes with the first threat row of the
/// same sign, and an odd removed row with an odd added one.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn apply_combined_features(
    j: usize,
    acc: &mut Tile,
    psq_removed: &[u16],
    thr_removed: &[u16],
    psq_added: &[u16],
    thr_added: &[u16],
    psq_base: *const i8,
    thr_base: *const i8,
) {
    // SAFETY: tiles of the weights, as for the rows.
    unsafe {
        let psq_weights = psq_base.add(j);
        let thr_weights = thr_base.add(j);
        let odd_removed = apply_features_but_odd::<-1, _>(
            j,
            acc,
            psq_removed,
            thr_removed,
            psq_weights,
            thr_weights,
        );
        let odd_added =
            apply_features_but_odd::<1, _>(j, acc, psq_added, thr_added, psq_weights, thr_weights);
        apply_odd_features(acc, odd_removed, odd_added);
    }
}

/// Writes to `to` the accumulation of `from` less the rows of the features
/// removed and plus those of the features added: piece-square ones, from
/// the weights of a block, and threat ones, pawn pairs among them. A capture
/// removes up to ten piece-square features, so their rows are applied as a
/// refresh's are.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn apply_combined(
    feature_transformer: &FeatureTransformer,
    from_acc: &[i16; DIMENSIONS],
    to_acc: &mut [i16; DIMENSIONS],
    psq_added: &PsqIndexList,
    psq_removed: &PsqIndexList,
    thr_added: &ChangedThreatList,
    thr_removed: &ChangedThreatList,
    psq_weights: &[i8],
) {
    debug_assert!(
        psq_added
            .iter()
            .chain(psq_removed.iter())
            .all(|&i| usize::from(i) < K32H3::BLOCK_DIMENSIONS)
    );
    debug_assert!(
        thr_added
            .iter()
            .chain(thr_removed.iter())
            .all(|&i| usize::from(i) < THREAT_AND_PAIR_DIMENSIONS)
    );
    let from = from_acc.as_ptr();
    let to = to_acc.as_mut_ptr();
    let psq_base = psq_weights.as_ptr();
    let thr_base = feature_transformer.threat_weights().as_ptr();

    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, and rows of the lists, which
        // the weights hold.
        unsafe {
            let mut acc = load_tile(j, from);

            apply_combined_features(
                j,
                &mut acc,
                psq_removed.as_slice(),
                thr_removed.as_slice(),
                psq_added.as_slice(),
                thr_added.as_slice(),
                psq_base,
                thr_base,
            );

            store_tile(j, to, &acc);
        }
    }
}

fn update_accumulator_incremental<const FORWARD: bool>(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    ksq: Square,
    psq_block: usize,
    target_state: &mut AccumulatorState,
    computed: &AccumulatorState,
) {
    let p = perspective.idx();
    debug_assert!(computed.computed[p]);
    debug_assert!(!target_state.computed[p]);
    #[cfg(test)]
    count_update(0);

    let mut psq_removed = PsqIndexList::new();
    let mut psq_added = PsqIndexList::new();
    let mut thr_removed = ChangedThreatList::new();
    let mut thr_added = ChangedThreatList::new();

    let move_state = if FORWARD { &*target_state } else { computed };
    let (dirties, dirty_threats) = (&move_state.dirties, &move_state.dirty_threats);

    // Used solely for prefetching
    let threat_base = feature_transformer.threat_weights().as_ptr();

    // Going backward, what the move added is removed.
    let (psq_out, psq_in, thr_out, thr_in) = if FORWARD {
        (
            &mut psq_removed,
            &mut psq_added,
            &mut thr_removed,
            &mut thr_added,
        )
    } else {
        (
            &mut psq_added,
            &mut psq_removed,
            &mut thr_added,
            &mut thr_removed,
        )
    };
    full_threats::append_changed_indices(
        perspective,
        ksq,
        dirty_threats,
        thr_out,
        thr_in,
        threat_base,
        DIMENSIONS,
    );
    pp_3wide::append_changed_indices(
        perspective,
        ksq,
        &move_state.pawn_pairs(),
        thr_out,
        thr_in,
        threat_base,
        DIMENSIONS,
    );
    append_changed_indices(perspective, ksq, dirties, psq_out, psq_in);

    apply_combined(
        feature_transformer,
        &computed.accumulation[p],
        &mut target_state.accumulation[p],
        &psq_added,
        &psq_removed,
        &thr_added,
        &thr_removed,
        feature_transformer.psq_weights(psq_block),
    );

    target_state.computed[p] = true;
}

fn update_accumulator_incremental_both(
    feature_transformer: &FeatureTransformer,
    white_ksq: Square,
    black_ksq: Square,
    white_block: usize,
    black_block: usize,
    target_state: &mut AccumulatorState,
    computed: &AccumulatorState,
) {
    debug_assert!(computed.computed[WHITE.idx()]);
    debug_assert!(computed.computed[BLACK.idx()]);
    debug_assert!(!target_state.computed[WHITE.idx()]);
    debug_assert!(!target_state.computed[BLACK.idx()]);
    #[cfg(test)]
    count_update(1);

    let mut psq_removed = [PsqIndexList::new(); COLOR_NB];
    let mut psq_added = [PsqIndexList::new(); COLOR_NB];
    let mut thr_removed = [ChangedThreatList::new(); COLOR_NB];
    let mut thr_added = [ChangedThreatList::new(); COLOR_NB];

    let threat_base = feature_transformer.threat_weights().as_ptr();

    let dirties = &target_state.dirties;
    {
        let [white_removed, black_removed] = &mut thr_removed;
        let [white_added, black_added] = &mut thr_added;
        full_threats::append_changed_indices_both(
            white_ksq,
            black_ksq,
            &target_state.dirty_threats,
            white_removed,
            white_added,
            black_removed,
            black_added,
            threat_base,
            DIMENSIONS,
        );
        pp_3wide::append_changed_indices_both(
            white_ksq,
            black_ksq,
            &target_state.pawn_pairs(),
            white_removed,
            white_added,
            black_removed,
            black_added,
            threat_base,
            DIMENSIONS,
        );
    }
    for (perspective, ksq) in [(WHITE, white_ksq), (BLACK, black_ksq)] {
        let p = perspective.idx();
        append_changed_indices(
            perspective,
            ksq,
            dirties,
            &mut psq_removed[p],
            &mut psq_added[p],
        );
    }

    for (p, psq_block) in [(WHITE.idx(), white_block), (BLACK.idx(), black_block)] {
        apply_combined(
            feature_transformer,
            &computed.accumulation[p],
            &mut target_state.accumulation[p],
            &psq_added[p],
            &psq_removed[p],
            &thr_added[p],
            &thr_removed[p],
            feature_transformer.psq_weights(psq_block),
        );
    }

    target_state.computed = [true; COLOR_NB];
}

/// Updates the accumulator for a move of the perspective's king that keeps
/// its block and half of the board, as chess's does: the cache entry of the
/// new king square is brought up to date with the board, and the threats
/// and pawn pairs, whose features the king's move keeps, are carried over
/// from the previous accumulator, less the piece-square sums of the old king
/// square. The move, which captures nothing, moves no pawn.
fn update_accumulator_hybrid(
    perspective: Color,
    pos: &Position<Atomic>,
    feature_transformer: &FeatureTransformer,
    target: &mut AccumulatorState,
    computed: &AccumulatorState,
    cache: &mut AccumulatorCaches,
) {
    let p = perspective.idx();
    let dirty_piece = target.dirties.dirty_piece;

    debug_assert!(dirty_piece.pc == make_piece(perspective, KING));
    debug_assert!(dirty_piece.to != Square::NONE);
    debug_assert!(dirty_piece.remove_sq == Square::NONE);
    debug_assert!((dirty_piece.from.0 & 0b100) == (dirty_piece.to.0 & 0b100));
    debug_assert!(computed.computed[p]);
    debug_assert!(!target.computed[p]);
    debug_assert!({
        let pairs = target.pawn_pairs();
        pairs.before == pairs.after
    });
    #[cfg(test)]
    count_update(2);

    let old_ksq = dirty_piece.from;
    let new_ksq = dirty_piece.to;
    debug_assert!(old_ksq != new_ksq);

    // The board before the move: the king back on its square, and the
    // square it moved to empty, since kings never capture.
    let current_pieces = pos.piece_array();
    let mut previous_pieces = *current_pieces;
    let mut previous_piece_bb = pos.pieces();

    debug_assert!(previous_pieces[new_ksq.idx()] == dirty_piece.pc);
    previous_pieces[new_ksq.idx()] = NO_PIECE;
    previous_piece_bb &= !square_bb(new_ksq);

    debug_assert!(previous_pieces[old_ksq.idx()] == NO_PIECE);
    previous_pieces[old_ksq.idx()] = make_piece(perspective, KING);
    previous_piece_bb |= square_bb(old_ksq);

    let psq_block = k32h3::block_of(pos, perspective);
    let [old_entries, new_entries] = cache.entries[psq_block]
        .get_disjoint_mut([old_ksq.idx(), new_ksq.idx()])
        .expect("a king move changes its square");
    let old_entry = &old_entries[p];
    let new_entry = &mut new_entries[p];
    let psq_weights = feature_transformer.psq_weights(psq_block).as_ptr();
    let thr_base = feature_transformer.threat_weights().as_ptr();

    // "Remove" means we need to remove them from the cache entry,
    // "Add" means add them to the entry to get the accumulator we want
    let mut old_remove = PsqIndexList::new();
    let mut old_add = PsqIndexList::new();
    let mut new_remove = PsqIndexList::new();
    let mut new_add = PsqIndexList::new();

    append_changed_psq_indices(
        perspective,
        old_ksq,
        &old_entry.pieces,
        old_entry.piece_bb,
        &previous_pieces,
        previous_piece_bb,
        &mut old_remove,
        &mut old_add,
    );
    append_changed_psq_indices(
        perspective,
        new_ksq,
        &new_entry.pieces,
        new_entry.piece_bb,
        current_pieces,
        pos.pieces(),
        &mut new_remove,
        &mut new_add,
    );

    let mut thr_removed = ChangedThreatList::new();
    let mut thr_added = ChangedThreatList::new();
    full_threats::append_changed_indices(
        perspective,
        new_ksq,
        &target.dirty_threats,
        &mut thr_removed,
        &mut thr_added,
        thr_base,
        DIMENSIONS,
    );
    debug_assert!(
        thr_added
            .iter()
            .chain(thr_removed.iter())
            .all(|&i| usize::from(i) < THREAT_AND_PAIR_DIMENSIONS)
    );

    let from_acc = computed.accumulation[p].as_ptr();
    let to_acc = target.accumulation[p].as_mut_ptr();
    let old_acc = old_entry.accumulation.as_ptr();
    let new_acc = new_entry.accumulation.as_mut_ptr();

    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, and rows of the lists.
        unsafe {
            let mut acc = load_tile(j, new_acc);

            apply_changed_features(
                &mut acc,
                new_remove.as_slice(),
                new_add.as_slice(),
                psq_weights.add(j),
            );

            store_tile(j, new_acc, &acc);

            // adding the old accumulator adds (most of) the threat weights
            // that we need
            apply::<1>(&mut acc, from_acc.add(j));
            // But we have added a whole bunch of psq weights for the wrong
            // king bucket which we need to remove
            // first we remove the cached psq accumulation for the old king
            // position...
            apply::<-1>(&mut acc, old_acc.add(j));

            // ... then we adjust
            // The old entry's removed psq features have to be added back,
            // its added ones removed
            apply_combined_features(
                j,
                &mut acc,
                old_add.as_slice(),
                thr_removed.as_slice(),
                old_remove.as_slice(),
                thr_added.as_slice(),
                psq_weights,
                thr_base,
            );

            store_tile(j, to_acc, &acc);
        }
    }

    new_entry.pieces = *current_pieces;
    new_entry.piece_bb = pos.pieces();

    target.computed[p] = true;
}

/// The piece-square sums come from the cache entry of the perspective's
/// block and king square, brought up to date with the pieces that differ,
/// and the threats are summed anew from the active threat features. The
/// pawn pairs' sums come from the cache entry of the perspective and its
/// king's half of the board, brought up to date with the pairs of the pawns
/// that differ: a pawn move changes a few pairs of the dozens a board holds.
fn update_accumulator_refresh_cache(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    pos: &Position<Atomic>,
    accumulator: &mut AccumulatorState,
    cache: &mut AccumulatorCaches,
) {
    let p = perspective.idx();
    #[cfg(test)]
    count_update(3);
    let psq_block = k32h3::block_of(pos, perspective);
    let ksq = pos.king_square(perspective);
    let entry = &mut cache.entries[psq_block][ksq.idx()][p];
    let psq_weights = feature_transformer.psq_weights(psq_block).as_ptr();
    let thr_base = feature_transformer.threat_weights().as_ptr();
    let mut removed = PsqIndexList::new();
    let mut added = PsqIndexList::new();

    append_changed_psq_indices(
        perspective,
        ksq,
        &entry.pieces,
        entry.piece_bb,
        pos.piece_array(),
        pos.pieces(),
        &mut removed,
        &mut added,
    );

    entry.piece_bb = pos.pieces();
    entry.pieces = *pos.piece_array();

    let mut active = ActiveThreatList::new();
    full_threats::append_active_indices_of(perspective, pos, &mut active);
    // The pairs of the entry's pawns that the board's lack, and those of the
    // board's that the entry's lack, in lists of their own: at most 96 each,
    // the most pairs a board holds, but with the threats more than one list
    // holds. An entry of the board's pawns is used as it is, and a board
    // without pawns, which has no pairs, leaves its entry as it is.
    let mut pairs_removed = ActiveThreatList::new();
    let mut pairs_added = ActiveThreatList::new();
    let mut pairs_changed = false;
    let mut pair_acc: *mut i16 = std::ptr::null_mut();
    let pawns = [pos.pieces_cpt(WHITE, PAWN), pos.pieces_cpt(BLACK, PAWN)];
    if (pawns[0] | pawns[1]).any() {
        let pair_entry = &mut cache.pairs[p][king_half(ksq)];
        if pair_entry.pawns != pawns {
            pp_3wide::append_changed_indices(
                perspective,
                ksq,
                &DirtyPawnPairs {
                    before: pair_entry.pawns,
                    after: pawns,
                },
                &mut pairs_removed,
                &mut pairs_added,
                thr_base,
                DIMENSIONS,
            );
            pair_entry.pawns = pawns;
            pairs_changed = true;
        }
        pair_acc = pair_entry.accumulation.as_mut_ptr();
    }
    debug_assert!(
        active
            .iter()
            .chain(pairs_removed.iter())
            .chain(pairs_added.iter())
            .all(|&i| usize::from(i) < THREAT_AND_PAIR_DIMENSIONS)
    );

    accumulator.computed[p] = true;

    let entry_acc = entry.accumulation.as_mut_ptr();
    let to_acc = accumulator.accumulation[p].as_mut_ptr();

    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, the pairs' entry's among them
        // where it is used, and rows of the lists.
        unsafe {
            if pairs_changed {
                let mut acc = load_tile(j, pair_acc);

                apply_changed_features(
                    &mut acc,
                    pairs_removed.as_slice(),
                    pairs_added.as_slice(),
                    thr_base.add(j),
                );

                store_tile(j, pair_acc, &acc);
            }

            let mut acc = load_tile(j, entry_acc);

            apply_changed_features(
                &mut acc,
                removed.as_slice(),
                added.as_slice(),
                psq_weights.add(j),
            );

            store_tile(j, entry_acc, &acc);

            apply_features::<1>(&mut acc, active.as_slice(), thr_base.add(j));
            if !pair_acc.is_null() {
                apply::<1>(&mut acc, pair_acc.add(j));
            }

            store_tile(j, to_acc, &acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::misc::Prng;
    use crate::movegen::{ALL, MoveList};
    use crate::nnue::features::full_threats::ThreatBoard;

    /// A network whose feature transformer has random biases and weights,
    /// over their whole range, so that a feature missed or counted twice
    /// shows in the accumulations.
    fn random_network(rng: &mut Prng) -> LargePageBox<Network> {
        let mut network = Network::zeroed(1);
        let ft = &mut network.feature_transformer;
        for values in ft.biases.0.chunks_mut(4) {
            let bits = rng.rand64();
            for (i, value) in values.iter_mut().enumerate() {
                *value = (bits >> (16 * i)) as i16;
            }
        }
        for values in ft
            .weights
            .0
            .chunks_mut(8)
            .chain(ft.threat_weights.0.as_flattened_mut().chunks_mut(8))
        {
            for (value, byte) in values.iter_mut().zip(rng.rand64().to_le_bytes()) {
                *value = byte as i8;
            }
        }
        network
    }

    /// A row of weights in the order of the accumulation's values, which
    /// the rows are not kept in where they widen in pairs.
    fn in_accumulation_order(row: &[i8]) -> Vec<i8> {
        let row = row.to_vec();
        #[cfg(use_paired_widening_add)]
        let row = {
            let mut row = row;
            crate::nnue::feature_transformer::permute(
                &mut row,
                &crate::nnue::feature_transformer::INVERSE_WIDENING_PAIR_ORDER,
            );
            row
        };
        row
    }

    /// The accumulation of a perspective, summed anew from the biases and
    /// the weights of the active features.
    fn summed(ft: &FeatureTransformer, pos: &Position<Atomic>, perspective: Color) -> [i16; L1] {
        let mut accumulation = ft.biases.0;
        let mut add = |row: &[i8]| {
            for (value, &weight) in accumulation.iter_mut().zip(&in_accumulation_order(row)) {
                *value = value.wrapping_add(i16::from(weight));
            }
        };
        let ksq = pos.king_square(perspective);
        let psq_block = k32h3::block_of(pos, perspective);
        for sq in pos.pieces() {
            let index = usize::from(k32h3::make_index(perspective, sq, pos.piece_on(sq), ksq));
            add(&ft.psq_weights(psq_block)[index * L1..][..L1]);
        }
        let mut active = ActiveThreatList::new();
        full_threats::append_active_indices_of(perspective, pos, &mut active);
        pp_3wide::append_active_indices_of(perspective, pos, &mut active);
        for &index in &active {
            add(&ft.threat_weights.0[usize::from(index)]);
        }
        accumulation
    }

    /// Starts, from openings full of captures to endgames
    /// where kings move often, through captures that explode queens and
    /// rooks, promotions, castling and en passant; then middlegames of many
    /// pieces where kings step within their half of the board, which updates
    /// by the king-move path, and crowds of sliders, whose captures change
    /// the most threats.
    const STARTS: [(&str, bool); 14] = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false,
        ),
        (
            "rn2kb1r/1pp1p2p/p2q1pp1/3P4/2P3b1/4PN2/PP3PPP/R2QKB1R b KQkq - 0 1",
            false,
        ),
        (
            "r4b1r/2kb1N2/p2Bpnp1/8/2Pp3p/1P1PPP2/P5PP/R3K2R b KQ - 0 1",
            false,
        ),
        (
            "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1",
            true,
        ),
        ("8/8/8/8/8/8/NkN5/1K6 w - - 0 1", false),
        ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", false),
        ("4K1R1/RR6/8/8/8/8/qqq3p1/7k b - - 0 1", false),
        ("3qk3/8/8/3Q4/8/8/8/4K3 w - - 0 1", false),
        ("4k3/1P6/8/8/8/8/1p2r3/4K3 w - - 0 1", false),
        ("r3k2r/8/8/3q4/3Q4/8/8/R3K2R w KQkq - 0 1", false),
        (
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
            false,
        ),
        (
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            false,
        ),
        (
            "q2rr1k1/1b1n1pbp/1p1p1np1/8/8/1P1P1NP1/1B1N1PBP/Q2RR1K1 w - - 0 1",
            false,
        ),
        ("1k6/2qqq3/2rbr3/8/8/3RBR2/3QQQ2/6K1 w - - 0 1", false),
    ];

    /// Positions without both kings end the game, and are never evaluated.
    fn evaluable(pos: &Position<Atomic>) -> bool {
        pos.has_king(WHITE) && pos.has_king(BLACK)
    }

    /// Along random games, every way of updating gives each perspective's
    /// sums from scratch.
    #[test]
    fn incremental_updates_match_sums_from_scratch() {
        let mut rng = Prng::new(20261104);
        let network = random_network(&mut rng);
        let ft = &network.feature_transformer;
        let mut caches = AccumulatorCaches::new(&network);
        let mut stack = AccumulatorStack::new();
        let mut explosions = 0;
        UPDATES.with(|updates| updates.set([0; 4]));

        let mut check = |stack: &mut AccumulatorStack, pos: &Position<Atomic>| {
            stack.evaluate(pos, ft, &mut caches);
            for perspective in [WHITE, BLACK] {
                assert!(
                    stack.latest().accumulation[perspective.idx()] == summed(ft, pos, perspective),
                    "the accumulation of {} in {}",
                    ["white", "black"][perspective.idx()],
                    pos.fen()
                );
            }
        };

        // As a search does, the walks evaluate some positions along a line,
        // updating from the last one evaluated, and some a move away; half
        // the walks evaluate every position, which updates both sides at
        // once.
        for walk in 0..4 {
            for (fen, chess960) in STARTS {
                let mut pos = Position::<Atomic>::from_fen(fen, chess960).unwrap();
                stack.reset();
                for _ in 0..80 {
                    let moves = MoveList::new::<ALL>(&pos);
                    if moves.is_empty() || !evaluable(&pos) {
                        break;
                    }
                    if walk % 2 == 1 || rng.rand64().is_multiple_of(3) {
                        check(&mut stack, &pos);
                    }
                    for &m in &moves {
                        if rng.rand64().is_multiple_of(4) {
                            let gives_check = pos.gives_check(m);
                            pos.do_move(m, gives_check, stack.push(&pos));
                            if evaluable(&pos) {
                                explosions +=
                                    usize::from(stack.latest().dirties.blast.squares().any());
                                check(&mut stack, &pos);
                            }
                            pos.undo_move(m);
                            stack.pop();
                        }
                    }
                    let m = moves.as_slice()[(rng.rand64() % moves.len() as u64) as usize];
                    let gives_check = pos.gives_check(m);
                    pos.do_move(m, gives_check, stack.push(&pos));
                }
            }
        }

        let updates = UPDATES.with(|updates| updates.get());
        for (path, count) in UPDATE_PATHS.iter().zip(updates) {
            assert!(count > 0, "no {path} updates: {updates:?}");
        }
        assert!(explosions > 0);
    }

    /// Each position of the lines is evaluated after the one before it.
    fn lines_update_both_sides(lines: &[(&str, &str)]) {
        let mut rng = Prng::new(20261105);
        let network = random_network(&mut rng);
        let ft = &network.feature_transformer;
        let mut caches = AccumulatorCaches::new(&network);
        let mut stack = AccumulatorStack::new();

        for &(fen, line) in lines {
            let mut pos = Position::<Atomic>::from_fen(fen, false).unwrap();
            stack.reset();
            stack.evaluate(&pos, ft, &mut caches);
            for uci in line.split(' ') {
                let m = crate::uci::to_move(&pos, uci);
                assert!(m != Move::none(), "{uci} in {}", pos.fen());
                let gives_check = pos.gives_check(m);
                pos.do_move(m, gives_check, stack.push(&pos));
                assert!(evaluable(&pos), "{uci} in {}", pos.fen());
                stack.evaluate(&pos, ft, &mut caches);
                for perspective in [WHITE, BLACK] {
                    assert!(
                        stack.latest().accumulation[perspective.idx()]
                            == summed(ft, &pos, perspective),
                        "the accumulation of {} after {uci} in {}",
                        ["white", "black"][perspective.idx()],
                        pos.fen()
                    );
                }
            }
        }
    }

    /// Kings step within their half among many pieces (the king-move path),
    /// or across it, captures explode queens and rooks, which changes the
    /// other side's block, and queens are made and exploded.
    const KING_AND_BLAST_LINES: [(&str, &str); 5] = [
        (
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
            "g1h1 g8h8 h1g1 c6d4 f3d4",
        ),
        (
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "e1f1 e8f8 f1e1 f8e8 e1d1 e8d8 c4e6",
        ),
        (
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "e1g1 e8c8 e3c5 f6e4",
        ),
        (
            "1k6/2qqq3/2rbr3/8/8/3RBR2/3QQQ2/6K1 w - - 0 1",
            "e3g5 b8a8 d3d6 a8b8",
        ),
        (
            "8/1P6/8/3k4/8/8/6p1/4K3 w - - 0 1",
            "b7b8q g2g1q e1e2 g1b1 b8b1",
        ),
    ];

    /// Pawns pushed two squares, taking pawns and taken by pawns and by
    /// pieces, en passant, beside blasts that spare them and take a knight,
    /// and promoted, with a capture and without.
    const PAWN_LINES: [(&str, &str); 4] = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e2e4 e7e5 g1f3 b8c6 f3e5 d7d5 e4d5 c7c5 d2d4 c5d4",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e2e4 a7a6 e4e5 d7d5 e5d6 e7e5 d2d4 e5d4",
        ),
        (
            "r3k3/1P6/8/8/8/8/6p1/4K2R w - - 0 1",
            "b7a8q g2h1q e1d2 e8d7",
        ),
        ("8/1P6/8/3k4/8/8/6p1/4K3 w - - 0 1", "b7b8n g2g1r"),
    ];

    #[test]
    fn lines_of_kings_blasts_and_pawns_update_both_sides() {
        lines_update_both_sides(&KING_AND_BLAST_LINES);
        lines_update_both_sides(&PAWN_LINES);
    }

    /// Caches cleared for another network forget the pawn pairs' sums of the
    /// one before, which the same pawns would otherwise reuse.
    #[test]
    fn caches_cleared_for_another_network_forget_its_pairs() {
        let mut rng = Prng::new(20261107);
        let before = random_network(&mut rng);
        let network = random_network(&mut rng);
        let ft = &network.feature_transformer;
        let mut caches = AccumulatorCaches::new(&before);
        let mut stack = AccumulatorStack::new();
        let pos = Position::<Atomic>::from_fen(STARTS[0].0, false).unwrap();

        stack.evaluate(&pos, &before.feature_transformer, &mut caches);
        caches.clear(&network);
        stack.reset();
        stack.evaluate(&pos, ft, &mut caches);
        for perspective in [WHITE, BLACK] {
            assert!(
                stack.latest().accumulation[perspective.idx()] == summed(ft, &pos, perspective)
            );
        }
    }

    /// `pos` as the features read it with the kings on the squares given,
    /// which orient the features.
    struct KingsOn<'a>(&'a Position<Atomic>, [Square; COLOR_NB]);

    impl ThreatBoard for KingsOn<'_> {
        fn king_square(&self, c: Color) -> Square {
            self.1[c.idx()]
        }

        fn pieces(&self) -> Bitboard {
            self.0.pieces()
        }

        fn pieces_pt(&self, pt: PieceType) -> Bitboard {
            self.0.pieces_pt(pt)
        }

        fn pieces_pt2(&self, pt1: PieceType, pt2: PieceType) -> Bitboard {
            self.0.pieces_pt2(pt1, pt2)
        }

        fn pieces_cpt(&self, c: Color, pt: PieceType) -> Bitboard {
            self.0.pieces_cpt(c, pt)
        }

        fn piece_on(&self, s: Square) -> Piece {
            self.0.piece_on(s)
        }
    }

    /// How many times each threat feature of `perspective` is active.
    fn active(board: &KingsOn, perspective: Color) -> HashMap<u16, i32> {
        let mut list = ActiveThreatList::new();
        full_threats::append_active_indices_of(perspective, board, &mut list);
        let mut counts = HashMap::new();
        for &index in &list {
            *counts.entry(index).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn moves_record_the_threats_and_pawns_they_change() {
        let mut rng = Prng::new(20261106);
        let mut stack = AccumulatorStack::new();
        let mut most = 0;
        // Moves that changed the pawns, without a capture and with one
        let mut pawn_moves = [0; 2];

        for _ in 0..4 {
            for (fen, chess960) in STARTS {
                let mut pos = Position::<Atomic>::from_fen(fen, chess960).unwrap();
                stack.reset();
                for _ in 0..80 {
                    let moves = MoveList::new::<ALL>(&pos);
                    if moves.is_empty() || !evaluable(&pos) {
                        break;
                    }
                    for &m in &moves {
                        let gives_check = pos.gives_check(m);
                        let before = pos.clone();
                        pos.do_move(m, gives_check, stack.push(&pos));
                        // The pawns before the move and after it
                        let pawns = |pos: &Position<Atomic>| {
                            [pos.pieces_cpt(WHITE, PAWN), pos.pieces_cpt(BLACK, PAWN)]
                        };
                        let pairs = stack.latest().pawn_pairs();
                        assert_eq!(pairs.before, pawns(&before), "{m:?} in {}", before.fen());
                        assert_eq!(pairs.after, pawns(&pos), "{m:?} in {}", before.fen());
                        let changed = pairs.before != pairs.after;
                        pawn_moves[usize::from(pos.captured_piece().is_some())] +=
                            usize::from(changed);
                        let threats = &stack.latest().dirty_threats;
                        most = most.max(threats.len());
                        // A capture changes at most 320, as the list counts.
                        assert!(threats.len() <= 320, "{} after {m:?}", threats.len());
                        if evaluable(&pos) {
                            // The features of both sides, oriented by the
                            // kings after the move, before and after it
                            let kings = [WHITE, BLACK].map(|c| pos.king_square(c));
                            for perspective in [WHITE, BLACK] {
                                let mut counts = active(&KingsOn(&before, kings), perspective);
                                let mut removed = ChangedThreatList::new();
                                let mut added = ChangedThreatList::new();
                                let row = [0i8; 1];
                                full_threats::append_changed_indices(
                                    perspective,
                                    kings[perspective.idx()],
                                    threats,
                                    &mut removed,
                                    &mut added,
                                    row.as_ptr(),
                                    0,
                                );
                                for &index in &removed {
                                    *counts.entry(index).or_insert(0) -= 1;
                                }
                                for &index in &added {
                                    *counts.entry(index).or_insert(0) += 1;
                                }
                                counts.retain(|_, n| *n != 0);
                                assert_eq!(
                                    counts,
                                    active(&KingsOn(&pos, kings), perspective),
                                    "{} after {m:?} from {}",
                                    ["white", "black"][perspective.idx()],
                                    before.fen()
                                );
                            }
                        }
                        pos.undo_move(m);
                        stack.pop();
                    }
                    let m = moves.as_slice()[(rng.rand64() % moves.len() as u64) as usize];
                    let gives_check = pos.gives_check(m);
                    pos.do_move(m, gives_check, stack.push(&pos));
                }
            }
        }
        // Blasts that change many threats were among the moves: 38 at most
        // in these walks, far below the list's bound.
        assert!(most > 30, "at most {most} threats changed");
        assert!(pawn_moves.iter().all(|&n| n > 100), "{pawn_moves:?}");
    }

    #[test]
    fn other_networks_record_no_threats() {
        let mut pos = Position::<Atomic>::from_fen(STARTS[13].0, false).unwrap();
        let m = crate::uci::to_move(&pos, "d3d6");
        let mut dirties = AtomicDirties::default();
        pos.do_move(m, pos.gives_check(m), &mut dirties);
        assert!(dirties.threats == ThreatRecord::default());
        assert!(dirties.blast.squares().any());
    }
}
