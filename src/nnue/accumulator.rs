//! The accumulators: for each position of the line being searched, the sums
//! of the feature transformer's weights over the active features of each
//! perspective, each brought up to date from another rather than summed
//! anew, and the caches of piece-square sums by king square that full
//! refreshes start from.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/nnue_accumulator.h` and `.cpp`: Stockfish, a UCI chess playing
//! engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

// The tile functions are unsafe for their pointers; the operations they
// call are unsafe only on some platforms.
#![allow(unused_unsafe)]

#[cfg(use_neon)]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::architecture::L1;
use super::feature_transformer::{
    FeatureTransformer, OUTPUT_DIMENSIONS, PSQ_DIMENSIONS, THREAT_AND_PP_DIMENSIONS,
};
use super::features::{full_threats, k32q2, pp_3wide};
use super::network::Network;
#[cfg(not(any(use_neon, use_paired_widening_add)))]
use super::simd::vec_load_convert_8_16;
use super::simd::{
    LANES, NUM_REGS, TILE_HEIGHT, VecT, vec_add_16, vec_load, vec_store, vec_sub_16,
};
#[cfg(use_paired_widening_add)]
use super::simd::{
    vec_maddubs_16, vec_set_8, vec_srai_16, vec_unpackhi_8, vec_unpacklo_8, vec_zero,
};
use crate::bitboard::{Bitboard, square_bb};
use crate::memory::LargePageBox;
#[cfg(use_paired_widening_add)]
use crate::misc::ValueList;
use crate::position::Position;
use crate::types::*;

type PsqIndexList = k32q2::IndexList;
/// Threat and pawn pair indices, which share the threat weights' rows.
type ThreatIndexList = full_threats::IndexList;

const DIMENSIONS: usize = OUTPUT_DIMENSIONS;

/// The result of the affine transformation of a position's input features
/// for both perspectives, which of them are up to date, and what the move
/// to the position changed.
#[repr(C, align(64))]
pub struct AccumulatorState {
    pub accumulation: [[i16; L1]; COLOR_NB],
    pub computed: [bool; COLOR_NB],
    pub dirties: Dirties,
}

/// AccumulatorCaches provides per-thread accumulator caches, where each
/// cache contains multiple entries for each of the possible king squares.
/// When the accumulator needs to be refreshed, the cached entry is used to
/// more efficiently update the accumulator, instead of rebuilding it from
/// scratch. This idea was first described by Luecx (author of Koivisto) and
/// is commonly referred to as "Finny Tables".
pub struct AccumulatorCaches {
    /// The entries by whether the perspective's opponent has a queen, then
    /// by king square
    entries: [[[Entry; COLOR_NB]; SQUARE_NB]; 2],
}

#[repr(C, align(64))]
struct Entry {
    accumulation: [i16; L1],
    pieces: [Piece; SQUARE_NB],
    piece_bb: Bitboard,
}

impl Entry {
    /// To initialize a refresh entry, we set all its bitboards empty, so we
    /// put the biases in the accumulation, without any weights on top
    fn clear(&mut self, biases: &[i16; L1]) {
        self.accumulation = *biases;
        self.pieces = [NO_PIECE; SQUARE_NB];
        self.piece_bb = Bitboard(0);
    }
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
        root.dirties.dirty_piece = DirtyPiece::default();
        root.dirties.dirty_threats.clear();
        root.dirties.dirty_pawn_pairs = DirtyPawnPairs::default();
        root.computed = [false; COLOR_NB];
        self.size = 1;
    }

    /// Adds the state of the position after a move, whose changes the move
    /// records in the dirties returned.
    #[inline(always)]
    pub fn push(&mut self) -> &mut Dirties {
        debug_assert!(self.size < Self::MAX_SIZE);
        let st = &mut self.accumulators[self.size];
        st.computed = [false; COLOR_NB];
        st.dirties.dirty_threats.clear();
        self.size += 1;
        &mut st.dirties
    }

    #[inline(always)]
    pub fn pop(&mut self) {
        debug_assert!(self.size > 1);
        self.size -= 1;
    }

    /// Brings the latest state up to date for both perspectives.
    pub fn evaluate(
        &mut self,
        pos: &Position,
        feature_transformer: &FeatureTransformer,
        cache: &mut AccumulatorCaches,
    ) {
        let last_white = self.find_last_usable_accumulator(WHITE);
        let last_black = self.find_last_usable_accumulator(BLACK);

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
        pos: &Position,
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
            // before it.
            if dirty_piece.pc == make_piece(perspective, KING)
                && dirty_piece.remove_pc != make_piece(!perspective, QUEEN)
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
    /// accumulator or the accumulator state just before a change that
    /// requires full refresh.
    fn find_last_usable_accumulator(&self, perspective: Color) -> usize {
        for curr_idx in (1..self.size).rev() {
            let state = &self.accumulators[curr_idx];
            if state.computed[perspective.idx()] {
                return curr_idx;
            }

            // Threat feature set refreshes require a king move across the
            // center, i.e., a subset of halfka refreshes
            if k32q2::requires_refresh(&state.dirties.dirty_piece, perspective) {
                return curr_idx;
            }
        }

        0
    }

    fn forward_update_incremental(
        &mut self,
        perspective: Color,
        pos: &Position,
        feature_transformer: &FeatureTransformer,
        begin: usize,
    ) {
        debug_assert!(begin < self.size);
        debug_assert!(self.accumulators[begin].computed[perspective.idx()]);

        let ksq = pos.king_square(perspective);
        let opponent_has_queen = pos.pieces_cpt(!perspective, QUEEN).any();

        for next in begin + 1..self.size {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next - 1);
            update_accumulator_incremental::<true>(
                perspective,
                feature_transformer,
                ksq,
                opponent_has_queen,
                target,
                computed,
            );
        }

        debug_assert!(self.latest().computed[perspective.idx()]);
    }

    fn backward_update_incremental(
        &mut self,
        perspective: Color,
        pos: &Position,
        feature_transformer: &FeatureTransformer,
        end: usize,
    ) {
        debug_assert!(end < self.size);
        debug_assert!(self.latest().computed[perspective.idx()]);

        let ksq = pos.king_square(perspective);
        let opponent_has_queen = pos.pieces_cpt(!perspective, QUEEN).any();

        for next in (end..self.size - 1).rev() {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next + 1);
            update_accumulator_incremental::<false>(
                perspective,
                feature_transformer,
                ksq,
                opponent_has_queen,
                target,
                computed,
            );
        }

        debug_assert!(self.accumulators[end].computed[perspective.idx()]);
    }

    fn forward_update_incremental_both(
        &mut self,
        pos: &Position,
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
        let white_opp_queen = pos.pieces_cpt(BLACK, QUEEN).any();
        let black_opp_queen = pos.pieces_cpt(WHITE, QUEEN).any();
        let shared_begin = white_begin.max(black_begin);

        // Catch up the lagging perspective, then traverse the common suffix
        // once.
        for next in white_begin + 1..=shared_begin {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next - 1);
            update_accumulator_incremental::<true>(
                WHITE,
                feature_transformer,
                white_ksq,
                white_opp_queen,
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
                black_opp_queen,
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
                white_opp_queen,
                black_opp_queen,
                target,
                computed,
            );
        }

        debug_assert!(self.latest().computed[WHITE.idx()]);
        debug_assert!(self.latest().computed[BLACK.idx()]);
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

impl crate::nnue::Accumulators for AccumulatorStack {
    type Dirties = Dirties;

    #[inline(always)]
    fn new() -> LargePageBox<Self> {
        AccumulatorStack::new()
    }

    #[inline(always)]
    fn reset(&mut self) {
        AccumulatorStack::reset(self);
    }

    #[inline(always)]
    fn push(&mut self) -> &mut Dirties {
        AccumulatorStack::push(self)
    }

    #[inline(always)]
    fn pop(&mut self) {
        AccumulatorStack::pop(self);
    }
}

impl crate::nnue::Caches for AccumulatorCaches {
    type Network = Network;

    fn new(network: Option<&Network>) -> LargePageBox<Self> {
        match network {
            Some(network) => AccumulatorCaches::new(network),
            None => AccumulatorCaches::empty(),
        }
    }

    fn clear(&mut self, network: &Network) {
        AccumulatorCaches::clear(self, network);
    }
}

// The accumulations are updated a tile at a time, the vectors of each kept
// in registers while every weight row is applied to it. The functions on
// tiles take the index `j` of the tile's first value, a multiple of
// TILE_HEIGHT below DIMENSIONS, and pointers to accumulations or weight rows
// of DIMENSIONS values; the index lists hold indices of weight rows.

const _: () = assert!(DIMENSIONS.is_multiple_of(TILE_HEIGHT));

type Tile = [VecT; NUM_REGS];

#[inline(always)]
unsafe fn load_tile(j: usize, data: *const i16) -> Tile {
    std::array::from_fn(|k| unsafe { vec_load(data.add(j + k * LANES)) })
}

#[inline(always)]
unsafe fn store_tile(j: usize, dest: *mut i16, acc: &Tile) {
    for (k, &value) in acc.iter().enumerate() {
        unsafe { vec_store(dest.add(j + k * LANES), value) };
    }
}

/// Adds (SIGN 1) or subtracts (SIGN -1) the tile of `data`.
#[inline(always)]
unsafe fn apply<const SIGN: i32>(j: usize, acc: &mut Tile, data: *const i16) {
    const { assert!(SIGN == 1 || SIGN == -1) };
    for (k, acc) in acc.iter_mut().enumerate() {
        let column = unsafe { vec_load(data.add(j + k * LANES)) };
        *acc = if SIGN == 1 {
            vec_add_16(*acc, column)
        } else {
            vec_sub_16(*acc, column)
        };
    }
}

/// Adds or subtracts the tile of a row of 8-bit weights.
#[inline(always)]
unsafe fn apply_feature_single<const SIGN: i32>(j: usize, acc: &mut Tile, data: *const i8) {
    const { assert!(SIGN == 1 || SIGN == -1) };
    let column = unsafe { data.add(j) };

    // A vector of 16 weights widens into two of the accumulators'.
    #[cfg(use_neon)]
    for pair in 0..NUM_REGS / 2 {
        let k = 2 * pair;
        unsafe {
            let weight = vld1q_s8(column.add(k * LANES));
            if SIGN == 1 {
                acc[k] = vaddw_s8(acc[k], vget_low_s8(weight));
                acc[k + 1] = vaddw_high_s8(acc[k + 1], weight);
            } else {
                acc[k] = vsubw_s8(acc[k], vget_low_s8(weight));
                acc[k + 1] = vsubw_high_s8(acc[k + 1], weight);
            }
        }
    }

    // A vector of weights widens into two of the accumulators': each byte
    // goes to the high byte of its lane, which an arithmetic shift brings
    // down with its sign. The weights are permuted at load time for it.
    #[cfg(use_paired_widening_add)]
    {
        const { assert!(NUM_REGS.is_multiple_of(2)) };
        let zero = vec_zero();
        unrolled::<{ NUM_REGS / 2 }>(|pair| {
            let k = 2 * pair;
            let weight = unsafe { vec_load(column.add(k * LANES).cast()) };
            let lo = vec_srai_16::<8>(vec_unpacklo_8(zero, weight));
            let hi = vec_srai_16::<8>(vec_unpackhi_8(zero, weight));
            if SIGN == 1 {
                acc[k] = vec_add_16(acc[k], lo);
                acc[k + 1] = vec_add_16(acc[k + 1], hi);
            } else {
                acc[k] = vec_sub_16(acc[k], lo);
                acc[k + 1] = vec_sub_16(acc[k + 1], hi);
            }
        });
    }

    #[cfg(not(any(use_neon, use_paired_widening_add)))]
    for (k, acc) in acc.iter_mut().enumerate() {
        let weight = unsafe { vec_load_convert_8_16(column.add(k * LANES)) };
        *acc = if SIGN == 1 {
            vec_add_16(*acc, weight)
        } else {
            vec_sub_16(*acc, weight)
        };
    }
}

/// Adds or subtracts the tiles of two rows of 8-bit weights at once: their
/// bytes interleaved, each pair is summed into an i16 lane by maddubs
/// against a vector of ones, which takes one shuffle per two rows instead of
/// one per row.
#[cfg(use_paired_widening_add)]
#[inline(always)]
unsafe fn apply_feature_pair<const SIGN: i32>(
    j: usize,
    acc: &mut Tile,
    data0: *const i8,
    data1: *const i8,
) {
    const { assert!(SIGN == 1 || SIGN == -1) };
    const { assert!(NUM_REGS.is_multiple_of(2)) };
    let ones = vec_set_8(1);
    unrolled::<{ NUM_REGS / 2 }>(|pair| {
        let k = 2 * pair;
        let (w0, w1) = unsafe {
            (
                vec_load(data0.add(j + k * LANES).cast()),
                vec_load(data1.add(j + k * LANES).cast()),
            )
        };
        let lo = vec_maddubs_16(ones, vec_unpacklo_8(w0, w1));
        let hi = vec_maddubs_16(ones, vec_unpackhi_8(w0, w1));
        if SIGN == 1 {
            acc[k] = vec_add_16(acc[k], lo);
            acc[k + 1] = vec_add_16(acc[k + 1], hi);
        } else {
            acc[k] = vec_sub_16(acc[k], lo);
            acc[k + 1] = vec_sub_16(acc[k + 1], hi);
        }
    });
}

/// Calls `f` for each of `0..N`, N at most 16, in a sequence of calls
/// rather than a loop: LLVM leaves the loops over the pairs of an AVX-512
/// tile rolled, which keeps the tile in memory instead of in registers.
#[cfg(use_paired_widening_add)]
#[inline(always)]
fn unrolled<const N: usize>(mut f: impl FnMut(usize)) {
    const { assert!(N <= 16) };
    macro_rules! calls {
        ($($k:literal)*) => {
            $(if $k < N {
                f($k);
            })*
        };
    }
    calls!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
}

/// The int8 weight rows (piece-square, threat and pawn-pair features alike)
/// that have to be added to or removed from an accumulator. Resolving the
/// feature indices to row pointers once, outside the tile loop, also lets
/// rows of different feature sets be processed together in pairs.
#[cfg(use_paired_widening_add)]
struct WeightRows<const MAX_ROWS: usize>(ValueList<*const i8, MAX_ROWS>);

#[cfg(use_paired_widening_add)]
type PsqWeightRows = WeightRows<{ k32q2::MAX_ACTIVE_DIMENSIONS }>;
/// Room for the rows of both feature sets, rounded up to a power of two as
/// a ValueList's capacity is.
#[cfg(use_paired_widening_add)]
type AllWeightRows = WeightRows<
    { (k32q2::MAX_ACTIVE_DIMENSIONS + full_threats::MAX_ACTIVE_DIMENSIONS).next_power_of_two() },
>;
#[cfg(use_paired_widening_add)]
type IncrementalWeightRows = AllWeightRows;

/// Without paired widening, gathering the rows gains nothing and costs
/// time, so they stay in their lists of indices, which the tile loop
/// resolves as it did before Stockfish's e08dfa80. An incremental update
/// changes one or two piece-square features.
#[cfg(not(use_paired_widening_add))]
struct WeightRows<'a, const INCREMENTAL: bool = false> {
    psq: Option<(&'a PsqIndexList, *const i8)>,
    threats: Option<(&'a ThreatIndexList, *const i8)>,
}

#[cfg(not(use_paired_widening_add))]
type PsqWeightRows<'a> = WeightRows<'a>;
#[cfg(not(use_paired_widening_add))]
type AllWeightRows<'a> = WeightRows<'a>;
#[cfg(not(use_paired_widening_add))]
type IncrementalWeightRows<'a> = WeightRows<'a, true>;

#[cfg(not(use_paired_widening_add))]
impl<'a, const INCREMENTAL: bool> WeightRows<'a, INCREMENTAL> {
    #[inline(always)]
    fn new() -> Self {
        WeightRows {
            psq: None,
            threats: None,
        }
    }

    #[inline(always)]
    fn append_psq(&mut self, list: &'a PsqIndexList, feature_transformer: &FeatureTransformer) {
        debug_assert!(self.psq.is_none());
        self.psq = Some((list, feature_transformer.weights.0.as_ptr()));
    }

    #[inline(always)]
    fn append_threat(
        &mut self,
        list: &'a ThreatIndexList,
        feature_transformer: &FeatureTransformer,
    ) {
        debug_assert!(self.threats.is_none());
        self.threats = Some((list, feature_transformer.threat_and_pp_weights.0.as_ptr()));
    }

    /// Adds (SIGN 1) or subtracts (SIGN -1) the tile of each row.
    #[inline(always)]
    unsafe fn apply<const SIGN: i32>(&self, j: usize, acc: &mut Tile) {
        const { assert!(SIGN == 1 || SIGN == -1) };
        if let Some((list, weights)) = self.psq {
            let row = |index: u16| {
                debug_assert!(usize::from(index) < PSQ_DIMENSIONS);
                // SAFETY: the index is a feature's, whose row is in the weights.
                unsafe { weights.add(usize::from(index) * DIMENSIONS) }
            };
            if INCREMENTAL {
                debug_assert!(list.len() == 1 || list.len() == 2);
                if let [first, rest @ ..] = list.as_slice() {
                    unsafe { apply_feature_single::<SIGN>(j, acc, row(*first)) };
                    if let [second, ..] = rest {
                        unsafe { apply_feature_single::<SIGN>(j, acc, row(*second)) };
                    }
                }
            } else {
                for &index in list {
                    unsafe { apply_feature_single::<SIGN>(j, acc, row(index)) };
                }
            }
        }
        if let Some((list, weights)) = self.threats {
            for &index in list {
                debug_assert!(usize::from(index) < THREAT_AND_PP_DIMENSIONS);
                // SAFETY: as for the piece-square rows.
                let row = unsafe { weights.add(usize::from(index) * DIMENSIONS) };
                unsafe { apply_feature_single::<SIGN>(j, acc, row) };
            }
        }
    }
}

#[cfg(use_paired_widening_add)]
impl<const MAX_ROWS: usize> WeightRows<MAX_ROWS> {
    #[inline(always)]
    fn new() -> Self {
        WeightRows(ValueList::new())
    }

    #[inline(always)]
    fn append_psq(&mut self, list: &PsqIndexList, feature_transformer: &FeatureTransformer) {
        debug_assert!(self.0.len() + list.len() <= MAX_ROWS);
        let weights = feature_transformer.weights.0.as_ptr();
        for &index in list {
            debug_assert!(usize::from(index) < PSQ_DIMENSIONS);
            // SAFETY: the index is a feature's, whose row is in the weights.
            let row = unsafe { weights.add(usize::from(index) * DIMENSIONS) };
            self.0.push(row);
        }
    }

    #[inline(always)]
    fn append_threat(&mut self, list: &ThreatIndexList, feature_transformer: &FeatureTransformer) {
        debug_assert!(self.0.len() + list.len() <= MAX_ROWS);
        let weights = feature_transformer.threat_and_pp_weights.0.as_ptr();
        for &index in list {
            debug_assert!(usize::from(index) < THREAT_AND_PP_DIMENSIONS);
            // SAFETY: as for append_psq.
            let row = unsafe { weights.add(usize::from(index) * DIMENSIONS) };
            self.0.push(row);
        }
    }

    /// Adds (SIGN 1) or subtracts (SIGN -1) the tile of each row.
    #[inline(always)]
    unsafe fn apply<const SIGN: i32>(&self, j: usize, acc: &mut Tile) {
        const { assert!(SIGN == 1 || SIGN == -1) };
        let (pairs, rest) = self.0.as_slice().as_chunks::<2>();
        for &[row0, row1] in pairs {
            unsafe { apply_feature_pair::<SIGN>(j, acc, row0, row1) };
        }
        for &row in rest {
            unsafe { apply_feature_single::<SIGN>(j, acc, row) };
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_combined(
    feature_transformer: &FeatureTransformer,
    from_acc: &[i16; DIMENSIONS],
    to_acc: &mut [i16; DIMENSIONS],
    psq_added: &PsqIndexList,
    psq_removed: &PsqIndexList,
    thr_added: &ThreatIndexList,
    thr_removed: &ThreatIndexList,
) {
    let from = from_acc.as_ptr();
    let to = to_acc.as_mut_ptr();

    let mut removed = IncrementalWeightRows::new();
    let mut added = IncrementalWeightRows::new();
    removed.append_psq(psq_removed, feature_transformer);
    removed.append_threat(thr_removed, feature_transformer);
    added.append_psq(psq_added, feature_transformer);
    added.append_threat(thr_added, feature_transformer);

    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, and rows of the lists.
        unsafe {
            let mut acc = load_tile(j, from);

            removed.apply::<-1>(j, &mut acc);
            added.apply::<1>(j, &mut acc);

            store_tile(j, to, &acc);
        }
    }
}

fn update_accumulator_incremental<const FORWARD: bool>(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    ksq: Square,
    opponent_has_queen: bool,
    target_state: &mut AccumulatorState,
    computed: &AccumulatorState,
) {
    let p = perspective.idx();
    debug_assert!(computed.computed[p]);
    debug_assert!(!target_state.computed[p]);

    // The size must be enough to contain the largest possible update.
    // That might depend on the feature set and generally relies on the
    // feature set's update cost calculation to be correct and never allow
    // updates with more added/removed features than MaxActiveDimensions.
    let mut psq_removed = PsqIndexList::new();
    let mut psq_added = PsqIndexList::new();
    let mut thr_removed = ThreatIndexList::new();
    let mut thr_added = ThreatIndexList::new();

    let dirties = if FORWARD {
        &target_state.dirties
    } else {
        &computed.dirties
    };

    // Used solely for prefetching
    let threat_pp_base = feature_transformer.threat_and_pp_weights.0.as_ptr();
    let pf_stride = DIMENSIONS;

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
        &dirties.dirty_threats,
        thr_out,
        thr_in,
        threat_pp_base,
        pf_stride,
    );
    pp_3wide::append_changed_indices(
        perspective,
        ksq,
        &dirties.dirty_pawn_pairs,
        thr_out,
        thr_in,
        threat_pp_base,
        pf_stride,
    );
    k32q2::append_changed_indices(
        perspective,
        ksq,
        &dirties.dirty_piece,
        opponent_has_queen,
        psq_out,
        psq_in,
    );

    apply_combined(
        feature_transformer,
        &computed.accumulation[p],
        &mut target_state.accumulation[p],
        &psq_added,
        &psq_removed,
        &thr_added,
        &thr_removed,
    );

    target_state.computed[p] = true;
}

fn update_accumulator_incremental_both(
    feature_transformer: &FeatureTransformer,
    white_ksq: Square,
    black_ksq: Square,
    white_opp_queen: bool,
    black_opp_queen: bool,
    target_state: &mut AccumulatorState,
    computed: &AccumulatorState,
) {
    debug_assert!(computed.computed[WHITE.idx()]);
    debug_assert!(computed.computed[BLACK.idx()]);
    debug_assert!(!target_state.computed[WHITE.idx()]);
    debug_assert!(!target_state.computed[BLACK.idx()]);

    let mut psq_removed = [PsqIndexList::new(); COLOR_NB];
    let mut psq_added = [PsqIndexList::new(); COLOR_NB];
    let mut thr_removed = [ThreatIndexList::new(); COLOR_NB];
    let mut thr_added = [ThreatIndexList::new(); COLOR_NB];

    let threat_pp_base = feature_transformer.threat_and_pp_weights.0.as_ptr();
    let pf_stride = DIMENSIONS;

    let dirties = &target_state.dirties;
    {
        let [white_removed, black_removed] = &mut thr_removed;
        let [white_added, black_added] = &mut thr_added;
        full_threats::append_changed_indices_both(
            white_ksq,
            black_ksq,
            &dirties.dirty_threats,
            white_removed,
            white_added,
            black_removed,
            black_added,
            threat_pp_base,
            pf_stride,
        );
        pp_3wide::append_changed_indices_both(
            white_ksq,
            black_ksq,
            &dirties.dirty_pawn_pairs,
            white_removed,
            white_added,
            black_removed,
            black_added,
            threat_pp_base,
            pf_stride,
        );
    }
    for (perspective, ksq, opponent_has_queen) in [
        (WHITE, white_ksq, white_opp_queen),
        (BLACK, black_ksq, black_opp_queen),
    ] {
        let p = perspective.idx();
        k32q2::append_changed_indices(
            perspective,
            ksq,
            &dirties.dirty_piece,
            opponent_has_queen,
            &mut psq_removed[p],
            &mut psq_added[p],
        );
    }

    for p in [WHITE.idx(), BLACK.idx()] {
        apply_combined(
            feature_transformer,
            &computed.accumulation[p],
            &mut target_state.accumulation[p],
            &psq_added[p],
            &psq_removed[p],
            &thr_added[p],
            &thr_removed[p],
        );
    }

    target_state.computed = [true; COLOR_NB];
}

/// The squares whose pieces differ between two boards.
#[inline(always)]
pub(super) fn get_changed_pieces(
    old_pieces: &[Piece; SQUARE_NB],
    new_pieces: &[Piece; SQUARE_NB],
) -> Bitboard {
    const { assert!(size_of::<Piece>() == 1) };

    #[cfg(use_avx2)]
    // SAFETY: each load reads 32 of the 64 bytes.
    unsafe {
        let mut same_bb = 0u64;
        for i in [0, 32] {
            let old_v = _mm256_loadu_si256(old_pieces.as_ptr().add(i).cast());
            let new_v = _mm256_loadu_si256(new_pieces.as_ptr().add(i).cast());
            let cmp_equal = _mm256_cmpeq_epi8(old_v, new_v);
            let equal_mask = _mm256_movemask_epi8(cmp_equal) as u32;
            same_bb |= u64::from(equal_mask) << i;
        }
        Bitboard(!same_bb)
    }

    #[cfg(use_neon)]
    // SAFETY: the loads read the 64 bytes, de-interleaved by 4.
    unsafe {
        let old_v = vld4q_u8(old_pieces.as_ptr().cast());
        let new_v = vld4q_u8(new_pieces.as_ptr().cast());

        let cmp0_1 = vsriq_n_u8::<1>(vceqq_u8(old_v.1, new_v.1), vceqq_u8(old_v.0, new_v.0));
        let cmp2_3 = vsriq_n_u8::<1>(vceqq_u8(old_v.3, new_v.3), vceqq_u8(old_v.2, new_v.2));
        let merged = vsriq_n_u8::<2>(cmp2_3, cmp0_1);
        let merged = vsriq_n_u8::<4>(merged, merged);
        let same_bb = vshrn_n_u16::<4>(vreinterpretq_u16_u8(merged));

        Bitboard(!vget_lane_u64::<0>(vreinterpret_u64_u8(same_bb)))
    }

    #[cfg(all(target_arch = "x86_64", not(use_avx2)))]
    // SAFETY: x86-64 has SSE2; each load reads 16 of the 64 bytes.
    unsafe {
        let mut same_bb = 0u64;
        for i in [0, 16, 32, 48] {
            let old_v = _mm_loadu_si128(old_pieces.as_ptr().add(i).cast());
            let new_v = _mm_loadu_si128(new_pieces.as_ptr().add(i).cast());
            let same = _mm_cmpeq_epi8(old_v, new_v);
            same_bb |= u64::from(_mm_movemask_epi8(same) as u32) << i;
        }
        Bitboard(!same_bb)
    }

    #[cfg(not(any(use_avx2, use_neon, target_arch = "x86_64")))]
    get_changed_pieces_scalar(old_pieces, new_pieces)
}

#[cfg(any(test, not(any(use_avx2, use_neon, target_arch = "x86_64"))))]
fn get_changed_pieces_scalar(
    old_pieces: &[Piece; SQUARE_NB],
    new_pieces: &[Piece; SQUARE_NB],
) -> Bitboard {
    let mut changed = 0u64;
    for sq in 0..SQUARE_NB {
        changed |= u64::from(old_pieces[sq] != new_pieces[sq]) << sq;
    }
    Bitboard(changed)
}

/// Updates accumulator for a king move, and also updates the accumulator
/// cache for the new king position.
fn update_accumulator_hybrid(
    perspective: Color,
    pos: &Position,
    feature_transformer: &FeatureTransformer,
    target: &mut AccumulatorState,
    computed: &AccumulatorState,
    cache: &mut AccumulatorCaches,
) {
    let p = perspective.idx();
    let dirty_piece = target.dirties.dirty_piece;

    debug_assert!(dirty_piece.pc == make_piece(perspective, KING));
    debug_assert!(dirty_piece.to != Square::NONE);
    debug_assert!((dirty_piece.from.0 & 0b100) == (dirty_piece.to.0 & 0b100));
    debug_assert!(computed.computed[p]);
    debug_assert!(!target.computed[p]);

    let old_ksq = dirty_piece.from;
    let new_ksq = dirty_piece.to;
    debug_assert!(old_ksq != new_ksq);

    let current_pieces = pos.piece_array();
    let mut previous_pieces = *current_pieces; // copies 64 bytes!

    let mut previous_piece_bb = pos.pieces();

    debug_assert!(previous_pieces[new_ksq.idx()] == dirty_piece.pc);

    if dirty_piece.remove_sq != Square::NONE {
        debug_assert!(dirty_piece.remove_sq == new_ksq);
        previous_pieces[new_ksq.idx()] = dirty_piece.remove_pc;
    } else {
        previous_pieces[new_ksq.idx()] = NO_PIECE;
        previous_piece_bb &= !square_bb(new_ksq);
    }

    debug_assert!(previous_pieces[old_ksq.idx()] == NO_PIECE);
    previous_pieces[old_ksq.idx()] = make_piece(perspective, KING);
    previous_piece_bb |= square_bb(old_ksq);

    let opponent_has_queen = pos.pieces_cpt(!perspective, QUEEN).any();
    let [old_entries, new_entries] = cache.entries[usize::from(opponent_has_queen)]
        .get_disjoint_mut([old_ksq.idx(), new_ksq.idx()])
        .expect("a king move changes its square");
    let old_entry = &old_entries[p];
    let new_entry = &mut new_entries[p];

    // "Remove" means we need to remove them from the cache entry,
    // "Add" means add them to the entry to get the accumulator we want
    let mut old_remove = PsqIndexList::new();
    let mut old_add = PsqIndexList::new();
    let mut new_remove = PsqIndexList::new();
    let mut new_add = PsqIndexList::new();

    let old_changed_bb = get_changed_pieces(&old_entry.pieces, &previous_pieces);
    let old_removed_bb = old_changed_bb & old_entry.piece_bb;
    let old_added_bb = old_changed_bb & previous_piece_bb;

    let new_changed_bb = get_changed_pieces(&new_entry.pieces, current_pieces);
    let new_removed_bb = new_changed_bb & new_entry.piece_bb;
    let new_added_bb = new_changed_bb & pos.pieces();

    #[cfg(use_avx512icl)]
    {
        k32q2::write_indices(
            &old_entry.pieces,
            &previous_pieces,
            old_removed_bb,
            old_added_bb,
            perspective,
            old_ksq,
            opponent_has_queen,
            &mut old_remove,
            &mut old_add,
        );
        k32q2::write_indices(
            &new_entry.pieces,
            current_pieces,
            new_removed_bb,
            new_added_bb,
            perspective,
            new_ksq,
            opponent_has_queen,
            &mut new_remove,
            &mut new_add,
        );
    }
    #[cfg(not(use_avx512icl))]
    {
        for sq in old_removed_bb {
            old_remove.push(k32q2::make_index(
                perspective,
                sq,
                old_entry.pieces[sq.idx()],
                old_ksq,
                opponent_has_queen,
            ));
        }
        for sq in old_added_bb {
            old_add.push(k32q2::make_index(
                perspective,
                sq,
                previous_pieces[sq.idx()],
                old_ksq,
                opponent_has_queen,
            ));
        }
        for sq in new_removed_bb {
            new_remove.push(k32q2::make_index(
                perspective,
                sq,
                new_entry.pieces[sq.idx()],
                new_ksq,
                opponent_has_queen,
            ));
        }
        for sq in new_added_bb {
            new_add.push(k32q2::make_index(
                perspective,
                sq,
                current_pieces[sq.idx()],
                new_ksq,
                opponent_has_queen,
            ));
        }
    }

    // also contain pp indices
    let mut thr_removed = ThreatIndexList::new();
    let mut thr_added = ThreatIndexList::new();
    let threat_pp_base = feature_transformer.threat_and_pp_weights.0.as_ptr();
    let pf_stride = DIMENSIONS;
    full_threats::append_changed_indices(
        perspective,
        new_ksq,
        &target.dirties.dirty_threats,
        &mut thr_removed,
        &mut thr_added,
        threat_pp_base,
        pf_stride,
    );
    pp_3wide::append_changed_indices(
        perspective,
        new_ksq,
        &target.dirties.dirty_pawn_pairs,
        &mut thr_removed,
        &mut thr_added,
        threat_pp_base,
        pf_stride,
    );

    let from_acc = computed.accumulation[p].as_ptr();
    let to_acc = target.accumulation[p].as_mut_ptr();
    let old_acc = old_entry.accumulation.as_ptr();
    let new_acc = new_entry.accumulation.as_mut_ptr();

    let mut new_entry_removed = PsqWeightRows::new();
    let mut new_entry_added = PsqWeightRows::new();
    new_entry_removed.append_psq(&new_remove, feature_transformer);
    new_entry_added.append_psq(&new_add, feature_transformer);

    // The old entry's removed psq features have to be added back, its added
    // ones removed
    let mut removed = AllWeightRows::new();
    let mut added = AllWeightRows::new();
    added.append_psq(&old_remove, feature_transformer);
    added.append_threat(&thr_added, feature_transformer);
    removed.append_psq(&old_add, feature_transformer);
    removed.append_threat(&thr_removed, feature_transformer);

    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, and rows of the lists.
        unsafe {
            let mut acc = load_tile(j, new_acc);

            new_entry_removed.apply::<-1>(j, &mut acc);
            new_entry_added.apply::<1>(j, &mut acc);

            store_tile(j, new_acc, &acc);

            // adding the old accumulator adds (most of) the threats and pp
            // weights that we need
            apply::<1>(j, &mut acc, from_acc);
            // But we have added a whole bunch of psq weights for the wrong
            // king bucket which we need to remove
            // first we remove the cached psq accumulation for the old king
            // position...
            apply::<-1>(j, &mut acc, old_acc);

            // ... then we adjust
            removed.apply::<-1>(j, &mut acc);
            added.apply::<1>(j, &mut acc);

            store_tile(j, to_acc, &acc);
        }
    }

    new_entry.pieces = *current_pieces;
    new_entry.piece_bb = pos.pieces();

    target.computed[p] = true;
}

/// HalfKA data comes from the Finny table entry, while the threats are
/// built from the active threat features
fn update_accumulator_refresh_cache(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    pos: &Position,
    accumulator: &mut AccumulatorState,
    cache: &mut AccumulatorCaches,
) {
    let p = perspective.idx();
    let opponent_has_queen = pos.pieces_cpt(!perspective, QUEEN).any();
    let ksq = pos.king_square(perspective);
    let entry = &mut cache.entries[usize::from(opponent_has_queen)][ksq.idx()][p];
    let mut removed = PsqIndexList::new();
    let mut added = PsqIndexList::new();

    let changed_bb = get_changed_pieces(&entry.pieces, pos.piece_array());
    let removed_bb = changed_bb & entry.piece_bb;
    let added_bb = changed_bb & pos.pieces();

    #[cfg(use_avx512icl)]
    k32q2::write_indices(
        &entry.pieces,
        pos.piece_array(),
        removed_bb,
        added_bb,
        perspective,
        ksq,
        opponent_has_queen,
        &mut removed,
        &mut added,
    );
    #[cfg(not(use_avx512icl))]
    {
        for sq in removed_bb {
            removed.push(k32q2::make_index(
                perspective,
                sq,
                entry.pieces[sq.idx()],
                ksq,
                opponent_has_queen,
            ));
        }
        for sq in added_bb {
            added.push(k32q2::make_index(
                perspective,
                sq,
                pos.piece_on(sq),
                ksq,
                opponent_has_queen,
            ));
        }
    }

    entry.piece_bb = pos.pieces();
    entry.pieces = *pos.piece_array();

    let mut active = ThreatIndexList::new();
    full_threats::append_active_indices(perspective, pos, &mut active);
    pp_3wide::append_active_indices(perspective, pos, &mut active);

    accumulator.computed[p] = true;

    let entry_acc = entry.accumulation.as_mut_ptr();
    let to_acc = accumulator.accumulation[p].as_mut_ptr();

    let mut entry_removed = PsqWeightRows::new();
    let mut entry_added = PsqWeightRows::new();
    let mut active_threats = AllWeightRows::new();
    entry_removed.append_psq(&removed, feature_transformer);
    entry_added.append_psq(&added, feature_transformer);
    active_threats.append_threat(&active, feature_transformer);

    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, and rows of the lists.
        unsafe {
            let mut acc = load_tile(j, entry_acc);

            entry_removed.apply::<-1>(j, &mut acc);
            entry_added.apply::<1>(j, &mut acc);

            store_tile(j, entry_acc, &acc);

            active_threats.apply::<1>(j, &mut acc);

            store_tile(j, to_acc, &acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::Prng;
    use crate::movegen::{ALL, MoveList};

    /// A network whose feature transformer has random biases and weights,
    /// over their whole range, so that a feature missed or counted twice
    /// shows in the accumulations.
    fn random_network(rng: &mut Prng) -> LargePageBox<Network> {
        let mut network = Network::zeroed();
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
            .chain(ft.threat_and_pp_weights.0.chunks_mut(8))
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
    fn summed(ft: &FeatureTransformer, pos: &Position, perspective: Color) -> [i16; L1] {
        let mut accumulation = ft.biases.0;
        let ksq = pos.king_square(perspective);
        let opponent_has_queen = pos.pieces_cpt(!perspective, QUEEN).any();
        for sq in pos.pieces() {
            let index =
                k32q2::make_index(perspective, sq, pos.piece_on(sq), ksq, opponent_has_queen);
            let row = in_accumulation_order(&ft.weights.0[usize::from(index) * L1..][..L1]);
            for (value, &weight) in accumulation.iter_mut().zip(&row) {
                *value = value.wrapping_add(i16::from(weight));
            }
        }

        let mut active = ThreatIndexList::new();
        full_threats::append_active_indices(perspective, pos, &mut active);
        pp_3wide::append_active_indices(perspective, pos, &mut active);
        for &index in &active {
            let row =
                in_accumulation_order(&ft.threat_and_pp_weights.0[usize::from(index) * L1..][..L1]);
            for (value, &weight) in accumulation.iter_mut().zip(&row) {
                *value = value.wrapping_add(i16::from(weight));
            }
        }
        accumulation
    }

    /// Middlegames, where king moves update by the hybrid path, endgames,
    /// where kings move often, crossing the board's halves, and positions
    /// where a queen is taken, by the king among them, or made, which refresh
    /// the other side's features.
    const STARTS: [(&str, bool); 9] = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false,
        ),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            false,
        ),
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            false,
        ),
        (
            "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 9",
            true,
        ),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", false),
        ("8/8/4k3/2pp4/3P4/2P1K3/8/8 w - - 0 1", false),
        (
            "rnb1kbnr/pppp1ppp/8/4p3/8/5P2/PPPPPqPP/RNBQKBNR w KQkq - 0 3",
            false,
        ),
        (
            "rnb1kbnr/ppp2ppp/8/3qp3/3Q4/8/PPP1PPPP/RNB1KBNR w KQkq - 0 4",
            false,
        ),
        ("8/1P6/8/3k4/8/8/6p1/4K3 w - - 0 1", false),
    ];

    #[test]
    fn incremental_updates_match_sums_from_scratch() {
        let mut rng = Prng::new(20261012);
        let network = random_network(&mut rng);
        let ft = &network.feature_transformer;
        let mut caches = AccumulatorCaches::new(&network);
        let mut stack = AccumulatorStack::new();

        let mut check = |stack: &mut AccumulatorStack, pos: &Position| {
            stack.evaluate(pos, ft, &mut caches);
            for perspective in [WHITE, BLACK] {
                let accumulation = &stack.latest().accumulation[perspective.idx()];
                assert!(
                    *accumulation == summed(ft, pos, perspective),
                    "the accumulation of {} in {}",
                    ["white", "black"][perspective.idx()],
                    pos.fen()
                );
            }
        };

        // As a search does, the walks evaluate some positions along a line,
        // updating from the last one evaluated, and some a move away.
        for _ in 0..2 {
            for (fen, chess960) in STARTS {
                let mut pos = Position::from_fen(fen, chess960).unwrap();
                stack.reset();
                for _ in 0..80 {
                    let moves = MoveList::new::<ALL>(&pos);
                    if moves.is_empty() {
                        break;
                    }
                    if rng.rand64().is_multiple_of(3) {
                        check(&mut stack, &pos);
                    }
                    for &m in &moves {
                        if rng.rand64().is_multiple_of(4) {
                            let gives_check = pos.gives_check(m);
                            pos.do_move(m, gives_check, stack.push());
                            check(&mut stack, &pos);
                            pos.undo_move(m);
                            stack.pop();
                        }
                    }
                    let m = moves.as_slice()[(rng.rand64() % moves.len() as u64) as usize];
                    let gives_check = pos.gives_check(m);
                    pos.do_move(m, gives_check, stack.push());
                }
            }
        }
    }

    #[test]
    fn queens_taken_or_made_update_both_sides() {
        let mut rng = Prng::new(20261014);
        let network = random_network(&mut rng);
        let ft = &network.feature_transformer;
        let mut caches = AccumulatorCaches::new(&network);
        let mut stack = AccumulatorStack::new();

        // Each position is evaluated after the one before it, so that the
        // king taking the queen among 31 pieces would update by the hybrid
        // path but for the queen, and the king's return does.
        for (fen, line) in [
            (
                "rnb1kbnr/pppp1ppp/8/4p3/8/5P2/PPPPPqPP/RNBQKBNR w KQkq - 0 3",
                "e1f2 b8c6 f2e1",
            ),
            (
                "rnb1kbnr/ppp2ppp/8/3qp3/3Q4/8/PPP1PPPP/RNB1KBNR w KQkq - 0 4",
                "d4d5 g8f6 d5b7 c8b7",
            ),
            (
                "8/1P6/8/3k4/8/8/6p1/4K3 w - - 0 1",
                "b7b8q g2g1q e1e2 g1b1 b8b1",
            ),
        ] {
            let mut pos = Position::from_fen(fen, false).unwrap();
            stack.reset();
            stack.evaluate(&pos, ft, &mut caches);
            for uci in line.split(' ') {
                let m = crate::uci::to_move(&pos, uci);
                assert!(m != Move::none(), "{uci} in {}", pos.fen());
                let gives_check = pos.gives_check(m);
                pos.do_move(m, gives_check, stack.push());
                stack.evaluate(&pos, ft, &mut caches);
                for perspective in [WHITE, BLACK] {
                    let accumulation = &stack.latest().accumulation[perspective.idx()];
                    assert!(
                        *accumulation == summed(ft, &pos, perspective),
                        "the accumulation of {} after {uci} in {}",
                        ["white", "black"][perspective.idx()],
                        pos.fen()
                    );
                }
            }
        }
    }

    #[test]
    fn changed_pieces_are_the_squares_that_differ() {
        let mut rng = crate::misc::Prng::new(1070372);
        for _ in 0..1000 {
            let old: [Piece; SQUARE_NB] = std::array::from_fn(|_| Piece(rng.rand64() as u8 & 15));
            let mut new = old;
            let mask = rng.rand64() & rng.rand64();
            for sq in Bitboard(mask) {
                new[sq.idx()] = Piece(new[sq.idx()].0 ^ (1 + rng.rand64() as u8 % 15));
            }
            assert_eq!(get_changed_pieces(&old, &new), Bitboard(mask));
            assert_eq!(get_changed_pieces_scalar(&old, &new), Bitboard(mask));
        }
    }
}
