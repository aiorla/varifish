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
    FeatureTransformer, OUTPUT_DIMENSIONS, PSQ_BLOCK_DIMENSIONS, THREAT_AND_PP_DIMENSIONS,
};
use super::features::k32h3::{self, K32H3};
use super::features::{full_threats, pp_3wide};
use super::network::Network;
#[cfg(not(any(use_neon, use_paired_widening_add)))]
use super::simd::vec_load_convert_8_16;
use super::simd::{
    LANES, NUM_REGS, TILE_HEIGHT, VecT, vec_add_16, vec_load, vec_store, vec_sub_16,
};
#[cfg(use_paired_widening_add)]
use super::simd::{
    vec_maddubs_16, vec_set_8, vec_set_16, vec_srai_16, vec_unpackhi_8, vec_unpacklo_8, vec_xor,
    vec_zero,
};
use crate::bitboard::{Bitboard, square_bb};
use crate::memory::LargePageBox;
use crate::position::Position;
use crate::types::*;

type PsqIndexList = k32h3::IndexList;
/// Threat and pawn pair indices, which share the threat weights' rows.
type ThreatIndexList = full_threats::IndexList;

pub(super) const DIMENSIONS: usize = OUTPUT_DIMENSIONS;

/// The result of the affine transformation of a position's input features
/// for both perspectives, which of them are up to date, what the move to the
/// position changed, and the state of the piece-square blocks before it.
#[repr(C, align(64))]
pub struct AccumulatorState {
    pub accumulation: [[i16; L1]; COLOR_NB],
    pub computed: [bool; COLOR_NB],
    pub dirties: Dirties,
    pub block_before: k32h3::BlockState,
}

/// AccumulatorCaches provides per-thread accumulator caches, where each
/// cache contains multiple entries for each of the possible king squares.
/// When the accumulator needs to be refreshed, the cached entry is used to
/// more efficiently update the accumulator, instead of rebuilding it from
/// scratch. This idea was first described by Luecx (author of Koivisto) and
/// is commonly referred to as "Finny Tables".
pub struct AccumulatorCaches {
    /// The entries by the perspective's piece-square block, then by king
    /// square
    entries: [[[Entry; COLOR_NB]; SQUARE_NB]; K32H3::NUM_BLOCKS],
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
        self.clear_with_biases(&network.feature_transformer.biases.0);
    }

    /// Empties the caches with the given feature-transformer biases.
    pub fn clear_with_biases(&mut self, biases: &[i16; L1]) {
        for entries in self.entries.as_flattened_mut() {
            for entry in entries {
                entry.clear(biases);
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

    /// Adds the state of the position after a move from `pos`, whose
    /// changes the move records in the dirties returned.
    #[inline(always)]
    pub fn push<V: crate::variant::Variant>(&mut self, pos: &Position<V>) -> &mut Dirties {
        debug_assert!(self.size < Self::MAX_SIZE);
        let st = &mut self.accumulators[self.size];
        st.computed = [false; COLOR_NB];
        st.block_before = k32h3::block_state(pos);
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
    pub fn evaluate<V: crate::variant::Variant>(
        &mut self,
        pos: &Position<V>,
        feature_transformer: &FeatureTransformer,
        cache: &mut AccumulatorCaches,
    ) {
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

    fn evaluate_side<V: crate::variant::Variant>(
        &mut self,
        perspective: Color,
        pos: &Position<V>,
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
    /// requires full refresh.
    fn find_last_usable_accumulator<V: crate::variant::Variant>(
        &self,
        perspective: Color,
        pos: &Position<V>,
    ) -> usize {
        // All traversed states must use the current block.
        let psq_block = k32h3::block_of(pos, perspective);

        for curr_idx in (1..self.size).rev() {
            let state = &self.accumulators[curr_idx];
            if state.computed[perspective.idx()] {
                return curr_idx;
            }

            let before = usize::from(state.block_before[perspective.idx()]);

            // Threat feature set refreshes require a king move across the
            // center, i.e., a subset of halfka refreshes
            if k32h3::requires_refresh(&state.dirties.dirty_piece, perspective, before, psq_block) {
                return curr_idx;
            }
        }

        0
    }

    fn forward_update_incremental<V: crate::variant::Variant>(
        &mut self,
        perspective: Color,
        pos: &Position<V>,
        feature_transformer: &FeatureTransformer,
        begin: usize,
    ) {
        debug_assert!(begin < self.size);
        debug_assert!(self.accumulators[begin].computed[perspective.idx()]);

        let ksq = pos.king_square(perspective);
        let psq_block = k32h3::block_of(pos, perspective);

        for next in begin + 1..self.size {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next - 1);
            update_accumulator_incremental::<true, V>(
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

    fn backward_update_incremental<V: crate::variant::Variant>(
        &mut self,
        perspective: Color,
        pos: &Position<V>,
        feature_transformer: &FeatureTransformer,
        end: usize,
    ) {
        debug_assert!(end < self.size);
        debug_assert!(self.latest().computed[perspective.idx()]);

        let ksq = pos.king_square(perspective);
        let psq_block = k32h3::block_of(pos, perspective);

        for next in (end..self.size - 1).rev() {
            let (target, computed) = target_and_source(&mut self.accumulators, next, next + 1);
            update_accumulator_incremental::<false, V>(
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

    fn forward_update_incremental_both<V: crate::variant::Variant>(
        &mut self,
        pos: &Position<V>,
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
            update_accumulator_incremental::<true, V>(
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
            update_accumulator_incremental::<true, V>(
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
            update_accumulator_incremental_both::<V>(
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

impl crate::nnue::Accumulators<crate::variant::Chess> for AccumulatorStack {
    type Dirties = Dirties;

    #[inline(always)]
    fn new() -> LargePageBox<Self> {
        AccumulatorStack::new()
    }

    #[inline(always)]
    fn reset(&mut self, _network: Option<&Network>) {
        AccumulatorStack::reset(self);
    }

    #[inline(always)]
    fn push(&mut self, pos: &Position) -> &mut Dirties {
        AccumulatorStack::push(self, pos)
    }

    #[inline(always)]
    fn pop(&mut self) {
        AccumulatorStack::pop(self);
    }
}

impl crate::nnue::Accumulators<crate::variant::ThreeCheck> for AccumulatorStack {
    type Dirties = Dirties;

    #[inline(always)]
    fn new() -> LargePageBox<Self> {
        AccumulatorStack::new()
    }

    #[inline(always)]
    fn reset(&mut self, _network: Option<&super::threecheck::ThreeCheckNetwork>) {
        AccumulatorStack::reset(self);
    }

    #[inline(always)]
    fn push(&mut self, pos: &Position<crate::variant::ThreeCheck>) -> &mut Dirties {
        AccumulatorStack::push(self, pos)
    }

    #[inline(always)]
    fn pop(&mut self) {
        AccumulatorStack::pop(self);
    }
}

impl crate::nnue::Accumulators<crate::variant::Crazyhouse> for AccumulatorStack {
    type Dirties = Dirties;

    #[inline(always)]
    fn new() -> LargePageBox<Self> {
        AccumulatorStack::new()
    }

    #[inline(always)]
    fn reset(&mut self, _network: Option<&super::crazyhouse::CrazyhouseNetwork>) {
        AccumulatorStack::reset(self);
    }

    #[inline(always)]
    fn push(&mut self, pos: &Position<crate::variant::Crazyhouse>) -> &mut Dirties {
        AccumulatorStack::push(self, pos)
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
// TILE_HEIGHT below DIMENSIONS, or pointers to the tile's first value in an
// accumulation or a weight row of DIMENSIONS values; the index lists hold
// indices of weight rows.

const _: () = assert!(DIMENSIONS.is_multiple_of(TILE_HEIGHT));

pub(super) type Tile = [VecT; NUM_REGS];

#[inline(always)]
pub(super) unsafe fn load_tile(j: usize, data: *const i16) -> Tile {
    std::array::from_fn(|k| unsafe { vec_load(data.add(j + k * LANES)) })
}

#[inline(always)]
pub(super) unsafe fn store_tile(j: usize, dest: *mut i16, acc: &Tile) {
    for (k, &value) in acc.iter().enumerate() {
        unsafe { vec_store(dest.add(j + k * LANES), value) };
    }
}

/// Adds (SIGN 1) or subtracts (SIGN -1) the tile that `data` points at.
#[inline(always)]
pub(super) unsafe fn apply<const SIGN: i32>(acc: &mut Tile, data: *const i16) {
    const { assert!(SIGN == 1 || SIGN == -1) };
    for (k, acc) in acc.iter_mut().enumerate() {
        let column = unsafe { vec_load(data.add(k * LANES)) };
        *acc = if SIGN == 1 {
            vec_add_16(*acc, column)
        } else {
            vec_sub_16(*acc, column)
        };
    }
}

// The tile loops below walk all changed weight rows together, one tile at a
// time, and the hardware prefetcher doesn't keep up with that many rows:
// prefetch the next tile of a row while its current tile is being added. On
// a row's last tile this hits the next row, which is harmless since
// prefetches never fault. Rows of only two tiles (AVX-512) get no lead time
// from this and are faster without it, and so are AMD Zen 1/2, which only
// get builds without pext (Stockfish's 9d24e938).
#[cfg(use_pext)]
#[inline(always)]
fn prefetch_next_tile(data: *const i8) {
    const CACHE_LINE_SIZE: usize = 64;
    if DIMENSIONS / TILE_HEIGHT > 2 {
        for k in (0..TILE_HEIGHT).step_by(CACHE_LINE_SIZE) {
            crate::misc::prefetch(data.wrapping_add(TILE_HEIGHT + k));
        }
    }
}

#[cfg(all(use_paired_widening_add, not(use_pext)))]
#[inline(always)]
fn prefetch_next_tile(_: *const i8) {}

/// Adds or subtracts the tile of a row of 8-bit weights that `column`
/// points at.
#[inline(always)]
unsafe fn apply_feature_single<const SIGN: i32>(acc: &mut Tile, column: *const i8) {
    const { assert!(SIGN == 1 || SIGN == -1) };

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
        prefetch_next_tile(column);
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
pub(super) unsafe fn apply_feature_pair<const SIGN: i32>(
    acc: &mut Tile,
    data0: *const i8,
    data1: *const i8,
) {
    const { assert!(SIGN == 1 || SIGN == -1) };
    const { assert!(NUM_REGS.is_multiple_of(2)) };
    prefetch_next_tile(data0);
    prefetch_next_tile(data1);
    let ones = vec_set_8(1);
    unrolled::<{ NUM_REGS / 2 }>(|pair| {
        let k = 2 * pair;
        let (w0, w1) = unsafe {
            (
                vec_load(data0.add(k * LANES).cast()),
                vec_load(data1.add(k * LANES).cast()),
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

/// Adds the tile of the row `add` and subtracts that of the row `sub` in one
/// maddubs per half: both rows are biased to u8, so that maddubs' unsigned
/// operand is valid; the bias cancels, and |add - sub| <= 255 cannot
/// saturate.
#[cfg(use_paired_widening_add)]
#[inline(always)]
unsafe fn apply_feature_diff(acc: &mut Tile, add: *const i8, sub: *const i8) {
    const { assert!(NUM_REGS.is_multiple_of(2)) };
    prefetch_next_tile(add);
    prefetch_next_tile(sub);
    let bias = vec_set_8(-128);
    let signs = vec_set_16(-255); // i8 pairs (+1, -1)
    unrolled::<{ NUM_REGS / 2 }>(|pair| {
        let k = 2 * pair;
        let (ua, us) = unsafe {
            (
                vec_xor(vec_load(add.add(k * LANES).cast()), bias),
                vec_xor(vec_load(sub.add(k * LANES).cast()), bias),
            )
        };
        acc[k] = vec_add_16(acc[k], vec_maddubs_16(vec_unpacklo_8(ua, us), signs));
        acc[k + 1] = vec_add_16(acc[k + 1], vec_maddubs_16(vec_unpackhi_8(ua, us), signs));
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

/// Adds (SIGN 1) or subtracts (SIGN -1) the tiles of the rows of
/// `indices`, `weights` pointing at the tile of the weights' first row: the
/// base already includes the tile's offset.
#[inline(always)]
pub(super) unsafe fn apply_features<const SIGN: i32>(
    acc: &mut Tile,
    indices: &[u16],
    weights: *const i8,
) {
    const { assert!(SIGN == 1 || SIGN == -1) };
    // SAFETY: the index is a feature's, whose row is in the weights.
    let row = |index: u16| unsafe { weights.add(usize::from(index) * DIMENSIONS) };

    #[cfg(use_paired_widening_add)]
    let indices = {
        let (pairs, rest) = indices.as_chunks::<2>();
        for &[first, second] in pairs {
            unsafe { apply_feature_pair::<SIGN>(acc, row(first), row(second)) };
        }
        rest
    };
    for &index in indices {
        unsafe { apply_feature_single::<SIGN>(acc, row(index)) };
    }
}

/// The piece-square rows of one sign: the indices of rows of a block's
/// weights, or, for an incremental update, the addresses of the rows, found
/// once for all the tiles.
pub(super) trait PsqRows {
    fn len(&self) -> usize;

    /// The tile `j` of the `i`th row, `base` pointing at the tile of the
    /// weights' first row.
    ///
    /// # Safety
    ///
    /// `i` is below `len()`, and the row's tile is in the weights.
    unsafe fn row(&self, i: usize, base: *const i8, j: usize) -> *const i8;
}

impl PsqRows for [u16] {
    #[inline(always)]
    fn len(&self) -> usize {
        <[u16]>::len(self)
    }

    #[inline(always)]
    unsafe fn row(&self, i: usize, base: *const i8, _j: usize) -> *const i8 {
        unsafe { base.add(usize::from(*self.get_unchecked(i)) * DIMENSIONS) }
    }
}

impl<const N: usize> PsqRows for [*const i8; N] {
    #[inline(always)]
    fn len(&self) -> usize {
        N
    }

    #[inline(always)]
    unsafe fn row(&self, i: usize, _base: *const i8, j: usize) -> *const i8 {
        unsafe { self.get_unchecked(i).add(j) }
    }
}

/// Applies one sign's piece-square rows and threat and pawn-pair rows to
/// the tile `j`, `psq_weights` and `thr_weights` pointing at the tiles of
/// their weights' first rows. With paired widening, an odd row is held back
/// and returned, so that it can be combined with the other sign's odd row.
#[inline(always)]
pub(super) unsafe fn apply_features_but_odd<const SIGN: i32, P: PsqRows + ?Sized>(
    j: usize,
    acc: &mut Tile,
    psq: &P,
    thr: &[u16],
    psq_weights: *const i8,
    thr_weights: *const i8,
) -> Option<*const i8> {
    // SAFETY: the rows' tiles are in their weights.
    let psq_row = |i: usize| unsafe { psq.row(i, psq_weights, j) };
    let psq_count = psq.len();

    #[cfg(use_paired_widening_add)]
    {
        let thr_row = |index: u16| unsafe { thr_weights.add(usize::from(index) * DIMENSIONS) };
        let mut thr = thr;
        let mut i = 0;
        while i + 1 < psq_count {
            unsafe { apply_feature_pair::<SIGN>(acc, psq_row(i), psq_row(i + 1)) };
            i += 2;
        }

        // Pair the odd psq row with the first threat/pp row of the same
        // sign, if any.
        if i < psq_count {
            let Some((&first, rest)) = thr.split_first() else {
                return Some(psq_row(i));
            };
            unsafe { apply_feature_pair::<SIGN>(acc, psq_row(i), thr_row(first)) };
            thr = rest;
        }

        let mut odd = None;
        if !thr.len().is_multiple_of(2)
            && let Some((&last, rest)) = thr.split_last()
        {
            odd = Some(thr_row(last));
            thr = rest;
        }
        unsafe { apply_features::<SIGN>(acc, thr, thr_weights) };
        odd
    }

    #[cfg(not(use_paired_widening_add))]
    {
        for i in 0..psq_count {
            unsafe { apply_feature_single::<SIGN>(acc, psq_row(i)) };
        }
        unsafe { apply_features::<SIGN>(acc, thr, thr_weights) };
        None
    }
}

/// Applies the rows that apply_features_but_odd held back: with paired
/// widening, a removed row and an added one in one difference.
#[inline(always)]
pub(super) unsafe fn apply_odd_features(
    acc: &mut Tile,
    removed: Option<*const i8>,
    added: Option<*const i8>,
) {
    #[cfg(use_paired_widening_add)]
    if let (Some(removed), Some(added)) = (removed, added) {
        unsafe { apply_feature_diff(acc, added, removed) };
        return;
    }
    if let Some(removed) = removed {
        unsafe { apply_feature_single::<-1>(acc, removed) };
    }
    if let Some(added) = added {
        unsafe { apply_feature_single::<1>(acc, added) };
    }
}

/// Removes the rows of `removed` and adds those of `added`, `weights`
/// pointing at the tile of the weights' first row: with paired widening, an
/// odd row of each goes with the other in one difference.
#[inline(always)]
pub(super) unsafe fn apply_changed_features(
    acc: &mut Tile,
    removed: &[u16],
    added: &[u16],
    weights: *const i8,
) {
    // No threat rows: their weights are never read. Index lists read no
    // tile index.
    let none = std::ptr::null();
    unsafe {
        let odd_removed = apply_features_but_odd::<-1, _>(0, acc, removed, &[], weights, none);
        let odd_added = apply_features_but_odd::<1, _>(0, acc, added, &[], weights, none);
        apply_odd_features(acc, odd_removed, odd_added);
    }
}

/// Removes and adds the rows of piece-square features, from `psq_weights`,
/// those of their block.
#[inline(always)]
unsafe fn apply_psq_features(
    j: usize,
    acc: &mut Tile,
    removed: &PsqIndexList,
    added: &PsqIndexList,
    psq_weights: *const i8,
) {
    debug_assert!(
        removed
            .iter()
            .chain(added.iter())
            .all(|&i| usize::from(i) < PSQ_BLOCK_DIMENSIONS)
    );
    unsafe {
        apply_changed_features(
            acc,
            removed.as_slice(),
            added.as_slice(),
            psq_weights.add(j),
        );
    }
}

/// Applies the rows of threat and pawn-pair features.
#[inline(always)]
unsafe fn apply_threat_features<const SIGN: i32>(
    j: usize,
    acc: &mut Tile,
    list: &ThreatIndexList,
    feature_transformer: &FeatureTransformer,
) {
    debug_assert!(
        list.iter()
            .all(|&i| usize::from(i) < THREAT_AND_PP_DIMENSIONS)
    );
    let weights = feature_transformer.threat_and_pp_weights.0.as_ptr();
    unsafe { apply_features::<SIGN>(acc, list.as_slice(), weights.add(j)) };
}

/// Removes and adds the rows of piece-square features and of threat and
/// pawn-pair features: with paired widening, the last row of an odd number
/// of piece-square rows goes with the first threat row of the same sign,
/// and an odd removed row with an odd added one.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn apply_combined_features<R: PsqRows + ?Sized, A: PsqRows + ?Sized>(
    j: usize,
    acc: &mut Tile,
    psq_removed: &R,
    thr_removed: &ThreatIndexList,
    psq_added: &A,
    thr_added: &ThreatIndexList,
    feature_transformer: &FeatureTransformer,
    psq_base: *const i8,
) {
    // SAFETY: tiles of the weights, as for the rows.
    unsafe {
        let psq_weights = psq_base.add(j);
        let thr_weights = feature_transformer.threat_and_pp_weights.0.as_ptr().add(j);
        let odd_removed = apply_features_but_odd::<-1, R>(
            j,
            acc,
            psq_removed,
            thr_removed.as_slice(),
            psq_weights,
            thr_weights,
        );
        let odd_added = apply_features_but_odd::<1, A>(
            j,
            acc,
            psq_added,
            thr_added.as_slice(),
            psq_weights,
            thr_weights,
        );
        apply_odd_features(acc, odd_removed, odd_added);
    }
}

/// An incremental update of one perspective's accumulation by REMOVED
/// piece-square rows removed and ADDED added, whose addresses are found
/// once for all the tiles.
#[allow(clippy::too_many_arguments)]
fn apply_combined_counted<const REMOVED: usize, const ADDED: usize>(
    feature_transformer: &FeatureTransformer,
    from_acc: &[i16; DIMENSIONS],
    to_acc: &mut [i16; DIMENSIONS],
    psq_added: &PsqIndexList,
    psq_removed: &PsqIndexList,
    thr_added: &ThreatIndexList,
    thr_removed: &ThreatIndexList,
    psq_weights: &[i8],
) {
    const { assert!(REMOVED <= 2 && ADDED <= 2) };

    let from = from_acc.as_ptr();
    let to = to_acc.as_mut_ptr();
    let psq_weights = psq_weights.as_ptr();

    // SAFETY: the index is a feature's, whose row is in the weights.
    let row = |index: u16| unsafe { psq_weights.add(usize::from(index) * DIMENSIONS) };
    let removed_weights: [*const i8; REMOVED] = std::array::from_fn(|i| row(psq_removed[i]));
    let added_weights: [*const i8; ADDED] = std::array::from_fn(|i| row(psq_added[i]));

    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, and rows of the lists.
        unsafe {
            let mut acc = load_tile(j, from);

            apply_combined_features(
                j,
                &mut acc,
                &removed_weights,
                thr_removed,
                &added_weights,
                thr_added,
                feature_transformer,
                psq_weights,
            );

            store_tile(j, to, &acc);
        }
    }
}

/// An incremental update of one perspective's accumulation, which removes
/// and adds one or two piece-square rows each (or (0, 1) / (1, 0) for a
/// Crazyhouse drop).
#[allow(clippy::too_many_arguments)]
fn apply_combined<V: crate::variant::Variant>(
    feature_transformer: &FeatureTransformer,
    from_acc: &[i16; DIMENSIONS],
    to_acc: &mut [i16; DIMENSIONS],
    psq_added: &PsqIndexList,
    psq_removed: &PsqIndexList,
    thr_added: &ThreatIndexList,
    thr_removed: &ThreatIndexList,
    psq_weights: &[i8],
) {
    debug_assert!(if V::CRAZYHOUSE {
        psq_removed.len() <= 2 && psq_added.len() <= 2
    } else {
        (psq_removed.len() == 1 || psq_removed.len() == 2)
            && (psq_added.len() == 1 || psq_added.len() == 2)
    });
    debug_assert!(
        psq_removed
            .iter()
            .chain(psq_added.iter())
            .all(|&i| usize::from(i) < PSQ_BLOCK_DIMENSIONS)
    );
    debug_assert!(
        thr_removed
            .iter()
            .chain(thr_added.iter())
            .all(|&i| usize::from(i) < THREAT_AND_PP_DIMENSIONS)
    );

    // Dispatch once; feature counts and row addresses stay fixed across
    // tiles.
    let apply = if V::CRAZYHOUSE {
        match (psq_removed.len(), psq_added.len()) {
            (0, 1) => apply_combined_counted::<0, 1>,
            (1, 0) => apply_combined_counted::<1, 0>,
            (1, 1) => apply_combined_counted::<1, 1>,
            (1, _) => apply_combined_counted::<1, 2>,
            (_, 1) => apply_combined_counted::<2, 1>,
            _ => apply_combined_counted::<2, 2>,
        }
    } else {
        match (psq_removed.len(), psq_added.len()) {
            (1, 1) => apply_combined_counted::<1, 1>,
            (1, _) => apply_combined_counted::<1, 2>,
            (_, 1) => apply_combined_counted::<2, 1>,
            _ => apply_combined_counted::<2, 2>,
        }
    };
    apply(
        feature_transformer,
        from_acc,
        to_acc,
        psq_added,
        psq_removed,
        thr_added,
        thr_removed,
        psq_weights,
    );
}

#[inline(always)]
fn update_accumulator_incremental<const FORWARD: bool, V: crate::variant::Variant>(
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
    k32h3::append_changed_indices::<V>(perspective, ksq, &dirties.dirty_piece, psq_out, psq_in);

    apply_combined::<V>(
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

#[inline(always)]
fn update_accumulator_incremental_both<V: crate::variant::Variant>(
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
    for (perspective, ksq) in [(WHITE, white_ksq), (BLACK, black_ksq)] {
        let p = perspective.idx();
        k32h3::append_changed_indices::<V>(
            perspective,
            ksq,
            &dirties.dirty_piece,
            &mut psq_removed[p],
            &mut psq_added[p],
        );
    }

    for (p, psq_block) in [(WHITE.idx(), white_block), (BLACK.idx(), black_block)] {
        apply_combined::<V>(
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

/// Appends the piece-square feature indices that differ between two
/// boards, given the bitboards of the pieces on them before and after the
/// change.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn append_changed_psq_indices(
    perspective: Color,
    ksq: Square,
    old_pieces: &[Piece; SQUARE_NB],
    old_bb: Bitboard,
    new_pieces: &[Piece; SQUARE_NB],
    new_bb: Bitboard,
    removed: &mut PsqIndexList,
    added: &mut PsqIndexList,
) {
    let changed_bb = get_changed_pieces(old_pieces, new_pieces);
    let removed_bb = changed_bb & old_bb;
    let added_bb = changed_bb & new_bb;

    #[cfg(use_avx512icl)]
    k32h3::write_indices(
        old_pieces,
        new_pieces,
        removed_bb,
        added_bb,
        perspective,
        ksq,
        removed,
        added,
    );
    #[cfg(not(use_avx512icl))]
    {
        for sq in removed_bb {
            removed.push(k32h3::make_index(
                perspective,
                sq,
                old_pieces[sq.idx()],
                ksq,
            ));
        }
        for sq in added_bb {
            added.push(k32h3::make_index(
                perspective,
                sq,
                new_pieces[sq.idx()],
                ksq,
            ));
        }
    }
}

/// Updates accumulator for a king move, and also updates the accumulator
/// cache for the new king position.
fn update_accumulator_hybrid<V: crate::variant::Variant>(
    perspective: Color,
    pos: &Position<V>,
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

    let psq_block = k32h3::block_of(pos, perspective);
    let [old_entries, new_entries] = cache.entries[psq_block]
        .get_disjoint_mut([old_ksq.idx(), new_ksq.idx()])
        .expect("a king move changes its square");
    let old_entry = &old_entries[p];
    let new_entry = &mut new_entries[p];
    let psq_weights = feature_transformer.psq_weights(psq_block).as_ptr();

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

    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, and rows of the lists.
        unsafe {
            let mut acc = load_tile(j, new_acc);

            apply_psq_features(j, &mut acc, &new_remove, &new_add, psq_weights);

            store_tile(j, new_acc, &acc);

            // adding the old accumulator adds (most of) the threats and pp
            // weights that we need
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
                &thr_removed,
                old_remove.as_slice(),
                &thr_added,
                feature_transformer,
                psq_weights,
            );

            store_tile(j, to_acc, &acc);
        }
    }

    new_entry.pieces = *current_pieces;
    new_entry.piece_bb = pos.pieces();

    target.computed[p] = true;
}

/// HalfKA data comes from the Finny table entry, while the threats are
/// built from the active threat features
fn update_accumulator_refresh_cache<V: crate::variant::Variant>(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    pos: &Position<V>,
    accumulator: &mut AccumulatorState,
    cache: &mut AccumulatorCaches,
) {
    let p = perspective.idx();
    let psq_block = k32h3::block_of(pos, perspective);
    let ksq = pos.king_square(perspective);
    let entry = &mut cache.entries[psq_block][ksq.idx()][p];
    let psq_weights = feature_transformer.psq_weights(psq_block).as_ptr();
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

    let mut active = ThreatIndexList::new();
    full_threats::append_active_indices(perspective, pos, &mut active);
    pp_3wide::append_active_indices(perspective, pos, &mut active);

    accumulator.computed[p] = true;

    let entry_acc = entry.accumulation.as_mut_ptr();
    let to_acc = accumulator.accumulation[p].as_mut_ptr();

    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, and rows of the lists.
        unsafe {
            let mut acc = load_tile(j, entry_acc);

            apply_psq_features(j, &mut acc, &removed, &added, psq_weights);

            store_tile(j, entry_acc, &acc);

            apply_threat_features::<1>(j, &mut acc, &active, feature_transformer);

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
        let psq_block = k32h3::block_of(pos, perspective);
        for sq in pos.pieces() {
            let index = k32h3::make_index(perspective, sq, pos.piece_on(sq), ksq);
            let row =
                in_accumulation_order(&ft.psq_weights(psq_block)[usize::from(index) * L1..][..L1]);
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
                            pos.do_move(m, gives_check, stack.push(&pos));
                            check(&mut stack, &pos);
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
    }

    #[test]
    fn heavy_pieces_taken_or_made_update_both_sides() {
        let mut rng = Prng::new(20261014);
        let network = random_network(&mut rng);
        let ft = &network.feature_transformer;
        let mut caches = AccumulatorCaches::new(&network);
        let mut stack = AccumulatorStack::new();

        // Each position is evaluated after the one before it, so that the
        // king taking the queen among 31 pieces, or the last rook among 26,
        // would update by the hybrid path but for the piece-square blocks
        // that change, and the king's return does. Queens, and the last
        // rooks or queens, taken or made change the blocks.
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
            (
                "4k3/1P6/8/8/8/8/1p2r3/4K3 w - - 0 1",
                "e1e2 b2b1r b7b8q b1b8 e2d3",
            ),
            (
                "1nb1kbn1/pppppppp/8/8/8/8/PPPPPrPP/1NB1KBN1 w - - 0 1",
                "e1f2 b8c6 f2e1",
            ),
        ] {
            let mut pos = Position::from_fen(fen, false).unwrap();
            stack.reset();
            stack.evaluate(&pos, ft, &mut caches);
            for uci in line.split(' ') {
                let m = crate::uci::to_move(&pos, uci);
                assert!(m != Move::none(), "{uci} in {}", pos.fen());
                let gives_check = pos.gives_check(m);
                pos.do_move(m, gives_check, stack.push(&pos));
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

    /// Rows applied by sign with an odd one held back, and the rows held
    /// back then, add up as the rows applied one at a time do, for up to
    /// three piece-square and threat rows of each sign, with weights over
    /// their whole range: with paired widening, the held-back rows go in one
    /// difference, whose bias must cancel without saturating.
    #[test]
    fn rows_held_back_add_up_as_single_rows() {
        const ROWS: u16 = 4;
        fn weights(rng: &mut Prng) -> Vec<i8> {
            (0..usize::from(ROWS) * DIMENSIONS)
                .map(|i| match i % 5 {
                    0 => i8::MIN,
                    1 => i8::MAX,
                    _ => rng.rand64() as i8,
                })
                .collect()
        }
        let mut rng = Prng::new(20261107);
        let psq_weights = weights(&mut rng);
        let thr_weights = weights(&mut rng);
        let start: Vec<i16> = (0..DIMENSIONS).map(|_| rng.rand64() as i16).collect();
        let first = |n: usize| (0..n as u16).collect::<Vec<u16>>();
        let last = |n: usize| (0..n as u16).map(|i| ROWS - 1 - i).collect::<Vec<u16>>();

        for counts in 0..4 * 4 * 4 * 4 {
            let [pr, tr, pa, ta] = [0, 1, 2, 3].map(|k| (counts >> (2 * k)) & 3);
            let (psq_removed, thr_removed) = (first(pr), first(tr));
            let (psq_added, thr_added) = (last(pa), last(ta));
            let mut expected = start.clone();
            let mut held_back = start.clone();
            let mut incremental = start.clone();
            let mut changed = start.clone();
            for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
                // SAFETY: tiles of the accumulations and of the weights'
                // rows, which the lists index.
                unsafe {
                    let psq = psq_weights.as_ptr().add(j);
                    let thr = thr_weights.as_ptr().add(j);
                    let row = |base: *const i8, i: u16| base.add(usize::from(i) * DIMENSIONS);

                    let mut acc = load_tile(j, expected.as_ptr());
                    for (rows, base, sign) in [
                        (&psq_removed, psq, -1),
                        (&thr_removed, thr, -1),
                        (&psq_added, psq, 1),
                        (&thr_added, thr, 1),
                    ] {
                        for &i in rows {
                            if sign == 1 {
                                apply_feature_single::<1>(&mut acc, row(base, i));
                            } else {
                                apply_feature_single::<-1>(&mut acc, row(base, i));
                            }
                        }
                    }
                    store_tile(j, expected.as_mut_ptr(), &acc);

                    let mut acc = load_tile(j, held_back.as_ptr());
                    let removed = apply_features_but_odd::<-1, [u16]>(
                        j,
                        &mut acc,
                        &psq_removed,
                        &thr_removed,
                        psq,
                        thr,
                    );
                    let added = apply_features_but_odd::<1, [u16]>(
                        j, &mut acc, &psq_added, &thr_added, psq, thr,
                    );
                    apply_odd_features(&mut acc, removed, added);
                    store_tile(j, held_back.as_mut_ptr(), &acc);

                    // An incremental update changes one or two piece-square
                    // rows of each sign, whose addresses it finds once.
                    if (1..=2).contains(&pr) && (1..=2).contains(&pa) {
                        let base = psq_weights.as_ptr();
                        let rows =
                            |list: &[u16]| [row(base, list[0]), row(base, *list.last().unwrap())];
                        let mut acc = load_tile(j, incremental.as_ptr());
                        let removed = if pr == 1 {
                            let rows: [*const i8; 1] = [rows(&psq_removed)[0]];
                            apply_features_but_odd::<-1, _>(
                                j,
                                &mut acc,
                                &rows,
                                &thr_removed,
                                psq,
                                thr,
                            )
                        } else {
                            let rows = rows(&psq_removed);
                            apply_features_but_odd::<-1, _>(
                                j,
                                &mut acc,
                                &rows,
                                &thr_removed,
                                psq,
                                thr,
                            )
                        };
                        let added = if pa == 1 {
                            let rows: [*const i8; 1] = [rows(&psq_added)[0]];
                            apply_features_but_odd::<1, _>(j, &mut acc, &rows, &thr_added, psq, thr)
                        } else {
                            let rows = rows(&psq_added);
                            apply_features_but_odd::<1, _>(j, &mut acc, &rows, &thr_added, psq, thr)
                        };
                        apply_odd_features(&mut acc, removed, added);
                        store_tile(j, incremental.as_mut_ptr(), &acc);
                    }

                    if tr == 0 && ta == 0 {
                        let mut acc = load_tile(j, changed.as_ptr());
                        apply_changed_features(&mut acc, &psq_removed, &psq_added, psq);
                        store_tile(j, changed.as_mut_ptr(), &acc);
                    }
                }
            }
            let rows = format!("{pr} {tr} {pa} {ta}");
            assert_eq!(held_back, expected, "{rows}");
            if (1..=2).contains(&pr) && (1..=2).contains(&pa) {
                assert_eq!(incremental, expected, "{rows}");
            }
            if tr == 0 && ta == 0 {
                assert_eq!(changed, expected, "{rows}");
            }
        }
    }
}
