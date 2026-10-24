//! The accumulators of the Atomic network: for each position of the line
//! being searched, the sums of the feature transformer's weights and of its
//! PSQT weights over the active features of each perspective, each brought
//! up to date from another rather than summed anew, and the caches of those
//! sums by king square that full refreshes start from.
//!
//! Derived, with changes for varifish in 2026, from Atomic-Stockfish's
//! `nnue/nnue_accumulator.h` and `.cpp`
//! (https://github.com/Belzedar94/Atomic-Stockfish) and from Stockfish's,
//! Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under the
//! GNU General Public License, version 3 or later. Atomic-Stockfish sums
//! every feature anew when a king moves or explodes; varifish starts from
//! the sums cached for the king's square, as Stockfish does, and fills the
//! states between back from the one refreshed. The sums are the same, as
//! they wrap.

// The tile functions are unsafe for their pointers; the operations they
// call are unsafe only on some platforms.
#![allow(unused_unsafe)]

use super::features::{self, AddedList, IndexList, RemovedList};
use super::network::{FeatureTransformer, HALF_DIMENSIONS, Network, PSQT_BUCKETS};
use crate::bitboard::Bitboard;
use crate::memory::LargePageBox;
use crate::nnue::accumulator::get_changed_pieces;
use crate::nnue::simd::{
    LANES, NUM_REGISTERS_SIMD, VecT, best_register_count, vec_add_16, vec_load, vec_store,
    vec_sub_16,
};
use crate::position::Position;
use crate::types::*;
use crate::variant::Atomic;

/// The result of the affine transformation of a position's input features
/// for both perspectives, which of them are up to date, and what the move
/// to the position changed.
#[repr(C, align(64))]
pub struct AccumulatorState {
    pub accumulation: [[i16; HALF_DIMENSIONS]; COLOR_NB],
    pub psqt_accumulation: [[i32; PSQT_BUCKETS]; COLOR_NB],
    pub computed: [bool; COLOR_NB],
    pub dirties: AtomicDirties,
}

/// For each perspective and square its king is oriented to, the sums of
/// the board last refreshed there, which a refresh brings up to date with
/// the pieces that differ ("Finny tables", first described by Luecx, the
/// author of Koivisto). An exploded king counts as on A1, as its features
/// do.
pub struct AccumulatorCaches {
    entries: [[Entry; COLOR_NB]; SQUARE_NB],
}

#[repr(C, align(64))]
struct Entry {
    accumulation: [i16; HALF_DIMENSIONS],
    psqt_accumulation: [i32; PSQT_BUCKETS],
    pieces: [Piece; SQUARE_NB],
    piece_bb: Bitboard,
}

impl Entry {
    /// An entry of an empty board: the biases, without any weights on top.
    fn clear(&mut self, biases: &[i16; HALF_DIMENSIONS]) {
        self.accumulation = *biases;
        self.psqt_accumulation = [0; PSQT_BUCKETS];
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
        for entries in &mut self.entries {
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
        root.dirties = AtomicDirties::default();
        root.computed = [false; COLOR_NB];
        self.size = 1;
    }

    /// Adds the state of the position after a move, whose changes the move
    /// records in the dirties returned.
    #[inline(always)]
    pub fn push(&mut self) -> &mut AtomicDirties {
        debug_assert!(self.size < Self::MAX_SIZE);
        let st = &mut self.accumulators[self.size];
        st.computed = [false; COLOR_NB];
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
        pos: &Position<Atomic>,
        feature_transformer: &FeatureTransformer,
        cache: &mut AccumulatorCaches,
    ) {
        for perspective in [WHITE, BLACK] {
            self.evaluate_side(perspective, pos, feature_transformer, cache);
        }
    }

    fn evaluate_side(
        &mut self,
        perspective: Color,
        pos: &Position<Atomic>,
        feature_transformer: &FeatureTransformer,
        cache: &mut AccumulatorCaches,
    ) {
        let last_usable_accum = self.find_last_usable_accumulator(perspective);

        if self.accumulators[last_usable_accum].computed[perspective.idx()] {
            self.forward_update_incremental(
                perspective,
                pos,
                feature_transformer,
                last_usable_accum,
            );
        } else {
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
            if state.computed[perspective.idx()]
                || features::requires_refresh(&state.dirties, perspective)
            {
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

        let ksq = features::king_square(pos, perspective);

        for next in begin + 1..self.size {
            let [computed, target] = self
                .accumulators
                .get_disjoint_mut([next - 1, next])
                .expect("two states of the stack");
            update_accumulator_incremental::<true>(
                perspective,
                feature_transformer,
                ksq,
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

        let ksq = features::king_square(pos, perspective);

        for next in (end..self.size - 1).rev() {
            let [target, computed] = self
                .accumulators
                .get_disjoint_mut([next, next + 1])
                .expect("two states of the stack");
            update_accumulator_incremental::<false>(
                perspective,
                feature_transformer,
                ksq,
                target,
                computed,
            );
        }

        debug_assert!(self.accumulators[end].computed[perspective.idx()]);
    }
}

impl crate::nnue::Accumulators for AccumulatorStack {
    type Dirties = AtomicDirties;

    #[inline(always)]
    fn new() -> LargePageBox<Self> {
        AccumulatorStack::new()
    }

    #[inline(always)]
    fn reset(&mut self) {
        AccumulatorStack::reset(self);
    }

    #[inline(always)]
    fn push(&mut self) -> &mut AtomicDirties {
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
// TILE_HEIGHT below HALF_DIMENSIONS, and pointers to accumulations or weight
// rows of HALF_DIMENSIONS values; the index lists hold indices of weight
// rows.

const NUM_REGS: usize = best_register_count(
    size_of::<VecT>(),
    size_of::<i16>(),
    HALF_DIMENSIONS,
    NUM_REGISTERS_SIMD,
);
const TILE_HEIGHT: usize = NUM_REGS * LANES;

const _: () = assert!(HALF_DIMENSIONS.is_multiple_of(TILE_HEIGHT));

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

/// Adds (SIGN 1) or subtracts (SIGN -1) the rows of the features listed.
#[inline(always)]
unsafe fn apply_features<const SIGN: i32>(
    j: usize,
    acc: &mut Tile,
    list: &[u32],
    feature_transformer: &FeatureTransformer,
) {
    const { assert!(SIGN == 1 || SIGN == -1) };
    let weights = feature_transformer.weights.0.as_ptr();

    for &index in list {
        debug_assert!((index as usize) < features::DIMENSIONS);
        let row = unsafe { weights.add(index as usize * HALF_DIMENSIONS) };
        for (k, acc) in acc.iter_mut().enumerate() {
            let column = unsafe { vec_load(row.add(j + k * LANES)) };
            *acc = if SIGN == 1 {
                vec_add_16(*acc, column)
            } else {
                vec_sub_16(*acc, column)
            };
        }
    }
}

/// Adds or subtracts the PSQT rows of the features listed.
#[inline(always)]
fn apply_psqt_features<const SIGN: i32>(
    psqt: &mut [i32; PSQT_BUCKETS],
    list: &[u32],
    feature_transformer: &FeatureTransformer,
) {
    for &index in list {
        let row = feature_transformer.psqt_row(index);
        for (value, &weight) in psqt.iter_mut().zip(row) {
            *value = if SIGN == 1 {
                value.wrapping_add(weight)
            } else {
                value.wrapping_sub(weight)
            };
        }
    }
}

/// Writes to `to` the accumulations of `from` less the features removed
/// and plus those added; `from` and `to` may be the same.
#[inline(always)]
fn apply_changes(
    feature_transformer: &FeatureTransformer,
    from: (*const i16, &[i32; PSQT_BUCKETS]),
    to: (*mut i16, &mut [i32; PSQT_BUCKETS]),
    removed: &[u32],
    added: &[u32],
) {
    for j in (0..HALF_DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: a tile of the accumulations, and rows of the lists.
        unsafe {
            let mut acc = load_tile(j, from.0);
            apply_features::<-1>(j, &mut acc, removed, feature_transformer);
            apply_features::<1>(j, &mut acc, added, feature_transformer);
            store_tile(j, to.0, &acc);
        }
    }

    let mut psqt = *from.1;
    apply_psqt_features::<-1>(&mut psqt, removed, feature_transformer);
    apply_psqt_features::<1>(&mut psqt, added, feature_transformer);
    *to.1 = psqt;
}

fn update_accumulator_incremental<const FORWARD: bool>(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    ksq: Square,
    target_state: &mut AccumulatorState,
    computed: &AccumulatorState,
) {
    let p = perspective.idx();
    debug_assert!(computed.computed[p]);
    debug_assert!(!target_state.computed[p]);

    let mut removed = RemovedList::new();
    let mut added = AddedList::new();
    let dirties = if FORWARD {
        &target_state.dirties
    } else {
        &computed.dirties
    };
    features::append_changed_indices(perspective, ksq, dirties, &mut removed, &mut added);

    // Going backward, what the move added is removed.
    let (removed, added) = if FORWARD {
        (removed.as_slice(), added.as_slice())
    } else {
        (added.as_slice(), removed.as_slice())
    };

    apply_changes(
        feature_transformer,
        (
            computed.accumulation[p].as_ptr(),
            &computed.psqt_accumulation[p],
        ),
        (
            target_state.accumulation[p].as_mut_ptr(),
            &mut target_state.psqt_accumulation[p],
        ),
        removed,
        added,
    );

    target_state.computed[p] = true;
}

/// Brings the cache entry of the perspective's king square up to date with
/// the pieces that differ, and copies its sums to the accumulator.
fn update_accumulator_refresh_cache(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    pos: &Position<Atomic>,
    accumulator: &mut AccumulatorState,
    cache: &mut AccumulatorCaches,
) {
    let p = perspective.idx();
    let ksq = features::king_square(pos, perspective);
    let entry = &mut cache.entries[features::orient(perspective, ksq).idx()][p];
    let mut removed = IndexList::new();
    let mut added = IndexList::new();

    let changed_bb = get_changed_pieces(&entry.pieces, pos.piece_array());
    let removed_bb = changed_bb & entry.piece_bb;
    let added_bb = changed_bb & pos.pieces();

    for sq in removed_bb {
        removed.push(features::make_index(
            perspective,
            sq,
            entry.pieces[sq.idx()],
            ksq,
        ));
    }
    for sq in added_bb {
        added.push(features::make_index(perspective, sq, pos.piece_on(sq), ksq));
    }

    entry.piece_bb = pos.pieces();
    entry.pieces = *pos.piece_array();

    let entry_acc = entry.accumulation.as_mut_ptr();
    let mut psqt = entry.psqt_accumulation;
    apply_changes(
        feature_transformer,
        (entry_acc, &entry.psqt_accumulation),
        (entry_acc, &mut psqt),
        removed.as_slice(),
        added.as_slice(),
    );
    entry.psqt_accumulation = psqt;

    accumulator.accumulation[p] = entry.accumulation;
    accumulator.psqt_accumulation[p] = entry.psqt_accumulation;
    accumulator.computed[p] = true;
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
        for values in ft.biases.0.chunks_mut(4).chain(ft.weights.0.chunks_mut(4)) {
            let bits = rng.rand64();
            for (i, value) in values.iter_mut().enumerate() {
                *value = (bits >> (16 * i)) as i16;
            }
        }
        for values in ft.psqt_weights.0.chunks_mut(2) {
            let bits = rng.rand64();
            for (i, value) in values.iter_mut().enumerate() {
                *value = (bits >> (32 * i)) as i32;
            }
        }
        network
    }

    /// The sums of a perspective, summed anew from the biases and the
    /// weights of the active features.
    fn summed(
        ft: &FeatureTransformer,
        pos: &Position<Atomic>,
        perspective: Color,
    ) -> ([i16; HALF_DIMENSIONS], [i32; PSQT_BUCKETS]) {
        let mut accumulation = ft.biases.0;
        let mut psqt = [0i32; PSQT_BUCKETS];
        let ksq = features::king_square(pos, perspective);
        for sq in pos.pieces() {
            let index = features::make_index(perspective, sq, pos.piece_on(sq), ksq);
            let row = &ft.weights.0[index as usize * HALF_DIMENSIONS..][..HALF_DIMENSIONS];
            for (value, &weight) in accumulation.iter_mut().zip(row) {
                *value = value.wrapping_add(weight);
            }
            for (value, &weight) in psqt.iter_mut().zip(ft.psqt_row(index)) {
                *value = value.wrapping_add(weight);
            }
        }
        (accumulation, psqt)
    }

    /// Openings and middlegames full of captures, endgames where kings move
    /// often, and positions a capture next to a king ends.
    const STARTS: [(&str, bool); 7] = [
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
    ];

    #[test]
    fn incremental_updates_match_sums_from_scratch() {
        let mut rng = Prng::new(20261012);
        let network = random_network(&mut rng);
        let ft = &network.feature_transformer;
        let mut caches = AccumulatorCaches::new(&network);
        let mut stack = AccumulatorStack::new();
        let mut exploded_kings = 0;

        let mut check = |stack: &mut AccumulatorStack, pos: &Position<Atomic>| {
            stack.evaluate(pos, ft, &mut caches);
            for perspective in [WHITE, BLACK] {
                let state = stack.latest();
                let (accumulation, psqt) = summed(ft, pos, perspective);
                assert!(
                    state.accumulation[perspective.idx()] == accumulation
                        && state.psqt_accumulation[perspective.idx()] == psqt,
                    "the sums of {} in {}",
                    ["white", "black"][perspective.idx()],
                    pos.fen()
                );
            }
        };

        // As a search does, the walks evaluate some positions along a line,
        // updating from the last one evaluated, and some a move away, among
        // them those where a king exploded.
        for _ in 0..4 {
            for (fen, chess960) in STARTS {
                let mut pos = Position::<Atomic>::from_fen(fen, chess960).unwrap();
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
                            exploded_kings += usize::from(!pos.has_king(pos.side_to_move()));
                            check(&mut stack, &pos);
                            pos.undo_move(m);
                            stack.pop();
                        }
                    }
                    let m = moves.as_slice()[(rng.rand64() % moves.len() as u64) as usize];
                    let gives_check = pos.gives_check(m);
                    pos.do_move(m, gives_check, stack.push());
                    if !pos.has_king(pos.side_to_move()) {
                        check(&mut stack, &pos);
                        break;
                    }
                }
            }
        }
        assert!(exploded_kings > 0);
    }
}
