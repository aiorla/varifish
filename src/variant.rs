//! The variants varifish plays. A variant is a type parameter of the position
//! and of everything built on it, down to the search, so that each variant
//! compiles to its own code and none tests at run time which variant it
//! plays. What differs between variants goes through the `Variant` trait,
//! whose functions each variant implements with its own rules, or is decided
//! by its constants, which compile away.

use std::fmt;

use crate::movegen::{GenType, Moves};
use crate::nnue::eval_file::{EvalFile, NetworkFile, Networks, UseNnue};
use crate::nnue::{Accumulators, Caches};
use crate::position::{Position, Prefetcher};
use crate::search::{SharedState, Worker};
use crate::syzygy::{SyzygyFormat, Tablebases, Tables};
use crate::types::*;

/// The rules of a variant.
pub trait Variant: Copy + Default + fmt::Debug + Eq + Send + Sync + 'static {
    /// The variant's name as UCI_Variant gives it
    const NAME: &'static str;
    /// Whether captures explode, as in Atomic chess
    const ATOMIC: bool;

    /// The commands and positions `bench` searches by default
    const BENCH: &'static [&'static str];
    /// How the variant's Syzygy tables are stored
    const SYZYGY: SyzygyFormat;

    /// What a state keeps beyond chess's to undo a move
    type StateExtra: Copy + Default + fmt::Debug + Eq + Send + Sync;
    /// What a move changes for the variant's network
    type Dirties: MoveDirties;
    /// The network that evaluates the variant
    type Network: NetworkFile;
    /// The network's accumulators along a search's line
    type Accumulators: Accumulators<Dirties = Self::Dirties>;
    /// A search thread's caches of the network's accumulators
    type Caches: Caches<Network = Self::Network>;

    /// The variant's network among those loaded.
    fn eval_file(networks: &Networks) -> &EvalFile<Self::Network>;

    /// The variant's tables among those found.
    fn tables(tablebases: &Tablebases) -> &Tables<Self>;

    /// The evaluation of a position not in check, from the side to move's
    /// point of view, whose state the stack's latest state holds, by the
    /// network if `use_nnue` asks for it, which is then loaded.
    fn evaluate(
        network: Option<&Self::Network>,
        pos: &Position<Self>,
        accumulators: &mut Self::Accumulators,
        caches: &mut Self::Caches,
        optimism: i32,
        use_nnue: UseNnue,
    ) -> Value;

    /// The terms of the evaluation, for `eval`, by the network if
    /// `use_nnue` asks for it, which is then loaded.
    fn trace(network: Option<&Self::Network>, pos: &Position<Self>, use_nnue: UseNnue) -> String;

    /// Appends the legal moves of kind `T` to `moves`.
    fn generate<const T: GenType>(pos: &Position<Self>, moves: &mut Moves);

    /// Tests whether a move, possibly corrupt as moves from the transposition
    /// table can be, is legal.
    fn legal(pos: &Position<Self>, m: Move) -> bool;

    /// Tests whether a legal move gives check.
    fn gives_check(pos: &Position<Self>, m: Move) -> bool;

    /// Plays a legal move; `gives_check` tells whether it gives check.
    fn do_move<P: Prefetcher>(
        pos: &mut Position<Self>,
        m: Move,
        gives_check: bool,
        dirties: &mut Self::Dirties,
        prefetcher: &P,
    );

    /// Unmakes the last move played, `m`.
    fn undo_move(pos: &mut Position<Self>, m: Move);

    /// Tests whether the static exchange evaluation of a legal move is at
    /// least `threshold`.
    fn see_ge(pos: &Position<Self>, m: Move, threshold: i32) -> bool;

    /// Searches the root of `worker`'s search to `depth` in the window
    /// (alpha, beta), with the variant's search: Stockfish's for chess,
    /// Atomic-Stockfish's for Atomic.
    fn search_root(
        worker: &mut Worker<Self>,
        sh: &SharedState,
        pos: &mut Position<Self>,
        alpha: Value,
        beta: Value,
        depth: Depth,
    ) -> Value;
}

/// Chess, and Chess960, as Stockfish plays them
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Chess;

/// Atomic chess, by Lichess's rules: a capture explodes the capturing piece,
/// the captured one and every piece but pawns next to the capture, and
/// exploding the opponent's king wins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Atomic;

/// A variant chosen at run time, as UCI_Variant names it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VariantKind {
    #[default]
    Chess,
    Atomic,
}

impl VariantKind {
    pub const ALL: [VariantKind; 2] = [VariantKind::Chess, VariantKind::Atomic];

    pub fn name(self) -> &'static str {
        match self {
            VariantKind::Chess => Chess::NAME,
            VariantKind::Atomic => Atomic::NAME,
        }
    }

    pub fn from_name(name: &str) -> Option<VariantKind> {
        VariantKind::ALL
            .into_iter()
            .find(|v| v.name().eq_ignore_ascii_case(name))
    }

    /// The commands and positions `bench` searches by default
    pub fn bench(self) -> &'static [&'static str] {
        match self {
            VariantKind::Chess => Chess::BENCH,
            VariantKind::Atomic => Atomic::BENCH,
        }
    }
}

impl Variant for Chess {
    const NAME: &'static str = "chess";
    const ATOMIC: bool = false;
    const BENCH: &'static [&'static str] = crate::benchmark::DEFAULTS;
    const SYZYGY: SyzygyFormat = crate::syzygy::CHESS;

    type StateExtra = ();
    type Dirties = Dirties;
    type Network = crate::nnue::network::Network;
    type Accumulators = crate::nnue::accumulator::AccumulatorStack;
    type Caches = crate::nnue::accumulator::AccumulatorCaches;

    #[inline(always)]
    fn eval_file(networks: &Networks) -> &EvalFile<Self::Network> {
        &networks.chess
    }

    #[inline(always)]
    fn tables(tablebases: &Tablebases) -> &Tables<Chess> {
        &tablebases.chess
    }

    /// Chess is always evaluated by its network.
    #[inline(always)]
    fn evaluate(
        network: Option<&Self::Network>,
        pos: &Position<Chess>,
        accumulators: &mut Self::Accumulators,
        caches: &mut Self::Caches,
        optimism: i32,
        _use_nnue: UseNnue,
    ) -> Value {
        let network = network.expect("searches check that the network is loaded");
        crate::evaluate::evaluate(network, pos, accumulators, caches, optimism)
    }

    fn trace(network: Option<&Self::Network>, pos: &Position<Chess>, _use_nnue: UseNnue) -> String {
        crate::evaluate::trace(pos, network.expect("the network is verified"))
    }

    #[inline(always)]
    fn generate<const T: GenType>(pos: &Position<Chess>, moves: &mut Moves) {
        crate::movegen::generate::<T>(pos, moves);
    }

    #[inline(always)]
    fn legal(pos: &Position<Chess>, m: Move) -> bool {
        pos.chess_legal(m)
    }

    #[inline(always)]
    fn gives_check(pos: &Position<Chess>, m: Move) -> bool {
        pos.chess_gives_check(m)
    }

    #[inline(always)]
    fn do_move<P: Prefetcher>(
        pos: &mut Position<Chess>,
        m: Move,
        gives_check: bool,
        dirties: &mut Dirties,
        prefetcher: &P,
    ) {
        pos.chess_do_move(m, gives_check, dirties, prefetcher);
    }

    #[inline(always)]
    fn undo_move(pos: &mut Position<Chess>, m: Move) {
        pos.chess_undo_move(m);
    }

    #[inline(always)]
    fn see_ge(pos: &Position<Chess>, m: Move, threshold: i32) -> bool {
        pos.chess_see_ge(m, threshold)
    }

    #[inline(always)]
    fn search_root(
        worker: &mut Worker<Chess>,
        sh: &SharedState,
        pos: &mut Position<Chess>,
        alpha: Value,
        beta: Value,
        depth: Depth,
    ) -> Value {
        worker.chess_search_root(sh, pos, alpha, beta, depth)
    }
}

impl Variant for Atomic {
    const NAME: &'static str = "atomic";
    const ATOMIC: bool = true;
    const BENCH: &'static [&'static str] = crate::benchmark::ATOMIC_DEFAULTS;
    const SYZYGY: SyzygyFormat = crate::syzygy::ATOMIC;

    type StateExtra = crate::position::AtomicState;
    type Dirties = AtomicDirties;
    type Network = crate::nnue::atomic::network::Network;
    type Accumulators = crate::nnue::atomic::accumulator::AccumulatorStack;
    type Caches = crate::nnue::atomic::accumulator::AccumulatorCaches;

    #[inline(always)]
    fn eval_file(networks: &Networks) -> &EvalFile<Self::Network> {
        &networks.atomic
    }

    #[inline(always)]
    fn tables(tablebases: &Tablebases) -> &Tables<Atomic> {
        &tablebases.atomic
    }

    #[inline(always)]
    fn evaluate(
        network: Option<&Self::Network>,
        pos: &Position<Atomic>,
        accumulators: &mut Self::Accumulators,
        caches: &mut Self::Caches,
        _optimism: i32,
        use_nnue: UseNnue,
    ) -> Value {
        crate::evaluate::atomic::evaluate(network, pos, accumulators, caches, use_nnue)
    }

    fn trace(network: Option<&Self::Network>, pos: &Position<Atomic>, use_nnue: UseNnue) -> String {
        crate::evaluate::atomic::trace(pos, network, use_nnue)
    }

    #[inline(always)]
    fn generate<const T: GenType>(pos: &Position<Atomic>, moves: &mut Moves) {
        pos.atomic_generate::<T>(moves);
    }

    #[inline(always)]
    fn legal(pos: &Position<Atomic>, m: Move) -> bool {
        pos.atomic_legal(m)
    }

    #[inline(always)]
    fn gives_check(pos: &Position<Atomic>, m: Move) -> bool {
        pos.atomic_gives_check(m)
    }

    #[inline(always)]
    fn do_move<P: Prefetcher>(
        pos: &mut Position<Atomic>,
        m: Move,
        gives_check: bool,
        dirties: &mut AtomicDirties,
        prefetcher: &P,
    ) {
        pos.atomic_do_move(m, gives_check, dirties, prefetcher);
    }

    #[inline(always)]
    fn undo_move(pos: &mut Position<Atomic>, m: Move) {
        pos.atomic_undo_move(m);
    }

    #[inline(always)]
    fn see_ge(pos: &Position<Atomic>, m: Move, threshold: i32) -> bool {
        pos.atomic_see_ge(m, threshold)
    }

    #[inline(always)]
    fn search_root(
        worker: &mut Worker<Atomic>,
        sh: &SharedState,
        pos: &mut Position<Atomic>,
        alpha: Value,
        beta: Value,
        depth: Depth,
    ) -> Value {
        worker.atomic_search_root(sh, pos, alpha, beta, depth)
    }
}
