//! The variants varifish plays. A variant is a type parameter of the position
//! and of everything built on it, down to the search, so that each variant
//! compiles to its own code and none tests at run time which variant it
//! plays. What differs between variants goes through the `Variant` trait,
//! whose functions each variant implements with its own rules, or is decided
//! by its constants, which compile away.

use std::fmt;
use std::path::Path;

use crate::memory::LargePageBox;
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
    /// Whether a third check wins immediately, as in Three-check
    const THREE_CHECK: bool = false;
    /// Whether captured pieces enter pockets and can be dropped, as in Crazyhouse
    const CRAZYHOUSE: bool = false;
    /// Whether captures are mandatory, there is no check/castling, and losing
    /// all pieces or being stalemated wins, as in Antichess
    const ANTICHESS: bool = false;

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
    type Accumulators: Accumulators<Self, Dirties = Self::Dirties>;
    /// A search thread's caches of the network's accumulators
    type Caches: Caches<Network = Self::Network>;

    /// Checks remaining for `c` (meaningful only when `Self::THREE_CHECK`).
    #[inline(always)]
    fn checks_remaining(_extra: &Self::StateExtra, _c: Color) -> u8 {
        0
    }

    /// Sets checks remaining for `c` (meaningful only when `Self::THREE_CHECK`).
    #[inline(always)]
    fn set_checks_remaining(_extra: &mut Self::StateExtra, _c: Color, _n: u8) {}

    /// Count of `pt` in `c`'s pocket (meaningful only when `Self::CRAZYHOUSE`).
    #[inline(always)]
    fn hand_count(_extra: &Self::StateExtra, _c: Color, _pt: PieceType) -> u8 {
        0
    }

    /// Whether `c` has any piece in hand (meaningful only when `Self::CRAZYHOUSE`).
    #[inline(always)]
    fn has_hand(_extra: &Self::StateExtra, _c: Color) -> bool {
        false
    }

    /// Adds one `pt` to `c`'s pocket (meaningful only when `Self::CRAZYHOUSE`).
    #[inline(always)]
    fn add_hand(_extra: &mut Self::StateExtra, _c: Color, _pt: PieceType) {}

    /// Whether `sq` holds a promoted piece (meaningful only when `Self::CRAZYHOUSE`).
    #[inline(always)]
    fn is_promoted(_extra: &Self::StateExtra, _sq: Square) -> bool {
        false
    }

    /// Marks `sq` as a promoted piece (meaningful only when `Self::CRAZYHOUSE`).
    #[inline(always)]
    fn set_promoted(_extra: &mut Self::StateExtra, _sq: Square) {}

    /// Bitboard of promoted pieces (meaningful only when `Self::CRAZYHOUSE`).
    #[inline(always)]
    fn promoted_bb(_extra: &Self::StateExtra) -> crate::bitboard::Bitboard {
        crate::bitboard::Bitboard::EMPTY
    }

    /// Zobrist contribution of Crazyhouse's extra state.
    #[inline(always)]
    fn crazyhouse_key(_extra: &Self::StateExtra) -> Key {
        0
    }

    /// Appends `[QRBNPqrbnp]` to `out` when `Self::CRAZYHOUSE`.
    #[inline(always)]
    fn write_hand_fen(_extra: &Self::StateExtra, _out: &mut String) {}

    /// Updates Crazyhouse extra state for a capture and returns the Zobrist delta.
    #[inline(always)]
    fn on_crazyhouse_capture(
        _extra: &mut Self::StateExtra,
        _captor: Color,
        _capsq: Square,
        _captured_pt: PieceType,
    ) -> Key {
        0
    }

    /// Updates Crazyhouse extra state for a non-promotion, non-castling board
    /// move and returns the Zobrist delta.
    #[inline(always)]
    fn on_crazyhouse_move(_extra: &mut Self::StateExtra, _from: Square, _to: Square) -> Key {
        0
    }

    /// Updates Crazyhouse extra state for a promotion and returns the Zobrist delta.
    #[inline(always)]
    fn on_crazyhouse_promotion(_extra: &mut Self::StateExtra, _to: Square) -> Key {
        0
    }

    /// Copies persistent extra state across a move or null move.
    #[inline(always)]
    fn carry_extra(_dst: &mut Self::StateExtra, _src: &Self::StateExtra) {}

    /// Whether two states' extra fields denote the same position for
    /// repetition detection.
    #[inline(always)]
    fn same_extra(_a: &Self::StateExtra, _b: &Self::StateExtra) -> bool {
        true
    }

    /// The variant's network among those loaded.
    fn eval_file(networks: &Networks) -> &EvalFile<Self::Network>;

    /// Verifies that the variant has a usable network loaded for `eval_file_path`.
    fn verify_network(networks: &Networks, eval_file_path: &Path) -> Result<String, String> {
        networks.verify(Self::eval_file(networks), eval_file_path)
    }

    /// Allocates accumulator caches initialized for whichever network evaluates
    /// this variant.
    fn new_caches(networks: &Networks) -> LargePageBox<Self::Caches> {
        Self::Caches::new(Self::eval_file(networks).network())
    }

    /// Clears accumulator caches for whichever network evaluates this variant.
    fn clear_caches(caches: &mut Self::Caches, networks: &Networks) {
        if let Some(network) = Self::eval_file(networks).network() {
            caches.clear(network);
        }
    }

    /// The variant's tables among those found.
    fn tables(tablebases: &Tablebases) -> &Tables<Self>;

    /// The evaluation of a position not in check, from the side to move's
    /// point of view, whose state the stack's latest state holds, by the
    /// network if `use_nnue` asks for it, which is then loaded.
    fn evaluate(
        networks: &Networks,
        pos: &Position<Self>,
        accumulators: &mut Self::Accumulators,
        caches: &mut Self::Caches,
        use_nnue: UseNnue,
    ) -> Value;

    /// The terms of the evaluation, for `eval`, by the network if
    /// `use_nnue` asks for it, which is then loaded.
    fn trace(networks: &Networks, pos: &Position<Self>, use_nnue: UseNnue) -> String;

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
    /// (alpha, beta), with the variant's search: Stockfish's for chess and
    /// Three-check, Atomic-Stockfish's for Atomic.
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

/// Three-check, by Lichess's rules: chess's moves, and the first side to give
/// three checks wins immediately.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThreeCheck;

/// Crazyhouse, by Lichess's rules: captured pieces enter the captor's pocket
/// (promoted pieces as pawns) and can be dropped onto empty squares.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Crazyhouse;

/// Antichess, by Lichess's rules: captures are mandatory, there is no check or
/// castling, kings may be captured or promoted to, and losing all pieces or
/// being stalemated wins immediately.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Antichess;

/// A variant chosen at run time, as UCI_Variant names it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VariantKind {
    #[default]
    Chess,
    Atomic,
    ThreeCheck,
    Crazyhouse,
    Antichess,
}

impl VariantKind {
    pub const ALL: [VariantKind; 5] = [
        VariantKind::Chess,
        VariantKind::Atomic,
        VariantKind::ThreeCheck,
        VariantKind::Crazyhouse,
        VariantKind::Antichess,
    ];

    pub fn name(self) -> &'static str {
        match self {
            VariantKind::Chess => Chess::NAME,
            VariantKind::Atomic => Atomic::NAME,
            VariantKind::ThreeCheck => ThreeCheck::NAME,
            VariantKind::Crazyhouse => Crazyhouse::NAME,
            VariantKind::Antichess => Antichess::NAME,
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
            VariantKind::ThreeCheck => ThreeCheck::BENCH,
            VariantKind::Crazyhouse => Crazyhouse::BENCH,
            VariantKind::Antichess => Antichess::BENCH,
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
        networks: &Networks,
        pos: &Position<Chess>,
        accumulators: &mut Self::Accumulators,
        caches: &mut Self::Caches,
        _use_nnue: UseNnue,
    ) -> Value {
        let network = networks
            .chess
            .network()
            .expect("searches check that the network is loaded");
        crate::evaluate::evaluate(network, pos, accumulators, caches)
    }

    fn trace(networks: &Networks, pos: &Position<Chess>, _use_nnue: UseNnue) -> String {
        crate::evaluate::trace(
            pos,
            networks.chess.network().expect("the network is verified"),
        )
    }

    #[inline(always)]
    fn generate<const T: GenType>(pos: &Position<Chess>, moves: &mut Moves) {
        crate::movegen::generate::<T, Chess>(pos, moves);
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
    type Network = crate::nnue::atomic::AtomicNetwork;
    type Accumulators = crate::nnue::atomic::AtomicAccumulators;
    type Caches = crate::nnue::atomic::AtomicCaches;

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
        networks: &Networks,
        pos: &Position<Atomic>,
        accumulators: &mut Self::Accumulators,
        caches: &mut Self::Caches,
        use_nnue: UseNnue,
    ) -> Value {
        crate::evaluate::atomic::evaluate(
            networks.atomic.network(),
            pos,
            accumulators,
            caches,
            use_nnue,
        )
    }

    fn trace(networks: &Networks, pos: &Position<Atomic>, use_nnue: UseNnue) -> String {
        crate::evaluate::atomic::trace(pos, networks.atomic.network(), use_nnue)
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

impl Variant for ThreeCheck {
    const NAME: &'static str = "3check";
    const ATOMIC: bool = false;
    const THREE_CHECK: bool = true;
    const BENCH: &'static [&'static str] = crate::benchmark::THREE_CHECK_DEFAULTS;
    const SYZYGY: SyzygyFormat = crate::syzygy::THREE_CHECK;

    type StateExtra = crate::position::ChecksRemaining;
    type Dirties = Dirties;
    type Network = crate::nnue::threecheck::ThreeCheckNetwork;
    type Accumulators = crate::nnue::accumulator::AccumulatorStack;
    type Caches = crate::nnue::threecheck::ThreeCheckCaches;

    #[inline(always)]
    fn checks_remaining(extra: &Self::StateExtra, c: Color) -> u8 {
        extra.get(c)
    }

    #[inline(always)]
    fn set_checks_remaining(extra: &mut Self::StateExtra, c: Color, n: u8) {
        extra.set(c, n);
    }

    #[inline(always)]
    fn carry_extra(dst: &mut Self::StateExtra, src: &Self::StateExtra) {
        *dst = *src;
    }

    #[inline(always)]
    fn same_extra(a: &Self::StateExtra, b: &Self::StateExtra) -> bool {
        a == b
    }

    #[inline(always)]
    fn eval_file(networks: &Networks) -> &EvalFile<Self::Network> {
        &networks.threecheck
    }

    fn verify_network(networks: &Networks, eval_file_path: &Path) -> Result<String, String> {
        networks.verify_threecheck(eval_file_path)
    }

    fn new_caches(networks: &Networks) -> LargePageBox<Self::Caches> {
        let mut caches = Self::Caches::new(None);
        Self::clear_caches(&mut caches, networks);
        caches
    }

    fn clear_caches(caches: &mut Self::Caches, networks: &Networks) {
        if let Some(network) = networks.threecheck.network() {
            caches.clear(network);
        } else if let Some(chess) = networks.chess.network() {
            caches.0.clear(chess);
        }
    }

    #[inline(always)]
    fn tables(tablebases: &Tablebases) -> &Tables<ThreeCheck> {
        &tablebases.threecheck
    }

    #[inline(always)]
    fn evaluate(
        networks: &Networks,
        pos: &Position<ThreeCheck>,
        accumulators: &mut Self::Accumulators,
        caches: &mut Self::Caches,
        use_nnue: UseNnue,
    ) -> Value {
        crate::evaluate::threecheck::evaluate(networks, pos, accumulators, caches, use_nnue)
    }

    fn trace(networks: &Networks, pos: &Position<ThreeCheck>, use_nnue: UseNnue) -> String {
        crate::evaluate::threecheck::trace(networks, pos, use_nnue)
    }

    #[inline(always)]
    fn generate<const T: GenType>(pos: &Position<ThreeCheck>, moves: &mut Moves) {
        pos.three_check_generate::<T>(moves);
    }

    #[inline(always)]
    fn legal(pos: &Position<ThreeCheck>, m: Move) -> bool {
        pos.three_check_legal(m)
    }

    #[inline(always)]
    fn gives_check(pos: &Position<ThreeCheck>, m: Move) -> bool {
        pos.chess_gives_check(m)
    }

    #[inline(always)]
    fn do_move<P: Prefetcher>(
        pos: &mut Position<ThreeCheck>,
        m: Move,
        gives_check: bool,
        dirties: &mut Dirties,
        prefetcher: &P,
    ) {
        pos.chess_do_move(m, gives_check, dirties, prefetcher);
    }

    #[inline(always)]
    fn undo_move(pos: &mut Position<ThreeCheck>, m: Move) {
        pos.chess_undo_move(m);
    }

    #[inline(always)]
    fn see_ge(pos: &Position<ThreeCheck>, m: Move, threshold: i32) -> bool {
        pos.chess_see_ge(m, threshold)
    }

    #[inline(always)]
    fn search_root(
        worker: &mut Worker<ThreeCheck>,
        sh: &SharedState,
        pos: &mut Position<ThreeCheck>,
        alpha: Value,
        beta: Value,
        depth: Depth,
    ) -> Value {
        worker.chess_search_root(sh, pos, alpha, beta, depth)
    }
}

impl Variant for Crazyhouse {
    const NAME: &'static str = "crazyhouse";
    const ATOMIC: bool = false;
    const CRAZYHOUSE: bool = true;
    const BENCH: &'static [&'static str] = crate::benchmark::CRAZYHOUSE_DEFAULTS;
    const SYZYGY: SyzygyFormat = crate::syzygy::CRAZYHOUSE;

    type StateExtra = crate::position::CrazyhouseState;
    type Dirties = Dirties;
    type Network = crate::nnue::crazyhouse::CrazyhouseNetwork;
    type Accumulators = crate::nnue::accumulator::AccumulatorStack;
    type Caches = crate::nnue::crazyhouse::CrazyhouseCaches;

    #[inline(always)]
    fn hand_count(extra: &Self::StateExtra, c: Color, pt: PieceType) -> u8 {
        extra.count(c, pt)
    }

    #[inline(always)]
    fn has_hand(extra: &Self::StateExtra, c: Color) -> bool {
        extra.has_any(c)
    }

    #[inline(always)]
    fn add_hand(extra: &mut Self::StateExtra, c: Color, pt: PieceType) {
        extra.add(c, pt);
    }

    #[inline(always)]
    fn is_promoted(extra: &Self::StateExtra, sq: Square) -> bool {
        extra.is_promoted(sq)
    }

    #[inline(always)]
    fn set_promoted(extra: &mut Self::StateExtra, sq: Square) {
        extra.set_promoted(sq);
    }

    #[inline(always)]
    fn promoted_bb(extra: &Self::StateExtra) -> crate::bitboard::Bitboard {
        extra.promoted
    }

    #[inline(always)]
    fn crazyhouse_key(extra: &Self::StateExtra) -> Key {
        crate::position::crazyhouse_extra_key(extra)
    }

    #[inline(always)]
    fn write_hand_fen(extra: &Self::StateExtra, out: &mut String) {
        crate::position::write_hand_fen(extra, out);
    }

    #[inline(always)]
    fn on_crazyhouse_capture(
        extra: &mut Self::StateExtra,
        captor: Color,
        capsq: Square,
        captured_pt: PieceType,
    ) -> Key {
        crate::position::on_capture(extra, captor, capsq, captured_pt)
    }

    #[inline(always)]
    fn on_crazyhouse_move(extra: &mut Self::StateExtra, from: Square, to: Square) -> Key {
        crate::position::on_board_move(extra, from, to)
    }

    #[inline(always)]
    fn on_crazyhouse_promotion(extra: &mut Self::StateExtra, to: Square) -> Key {
        crate::position::on_promotion(extra, to)
    }

    #[inline(always)]
    fn carry_extra(dst: &mut Self::StateExtra, src: &Self::StateExtra) {
        *dst = *src;
    }

    #[inline(always)]
    fn same_extra(a: &Self::StateExtra, b: &Self::StateExtra) -> bool {
        a == b
    }

    #[inline(always)]
    fn eval_file(networks: &Networks) -> &EvalFile<Self::Network> {
        &networks.crazyhouse
    }

    fn verify_network(networks: &Networks, eval_file_path: &Path) -> Result<String, String> {
        networks.verify_crazyhouse(eval_file_path)
    }

    fn new_caches(networks: &Networks) -> LargePageBox<Self::Caches> {
        let mut caches = Self::Caches::new(None);
        Self::clear_caches(&mut caches, networks);
        caches
    }

    fn clear_caches(caches: &mut Self::Caches, networks: &Networks) {
        if let Some(network) = networks.crazyhouse.network() {
            caches.clear(network);
        } else if let Some(chess) = networks.chess.network() {
            caches.0.clear(chess);
        }
    }

    #[inline(always)]
    fn tables(tablebases: &Tablebases) -> &Tables<Crazyhouse> {
        &tablebases.crazyhouse
    }

    #[inline(always)]
    fn evaluate(
        networks: &Networks,
        pos: &Position<Crazyhouse>,
        accumulators: &mut Self::Accumulators,
        caches: &mut Self::Caches,
        use_nnue: UseNnue,
    ) -> Value {
        crate::evaluate::crazyhouse::evaluate(networks, pos, accumulators, caches, use_nnue)
    }

    fn trace(networks: &Networks, pos: &Position<Crazyhouse>, use_nnue: UseNnue) -> String {
        crate::evaluate::crazyhouse::trace(networks, pos, use_nnue)
    }

    #[inline(always)]
    fn generate<const T: GenType>(pos: &Position<Crazyhouse>, moves: &mut Moves) {
        pos.crazyhouse_generate::<T>(moves);
    }

    #[inline(always)]
    fn legal(pos: &Position<Crazyhouse>, m: Move) -> bool {
        pos.crazyhouse_legal(m)
    }

    #[inline(always)]
    fn gives_check(pos: &Position<Crazyhouse>, m: Move) -> bool {
        pos.crazyhouse_gives_check(m)
    }

    #[inline(always)]
    fn do_move<P: Prefetcher>(
        pos: &mut Position<Crazyhouse>,
        m: Move,
        gives_check: bool,
        dirties: &mut Dirties,
        prefetcher: &P,
    ) {
        pos.crazyhouse_do_move(m, gives_check, dirties, prefetcher);
    }

    #[inline(always)]
    fn undo_move(pos: &mut Position<Crazyhouse>, m: Move) {
        pos.crazyhouse_undo_move(m);
    }

    #[inline(always)]
    fn see_ge(pos: &Position<Crazyhouse>, m: Move, threshold: i32) -> bool {
        pos.chess_see_ge(m, threshold)
    }

    #[inline(always)]
    fn search_root(
        worker: &mut Worker<Crazyhouse>,
        sh: &SharedState,
        pos: &mut Position<Crazyhouse>,
        alpha: Value,
        beta: Value,
        depth: Depth,
    ) -> Value {
        worker.chess_search_root(sh, pos, alpha, beta, depth)
    }
}

impl Variant for Antichess {
    const NAME: &'static str = "antichess";
    const ATOMIC: bool = false;
    const ANTICHESS: bool = true;
    const BENCH: &'static [&'static str] = crate::benchmark::ANTICHESS_DEFAULTS;
    const SYZYGY: SyzygyFormat = crate::syzygy::ANTICHESS;

    type StateExtra = ();
    type Dirties = Dirties;
    type Network = crate::nnue::antichess::AntichessNetwork;
    type Accumulators = crate::nnue::antichess::AntichessAccumulators;
    type Caches = crate::nnue::antichess::AntichessCaches;

    #[inline(always)]
    fn eval_file(networks: &Networks) -> &EvalFile<Self::Network> {
        &networks.antichess
    }

    fn verify_network(networks: &Networks, eval_file_path: &Path) -> Result<String, String> {
        networks.verify_antichess(eval_file_path)
    }

    #[inline(always)]
    fn tables(tablebases: &Tablebases) -> &Tables<Antichess> {
        &tablebases.antichess
    }

    #[inline(always)]
    fn evaluate(
        networks: &Networks,
        pos: &Position<Antichess>,
        accumulators: &mut Self::Accumulators,
        caches: &mut Self::Caches,
        use_nnue: UseNnue,
    ) -> Value {
        crate::evaluate::antichess::evaluate(networks, pos, accumulators, caches, use_nnue)
    }

    fn trace(networks: &Networks, pos: &Position<Antichess>, use_nnue: UseNnue) -> String {
        crate::evaluate::antichess::trace(networks, pos, use_nnue)
    }

    #[inline(always)]
    fn generate<const T: GenType>(pos: &Position<Antichess>, moves: &mut Moves) {
        pos.antichess_generate::<T>(moves);
    }

    #[inline(always)]
    fn legal(pos: &Position<Antichess>, m: Move) -> bool {
        pos.antichess_legal(m)
    }

    #[inline(always)]
    fn gives_check(_pos: &Position<Antichess>, _m: Move) -> bool {
        false
    }

    #[inline(always)]
    fn do_move<P: Prefetcher>(
        pos: &mut Position<Antichess>,
        m: Move,
        _gives_check: bool,
        dirties: &mut Dirties,
        prefetcher: &P,
    ) {
        pos.antichess_do_move(m, dirties, prefetcher);
    }

    #[inline(always)]
    fn undo_move(pos: &mut Position<Antichess>, m: Move) {
        pos.antichess_undo_move(m);
    }

    #[inline(always)]
    fn see_ge(_pos: &Position<Antichess>, _m: Move, _threshold: i32) -> bool {
        // Static exchange evaluation assumes material retention, which is
        // inverted in Antichess; search/movepick skip SEE pruning under
        // V::ANTICHESS.
        true
    }

    #[inline(always)]
    fn search_root(
        worker: &mut Worker<Antichess>,
        sh: &SharedState,
        pos: &mut Position<Antichess>,
        alpha: Value,
        beta: Value,
        depth: Depth,
    ) -> Value {
        worker.chess_search_root(sh, pos, alpha, beta, depth)
    }
}
