//! The search: iterative deepening with aspiration windows around a
//! principal variation search, a quiescence search at its leaves, and the
//! heuristics that prune, reduce and extend it.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `search.h` and `search.cpp`: Stockfish, a UCI chess playing engine derived
//! from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! AUTHORS), under the GNU General Public License, version 3 or later.
//!
//! Where Stockfish keeps per-node data on a stack of structs reached through
//! pointers, varifish indexes a stack owned by the worker: `ss` is the index
//! of the node's entry, the root's being `ROOT_SS`. The principal variation
//! that Stockfish builds in a local of each frame lives in `pv_table`, at
//! the frame's ply, and the continuation histories that its stack entries
//! point to are indices into the tables that hold them.
//!
//! Atomic searches its nodes with Atomic-Stockfish's search, in `atomic`,
//! and shares the iterative deepening with chess where they agree.

use std::collections::HashMap;
use std::ops::{Index, IndexMut};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::bitboard::pawn_single_push_bb;
use crate::history::{
    CORRECTION_HISTORY_LIMIT, Fill, FillShared, LOW_PLY_HISTORY_SIZE, PieceToHistory,
    SharedHistories, TTMoveHistory, WorkerHistories, continuation_correction_index,
    continuation_index,
};
use crate::memory::LargePageBox;
use crate::misc::{
    Prng, TimePoint, ValueList, clamp_f64, interpolate, interpolate_i64, min_f64, mul_add, prefetch,
};
use crate::movegen::{ALL, MOVE_LIST_SIZE, MoveList};
use crate::movepick::MovePicker;
use crate::nnue::eval_file::{Networks, UseNnue};
use crate::nnue::{Accumulators, Caches};
use crate::position::{Position, Prefetcher, StateInfo};
use crate::score::Score;
use crate::syzygy::{ProbeState, SyzygyOptions, Tablebases, Tables, TbConfig};
use crate::thread::ThreadHandle;
use crate::timeman::TimeManagement;
use crate::tt::TranspositionTable;
use crate::types::*;
use crate::uci::{move_to_uci, wdl};
use crate::variant::{Chess, Variant};

mod atomic;

// Different node types, used as a const generic parameter
type NodeType = u8;
const NON_PV: NodeType = 0;
const PV: NodeType = 1;
const ROOT: NodeType = 2;

#[inline(always)]
fn lmr_divisor(depth: i32) -> i32 {
    let d = depth.min(16);
    3000 + 7 * (d - 8) * (d - 8)
}

const NODES_LIMIT_OUTPUT: u64 = 10_000_000;

const SEARCHEDLIST_CAPACITY: i32 = 32;
type SearchedList = ValueList<Move, { SEARCHEDLIST_CAPACITY as usize }>;

/// The index of the root's entry in the search stack, which leaves room for
/// the seven entries before it that the root's continuation histories read.
const ROOT_SS: usize = 7;

/// The ply of the node at stack index `ss`.
#[inline(always)]
fn ply_of(ss: usize) -> i32 {
    ss as i32 - ROOT_SS as i32
}

// (*Scalers):
// The values with Scaler asterisks have proven non-linear scaling.
// They are optimized to time controls of 180 + 1.8 and longer,
// so changing them or adding conditions that are similar requires
// tests at these types of time controls.

// (*Scaler) All tuned parameters at time controls shorter than
// optimized for require verifications at longer time controls.

/// Reductions lookup table indexed by depth or move number, `scale` out of
/// 128 of the logarithm.
fn reductions_table(scale: i32) -> [i32; MOVE_LIST_SIZE] {
    let mut table = [0; MOVE_LIST_SIZE];
    for (i, r) in table.iter_mut().enumerate().skip(1) {
        *r = (scale as f64 / 128.0 * (i as f64).ln()) as i32;
    }
    table
}

/// The constants with which iterative deepening decides, after each
/// iteration, whether to search another.
struct TimeConstants {
    /// The falling eval's base, and its weights of the drops of the best
    /// value from the last search's average score and from the iteration's
    /// four before
    falling_eval: (f64, f64, f64),
    falling_eval_range: (f64, f64),
    /// The time reduction's line over the iterations since the best move
    /// last changed, as interpolate()'s (x0, x1, y0, y1), and its range
    time_reduction: (f64, f64, f64, f64),
    time_reduction_range: (f64, f64),
    /// The reduction's base and divisor
    reduction: (f64, f64),
    /// The best move instability's base and weight of its changes
    best_move_instability: (f64, f64),
    /// The high best move effort's line over the root move's share of the
    /// nodes, in hundred thousandths, and its range
    high_best_move_effort: (i64, i64, f64, f64),
    high_best_move_effort_range: (f64, f64),
    /// The time, in milliseconds, at most spent on a single legal move
    single_move_time: f64,
}

/// Stockfish's
const CHESS_TIME: TimeConstants = TimeConstants {
    falling_eval: (11.48, 2.30, 1.1),
    falling_eval_range: (0.576, 1.728),
    time_reduction: (4.96, 18.79, 0.639, 1.712),
    time_reduction_range: (0.629, 1.544),
    reduction: (1.468, 2.284),
    best_move_instability: (1.077, 2.229),
    high_best_move_effort: (75800, 104510, 0.969, 0.714),
    high_best_move_effort_range: (0.693, 0.838),
    single_move_time: 500.0,
};

/// Add correctionHistory value to raw staticEval and guarantee evaluation
/// does not hit the tablebase range.
#[inline(always)]
fn to_corrected_static_eval(v: Value, cv: i32) -> Value {
    (v + cv / 131072).clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

/// Add a small random component to draw evaluations to avoid 3-fold
/// blindness
#[inline(always)]
fn value_draw(nodes: u64) -> Value {
    VALUE_DRAW - 1 + (nodes & 0x2) as Value
}

/// Adjusts a mate or TB score from "plies to mate from the root" to "plies
/// to mate from the current position". Standard scores are unchanged. The
/// function is called before storing a value in the transposition table.
#[inline(always)]
fn value_to_tt(v: Value, ply: i32) -> Value {
    if is_win(v) {
        v + ply
    } else if is_loss(v) {
        v - ply
    } else {
        v
    }
}

/// Inverse of value_to_tt(): it adjusts a mate or TB score from the
/// transposition table (which refers to the plies to mate/be mated from
/// current position) to "plies to mate/be mated (TB win/loss) from the
/// root". However, to avoid potentially false mate or TB scores related to
/// the 50 moves rule and the graph history interaction, we return the
/// highest non-TB score instead.
#[inline(always)]
fn value_from_tt(v: Value, ply: i32, r50c: i32) -> Value {
    if !is_valid(v) {
        return VALUE_NONE;
    }

    // Handle TB win or better
    if is_win(v) {
        // Downgrade a potentially false mate score
        if is_mate(v) && VALUE_MATE - v > 100 - r50c {
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        }

        // Downgrade a potentially false TB score
        if VALUE_TB - v > 100 - r50c {
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        }

        return v - ply;
    }

    // Handle TB loss or worse
    if is_loss(v) {
        // Downgrade a potentially false mate score
        if is_mated(v) && VALUE_MATE + v > 100 - r50c {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }

        // Downgrade a potentially false TB score
        if VALUE_TB + v > 100 - r50c {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }

        return v + ply;
    }

    v
}

/// Increments a counter that only its thread writes.
#[inline(always)]
fn increment(counter: &AtomicU64) {
    counter.store(counter.load(Relaxed) + 1, Relaxed);
}

/// The principal variation of a node: its best move and the principal
/// variation of the node after it, at most MAX_PLY + 1 moves.
#[derive(Clone, Copy)]
pub struct PVMoves {
    moves: [Move; MAX_PLY as usize + 1],
    length: usize,
}

impl Default for PVMoves {
    fn default() -> Self {
        PVMoves {
            moves: [Move::none(); MAX_PLY as usize + 1],
            length: 0,
        }
    }
}

impl PVMoves {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.length]
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.length = 0;
    }

    /// Sets the variation to `mv` followed by `child_pv`, if any.
    #[inline(always)]
    pub fn update(&mut self, mv: Move, child_pv: Option<&PVMoves>) {
        debug_assert!(child_pv.is_none_or(|pv| pv.len() <= MAX_PLY as usize));
        self.length = child_pv.map_or(0, |pv| pv.length);

        if let Some(pv) = child_pv {
            self.moves[1..=self.length].copy_from_slice(&pv.moves[..self.length]);
        }

        self.moves[0] = mv;
        self.length += 1;
    }

    /// Sets the variation to a root move's, which syzygy_extend_pv() may
    /// have made longer than MAX_PLY moves.
    pub fn assign(&mut self, moves: &[Move]) {
        self.length = moves.len().min(MAX_PLY as usize);
        self.moves[..self.length].copy_from_slice(&moves[..self.length]);
    }
}

impl Index<usize> for PVMoves {
    type Output = Move;

    #[inline(always)]
    fn index(&self, index: usize) -> &Move {
        &self.moves[index]
    }
}

/// The principal variations of the nodes along the line being searched,
/// indexed by ply.
struct PvTable([PVMoves; 256]);

impl PvTable {
    /// Sets the principal variation of the node at `ply`, which its parent
    /// holds, to `mv` followed by the node's own if `child_pv`.
    #[inline(always)]
    fn update(&mut self, ply: i32, mv: Move, child_pv: bool) {
        let ply = ply as usize & 255;
        debug_assert!(ply > 0);
        let (parents, nodes) = self.0.split_at_mut(ply);
        parents[ply - 1].update(mv, child_pv.then_some(&nodes[0]));
    }
}

impl Index<i32> for PvTable {
    type Output = PVMoves;

    #[inline(always)]
    fn index(&self, ply: i32) -> &PVMoves {
        &self.0[ply as usize & 255]
    }
}

impl IndexMut<i32> for PvTable {
    #[inline(always)]
    fn index_mut(&mut self, ply: i32) -> &mut PVMoves {
        &mut self.0[ply as usize & 255]
    }
}

/// Stack struct keeps track of the information we need to remember from
/// nodes shallower and deeper in the tree during the search. Each search
/// thread has its own array of Stack objects, indexed by the current ply.
///
/// The continuation histories are indices: `continuation_history` into the
/// shared continuation histories and `continuation_correction_history` into
/// the thread's continuation correction histories, where 0 is the sentinel
/// that Stockfish's stack points to before the root and after null moves.
#[derive(Clone, Copy, Default)]
pub struct Stack {
    continuation_history: u16,
    continuation_correction_history: u16,
    current_move: Move,
    excluded_move: Move,
    static_eval: Value,
    stat_score: i32,
    move_count: i32,
    in_check: bool,
    tt_pv: bool,
    tt_hit: bool,
    follow_pv: bool,
    cutoff_cnt: i32,
    reduction: i32,
    prior_nmp_fail_high: i32,
}

/// The stack, from seven entries before the root, which the continuation
/// histories reach back to, to two after the deepest ply.
struct SearchStack([Stack; 256]);

const _: () = assert!(ROOT_SS + MAX_PLY as usize + 3 <= 256);

// Unchecked, as Stockfish's pointers are: a node's entry is ROOT_SS plus its
// ply, which search and qsearch keep within MAX_PLY, returning at it, and a
// node reaches at most seven entries before its own and two after. A mask in
// place of the check would keep every offset's address apart, where without
// it they are all one address and constant displacements.
impl Index<usize> for SearchStack {
    type Output = Stack;

    #[inline(always)]
    fn index(&self, ss: usize) -> &Stack {
        debug_assert!(ss < self.0.len());
        // SAFETY: in bounds, as above.
        unsafe { self.0.get_unchecked(ss) }
    }
}

impl IndexMut<usize> for SearchStack {
    #[inline(always)]
    fn index_mut(&mut self, ss: usize) -> &mut Stack {
        debug_assert!(ss < self.0.len());
        // SAFETY: in bounds, as above.
        unsafe { self.0.get_unchecked_mut(ss) }
    }
}

/// RootMove struct is used for moves at the root of the tree. For each root
/// move we store a score and a PV (really a refutation in the case of moves
/// which fail low). Score is normally set at -VALUE_INFINITE for all non-pv
/// moves.
#[derive(Clone, Debug)]
pub struct RootMove {
    pub effort: u64,
    pub score: Value,
    pub previous_score: Value,
    pub average_score: Value,
    pub mean_squared_score: Value,
    pub uci_score: Value,
    // By default root scores are exact, unless flagged as a one-sided bound
    // here. See also the bounds in types.rs.
    pub inexact_lower: bool,
    pub inexact_upper: bool,
    pub previous_score_exact: bool,
    pub sel_depth: i32,
    pub tb_rank: i32,
    pub tb_score: Value,
    // syzygy_extend_pv() may lead to PVs longer than MAX_PLY
    pub pv: Vec<Move>,
    pub previous_pv: Vec<Move>,
}

impl RootMove {
    pub fn new(m: Move) -> RootMove {
        let mut pv = Vec::with_capacity(MAX_PLY as usize);
        pv.push(m);
        RootMove {
            effort: 0,
            score: -VALUE_INFINITE,
            previous_score: -VALUE_INFINITE,
            average_score: -VALUE_INFINITE,
            mean_squared_score: -VALUE_INFINITE * VALUE_INFINITE,
            uci_score: -VALUE_INFINITE,
            inexact_lower: false,
            inexact_upper: false,
            previous_score_exact: false,
            sel_depth: 0,
            tb_rank: 0,
            tb_score: 0,
            pv,
            previous_pv: Vec::with_capacity(MAX_PLY as usize),
        }
    }

    pub fn is_inexact(&self) -> bool {
        self.inexact_lower || self.inexact_upper
    }

    pub fn is_exact_loss(&self) -> bool {
        self.score != -VALUE_INFINITE && is_loss(self.score) && !self.is_inexact()
    }

    pub fn unset_inexact(&mut self) {
        self.inexact_lower = false;
        self.inexact_upper = false;
    }

    /// Called in case we have no ponder move before exiting the search, for
    /// instance, in case we stop the search during a fail high at root. We
    /// try hard to have a ponder move to return to the GUI, otherwise in
    /// case of 'ponder on' we have nothing to think about.
    pub fn extract_ponder_from_tt<V: Variant>(
        &mut self,
        tt: &TranspositionTable,
        pos: &mut Position<V>,
        scratch: &mut V::Dirties,
    ) -> bool {
        debug_assert!(self.pv.len() == 1 && self.pv[0] != Move::none());

        pos.make_move(self.pv[0], scratch);

        if !pos.is_draw(1) {
            let (tt_hit, tt_data, _) = tt.probe(pos.key());
            if tt_hit && tt_data.mv != Move::none() && pos.legal(tt_data.mv) {
                self.pv.push(tt_data.mv);
            }
        }

        pos.undo_move(self.pv[0]);
        self.pv.len() > 1
    }
}

/// The order of root moves: descending by score, and then by previous
/// score, as Stockfish's `RootMove::operator<` sorts them.
fn root_order(a: &RootMove, b: &RootMove) -> std::cmp::Ordering {
    b.score
        .cmp(&a.score)
        .then(b.previous_score.cmp(&a.previous_score))
}

/// Whether `a` sorts before `b`.
fn root_less(a: &RootMove, b: &RootMove) -> bool {
    root_order(a, b) == std::cmp::Ordering::Less
}

/// LimitsType struct stores information sent by the caller about the
/// analysis required.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LimitsType {
    pub searchmoves: Vec<String>,
    pub time: [TimePoint; COLOR_NB],
    pub inc: [TimePoint; COLOR_NB],
    pub npmsec: TimePoint,
    pub movetime: TimePoint,
    pub start_time: TimePoint,
    pub movestogo: i32,
    pub depth: i32,
    pub mate: i32,
    pub perft: i32,
    pub infinite: i32,
    pub nodes: u64,
    pub ponder_mode: bool,
}

impl LimitsType {
    pub fn use_time_management(&self) -> bool {
        self.time[WHITE.idx()] != 0 || self.time[BLACK.idx()] != 0
    }
}

/// The UCI options that a search reads, as they are when it starts.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchOptions {
    pub multi_pv: usize,
    pub skill_level: i32,
    pub limit_strength: bool,
    pub uci_elo: i32,
    pub show_wdl: bool,
    pub move_overhead: i32,
    pub nodestime: i32,
    pub ponder: bool,
    /// How Atomic is evaluated
    pub use_nnue: UseNnue,
    /// How the tablebases are probed
    pub syzygy: SyzygyOptions,
}

/// What the threads of a search share: the transposition table, the shared
/// histories, the networks, the tablebases, and the flags that steer the
/// search.
pub struct SharedState {
    pub tt: TranspositionTable,
    pub histories: SharedHistories,
    pub networks: Networks,
    pub tablebases: Tablebases,
    pub stop: AtomicBool,
    pub increase_depth: AtomicBool,
    /// Whether the main thread is pondering, which "ponderhit" ends
    pub ponder: AtomicBool,
}

impl SharedState {
    pub fn new(networks: Networks) -> SharedState {
        SharedState {
            tt: TranspositionTable::default(),
            histories: SharedHistories::new(1),
            networks,
            tablebases: Tablebases::default(),
            stop: AtomicBool::new(false),
            increase_depth: AtomicBool::new(false),
            ponder: AtomicBool::new(false),
        }
    }
}

/// The counters of a thread that other threads read, on a cache line of
/// their own.
#[repr(align(64))]
#[derive(Debug, Default)]
pub struct Counters {
    pub nodes: AtomicU64,
    pub tb_hits: AtomicU64,
    pub best_move_changes: AtomicU64,
}

impl Counters {
    pub fn reset(&self) {
        self.nodes.store(0, Relaxed);
        self.tb_hits.store(0, Relaxed);
        self.best_move_changes.store(0, Relaxed);
    }
}

pub struct InfoShort {
    pub depth: i32,
    pub score: Score,
}

pub struct InfoFull<'a> {
    pub depth: i32,
    pub score: Score,
    pub sel_depth: i32,
    pub multi_pv: usize,
    pub wdl: &'a str,
    pub bound: &'a str,
    pub time_ms: usize,
    pub nodes: usize,
    pub nps: usize,
    pub tb_hits: usize,
    pub pv: &'a str,
    pub hashfull: i32,
}

pub struct InfoIteration<'a> {
    pub depth: i32,
    pub currmove: &'a str,
    pub currmovenumber: usize,
}

pub type UpdateShort = Arc<dyn Fn(&InfoShort) + Send + Sync>;
pub type UpdateFull = Arc<dyn Fn(&InfoFull) + Send + Sync>;
pub type UpdateIter = Arc<dyn Fn(&InfoIteration) + Send + Sync>;
pub type UpdateBestmove = Arc<dyn Fn(&str, &str) + Send + Sync>;
pub type UpdateStart = Arc<dyn Fn() + Send + Sync>;
pub type UpdateInfoString = Arc<dyn Fn(&str) + Send + Sync>;

/// Where a search reports: its lines, iterations, best move and start,
/// and the info strings that Stockfish's search prints itself.
#[derive(Clone)]
pub struct UpdateContext {
    pub on_update_no_moves: UpdateShort,
    pub on_update_full: UpdateFull,
    pub on_iter: UpdateIter,
    pub on_bestmove: UpdateBestmove,
    pub on_start: UpdateStart,
    pub on_info_string: UpdateInfoString,
}

impl Default for UpdateContext {
    fn default() -> Self {
        UpdateContext {
            on_update_no_moves: Arc::new(|_| {}),
            on_update_full: Arc::new(|_| {}),
            on_iter: Arc::new(|_| {}),
            on_bestmove: Arc::new(|_, _| {}),
            on_start: Arc::new(|| {}),
            on_info_string: Arc::new(|_| {}),
        }
    }
}

/// The generator of the skill level's random choices, seeded once from the
/// clock, so that its sequence is not deterministic.
static SKILL_RNG: Mutex<Option<Prng>> = Mutex::new(None);

/// Skill structure is used to implement strength limit. If we have a
/// UCI_Elo, we convert it to an appropriate skill level, anchored to the
/// Stash engine. This method is based on a fit of the Elo results for games
/// played between Stockfish at various skill levels and various versions of
/// the Stash engine. Skill 0 .. 19 now covers CCRL Blitz Elo from 1320 to
/// 3190, approximately. Reference:
/// https://github.com/vondele/Stockfish/commit/a08b8d4e9711c2
pub struct Skill {
    level: f64,
    best: Move,
}

impl Skill {
    // Lowest and highest Elo ratings used in the skill level calculation
    pub const LOWEST_ELO: i32 = 1320;
    pub const HIGHEST_ELO: i32 = 3190;

    pub fn new(skill_level: i32, uci_elo: i32) -> Skill {
        let level = if uci_elo != 0 {
            let e = f64::from(uci_elo - Self::LOWEST_ELO)
                / f64::from(Self::HIGHEST_ELO - Self::LOWEST_ELO);
            clamp_f64(
                mul_add(
                    mul_add(mul_add(37.2473, e, -40.8525), e, 22.2943),
                    e,
                    -0.311438,
                ),
                0.0,
                19.0,
            )
        } else {
            f64::from(skill_level)
        };
        Skill {
            level,
            best: Move::none(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.level < 20.0
    }

    fn time_to_pick(&self, depth: Depth) -> bool {
        depth == 1 + self.level as i32
    }

    /// When playing with strength handicap, choose the best move among a
    /// set of RootMoves using a statistical rule dependent on 'level'. Idea
    /// by Heinz van Saanen.
    fn pick_best(&mut self, root_moves: &[RootMove], multi_pv: usize) -> Move {
        let mut rng = SKILL_RNG.lock().unwrap_or_else(|e| e.into_inner());
        let rng = rng.get_or_insert_with(|| {
            let millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(1, |d| d.as_millis() as u64);
            Prng::new(millis.max(1))
        });

        // With tablebases at the root, rootMoves are ordered by tbRank
        // rather than by score, so compute the score range explicitly to
        // keep 'delta' non-negative.
        let mut top_score = root_moves[0].score;
        let mut min_score = root_moves[0].score;
        for rm in &root_moves[1..multi_pv] {
            top_score = top_score.max(rm.score);
            min_score = min_score.min(rm.score);
        }
        let delta = (top_score - min_score).min(PAWN_VALUE);
        let mut max_score = -VALUE_INFINITE;
        let weakness = 120.0 - 2.0 * self.level;

        // Choose best move. For each move score we add two terms dependent
        // on weakness. One is deterministic and bigger for weaker levels,
        // and one is random. Then we choose the move with the resulting
        // highest score.
        for rm in &root_moves[..multi_pv] {
            // This is our magic formula
            let random = (delta as u32).wrapping_mul(rng.rand64() as u32 % weakness as i32 as u32);
            let push =
                mul_add(weakness, f64::from(top_score - rm.score), f64::from(random)) as i32 / 128;

            if rm.score + push >= max_score {
                max_score = rm.score + push;
                self.best = rm.pv[0];
            }
        }

        self.best
    }
}

/// SearchManager manages the search from the main thread. It is responsible
/// for keeping track of the time, and storing data strictly related to the
/// main thread.
pub struct SearchManager<V: Variant> {
    pub tm: TimeManagement,
    pub original_time_adjust: f64,
    pub calls_cnt: i32,

    pub iter_value: [Value; 4],
    pub previous_time_reduction: f64,
    pub best_previous_score: Value,
    pub best_previous_average_score: Value,
    pub stop_on_ponderhit: bool,

    pub updates: UpdateContext,

    /// The other threads, which the main thread starts and waits for
    pub helpers: Vec<Arc<ThreadHandle<V>>>,
    /// The counters of all threads, the main thread's first
    pub counters: Vec<Arc<Counters>>,
}

impl<V: Variant> SearchManager<V> {
    pub fn new(
        helpers: Vec<Arc<ThreadHandle<V>>>,
        counters: Vec<Arc<Counters>>,
    ) -> SearchManager<V> {
        SearchManager {
            tm: TimeManagement::default(),
            original_time_adjust: -1.0,
            calls_cnt: 0,
            iter_value: [VALUE_ZERO; 4],
            previous_time_reduction: 0.85,
            best_previous_score: VALUE_INFINITE,
            best_previous_average_score: VALUE_INFINITE,
            stop_on_ponderhit: false,
            updates: UpdateContext::default(),
            helpers,
            counters,
        }
    }

    pub fn nodes_searched(&self) -> u64 {
        self.counters.iter().map(|c| c.nodes.load(Relaxed)).sum()
    }

    pub fn tb_hits(&self) -> u64 {
        self.counters.iter().map(|c| c.tb_hits.load(Relaxed)).sum()
    }

    /// Detects when we are out of available time and stops the search.
    #[inline(always)]
    fn check_time(&mut self, limits: &LimitsType, sh: &SharedState) {
        self.calls_cnt -= 1;
        if self.calls_cnt > 0 {
            return;
        }
        self.check_time_now(limits, sh);
    }

    #[inline(never)]
    fn check_time_now(&mut self, limits: &LimitsType, sh: &SharedState) {
        // When using nodes, ensure checking rate is not lower than 0.1% of
        // nodes
        self.calls_cnt = if limits.nodes != 0 {
            ((limits.nodes / 1024) as i32).min(512)
        } else {
            512
        };

        let elapsed = self.tm.elapsed(|| self.nodes_searched());

        // We should not stop pondering until told so by the GUI
        if sh.ponder.load(Relaxed) {
            return;
        }

        if (limits.use_time_management() && (elapsed > self.tm.maximum() || self.stop_on_ponderhit))
            || (limits.movetime != 0 && elapsed >= limits.movetime)
            || (limits.nodes != 0 && self.nodes_searched() >= limits.nodes)
        {
            sh.stop.store(true, Relaxed);
        }
    }

    /// Reports the principal variations of a thread's root moves.
    #[allow(clippy::too_many_arguments)]
    fn output_pv(
        &self,
        sh: &SharedState,
        root_moves: &mut [RootMove],
        pos: &mut Position<V>,
        options: &SearchOptions,
        limits: &LimitsType,
        tb_config: &TbConfig,
        depth: Depth,
    ) {
        let nodes = self.nodes_searched();
        let multi_pv = options.multi_pv.min(root_moves.len());
        let tb_hits = self.tb_hits()
            + if tb_config.root_in_tb {
                root_moves.len() as u64
            } else {
                0
            };

        for (i, rm) in root_moves.iter_mut().enumerate().take(multi_pv) {
            let use_previous_score = rm.score == -VALUE_INFINITE;

            if depth == 1 && use_previous_score && i > 0 {
                continue;
            }

            let d = if use_previous_score {
                (depth - 1).max(1)
            } else {
                depth
            };
            let mut v = if use_previous_score {
                rm.previous_score
            } else {
                rm.uci_score
            };

            if v == -VALUE_INFINITE {
                v = VALUE_ZERO;
            }

            let is_tb_score = tb_config.root_in_tb && !is_mate_or_mated(v);
            if is_tb_score {
                v = rm.tb_score;
            }

            // Potentially correct and extend the PV, and in exceptional
            // cases v. Previous PVs have already been extended. Inexact
            // flags indicate an unreliable PV.
            if is_decisive(v)
                && !is_mate_or_mated(v)
                && !use_previous_score
                && (!rm.is_inexact() || is_tb_score)
            {
                syzygy_extend_pv(
                    V::tables(&sh.tablebases),
                    options,
                    limits,
                    pos,
                    rm,
                    &mut v,
                    multi_pv,
                    &self.updates,
                );
            }

            let pv_moves = if use_previous_score {
                &rm.previous_pv
            } else {
                &rm.pv
            };
            let pv = pv_moves
                .iter()
                .map(|&m| move_to_uci(m, pos.is_chess960()))
                .collect::<Vec<_>>()
                .join(" ");

            let wdl = if options.show_wdl {
                wdl(v, pos).unwrap_or_default()
            } else {
                String::new()
            };

            // Scores cannot be both exact and inexact
            debug_assert!(!(rm.inexact_lower && rm.inexact_upper));
            let bound = if rm.inexact_lower {
                "lowerbound"
            } else if rm.inexact_upper {
                "upperbound"
            } else {
                ""
            };

            let time = self.tm.elapsed_time().max(1);
            let info = InfoFull {
                depth: d,
                score: Score::new(v, pos),
                sel_depth: rm.sel_depth,
                multi_pv: i + 1,
                wdl: &wdl,
                // TB and previous scores are exact, even though their flags
                // may say otherwise
                bound: if !(is_tb_score || use_previous_score) {
                    bound
                } else {
                    ""
                },
                time_ms: time as usize,
                nodes: nodes as usize,
                nps: (nodes * 1000 / time as u64) as usize,
                tb_hits: tb_hits as usize,
                pv: &pv,
                hashfull: sh.tt.hashfull(0),
            };

            (self.updates.on_update_full)(&info);
        }
    }
}

/// Used to correct and extend PVs for moves that have a TB (but not a mate)
/// score. Keeps the search based PV for as long as it is verified to
/// maintain the game outcome, truncates afterwards. Finally, extends to mate
/// the PV, providing a possible continuation (but not a proven mating line).
#[allow(clippy::too_many_arguments)]
fn syzygy_extend_pv<V: Variant>(
    tables: &Tables<V>,
    options: &SearchOptions,
    limits: &LimitsType,
    pos: &mut Position<V>,
    root_move: &mut RootMove,
    v: &mut Value,
    multi_pv: usize,
    updates: &UpdateContext,
) {
    let t_start = Instant::now();
    let move_overhead = options.move_overhead;
    let rule50 = options.syzygy.rule50;

    // Do not use more than moveOverhead / 2 ms, if time management is
    // active. Under 'nodestime' the pos.do_move() calls come for free.
    let time_abort = || {
        limits.npmsec == 0
            && limits.use_time_management()
            && 2.0 * multi_pv as f64 * t_start.elapsed().as_secs_f64() * 1000.0
                >= f64::from(move_overhead)
    };

    if time_abort() {
        return;
    }

    let mut scratch = V::Dirties::default();

    // Step 0, do the rootMove, no correction allowed, as needed for MultiPV
    // in TB
    pos.make_move(root_move.pv[0], &mut scratch);
    let mut ply = 1;

    // Step 1, walk the PV to the last position in TB with correct decisive
    // score
    while ply < root_move.pv.len() {
        let pv_move = root_move.pv[ply];

        let mut legal_moves: Vec<RootMove> = MoveList::new::<ALL>(pos)
            .iter()
            .map(|&m| RootMove::new(m))
            .collect();

        let config =
            tables.rank_root_moves(&options.syzygy, pos, &mut legal_moves, false, &time_abort);
        let rm = legal_moves
            .iter()
            .find(|rm| rm.pv[0] == pv_move)
            .expect("the PV is legal");

        if legal_moves[0].tb_rank != rm.tb_rank {
            break;
        }

        ply += 1;

        pos.make_move(pv_move, &mut scratch);

        // Do not allow for repetitions or drawing moves along the PV in TB
        // regime
        if config.root_in_tb
            && ((rule50 && pos.is_draw(ply as i32)) || pos.is_repetition(ply as i32))
        {
            pos.undo_move(pv_move);
            ply -= 1;
            break;
        }

        // Full PV shown will thus be validated and end in TB. If we cannot
        // validate the full PV in time, we do not show it.
        if config.root_in_tb && time_abort() {
            break;
        }
    }

    // Resize the PV to the correct part
    root_move.pv.truncate(ply);

    // Step 2, now extend the PV to mate, as if the user explored
    // syzygy-tables.info using top ranked moves (minimal DTZ), which gives
    // optimal mates only for simple endgames e.g. KRvK.
    while !(rule50 && pos.is_draw(0)) {
        if time_abort() {
            break;
        }

        let mut legal_moves = Vec::new();
        for &m in MoveList::new::<ALL>(pos).iter() {
            let mut rm = RootMove::new(m);
            pos.make_move(m, &mut scratch);
            // Give a score of each move to break DTZ ties restricting
            // opponent mobility, but not giving the opponent a capture.
            for &m_opp in MoveList::new::<ALL>(pos).iter() {
                rm.tb_rank -= if pos.capture(m_opp) { 100 } else { 1 };
            }
            pos.undo_move(m);
            legal_moves.push(rm);
        }

        // Mate found
        if legal_moves.is_empty() {
            break;
        }

        // Sort moves according to their above assigned rank. This will break
        // ties for moves with equal DTZ in rank_root_moves.
        legal_moves.sort_by_key(|m| std::cmp::Reverse(m.tb_rank));

        // The winning side tries to minimize DTZ, the losing side maximizes
        // it
        let config =
            tables.rank_root_moves(&options.syzygy, pos, &mut legal_moves, true, &time_abort);

        // If DTZ is not available we might not find a mate, so we bail out
        if !config.root_in_tb || config.cardinality > 0 {
            break;
        }

        let pv_move = legal_moves[0].pv[0];
        root_move.pv.push(pv_move);
        pos.make_move(pv_move, &mut scratch);
    }

    // Finding a draw in this function is an exceptional case, that cannot
    // happen when rule50 is false or during engine game play, since we have
    // a winning score, and play correctly with TB support. However, it can
    // be that a position is draw due to the 50 move rule if it has been
    // reached on the board with a non-optimal 50 move counter (e.g.
    // 8/8/6k1/3B4/3K4/4N3/8/8 w - - 54 106 ) which TB with dtz counter
    // rounding cannot always correctly rank. See also
    // https://github.com/official-stockfish/Stockfish/issues/5175#issuecomment-2058893495
    // We adjust the score to match the found PV. Note that a TB loss score
    // can be displayed if the engine did not find a drawing move yet, but
    // eventually search will figure it out (e.g. 1kq5/q2r4/5K2/8/8/8/8/7Q w
    // - - 96 1 )
    if pos.is_draw(0) {
        *v = VALUE_DRAW;
    }

    // Undo the PV moves
    for &m in root_move.pv.iter().rev() {
        pos.undo_move(m);
    }

    // Inform if we couldn't get a full extension in time
    if time_abort() {
        (updates.on_info_string)(
            "Syzygy based PV extension requires more time, increase Move Overhead as needed.",
        );
    }
}

/// What a search prefetches as it makes a move: the transposition table
/// entry and the shared histories of the position after it.
struct SearchPrefetch<'a> {
    tt: &'a TranspositionTable,
    histories: &'a SharedHistories,
}

impl Prefetcher for SearchPrefetch<'_> {
    #[inline(always)]
    fn prefetch<V: Variant>(&self, key: Key, st: &StateInfo<V>, pc: Piece, to: Square) {
        prefetch(self.tt.first_entry(key));
        prefetch(&self.histories.pawn_entry_at(st.pawn_key)[pc.idx()][to.idx()]);
        prefetch(self.histories.correction_entry(st.pawn_key));
        prefetch(self.histories.correction_entry(st.minor_piece_key));
        prefetch(
            self.histories
                .correction_entry(st.non_pawn_key[WHITE.idx()]),
        );
        prefetch(
            self.histories
                .correction_entry(st.non_pawn_key[BLACK.idx()]),
        );
    }
}

/// What start_searching() needs of the thread whose result it reports.
struct BestThread<'a> {
    root_moves: &'a mut [RootMove],
    limits: &'a LimitsType,
    options: &'a SearchOptions,
    tb_config: &'a TbConfig,
    root_depth: Depth,
}

/// Search::Worker is the class that does the actual search. It is
/// instantiated once per thread, and it is responsible for keeping track of
/// the search history, and storing data required for the search.
pub struct Worker<V: Variant> {
    // Public because they need to be updatable by the stats
    pub hist: LargePageBox<WorkerHistories>,
    pub tt_move_history: TTMoveHistory,

    pub limits: LimitsType,

    pv_idx: usize,
    pv_last: usize,
    pub counters: Arc<Counters>,
    sel_depth: i32,
    pub nmp_min_ply: i32,

    optimism: [Value; COLOR_NB],

    pub root_pos: Position<V>,
    pub root_moves: Vec<RootMove>,
    pub root_depth: Depth,
    seek_mate: bool,

    last_iteration_idx_pv: PVMoves,

    thread_idx: usize,
    num_threads: usize,

    // The main thread has a SearchManager, the others have None
    pub manager: Option<Box<SearchManager<V>>>,

    pub tb_config: TbConfig,
    pub options: SearchOptions,

    // Used by NNUE
    accumulator_stack: LargePageBox<V::Accumulators>,
    refresh_table: LargePageBox<V::Caches>,

    stack: SearchStack,
    pv_table: PvTable,
    // What the moves made without a network changes, which nothing reads
    scratch: V::Dirties,
    reductions: [i32; MOVE_LIST_SIZE],
}

impl<V: Variant> Worker<V> {
    /// The worker of thread `thread_idx` of `num_threads`, whose histories
    /// clear() sets before it searches.
    pub fn new(
        thread_idx: usize,
        num_threads: usize,
        counters: Arc<Counters>,
        manager: Option<Box<SearchManager<V>>>,
        networks: &Networks,
    ) -> Box<Worker<V>> {
        Box::new(Worker {
            hist: WorkerHistories::new(),
            tt_move_history: TTMoveHistory::default(),
            limits: LimitsType::default(),
            pv_idx: 0,
            pv_last: 0,
            counters,
            sel_depth: 0,
            nmp_min_ply: 0,
            optimism: [0; COLOR_NB],
            root_pos: Position::new(),
            root_moves: Vec::new(),
            root_depth: 0,
            seek_mate: false,
            last_iteration_idx_pv: PVMoves::default(),
            thread_idx,
            num_threads,
            manager,
            tb_config: TbConfig::default(),
            options: SearchOptions::default(),
            accumulator_stack: V::Accumulators::new(),
            refresh_table: V::Caches::new(V::eval_file(networks).network()),
            stack: SearchStack([Stack::default(); 256]),
            pv_table: PvTable([PVMoves::default(); 256]),
            scratch: V::Dirties::default(),
            reductions: reductions_table(if V::ATOMIC {
                atomic::LMR_LOG_SCALE
            } else {
                2872
            }),
        })
    }

    pub fn is_mainthread(&self) -> bool {
        self.thread_idx == 0
    }

    #[inline(always)]
    fn nodes(&self) -> u64 {
        self.counters.nodes.load(Relaxed)
    }

    /// Reset histories, usually before a new game
    ///
    /// # Safety
    ///
    /// Every thread of the pool clears at once, and none searches meanwhile:
    /// each writes its share of the shared histories by plain stores.
    pub unsafe fn clear(&mut self, sh: &SharedState) {
        // Atomic's values are Atomic-Stockfish's
        let (capture, correction, pawn, continuation) = if V::ATOMIC {
            (-699, -6, -1262, -552)
        } else {
            (-742, -5, -1338, -586)
        };

        self.hist.main_history.fill(-5);
        self.hist.capture_history.fill(capture);

        // Each thread clears its part of the dynamically-sized shared
        // histories. The constant-size continuation history is initialized
        // by thread 0.
        // SAFETY: by the caller, no other thread reads or writes this
        // thread's share, nor thread 0's continuation history.
        unsafe {
            sh.histories.correction_history.clear_range(
                correction,
                self.thread_idx,
                self.num_threads,
            );
            sh.histories
                .pawn_history
                .clear_range(pawn, self.thread_idx, self.num_threads);

            if self.thread_idx == 0 {
                sh.histories.continuation_history.fill_shared(continuation);
            }
        }

        self.tt_move_history.set(0);

        self.hist.continuation_correction_history.fill(5);

        // Without a network the caches are cleared when one is loaded.
        if let Some(network) = V::eval_file(&sh.networks).network() {
            self.refresh_table.clear(network);
        }
    }

    /// Called when the program receives the UCI 'go' command. It searches
    /// from the root position and, on the main thread, outputs the
    /// "bestmove".
    pub fn start_searching(&mut self, sh: &Arc<SharedState>) {
        self.accumulator_stack.reset();

        // The root position is searched outside the worker.
        let mut pos = std::mem::take(&mut self.root_pos);

        // Non-main threads go directly to iterative_deepening()
        if self.manager.is_none() {
            self.iterative_deepening(sh, &mut pos);
            self.root_pos = pos;
            return;
        }

        self.start_main_search(sh, &mut pos);
        self.root_pos = pos;
    }

    fn start_main_search(&mut self, sh: &Arc<SharedState>, pos: &mut Position<V>) {
        let us = pos.side_to_move();
        {
            let manager = self.manager.as_deref_mut().expect("the main thread");
            manager.tm.init::<V>(
                &mut self.limits,
                us,
                pos.game_ply(),
                &self.options,
                &mut manager.original_time_adjust,
            );
        }
        sh.tt.new_search();

        let updates = self.main_manager().updates.clone();
        (updates.on_start)();

        if self.root_moves.is_empty() {
            // Without moves, a side is lost when mated or, in Atomic, when
            // its king has exploded, won when the other king has, and is
            // otherwise stalemated.
            let v = if V::ATOMIC && !pos.has_king(us) {
                -VALUE_MATE
            } else if V::ATOMIC && !pos.has_king(!us) {
                VALUE_MATE
            } else if pos.in_check() {
                -VALUE_MATE
            } else {
                VALUE_DRAW
            };
            (updates.on_update_no_moves)(&InfoShort {
                depth: 0,
                score: Score::new(v, pos),
            });
            (updates.on_bestmove)(&move_to_uci(Move::none(), false), "");
            return;
        }

        // Main thread starts non-main threads, and begins own search
        for helper in &self.main_manager().helpers {
            let sh = Arc::clone(sh);
            helper.run_job(Box::new(move |worker| worker.start_searching(&sh)));
        }
        let mut uci_pv_sent = self.iterative_deepening(sh, pos);

        // When we reach the maximum depth, we can arrive here without a
        // raise of stop. However, if we are pondering or in an infinite
        // search, the UCI protocol states that we shouldn't print the best
        // move before the GUI sends a "stop" or "ponderhit" command. We
        // therefore simply wait here until the GUI sends one of those
        // commands.
        while !sh.stop.load(Relaxed) && (sh.ponder.load(Relaxed) || self.limits.infinite != 0) {
            std::hint::spin_loop();
        }

        // Stop the threads if not already stopped (also raise the stop if
        // "ponderhit" just reset ponder).
        sh.stop.store(true, Relaxed);

        // Wait until all threads have finished
        for helper in &self.main_manager().helpers {
            helper.wait_for_search_finished();
        }

        // When playing in 'nodes as time' mode, subtract the searched nodes
        // from the available ones before exiting.
        if self.limits.npmsec != 0 {
            let inc = self.limits.inc[us.idx()];
            let manager = self.main_manager();
            let nodes = manager.nodes_searched();
            manager
                .tm
                .advance_nodes_time((nodes as i64).wrapping_sub(inc));
        }

        let skill = Skill::new(
            self.options.skill_level,
            if self.options.limit_strength {
                self.options.uci_elo
            } else {
                0
            },
        );

        let best_thread = if self.limits.depth == 0 && !skill.enabled() {
            self.get_best_thread()
        } else {
            0
        };

        let mut manager = self.manager.take().expect("the main thread");
        if best_thread == 0 {
            let best = BestThread {
                root_moves: &mut self.root_moves,
                limits: &self.limits,
                options: &self.options,
                tb_config: &self.tb_config,
                root_depth: self.root_depth,
            };
            report_best(
                &mut manager,
                sh,
                pos,
                &mut self.scratch,
                best,
                true,
                uci_pv_sent,
            );
        } else {
            let helper = Arc::clone(&manager.helpers[best_thread - 1]);
            let mut worker = helper.worker.lock().unwrap_or_else(|e| e.into_inner());
            let worker = &mut **worker;
            let best = BestThread {
                root_moves: &mut worker.root_moves,
                limits: &worker.limits,
                options: &worker.options,
                tb_config: &worker.tb_config,
                root_depth: worker.root_depth,
            };
            uci_pv_sent = false;
            report_best(
                &mut manager,
                sh,
                pos,
                &mut self.scratch,
                best,
                false,
                uci_pv_sent,
            );
        }
        self.manager = Some(manager);
    }

    /// Pointer to the search manager, only allowed to be called by the main
    /// thread
    fn main_manager(&mut self) -> &mut SearchManager<V> {
        self.manager.as_deref_mut().expect("the main thread")
    }

    /// The index of the thread whose result to play: 0 for this, the main
    /// thread, or 1 + the index of a helper.
    fn get_best_thread(&self) -> usize {
        struct Candidate {
            score: Value,
            mv: Move,
            inexact: bool,
            pv_len: usize,
        }

        let of = |rm: &RootMove| Candidate {
            score: rm.score,
            mv: rm.pv[0],
            inexact: rm.is_inexact(),
            pv_len: rm.pv.len(),
        };

        let manager = self.manager.as_deref().expect("the main thread");
        let mut threads = vec![of(&self.root_moves[0])];
        for helper in &manager.helpers {
            let worker = helper.worker.lock().unwrap_or_else(|e| e.into_inner());
            threads.push(of(&worker.root_moves[0]));
        }

        let mut best_thread = 0;
        let mut min_score = VALUE_INFINITE;
        let mut votes: HashMap<Move, i64> = HashMap::new();

        for th in &threads {
            min_score = min_score.min(th.score);
        }

        // Vote according to score, and select the best thread
        for th in &threads {
            *votes.entry(th.mv).or_insert(0) += i64::from(th.score - min_score + 14);
        }

        for (i, new_thread_move) in threads.iter().enumerate() {
            let best_thread_move = &threads[best_thread];

            let best_thread_move_vote = votes[&best_thread_move.mv];
            let new_thread_move_vote = votes[&new_thread_move.mv];

            // Aborted (d1) searches may lead to inexact win (or loss)
            // scores.
            let best_thread_decisive = best_thread_move.score != -VALUE_INFINITE
                && is_decisive(best_thread_move.score)
                && !best_thread_move.inexact;
            let new_thread_decisive = new_thread_move.score != -VALUE_INFINITE
                && is_decisive(new_thread_move.score)
                && !new_thread_move.inexact;

            if best_thread_decisive {
                // Make sure we pick the shortest mate / TB conversion.
                if new_thread_decisive && new_thread_move.score.abs() > best_thread_move.score.abs()
                {
                    debug_assert!(
                        (is_win(best_thread_move.score) && is_win(new_thread_move.score))
                            || (is_loss(best_thread_move.score) && is_loss(new_thread_move.score))
                    );

                    best_thread = i;
                }
            } else if new_thread_decisive
                || (!is_loss(new_thread_move.score)
                    && (new_thread_move_vote > best_thread_move_vote
                        || (new_thread_move_vote == best_thread_move_vote
                            && new_thread_move.pv_len > best_thread_move.pv_len)))
            {
                best_thread = i;
            }
        }

        best_thread
    }

    /// Main iterative deepening loop. It calls search() repeatedly with
    /// increasing depth until the allocated thinking time has been consumed,
    /// the user stops the search, or the maximum search depth is reached.
    fn iterative_deepening(&mut self, sh: &SharedState, pos: &mut Position<V>) -> bool {
        let main_thread = self.manager.is_some();

        let mut last_best_move_pv: Vec<Move> = Vec::with_capacity(MAX_PLY as usize);
        let mut last_best_move_depth = 0;
        let mut last_best_move_score = -VALUE_INFINITE;

        let mut alpha;
        let mut beta;
        let mut best_value = -VALUE_INFINITE;
        let us = pos.side_to_move();
        let mut time_reduction = 1.0;
        let mut tot_best_move_changes = 0.0;
        let mut delta;
        let mut iter_idx = 0;

        // The stack starts at zero, with the seven entries before the root
        // pointing to the sentinel histories, as the continuation histories
        // of the root and the plies after it read them.
        self.stack.0 = [Stack::default(); 256];
        for i in 1..=ROOT_SS {
            self.stack[ROOT_SS - i].static_eval = VALUE_NONE;
        }

        if let Some(manager) = self.manager.as_deref_mut() {
            if manager.best_previous_score == VALUE_INFINITE {
                manager.iter_value = [VALUE_ZERO; 4];
            } else {
                manager.iter_value = [manager.best_previous_score; 4];
            }
        }

        let mut multi_pv = self.options.multi_pv;
        let mut skill = Skill::new(
            self.options.skill_level,
            if self.options.limit_strength {
                self.options.uci_elo
            } else {
                0
            },
        );

        // When playing with strength handicap enable MultiPV search that we
        // will use behind-the-scenes to retrieve a set of possible moves.
        if skill.enabled() {
            multi_pv = multi_pv.max(4);
        }

        multi_pv = multi_pv.min(self.root_moves.len());

        let mut search_again_counter = 0;
        let mut fail_high_recovery = 0;
        let mut uci_pv_sent = false;

        // Atomic's values here and below are Atomic-Stockfish's
        self.hist
            .low_ply_history
            .fill(if V::ATOMIC { 100 } else { 102 });

        // Both colors' entries, through one slice: indexed through the box,
        // each store reloads its pointer, and the loop is not vectorized.
        let main_history_keep = if V::ATOMIC { 789 } else { 729 };
        for entry in self.hist.main_history.as_flattened_mut() {
            entry.set(entry.get() * main_history_keep / 1024);
        }

        // Iterative deepening loop until requested to stop or the target
        // depth is reached
        while self.root_depth + 1 < MAX_PLY
            && !sh.stop.load(Relaxed)
            && !(self.limits.depth != 0 && main_thread && self.root_depth >= self.limits.depth)
        {
            self.root_depth += 1;

            // Signal the start of a new iteration
            if main_thread {
                uci_pv_sent = false;
            }

            // Save the last iteration's scores before the first PV line is
            // searched and all the move scores except the (new) PV are set
            // to -VALUE_INFINITE.
            for (i, rm) in self.root_moves.iter_mut().enumerate() {
                rm.previous_score = rm.score;
                rm.previous_pv.clone_from(&rm.pv);
                rm.previous_score_exact = i < multi_pv;
            }

            let mut pv_first = 0;
            self.pv_last = 0;

            if !sh.increase_depth.load(Relaxed) {
                search_again_counter += 1;
            }

            // MultiPV loop: we perform a full root search for each PV line
            self.pv_idx = 0;
            while self.pv_idx < multi_pv {
                let pv_idx = self.pv_idx;

                if pv_idx == self.pv_last {
                    pv_first = self.pv_last;
                    self.pv_last += 1;
                    while self.pv_last < self.root_moves.len() {
                        if self.root_moves[self.pv_last].tb_rank
                            != self.root_moves[pv_first].tb_rank
                        {
                            break;
                        }
                        self.pv_last += 1;
                    }
                }

                self.last_iteration_idx_pv
                    .assign(&self.root_moves[pv_idx].previous_pv);

                // Reset UCI info selDepth for each depth and each PV line
                self.sel_depth = 0;

                // Reset aspiration window starting size
                let mean_squared_div = if V::ATOMIC { 10588 } else { 10193 };
                delta = 5
                    + (self.thread_idx % 8) as i32
                    + self.root_moves[pv_idx].mean_squared_score.abs() / mean_squared_div;
                let avg = self.root_moves[pv_idx].average_score;
                alpha = (avg - delta).max(-VALUE_INFINITE);
                beta = (avg + delta).min(VALUE_INFINITE);

                // Do not tune these values. They are not intended for
                // playing strength. Atomic reads the previous score, as
                // Atomic-Stockfish does: searching an earlier MultiPV line
                // leaves the moves that failed low at -VALUE_INFINITE.
                let root_score = if V::ATOMIC {
                    self.root_moves[pv_idx].previous_score
                } else {
                    self.root_moves[pv_idx].score
                };
                self.seek_mate = (!V::ATOMIC || root_score != -VALUE_INFINITE)
                    && root_score.abs() >= 750 + 220000 / (self.root_depth * self.root_depth);

                // Adjust optimism based on root move's averageScore
                self.optimism[us.idx()] = if V::ATOMIC {
                    137 * avg / (avg.abs() + 81)
                } else {
                    114 * avg / (avg.abs() + 85)
                };
                self.optimism[(!us).idx()] = -self.optimism[us.idx()];

                // Start with a small aspiration window and, in the case of a
                // fail high/low, enlarge the window progressively.
                let mut failed_high_cnt = 0;
                if pv_idx == 0 {
                    fail_high_recovery = (fail_high_recovery - 2).max(0);
                }
                loop {
                    // Adjust the effective depth searched, but ensure at
                    // least one effective increment for every four
                    // searchAgain steps (see issue #2717).
                    let adjusted_depth = (self.root_depth
                        - failed_high_cnt
                        - fail_high_recovery
                        - 3 * (search_again_counter + 1) / 4)
                        .max(1);
                    best_value = V::search_root(self, sh, pos, alpha, beta, adjusted_depth);

                    // Bring the best move to the front. It is critical that
                    // sorting is done with a stable algorithm because all
                    // the values but the first and eventually the new best
                    // one is set to -VALUE_INFINITE and we want to keep the
                    // same order for all the moves except the new PV that
                    // goes to the front. Note that in the case of MultiPV
                    // search the already searched PV lines are preserved.
                    self.root_moves[pv_idx..self.pv_last].sort_by(root_order);

                    // If search has been stopped, we break immediately.
                    // Sorting is safe because RootMoves is still valid,
                    // although it refers to the previous iteration.
                    if sh.stop.load(Relaxed) {
                        break;
                    }

                    // When failing high/low give some update before a
                    // re-search. To avoid excessive output that could hang
                    // GUIs like Fritz 19, only start at nodes > 10M (rather
                    // than depth N, which can be reached quickly).
                    if main_thread
                        && multi_pv == 1
                        && (best_value <= alpha || best_value >= beta)
                        && self.nodes() > NODES_LIMIT_OUTPUT
                    {
                        self.output_pv(sh, pos, self.root_depth);
                    }

                    // In case of failing low/high increase aspiration window
                    // and re-search, otherwise exit the loop.
                    if best_value <= alpha {
                        beta = alpha;
                        alpha = (best_value - delta).max(-VALUE_INFINITE);

                        failed_high_cnt = 0;
                        if let Some(manager) = self.manager.as_deref_mut() {
                            manager.stop_on_ponderhit = false;
                        }
                    } else if best_value >= beta {
                        alpha = (beta - delta).max(alpha);
                        beta = (best_value + delta).min(VALUE_INFINITE);
                        failed_high_cnt += 1;
                    } else {
                        break;
                    }

                    let growth = if V::ATOMIC { 44 } else { 47 };
                    delta += growth * delta / 128;

                    debug_assert!(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
                }

                // Gradually increase depth after reduced depth search
                if failed_high_cnt > 0 && pv_idx == 0 {
                    fail_high_recovery = (failed_high_cnt + 1) / 2 + 2;
                }

                if sh.stop.load(Relaxed) && pv_idx != 0 {
                    let rms = &mut self.root_moves;

                    // In multiPV analysis we do not let aborted searches
                    // spoil mated-in/TB loss scores from a completed search
                    // in an earlier PV line. Hence we guard against an
                    // aborted pvIdx line overtaking pvIdx - 1 when pvIdx - 1
                    // is a proven loss. Moreover, we do not trust an exact
                    // loss score from an aborted search.
                    if (is_loss(rms[pv_idx - 1].score) && root_less(&rms[pv_idx], &rms[pv_idx - 1]))
                        || rms[pv_idx].is_exact_loss()
                    {
                        // If previousScore is exact and worse than pvIdx - 1,
                        // we can safely use it. If it is equal, we make sure
                        // it cannot overtake pvIdx - 1.
                        if rms[pv_idx].previous_score != -VALUE_INFINITE
                            && rms[pv_idx].previous_score_exact
                            && rms[pv_idx].previous_score <= rms[pv_idx - 1].score
                        {
                            let rm = &mut rms[pv_idx];
                            rm.score = rm.previous_score;
                            rm.uci_score = rm.previous_score;
                            rm.previous_score = -VALUE_INFINITE;
                            rm.pv.clone_from(&rm.previous_pv);
                            rm.unset_inexact();
                        }
                        // Otherwise, if we can, we cap the score to the best
                        // possible, and mark the score as inexact (also a
                        // valid excuse for the incomplete PV).
                        else {
                            if is_loss(rms[pv_idx - 1].score) {
                                let score = rms[pv_idx - 1].score;
                                let rm = &mut rms[pv_idx];
                                rm.score = score;
                                rm.uci_score = score;
                                rm.previous_score = -VALUE_INFINITE;
                                rm.pv.truncate(1);
                                rm.inexact_upper = true;
                            } else {
                                rms[pv_idx].inexact_upper = false;
                            }

                            rms[pv_idx].inexact_lower = !rms[pv_idx].inexact_upper;
                        }
                    }

                    // Finally, we mark all loss scores from partially
                    // searched moves as inexact.
                    for rm in &mut rms[pv_idx + 1..multi_pv] {
                        if rm.is_exact_loss() {
                            rm.inexact_lower = true;
                        }
                    }
                }

                // Sort the PV lines searched so far and update the GUI
                self.root_moves[pv_first..=pv_idx].sort_by(root_order);

                if main_thread
                    && !sh.stop.load(Relaxed)
                    && (pv_idx + 1 == multi_pv || self.nodes() > NODES_LIMIT_OUTPUT)
                {
                    self.output_pv(sh, pos, self.root_depth);
                    uci_pv_sent = pv_idx + 1 == multi_pv;
                }

                if sh.stop.load(Relaxed) {
                    break;
                }

                self.pv_idx += 1;
            }

            let forgotten_mate = last_best_move_score != -VALUE_INFINITE
                && is_mate_or_mated(last_best_move_score)
                && (self.root_moves[0].score.abs() < last_best_move_score.abs()
                    || self.root_moves[0].is_inexact());

            if !sh.stop.load(Relaxed) {
                if last_best_move_pv.is_empty() || last_best_move_pv[0] != self.root_moves[0].pv[0]
                {
                    last_best_move_depth = self.root_depth;
                }

                // Do not replace (shorter) mate scores from a previous
                // iteration
                if !forgotten_mate {
                    last_best_move_pv.clone_from(&self.root_moves[0].pv);
                    last_best_move_score = self.root_moves[0].score;
                }
            }

            let aborted_loss_search =
                sh.stop.load(Relaxed) && self.pv_idx == 0 && self.root_moves[0].is_exact_loss();

            // An exact mated-in/TB-loss score from an aborted search cannot
            // be trusted: the loss could be delayed or refuted upon
            // exploring the remaining root-moves. Thus here we roll back to
            // the score from the previous iteration. We do the same if a
            // search has failed to recover a mate score that was found in a
            // previous iteration.
            if aborted_loss_search
                || (self.root_moves[0].score != -VALUE_INFINITE && forgotten_mate)
            {
                // Bring the last best move to the front for best thread
                // selection
                if !last_best_move_pv.is_empty() {
                    if let Some(i) = self
                        .root_moves
                        .iter()
                        .position(|rm| rm.pv[0] == last_best_move_pv[0])
                    {
                        self.root_moves[..=i].rotate_right(1);
                    }
                    let rm = &mut self.root_moves[0];
                    rm.score = last_best_move_score;
                    rm.uci_score = last_best_move_score;
                    rm.pv.clone_from(&last_best_move_pv);
                    rm.unset_inexact();

                    if main_thread {
                        uci_pv_sent = false;
                    }
                }
                // For an aborted d1 search we label the loss score as
                // inexact
                else if aborted_loss_search {
                    self.root_moves[0].inexact_lower = true;
                }
            }

            // Have we found a "mate in x" after a completed iteration?
            if self.limits.mate != 0
                && !sh.stop.load(Relaxed)
                && is_mate_or_mated(self.root_moves[0].score)
                && VALUE_MATE - self.root_moves[0].score.abs() <= 2 * self.limits.mate
            {
                sh.stop.store(true, Relaxed);
            }

            if !main_thread {
                continue;
            }

            // If the skill level is enabled and time is up, pick a
            // sub-optimal best move
            if skill.enabled() && skill.time_to_pick(self.root_depth) {
                skill.pick_best(&self.root_moves, multi_pv);
            }

            // Do we have time for the next iteration? Can we stop searching
            // now?
            if self.limits.use_time_management()
                && !sh.stop.load(Relaxed)
                && !self.main_manager().stop_on_ponderhit
            {
                let nodes_effort = self.root_moves[0].effort * 100000 / self.nodes().max(1);
                let num_threads = self.num_threads;
                let root_depth = self.root_depth;
                let single_move = self.root_moves.len() == 1;
                let stop_score = self.root_moves[multi_pv - 1].score >= mate_in(3)
                    || self.root_moves[0].score == mated_in(2);
                let manager = self.main_manager();
                let tc = if V::ATOMIC {
                    &atomic::TIME
                } else {
                    &CHESS_TIME
                };

                let (base, average_weight, iter_weight) = tc.falling_eval;
                let falling_eval = mul_add(
                    iter_weight,
                    f64::from(manager.iter_value[iter_idx] - best_value),
                    mul_add(
                        average_weight,
                        f64::from(manager.best_previous_average_score - best_value),
                        base,
                    ),
                ) / 100.0;
                let (lo, hi) = tc.falling_eval_range;
                let falling_eval = clamp_f64(falling_eval, lo, hi);

                // If the bestMove is stable over several iterations, reduce
                // time accordingly
                let (x0, x1, y0, y1) = tc.time_reduction;
                let (lo, hi) = tc.time_reduction_range;
                time_reduction = clamp_f64(
                    interpolate(f64::from(root_depth - last_best_move_depth), x0, x1, y0, y1),
                    lo,
                    hi,
                );

                let (base, divisor) = tc.reduction;
                let reduction =
                    (base + manager.previous_time_reduction) / (divisor * time_reduction);

                // Age out and update PV variability metric
                tot_best_move_changes /= 2.0;
                for counters in &manager.counters {
                    tot_best_move_changes += counters.best_move_changes.load(Relaxed) as f64;
                    counters.best_move_changes.store(0, Relaxed);
                }

                let (base, weight) = tc.best_move_instability;
                let best_move_instability =
                    base + weight * tot_best_move_changes / num_threads as f64;

                let (x0, x1, y0, y1) = tc.high_best_move_effort;
                let (lo, hi) = tc.high_best_move_effort_range;
                let high_best_move_effort =
                    clamp_f64(interpolate_i64(nodes_effort as i64, x0, x1, y0, y1), lo, hi);

                let mut total_time = manager.tm.optimum() as f64
                    * falling_eval
                    * reduction
                    * best_move_instability
                    * high_best_move_effort;

                if single_move {
                    // Cap used time for a better viewer experience
                    total_time = min_f64(tc.single_move_time, total_time);
                }

                let elapsed_time = manager.tm.elapsed(|| manager.nodes_searched());

                // Stop the search if we have exceeded totalTime or maximum
                // time, or if we know that there are no better moves in the
                // analysed line(s).
                if elapsed_time as f64 > min_f64(total_time, manager.tm.maximum() as f64)
                    || stop_score
                {
                    // If we are allowed to ponder do not stop the search now
                    // but keep pondering until the GUI sends "ponderhit" or
                    // "stop".
                    if sh.ponder.load(Relaxed) {
                        manager.stop_on_ponderhit = true;
                    } else {
                        sh.stop.store(true, Relaxed);
                    }
                } else {
                    sh.increase_depth.store(
                        sh.ponder.load(Relaxed) || elapsed_time as f64 <= total_time * 0.50,
                        Relaxed,
                    );
                }
            }

            self.main_manager().iter_value[iter_idx] = best_value;
            iter_idx = (iter_idx + 1) & 3;
        }

        if !main_thread {
            return false;
        }

        self.main_manager().previous_time_reduction = time_reduction;

        // If the skill level is enabled, swap the best PV line with the
        // sub-optimal one
        if skill.enabled() {
            let best = if skill.best != Move::none() {
                skill.best
            } else {
                skill.pick_best(&self.root_moves, multi_pv)
            };
            let i = self
                .root_moves
                .iter()
                .position(|rm| rm.pv[0] == best)
                .expect("the skill picks a root move");
            self.root_moves.swap(0, i);
        }

        uci_pv_sent
    }

    /// Reports this thread's principal variations.
    fn output_pv(&mut self, sh: &SharedState, pos: &mut Position<V>, depth: Depth) {
        let manager = self.manager.as_deref().expect("the main thread");
        manager.output_pv(
            sh,
            &mut self.root_moves,
            pos,
            &self.options,
            &self.limits,
            &self.tb_config,
            depth,
        );
    }

    #[inline(always)]
    fn do_move(
        &mut self,
        sh: &SharedState,
        pos: &mut Position<V>,
        mv: Move,
        gives_check: bool,
        capture: bool,
        ss: usize,
    ) {
        // prefetch_key() does not model castling, en passant or promotion
        // exactly. The correction-history prefetches also approximate
        // castling and promotion. For these rare moves the prefetches land
        // on unused lines.
        prefetch(sh.tt.first_entry(pos.prefetch_key(mv)));

        let pc = pos.moved_piece(mv);
        let to = mv.to_sq();

        let cch = &self.hist.continuation_correction_history;
        for i in [1, 3, 5] {
            let index = self.stack[ss - i].continuation_correction_history as usize;
            prefetch(&cch[index & (PIECE_NB * SQUARE_NB - 1)][pc.idx()][to.idx()]);
        }

        increment(&self.counters.nodes);

        let dirties = self.accumulator_stack.push();
        pos.do_move_prefetching(
            mv,
            gives_check,
            dirties,
            &SearchPrefetch {
                tt: &sh.tt,
                histories: &sh.histories,
            },
        );

        let st = &mut self.stack[ss];
        st.current_move = mv;
        st.continuation_history = continuation_index(st.in_check, capture, pc, to);
        st.continuation_correction_history = continuation_correction_index(pc, to);
    }

    #[inline(always)]
    fn do_null_move(&mut self, pos: &mut Position<V>, ss: usize) {
        pos.do_null_move();
        let st = &mut self.stack[ss];
        st.current_move = Move::null();
        st.continuation_history = 0;
        st.continuation_correction_history = 0;
    }

    #[inline(always)]
    fn undo_move(&mut self, pos: &mut Position<V>, mv: Move) {
        pos.undo_move(mv);
        self.accumulator_stack.pop();
    }

    #[inline(always)]
    fn undo_null_move(&mut self, pos: &mut Position<V>) {
        pos.undo_null_move();
    }

    /// Main search function for both PV and non-PV nodes
    #[allow(clippy::too_many_arguments)]
    fn search<const NT: NodeType>(
        &mut self,
        sh: &SharedState,
        pos: &mut Position<V>,
        ss: usize,
        mut alpha: Value,
        mut beta: Value,
        mut depth: Depth,
        cut_node: bool,
    ) -> Value {
        let pv_node = NT != NON_PV;
        let root_node = NT == ROOT;
        let all_node = !(pv_node || cut_node);

        // Dive into quiescence search when the depth reaches zero
        if depth <= 0 {
            return if pv_node {
                self.qsearch::<PV>(sh, pos, ss, alpha, beta)
            } else {
                self.qsearch::<NON_PV>(sh, pos, ss, alpha, beta)
            };
        }

        // Limit the depth if extensions made it too large
        depth = depth.min(MAX_PLY - 1);

        let ply = ply_of(ss);

        // Check if we have an upcoming move that draws by repetition
        if !root_node && alpha < VALUE_DRAW && pos.upcoming_repetition(ply) {
            alpha = value_draw(self.nodes());
            if alpha >= beta {
                return alpha;
            }
        }

        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || alpha == beta - 1);
        debug_assert!(0 < depth && depth < MAX_PLY);
        debug_assert!(!(pv_node && cut_node));

        let mut captures_searched = SearchedList::new();
        let mut quiets_searched = SearchedList::new();

        // Step 1. Initialize node
        self.stack[ss].in_check = pos.checkers().any();
        let prior_capture = pos.captured_piece().is_some();
        let us = pos.side_to_move();
        self.stack[ss].move_count = 0;
        let mut best_value = -VALUE_INFINITE;
        let mut max_value = VALUE_INFINITE;

        self.stack[ss].follow_pv = root_node
            || (self.stack[ss - 1].follow_pv
                && ((ply - 1) as usize) < self.last_iteration_idx_pv.len()
                && self.stack[ss - 1].current_move
                    == self.last_iteration_idx_pv[(ply - 1) as usize]);

        // Check for the available remaining time
        if let Some(manager) = self.manager.as_deref_mut() {
            manager.check_time(&self.limits, sh);
        }

        // Used to send selDepth info to GUI (selDepth counts from 1, ply from
        // 0)
        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }

        if !root_node {
            // Step 2. Check for aborted search or immediate draw
            if sh.stop.load(Relaxed) || pos.is_draw(ply) || ply >= MAX_PLY {
                return if ply >= MAX_PLY && !self.stack[ss].in_check {
                    self.evaluate(sh, pos)
                } else {
                    value_draw(self.nodes())
                };
            }

            // Step 3. Mate distance pruning. Even if we mate at the next move
            // our score would be at best mate_in(ss->ply + 1), but if alpha
            // is already bigger because a shorter mate was found upward in
            // the tree then there is no need to search because we will
            // never beat the current alpha. Equal and opposite logic applies
            // when being mated. In either case, return a fail-high score.
            alpha = mated_in(ply).max(alpha);
            beta = mate_in(ply + 1).min(beta);
            if alpha >= beta {
                return alpha;
            }
        }

        debug_assert!((0..MAX_PLY).contains(&ply));

        let prev_move = self.stack[ss - 1].current_move;
        let prev_sq = if prev_move.is_ok() {
            prev_move.to_sq()
        } else {
            Square::NONE
        };
        let mut best_move = Move::none();
        let prior_reduction = self.stack[ss - 1].reduction;
        self.stack[ss - 1].reduction = 0;
        self.stack[ss].stat_score = 0;
        self.stack[ss + 2].cutoff_cnt = 0;
        self.stack[ss + 1].prior_nmp_fail_high = 0;

        let correction_value = self.correction_value(sh, pos, ss);

        // Step 4. Transposition table lookup
        let excluded_move = self.stack[ss].excluded_move;
        let pos_key = pos.key();
        let (tt_hit, mut tt_data, tt_writer) = sh.tt.probe(pos_key);

        self.stack[ss].tt_hit = tt_hit;
        tt_data.mv = if root_node {
            self.root_moves[self.pv_idx].pv[0]
        } else if tt_hit {
            tt_data.mv
        } else {
            Move::none()
        };
        tt_data.value = if tt_hit {
            value_from_tt(tt_data.value, ply, pos.rule50_count())
        } else {
            VALUE_NONE
        };
        self.stack[ss].tt_pv = if excluded_move != Move::none() {
            self.stack[ss].tt_pv
        } else {
            pv_node || (tt_hit && tt_data.is_pv)
        };
        let tt_capture = tt_data.mv != Move::none() && pos.capture_stage(tt_data.mv);

        // Step 5. Static evaluation of the position
        let mut unadjusted_static_eval = VALUE_NONE;
        let mut eval;

        // Skip early pruning when in check
        if self.stack[ss].in_check {
            eval = self.stack[ss - 2].static_eval;
            self.stack[ss].static_eval = eval;
        } else if excluded_move != Move::none() {
            eval = self.stack[ss].static_eval;
            unadjusted_static_eval = eval;
        } else if self.stack[ss].tt_hit {
            // Never assume anything about values stored in TT
            unadjusted_static_eval = tt_data.eval;
            if !is_valid(unadjusted_static_eval) {
                unadjusted_static_eval = self.evaluate(sh, pos);
            }

            eval = to_corrected_static_eval(unadjusted_static_eval, correction_value);
            self.stack[ss].static_eval = eval;

            // ttValue can be used as a better position evaluation
            if is_valid(tt_data.value)
                && (tt_data.bound
                    & if tt_data.value > eval {
                        BOUND_LOWER
                    } else {
                        BOUND_UPPER
                    })
                    != 0
            {
                eval = tt_data.value;
            }
        } else {
            unadjusted_static_eval = self.evaluate(sh, pos);
            eval = to_corrected_static_eval(unadjusted_static_eval, correction_value);
            self.stack[ss].static_eval = eval;

            // Static evaluation is saved as it was before adjustment by
            // correction history
            tt_writer.write(
                pos_key,
                VALUE_NONE,
                self.stack[ss].tt_pv,
                BOUND_NONE,
                DEPTH_UNSEARCHED,
                Move::none(),
                unadjusted_static_eval,
                sh.tt.generation(),
            );
        }

        // Set up the improving flag, which is true if current static
        // evaluation is bigger than the previous static evaluation at our
        // turn (if we were in check at our previous move we go back until we
        // weren't in check) and is false otherwise. The improving flag is
        // used in various pruning heuristics. Similarly, opponentWorsening is
        // true if our static evaluation is better for us than at the last
        // ply.
        let mut improving = self.stack[ss].static_eval > self.stack[ss - 2].static_eval;
        let opponent_worsening = self.stack[ss].static_eval > -self.stack[ss - 1].static_eval;

        // Hindsight adjustment of reductions based on static evaluation
        // difference
        if prior_reduction >= 3 && !opponent_worsening {
            depth += 1;
        }
        if prior_reduction >= 2
            && depth >= 2
            && self.stack[ss].static_eval + self.stack[ss - 1].static_eval > 166
        {
            depth -= 1;
        }

        // Step 6. At non-PV nodes we check for an early TT cutoff. Note that
        // we always check the validity of the TT value because of access
        // races.
        if !pv_node
            && excluded_move == Move::none()
            && is_valid(tt_data.value)
            && tt_data.depth > depth - (tt_data.value <= beta) as i32
        {
            // Case A: TT entry can produce a cutoff
            if (tt_data.bound
                & if tt_data.value >= beta {
                    BOUND_LOWER
                } else {
                    BOUND_UPPER
                })
                != 0
                && (cut_node == (tt_data.value >= beta) || depth > 4)
            {
                // If the ttMove is quiet, update move sorting heuristics on
                // TT hit
                if tt_data.mv != Move::none() && tt_data.value >= beta {
                    // Bonus for a quiet ttMove that fails high
                    if !tt_capture {
                        self.update_quiet_histories(sh, pos, ss, tt_data.mv, 131 * depth);
                    }

                    // Extra penalty for early quiet moves of the previous ply
                    if prev_sq != Square::NONE
                        && self.stack[ss - 1].move_count < 5
                        && !prior_capture
                    {
                        self.update_continuation_histories(
                            sh,
                            ss - 1,
                            pos.piece_on(prev_sq),
                            prev_sq,
                            -2210,
                        );
                    }
                }

                // Partial workaround for the graph history interaction
                // problem. For high rule50 counts don't produce
                // transposition table cutoffs.
                if pos.rule50_count() < 96 {
                    if depth >= 7
                        && tt_data.mv != Move::none()
                        && pos.legal(tt_data.mv)
                        && !is_decisive(tt_data.value)
                    {
                        pos.make_move(tt_data.mv, &mut self.scratch);
                        let next_pos_key = pos.key();
                        let (_, tt_data_next, _) = sh.tt.probe(next_pos_key);
                        pos.undo_move(tt_data.mv);

                        // Check that the ttValue after the tt move would also
                        // trigger a cutoff
                        if !is_valid(tt_data_next.value) {
                            return tt_data.value;
                        }

                        if (tt_data.value >= beta) == (-tt_data_next.value >= beta) {
                            return tt_data.value;
                        }
                    } else {
                        return tt_data.value;
                    }
                }
            }
            // Case B: No cutoff, but depth was sufficient, so the bound
            // points the wrong way.
            else if depth > 5 {
                // If such a mismatch is the only reason cutoff failed, the TT
                // entry is now useless (*Scaler) This is very close to
                // secondary aging, a known antiscaler. The depth and window
                // conditions seem to be key to decent scaling.
                tt_writer.penalize(1);
            }
        }

        // Step 7. Tablebases probe
        if !root_node && excluded_move == Move::none() && self.tb_config.cardinality != 0 {
            let pieces_count = pos.count_both(ALL_PIECES);

            if pieces_count <= self.tb_config.cardinality
                && (pieces_count < self.tb_config.cardinality
                    || depth >= self.tb_config.probe_depth)
                && pos.rule50_count() == 0
                && !pos.can_castle(ANY_CASTLING)
            {
                let mut err = ProbeState::Ok;
                let wdl = V::tables(&sh.tablebases).probe_wdl(pos, &mut err);

                // Force check of time on the next occasion
                if let Some(manager) = self.manager.as_deref_mut() {
                    manager.calls_cnt = 0;
                }

                if err != ProbeState::Fail {
                    increment(&self.counters.tb_hits);

                    let draw_score = i32::from(self.tb_config.use_rule50);

                    let tb_value = VALUE_TB - ply;

                    // Use the range VALUE_TB to VALUE_TB_WIN_IN_MAX_PLY to
                    // score
                    let value = if wdl < -draw_score {
                        -tb_value
                    } else if wdl > draw_score {
                        tb_value
                    } else {
                        VALUE_DRAW + 2 * wdl * draw_score
                    };

                    let b = if wdl < -draw_score {
                        BOUND_UPPER
                    } else if wdl > draw_score {
                        BOUND_LOWER
                    } else {
                        BOUND_EXACT
                    };

                    if b == BOUND_EXACT
                        || (if b == BOUND_LOWER {
                            value >= beta
                        } else {
                            value <= alpha
                        })
                    {
                        tt_writer.write(
                            pos_key,
                            value_to_tt(value, ply),
                            self.stack[ss].tt_pv,
                            b,
                            (MAX_PLY - 1).min(depth + 6),
                            Move::none(),
                            VALUE_NONE,
                            sh.tt.generation(),
                        );

                        return value;
                    }

                    if pv_node {
                        if b == BOUND_LOWER {
                            best_value = value;
                            alpha = alpha.max(best_value);
                        } else {
                            max_value = value;
                        }
                    }
                }
            }
        }

        if !self.stack[ss].in_check {
            // Use static evaluation difference to improve quiet move ordering
            if prev_move.is_ok() && !self.stack[ss - 1].in_check && !prior_capture {
                let eval_diff = (-(self.stack[ss - 1].static_eval + self.stack[ss].static_eval))
                    .clamp(-189, 194)
                    + 60;
                self.hist.main_history[(!us).idx()][prev_move.raw() as usize]
                    .update(eval_diff * 11);
                if !tt_hit
                    && pos.piece_on(prev_sq).type_of() != PAWN
                    && prev_move.type_of() != PROMOTION
                {
                    sh.histories.pawn_entry(pos)[pos.piece_on(prev_sq).idx()][prev_sq.idx()]
                        .update(eval_diff * 13);
                }
            }

            // Step 8. Razoring
            // If eval is really low, skip search entirely and return the
            // qsearch value
            if all_node && eval < alpha - 342 * depth && !self.seek_mate {
                return self.qsearch::<NON_PV>(sh, pos, ss, alpha, beta);
            }

            // Step 9. Futility pruning: child node
            // The depth condition is important for mate finding. It should
            // NOT be tuned.
            if !self.stack[ss].tt_pv
                && depth < if self.seek_mate { 6 } else { 19 }
                && eval >= beta
                && (tt_data.mv == Move::none() || tt_capture)
                && !is_loss(beta)
                && !is_win(eval)
            {
                let mut futility_mult = (45 + depth * 4).min(85);
                futility_mult -= 20 * (!self.stack[ss].tt_hit) as i32;

                let futility_margin = futility_mult * depth
                    - (2789 * improving as i32 + 335 * opponent_worsening as i32) * futility_mult
                        / 1024
                    + correction_value.abs() / 198435;

                if eval - futility_margin >= beta {
                    return (661 * beta + 363 * eval) / 1024;
                }
            }

            // Step 10. Null move search with verification search
            if cut_node
                && self.stack[ss].static_eval + 50 * self.stack[ss].prior_nmp_fail_high
                    >= beta - 13 * depth - 47 * improving as i32 + 365
                && excluded_move == Move::none()
                && pos.non_pawn_material(us) != 0
                && ply >= self.nmp_min_ply
                && beta >= -2000
            {
                debug_assert!(self.stack[ss - 1].current_move != Move::null());

                // Null move dynamic reduction based on depth
                let mut r = 7 + depth / 3 + ((self.stack[ss].static_eval - beta) / 256).max(0);
                r -= self.seek_mate as i32 * r / 2;
                self.do_null_move(pos, ss);

                let null_value =
                    -self.search::<NON_PV>(sh, pos, ss + 1, -beta, -beta + 1, depth - r, false);

                self.undo_null_move(pos);

                // Do not return unproven mate or TB scores
                if null_value >= beta && !is_win(null_value) {
                    if self.nmp_min_ply != 0 || depth < 16 {
                        self.stack[ss].prior_nmp_fail_high += 1;
                        return null_value;
                    }

                    // Recursive verification is not allowed
                    debug_assert!(self.nmp_min_ply == 0);

                    // Do verification search at high depths, with null move
                    // pruning disabled until ply exceeds nmpMinPly.
                    self.nmp_min_ply = ply + 3 * (depth - r) / 4;

                    let v = self.search::<NON_PV>(sh, pos, ss, beta - 1, beta, depth - r, false);

                    self.nmp_min_ply = 0;

                    if v >= beta {
                        self.stack[ss].prior_nmp_fail_high += 1;
                        return null_value;
                    }
                }
            }

            improving |= self.stack[ss].static_eval >= beta;

            // Step 11. Internal iterative reductions
            // At sufficient depth, reduce depth for PV/Cut nodes without a
            // TTMove. (*Scaler) Making IIR more aggressive scales poorly.
            if !self.stack[ss].follow_pv && !all_node && depth >= 6 && tt_data.mv == Move::none() {
                depth -= 1;
            }

            // Step 12. ProbCut
            // If we have a good enough capture (or queen promotion) and a
            // reduced search returns a value much above beta, we can
            // (almost) safely prune the previous move.
            let prob_cut_beta = beta + 241 - 64 * improving as i32;
            if depth >= 3
                && !is_decisive(beta)
                && !(is_valid(tt_data.value) && tt_data.value < prob_cut_beta)
            {
                debug_assert!(prob_cut_beta < VALUE_INFINITE && prob_cut_beta > beta);

                let mut mp = MovePicker::new_probcut(
                    pos,
                    tt_data.mv,
                    prob_cut_beta - self.stack[ss].static_eval,
                    &sh.histories,
                );
                let prob_cut_depth = depth - if improving { 5 } else { 3 };

                loop {
                    let mv = mp.next_move(pos, &self.hist, &sh.histories);
                    if mv == Move::none() {
                        break;
                    }

                    debug_assert!(mv.is_ok());

                    if mv == excluded_move {
                        continue;
                    }

                    let capture = pos.capture_stage(mv);
                    debug_assert!(capture);

                    self.do_move(sh, pos, mv, pos.gives_check(mv), capture, ss);
                    // Perform a preliminary qsearch to verify that the move
                    // holds
                    let mut value = -self.qsearch::<NON_PV>(
                        sh,
                        pos,
                        ss + 1,
                        -prob_cut_beta,
                        -prob_cut_beta + 1,
                    );

                    // If the qsearch held, perform the regular search
                    if value >= prob_cut_beta && prob_cut_depth > 0 {
                        value = -self.search::<NON_PV>(
                            sh,
                            pos,
                            ss + 1,
                            -prob_cut_beta,
                            -prob_cut_beta + 1,
                            prob_cut_depth,
                            !cut_node,
                        );
                    }

                    self.undo_move(pos, mv);

                    if value >= prob_cut_beta {
                        // Save ProbCut data into transposition table
                        tt_writer.write(
                            pos_key,
                            value_to_tt(value, ply),
                            self.stack[ss].tt_pv,
                            BOUND_LOWER,
                            prob_cut_depth + 1,
                            mv,
                            unadjusted_static_eval,
                            sh.tt.generation(),
                        );

                        if !is_decisive(value) {
                            return value - (prob_cut_beta - beta);
                        }
                    }
                }
            }
        }

        // moves_loop: When in check, search starts here

        // Step 13. A small ProbCut idea
        let prob_cut_beta = beta + 428;
        if (tt_data.bound & BOUND_LOWER) != 0
            && tt_data.depth >= depth - 4
            && tt_data.value >= prob_cut_beta
            && !is_decisive(beta)
            && is_valid(tt_data.value)
            && !is_decisive(tt_data.value)
            && !self.seek_mate
        {
            return prob_cut_beta;
        }

        let cont_hist: [&PieceToHistory; 6] = [
            sh.histories
                .continuation(self.stack[ss - 1].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 2].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 3].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 4].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 5].continuation_history),
            sh.histories
                .continuation(self.stack[ss - 6].continuation_history),
        ];

        let mut mp = MovePicker::new(pos, tt_data.mv, depth, cont_hist, ply);

        let mut value = best_value;

        let mut move_count = 0;

        // Step 14. Loop through all legal moves until no moves remain or a
        // beta cutoff occurs.
        loop {
            let mv = mp.next_move(pos, &self.hist, &sh.histories);
            if mv == Move::none() {
                break;
            }

            debug_assert!(mv.is_ok());

            if mv == excluded_move {
                continue;
            }

            // At root obey the "searchmoves" option and skip moves not listed
            // in Root Move List. In MultiPV mode we also skip PV moves that
            // have been already searched and those of lower "TB rank" if we
            // are in a TB root position.
            if root_node
                && !self.root_moves[self.pv_idx..self.pv_last]
                    .iter()
                    .any(|rm| rm.pv[0] == mv)
            {
                continue;
            }

            move_count += 1;
            self.stack[ss].move_count = move_count;

            if root_node && self.manager.is_some() && self.nodes() > NODES_LIMIT_OUTPUT {
                let manager = self.manager.as_deref().expect("the main thread");
                (manager.updates.on_iter)(&InfoIteration {
                    depth,
                    currmove: &move_to_uci(mv, pos.is_chess960()),
                    currmovenumber: move_count as usize + self.pv_idx,
                });
            }

            // Whether the child's principal variation is collected, which
            // Stockfish tells by its stack's pointer to it
            let mut child_pv = false;

            let mut extension = 0;
            let capture = pos.capture_stage(mv);
            let moved_piece = pos.moved_piece(mv);
            let gives_check = pos.gives_check(mv);

            // Calculate new depth for this move
            let mut new_depth = depth - 1;

            let mut r = self.reduction(improving, depth, move_count);

            // Decrease reduction for PvNodes
            if pv_node {
                r -= 512;
            }

            // Increase reduction for ttPv nodes
            // (*Scaler) Larger values scale well.
            if self.stack[ss].tt_pv {
                r += 929;
            }

            // Step 15. Pruning at shallow depths.
            // Depth conditions are important for mate finding.
            if !root_node && pos.non_pawn_material(us) != 0 && !is_loss(best_value) {
                // Skip quiet moves if movecount exceeds our threshold
                if move_count >= (3 + depth * depth) / (2 - improving as i32) {
                    mp.skip_quiet_moves();
                }

                // Reduced depth of the next LMR search
                let mut lmr_depth = new_depth - r / 1024;

                if capture || gives_check {
                    let captured_piece = pos.piece_on(mv.to_sq());
                    let capt_hist = self.hist.capture_history[moved_piece.idx()][mv.to_sq().idx()]
                        [captured_piece.type_of().idx()]
                    .get();

                    // Futility pruning for captures
                    if !gives_check && lmr_depth < 8 {
                        let futility_value = self.stack[ss].static_eval
                            + 234
                            + 247 * lmr_depth
                            + captured_piece.value()
                            + 134 * capt_hist / 1024;

                        if futility_value <= alpha {
                            continue;
                        }
                    }

                    // SEE based pruning for captures and checks.
                    // Avoid pruning sacrifices of our last piece for
                    // stalemate.
                    let margin = 177 * depth + capt_hist * 34 / 1024;
                    if (alpha >= VALUE_DRAW || pos.non_pawn_material(us) != moved_piece.value())
                        && !pos.see_ge(mv, -margin)
                    {
                        continue;
                    }
                } else if !self.stack[ss].follow_pv || !pv_node {
                    let mut history = cont_hist[0][moved_piece.idx()][mv.to_sq().idx()].get()
                        + cont_hist[1][moved_piece.idx()][mv.to_sq().idx()].get()
                        + sh.histories.pawn_entry(pos)[moved_piece.idx()][mv.to_sq().idx()].get();

                    // Continuation history based pruning
                    if history < -4136 * depth {
                        continue;
                    }

                    history += 69 * self.hist.main_history[us.idx()][mv.raw() as usize].get() / 32;

                    // (*Scaler): Generally, lower divisors scale well
                    lmr_depth += history / lmr_divisor(depth);

                    let futility_value = self.stack[ss].static_eval
                        + 119 * lmr_depth
                        + 90 * (self.stack[ss].static_eval > alpha) as i32
                        + 164;

                    // Futility pruning: parent node
                    // (*Scaler): Generally, more frequent futility pruning
                    // scales well
                    if !self.stack[ss].in_check && lmr_depth < 12 && futility_value <= alpha {
                        if best_value <= futility_value
                            && !is_decisive(best_value)
                            && !is_win(futility_value)
                        {
                            best_value = futility_value;
                        }
                        continue;
                    }

                    lmr_depth = lmr_depth.max(0);

                    // Prune moves with negative SEE
                    if !pos.see_ge(mv, -23 * lmr_depth * lmr_depth) {
                        continue;
                    }
                }
            }

            // Step 16. Singular Extensions
            //
            // We check for "only moves": if one move fails high on (alpha,
            // beta) but all others fail low on (alpha-s, beta-s), then that
            // move is singular. Singular moves are extended to better
            // estimate the result of the (putatively) forced line. If it's
            // non-singular, we may do some pruning or reduction.
            //
            // Recursive excluded search is avoided. The `excludedMove`
            // mechanism was historically a hack and remains a bit fragile.
            //
            // (*Scaler) Generally, higher singularBeta (i.e closer to
            // ttValue) and lower extension margins scale well.
            if !root_node
                && mv == tt_data.mv
                && excluded_move == Move::none()
                && depth >= 6 + self.stack[ss].tt_pv as i32
                && is_valid(tt_data.value)
                && !is_decisive(tt_data.value)
                && (tt_data.bound & BOUND_LOWER) != 0
                && tt_data.depth >= depth - 3
                && !self.is_shuffling(mv, ss, pos)
                && !self.seek_mate
            {
                let singular_beta = tt_data.value
                    - (59 + 66 * (self.stack[ss].tt_pv && !pv_node) as i32) * depth / 63;
                let singular_depth = new_depth / 2;

                self.stack[ss].excluded_move = mv;
                value = self.search::<NON_PV>(
                    sh,
                    pos,
                    ss,
                    singular_beta - 1,
                    singular_beta,
                    singular_depth,
                    cut_node,
                );
                self.stack[ss].excluded_move = Move::none();

                if value < singular_beta {
                    let corr_val_adj = correction_value.abs() / 198368;
                    let double_margin = -2 + 204 * pv_node as i32
                        - 152 * !tt_capture as i32
                        - corr_val_adj
                        - 1175 * self.tt_move_history.get() / 114178
                        - (ply > self.root_depth) as i32 * 38;
                    let triple_margin = 70 + 279 * pv_node as i32 - 188 * !tt_capture as i32
                        + 81 * self.stack[ss].tt_pv as i32
                        - corr_val_adj
                        - (ply > self.root_depth) as i32 * 43;

                    extension = 1
                        + (value < singular_beta - double_margin) as i32
                        + (value < singular_beta - triple_margin) as i32;

                    depth += 1;
                }
                // Multi-cut pruning
                // Our ttMove is assumed to fail high based on the bound of
                // the TT entry, and if after excluding the ttMove with a
                // reduced search we fail high over the original beta, we
                // assume this expected cut-node is not singular (multiple
                // moves fail high), and we can prune the whole subtree by
                // returning a softbound.
                else if value >= beta && !is_decisive(value) {
                    self.tt_move_history.update(-421 - 110 * depth);

                    if !self.stack[ss].in_check && value > self.stack[ss].static_eval {
                        let bonus = ((value - self.stack[ss].static_eval) * 664 / 1024)
                            .clamp(-CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
                        self.update_correction_history(sh, pos, ss, bonus);
                    }

                    return value;
                }
                // Negative extensions
                // If other moves failed high over (ttValue - margin) without
                // the ttMove on a reduced search, but we cannot do multi-cut
                // because (ttValue - margin) is lower than the original
                // beta, we do not know if the ttMove is singular or can do a
                // multi-cut, so we reduce the ttMove in favor of other moves
                // based on some conditions:

                // If the ttMove is assumed to fail high over current beta or
                // if we are on a cutNode
                else if tt_data.value >= beta || cut_node {
                    extension = -3;
                }
            }

            let node_count = if root_node { self.nodes() } else { 0 };

            // Step 17. Make the move
            self.do_move(sh, pos, mv, gives_check, capture, ss);

            // Add extension to new depth
            new_depth += extension;

            // Step 18. Compute and apply late moves reductions/extensions
            // (LMR)

            // Decrease reduction for PvNodes (*Scaler)
            if self.stack[ss].tt_pv {
                r -= 3023
                    + pv_node as i32 * 1004
                    + (tt_data.value > alpha) as i32 * 885
                    + (tt_data.depth >= depth) as i32 * (816 + cut_node as i32 * 940);
            }

            // Base reduction offset to compensate for other tweaks
            r += 697;

            r -= move_count * 65;
            r -= correction_value.abs() / 26310;

            // Increase reduction for cut nodes
            if cut_node {
                r += 4026 + 933 * (tt_data.mv == Move::none()) as i32;
            }

            // Increase reduction if ttMove is a capture
            if tt_capture {
                r += 1079;
            }

            // Increase reduction if next ply has a lot of fail high
            if self.stack[ss + 1].cutoff_cnt > 1 {
                r += 264
                    + 1095 * (self.stack[ss + 1].cutoff_cnt > 2) as i32
                    + 1138 * all_node as i32;
            }
            // For first picked move (ttMove) reduce reduction
            else if mv == tt_data.mv {
                r -= 2179;
            }

            if capture {
                let captured = pos.captured_piece();
                self.stack[ss].stat_score = 873 * captured.value() / 128
                    + self.hist.capture_history[moved_piece.idx()][mv.to_sq().idx()]
                        [captured.type_of().idx()]
                    .get();
            } else {
                self.stack[ss].stat_score = (2252
                    * self.hist.main_history[us.idx()][mv.raw() as usize].get()
                    + 1126 * cont_hist[0][moved_piece.idx()][mv.to_sq().idx()].get()
                    + 1093 * cont_hist[1][moved_piece.idx()][mv.to_sq().idx()].get())
                    / 1024;
            }

            // Decrease/increase reduction for moves with a good/bad history
            r -= self.stack[ss].stat_score * 439 / 4096;

            if !capture && !is_decisive(alpha) {
                r += 3 * (alpha - eval).clamp(-64, 96);
            }

            // Scale up reductions for expected ALL nodes
            if all_node {
                r += r * 276 / (256 * depth + 268);
            }

            // When seeking a mate, reduce non-checking moves more and
            // checking moves less
            if self.seek_mate && r > 0 {
                r += (if gives_check { -r } else { r }) / 4;
            }

            // Apply the computed LMR
            if depth >= 2 && move_count > 1 {
                // In general we want to cap the LMR depth search at
                // newDepth, but when reduction is negative, we allow this
                // move a limited search extension beyond the first move
                // depth. To avoid search explosion, extensions are not
                // allowed deep, relative to rootDepth, in the search tree.
                let d = (new_depth
                    + (-r / 1024).min(if ply < 2 * self.root_depth { 2 } else { 0 }))
                .max(1)
                    + pv_node as i32;

                self.stack[ss].reduction = new_depth - d;
                value = -self.search::<NON_PV>(sh, pos, ss + 1, -(alpha + 1), -alpha, d, true);
                self.stack[ss].reduction = 0;

                // Do a full-depth search when reduced LMR search fails high
                // (*Scaler) Shallower searches here don't scale well
                if value > alpha {
                    // Adjust full-depth search based on LMR results - if the
                    // result was good enough search deeper, if it was bad
                    // enough search shallower.
                    let do_deeper_search = d < new_depth && value > best_value + 53;
                    let do_shallower_search = value < best_value + 8;

                    new_depth += do_deeper_search as i32 - do_shallower_search as i32;

                    if new_depth > d {
                        value = -self.search::<NON_PV>(
                            sh,
                            pos,
                            ss + 1,
                            -(alpha + 1),
                            -alpha,
                            new_depth,
                            !cut_node,
                        );
                    }

                    // Post LMR continuation history updates
                    self.update_continuation_histories(sh, ss, moved_piece, mv.to_sq(), 1334);
                }
            }
            // Step 19. Full-depth search when LMR is skipped
            else if !pv_node || move_count > 1 {
                // Increase reduction if ttMove is not present
                if tt_data.mv == Move::none() {
                    r += 1127;
                }

                // If expected reduction is high, we reduce search depth here
                value = -self.search::<NON_PV>(
                    sh,
                    pos,
                    ss + 1,
                    -(alpha + 1),
                    -alpha,
                    new_depth - (r > 5234) as i32 - (r > 5487 && new_depth > 2) as i32,
                    !cut_node,
                );
            }

            // Step 20. For PV nodes only, do a full PV search on the first
            // move or after a fail high, otherwise let the parent node fail
            // low with value <= alpha and try another move.
            if pv_node && (move_count == 1 || value > alpha) {
                child_pv = true;
                self.pv_table[ply].clear();

                // Extend move from transposition table if we are about to
                // dive into qsearch. Decisive score handling improves mate
                // finding and retrograde analysis.
                if mv == tt_data.mv
                    && ((is_valid(tt_data.value)
                        && is_decisive(tt_data.value)
                        && tt_data.depth > 0)
                        || tt_data.depth > 1)
                {
                    new_depth = new_depth.max(1);
                }

                value = -self.search::<PV>(sh, pos, ss + 1, -beta, -alpha, new_depth, false);
            }

            // Step 21. Undo move
            self.undo_move(pos, mv);

            debug_assert!(value > -VALUE_INFINITE && value < VALUE_INFINITE);

            // Step 22. Check for a new best move
            // If a stop occurred, the value of the search cannot be trusted,
            // and we return immediately without updating the best move,
            // principal variation or transposition table.
            if sh.stop.load(Relaxed) {
                return VALUE_ZERO;
            }

            if root_node {
                let nodes = self.nodes();
                let rm = self
                    .root_moves
                    .iter_mut()
                    .find(|rm| rm.pv[0] == mv)
                    .expect("root moves include the moves searched at the root");

                rm.effort += nodes - node_count;

                let n = nodes - node_count;
                let e_prev = (rm.effort - n).max(1);

                // Exponential moving average parameters for root move
                const SCALE: u64 = 32;
                const CHI_NUMERATOR: u64 = 3;
                const CHI_DENOMINATOR: u64 = 2; // Chi = 3/2 = 1.5
                const MIN_WEIGHT: u64 = 12; // 37.5% minimum weight
                const MAX_WEIGHT: u64 = 24; // 75% maximum weight

                let w = ((SCALE * n * CHI_DENOMINATOR)
                    / (n * CHI_DENOMINATOR + CHI_NUMERATOR * e_prev))
                    .clamp(MIN_WEIGHT, MAX_WEIGHT);
                let w_mss = w.min(16);
                let v2 = i64::from(value) * i64::from(value.abs());

                // Stockfish averages in unsigned 64-bit arithmetic, which
                // amounts to rounding the averages down.
                if rm.average_score == -VALUE_INFINITE {
                    rm.average_score = value;
                } else {
                    rm.average_score = (i64::from(value) * w as i64
                        + i64::from(rm.average_score) * (SCALE - w) as i64)
                        .div_euclid(SCALE as i64) as Value;
                }

                if rm.mean_squared_score == -VALUE_INFINITE * VALUE_INFINITE {
                    rm.mean_squared_score = value * value.abs();
                } else {
                    rm.mean_squared_score = (v2 * w_mss as i64
                        + i64::from(rm.mean_squared_score) * (SCALE - w_mss) as i64)
                        .div_euclid(SCALE as i64)
                        as Value;
                }

                // PV move or new best move?
                if move_count == 1 || value > alpha {
                    rm.score = value;
                    rm.uci_score = value;
                    rm.sel_depth = self.sel_depth;
                    rm.unset_inexact();

                    if value >= beta {
                        rm.inexact_lower = true;
                        rm.uci_score = beta;
                    } else if value <= alpha {
                        rm.inexact_upper = true;
                        rm.uci_score = alpha;
                    }

                    rm.pv.truncate(1);

                    debug_assert!(child_pv);

                    rm.pv.extend_from_slice(self.pv_table[ply].as_slice());

                    // We record how often the best move has been changed in
                    // each iteration. This information is used for time
                    // management. In MultiPV mode, we must take care to only
                    // do this for the first PV line.
                    if move_count > 1 && self.pv_idx == 0 {
                        increment(&self.counters.best_move_changes);
                    }
                } else {
                    // All other moves but the PV are set to the lowest value:
                    // this is not a problem when sorting because the sort is
                    // stable and the move position in the list is preserved
                    // -- just the PV is pushed up.
                    rm.score = -VALUE_INFINITE;
                }
            }

            // If we have an alternative move equal in value to the current
            // bestmove, we sometimes promote it to bestmove by pretending it
            // just exceeds alpha (but not beta).
            let inc = (value == best_value
                && ply + 2 >= self.root_depth
                && (self.nodes() & 14) == 0
                && !is_win(value.abs() + 1)) as i32;

            if value + inc > best_value {
                best_value = value;

                if value + inc > alpha {
                    best_move = mv;

                    // Update PV even in fail-high case
                    if pv_node && !root_node {
                        self.pv_table.update(ply, mv, child_pv);
                    }

                    if value >= beta {
                        // (*Scaler) Infrequent and small updates scale well
                        self.stack[ss].cutoff_cnt += (extension < 2 || pv_node) as i32;
                        debug_assert!(value >= beta); // Fail high
                        break;
                    }

                    // Reduce other moves if we have found at least one score
                    // improvement
                    if depth > 3 && depth < 12 && !is_decisive(value) {
                        depth -= 3;
                    }

                    debug_assert!(depth > 0);
                    alpha = value; // Update alpha! Always alpha < beta
                }
            }

            // If the move is worse than some previously searched move,
            // remember it, to update its stats later.
            if mv != best_move && move_count <= SEARCHEDLIST_CAPACITY {
                if capture {
                    captures_searched.push(mv);
                } else {
                    quiets_searched.push(mv);
                }
            }
        }

        // Step 23. Check for mate and stalemate, otherwise update
        // bestmove/countermove stats

        debug_assert!(
            move_count != 0
                || !self.stack[ss].in_check
                || excluded_move != Move::none()
                || MoveList::new::<ALL>(pos).is_empty()
        );

        // Adjust best value for fail high cases
        if best_value >= beta && !is_decisive(best_value) && !is_decisive(alpha) {
            best_value = (best_value * depth + beta) / (depth + 1);
        }

        // All legal moves have been searched: if there are no legal moves,
        // it must be a mate or a stalemate (just a fail low score if we are
        // in a singular extension search).
        if move_count == 0 {
            best_value = if excluded_move != Move::none() {
                alpha
            } else if self.stack[ss].in_check {
                mated_in(ply)
            } else {
                VALUE_DRAW
            };
        }
        // If there is a move that produces search value greater than alpha,
        // we update the stats of searched moves.
        else if best_move != Move::none() {
            self.update_all_stats(
                sh,
                pos,
                ss,
                best_move,
                prev_sq,
                &quiets_searched,
                &captures_searched,
                depth,
                tt_data.mv,
                pv_node,
            );
            if !pv_node {
                self.tt_move_history
                    .update(if best_move == tt_data.mv { 918 } else { -747 });
            }
        }
        // Bonus for prior quiet countermove that caused the fail low
        else if !prior_capture && prev_sq != Square::NONE {
            let mut bonus_scale = -241;
            bonus_scale -= self.stack[ss - 1].stat_score / 98;
            bonus_scale += (59 * depth).min(420);
            bonus_scale += 186 * (self.stack[ss - 1].move_count > 9) as i32;
            bonus_scale += 142
                * (!self.stack[ss].in_check && best_value <= self.stack[ss].static_eval - 106)
                    as i32;
            bonus_scale += 159
                * (!self.stack[ss - 1].in_check
                    && best_value <= -self.stack[ss - 1].static_eval - 68) as i32;

            bonus_scale = bonus_scale.max(0);

            // scaledBonus ranges from 0 to roughly 2.3M, overflows happen for
            // multipliers larger than 900
            let scaled_bonus = (150 * depth - 85).min(1337) * bonus_scale;

            self.update_continuation_histories(
                sh,
                ss - 1,
                pos.piece_on(prev_sq),
                prev_sq,
                scaled_bonus * 263 / 16384,
            );

            self.hist.main_history[(!us).idx()][prev_move.raw() as usize]
                .update(scaled_bonus * 215 / 32768);

            if pos.piece_on(prev_sq).type_of() != PAWN && prev_move.type_of() != PROMOTION {
                sh.histories.pawn_entry(pos)[pos.piece_on(prev_sq).idx()][prev_sq.idx()]
                    .update(scaled_bonus * 324 / 8192);
            }
        }
        // Bonus for prior capture countermove that caused the fail low
        else if prev_sq != Square::NONE {
            debug_assert!(prior_capture);

            let captured_piece = pos.captured_piece();
            debug_assert!(captured_piece != NO_PIECE);
            self.hist.capture_history[pos.piece_on(prev_sq).idx()][prev_sq.idx()]
                [captured_piece.type_of().idx()]
            .update(892);
        }

        if pv_node {
            best_value = best_value.min(max_value);
        }

        // If no good move is found and the previous position was ttPv, then
        // the previous opponent move is probably good and the new position
        // is added to the search tree.
        if best_value <= alpha {
            self.stack[ss].tt_pv = self.stack[ss].tt_pv || self.stack[ss - 1].tt_pv;
        }

        // Step 24. Write gathered information in transposition table. Note
        // that the static evaluation is saved as it was before correction
        // history.
        if excluded_move == Move::none() && !(root_node && self.pv_idx != 0) {
            tt_writer.write(
                pos_key,
                value_to_tt(best_value, ply),
                self.stack[ss].tt_pv,
                if best_value >= beta {
                    BOUND_LOWER
                } else if pv_node && best_move != Move::none() {
                    BOUND_EXACT
                } else {
                    BOUND_UPPER
                },
                if move_count != 0 {
                    depth
                } else {
                    (depth + 6).min(MAX_PLY - 1)
                },
                best_move,
                unadjusted_static_eval,
                sh.tt.generation(),
            );
        }

        // Adjust correction history if the best move is not a capture and
        // the error direction matches whether we are above/below bounds.
        if !self.stack[ss].in_check
            && !(best_move != Move::none() && pos.capture(best_move))
            && (best_value > self.stack[ss].static_eval) == (best_move != Move::none())
        {
            let bonus = ((best_value - self.stack[ss].static_eval)
                * depth
                * if best_move != Move::none() { 12 } else { 18 }
                / 128)
                .clamp(-CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
            self.update_correction_history(sh, pos, ss, 1061 * bonus / 1024);
        }

        // The search is now complete
        debug_assert!(-VALUE_INFINITE < best_value && best_value < VALUE_INFINITE);
        best_value
    }

    /// Quiescence search function, which is called by the main search
    /// function with depth zero, or recursively with further decreasing
    /// depth. With depth <= 0, we "should" be using static eval only, but
    /// tactical moves may confuse the static eval. To fight this horizon
    /// effect, we implement this qsearch of tactical moves. See
    /// https://www.chessprogramming.org/Horizon_Effect and
    /// https://www.chessprogramming.org/Quiescence_Search
    fn qsearch<const NT: NodeType>(
        &mut self,
        sh: &SharedState,
        pos: &mut Position<V>,
        ss: usize,
        mut alpha: Value,
        beta: Value,
    ) -> Value {
        debug_assert!(NT != ROOT);
        let pv_node = NT == PV;

        debug_assert!(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || alpha == beta - 1);

        let ply = ply_of(ss);

        // Check if we have an upcoming move that draws by repetition
        if alpha < VALUE_DRAW && pos.upcoming_repetition(ply) {
            alpha = value_draw(self.nodes());
            if alpha >= beta {
                return alpha;
            }
        }

        // Step 1. Initialize node
        if pv_node {
            // The node's own principal variation starts empty, and so does
            // its parent's, which it collects.
            self.pv_table[ply].clear();
            self.pv_table[ply - 1].clear();
        }

        let mut best_move = Move::none();
        self.stack[ss].in_check = pos.checkers().any();
        let mut move_count = 0;

        // Used to send selDepth info to GUI (selDepth counts from 1, ply from
        // 0)
        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }

        // Step 2. Check for an immediate draw or maximum ply reached
        if pos.is_draw(ply) || ply >= MAX_PLY {
            return if ply >= MAX_PLY && !self.stack[ss].in_check {
                self.evaluate(sh, pos)
            } else {
                VALUE_DRAW
            };
        }

        debug_assert!((0..MAX_PLY).contains(&ply));

        // Step 3. Transposition table lookup
        let pos_key = pos.key();
        let (tt_hit, mut tt_data, tt_writer) = sh.tt.probe(pos_key);

        self.stack[ss].tt_hit = tt_hit;
        tt_data.mv = if tt_hit { tt_data.mv } else { Move::none() };
        tt_data.value = if tt_hit {
            value_from_tt(tt_data.value, ply, pos.rule50_count())
        } else {
            VALUE_NONE
        };
        let pv_hit = tt_hit && tt_data.is_pv;

        // At non-PV nodes we check for an early TT cutoff
        if !pv_node
            && tt_data.depth >= DEPTH_QS
            && is_valid(tt_data.value)
            && (tt_data.bound
                & if tt_data.value >= beta {
                    BOUND_LOWER
                } else {
                    BOUND_UPPER
                })
                != 0
        {
            return tt_data.value;
        }

        // Step 4. Static evaluation of the position
        let mut unadjusted_static_eval = VALUE_NONE;
        let mut best_value;
        let futility_base;
        if self.stack[ss].in_check {
            best_value = -VALUE_INFINITE;
            futility_base = -VALUE_INFINITE;
        } else {
            let correction_value = self.correction_value(sh, pos, ss);

            if self.stack[ss].tt_hit {
                // Never assume anything about values stored in TT
                unadjusted_static_eval = tt_data.eval;

                if !is_valid(unadjusted_static_eval) {
                    unadjusted_static_eval = self.evaluate(sh, pos);
                }

                best_value = to_corrected_static_eval(unadjusted_static_eval, correction_value);
                self.stack[ss].static_eval = best_value;

                // ttValue can be used as a better position evaluation
                if is_valid(tt_data.value)
                    && !is_decisive(tt_data.value)
                    && (tt_data.bound
                        & if tt_data.value > best_value {
                            BOUND_LOWER
                        } else {
                            BOUND_UPPER
                        })
                        != 0
                {
                    best_value = tt_data.value;
                }
            } else {
                unadjusted_static_eval = self.evaluate(sh, pos);
                best_value = to_corrected_static_eval(unadjusted_static_eval, correction_value);
                self.stack[ss].static_eval = best_value;
            }

            // Stand pat. Return immediately if static value is at least beta
            if best_value >= beta {
                debug_assert!(!is_decisive(best_value));
                best_value = (441 * best_value + 583 * beta) / 1024;

                if !self.stack[ss].tt_hit {
                    tt_writer.write(
                        pos_key,
                        VALUE_NONE,
                        false,
                        BOUND_LOWER,
                        DEPTH_UNSEARCHED,
                        Move::none(),
                        unadjusted_static_eval,
                        sh.tt.generation(),
                    );
                }
                return best_value;
            }

            if best_value > alpha {
                alpha = best_value;
            }

            futility_base = self.stack[ss].static_eval + 306;
        }

        let cont_hist = sh
            .histories
            .continuation(self.stack[ss - 1].continuation_history);

        let prev_move = self.stack[ss - 1].current_move;
        let prev_sq = if prev_move.is_ok() {
            prev_move.to_sq()
        } else {
            Square::NONE
        };

        // Initialize a MovePicker object for the current position, and
        // prepare to search the moves. We presently use two stages of move
        // generator in quiescence search: captures, or evasions only when in
        // check.
        let mut mp = MovePicker::new(pos, tt_data.mv, DEPTH_QS, [cont_hist; 6], ply);

        // Step 5. Loop through all legal moves until no moves remain or a
        // beta cutoff occurs.
        loop {
            let mv = mp.next_move(pos, &self.hist, &sh.histories);
            if mv == Move::none() {
                break;
            }

            debug_assert!(mv.is_ok());

            let gives_check = pos.gives_check(mv);
            let capture = pos.capture_stage(mv);

            move_count += 1;

            // Step 6. Pruning
            if !is_loss(best_value) {
                // Futility pruning and moveCount pruning
                if !gives_check
                    && mv.to_sq() != prev_sq
                    && !is_loss(futility_base)
                    && mv.type_of() != PROMOTION
                {
                    if move_count > 2 {
                        continue;
                    }

                    let futility_value = futility_base + pos.piece_on(mv.to_sq()).value();

                    // If static eval + value of piece we are going to capture
                    // is much lower than alpha, we can prune this move.
                    if futility_value <= alpha {
                        best_value = best_value.max(futility_value);
                        continue;
                    }

                    // If static exchange evaluation is low enough, we can
                    // prune
                    if !pos.see_ge(mv, alpha - futility_base) {
                        best_value = best_value.max(alpha.min(futility_base));
                        continue;
                    }
                }

                // Skip non-captures
                if !capture {
                    continue;
                }

                // Do not search moves with bad enough SEE values
                if !pos.see_ge(mv, -74) {
                    continue;
                }
            }

            // Step 7. Make and search the move
            self.do_move(sh, pos, mv, gives_check, capture, ss);

            let value = -self.qsearch::<NT>(sh, pos, ss + 1, -beta, -alpha);
            self.undo_move(pos, mv);

            debug_assert!(value > -VALUE_INFINITE && value < VALUE_INFINITE);

            // Step 8. Check for a new best move
            if value > best_value {
                best_value = value;

                if value > alpha {
                    best_move = mv;

                    // Update pv even in fail-high case
                    if pv_node {
                        self.pv_table.update(ply, mv, true);
                    }

                    if value < beta {
                        // Update alpha here!
                        alpha = value;
                    } else {
                        break; // Fail high
                    }
                }
            }
        }

        // Step 9. Check for mate and stalemate
        // All legal moves have been searched. A special case: if we are in
        // check and no legal moves were found, it is checkmate.
        if move_count == 0 {
            if self.stack[ss].in_check {
                // Checkmate!
                debug_assert!(MoveList::new::<ALL>(pos).is_empty());
                return mated_in(ply); // Plies to mate from the root
            }

            // Only check for stalemate under specific conditions
            let us = pos.side_to_move();
            if (pawn_single_push_bb(us, pos.pieces_cpt(us, PAWN)) & !pos.pieces()).is_empty()
                && pos.non_pawn_material(us) == 0
                && pos.captured_piece().type_of() >= KNIGHT
                && MoveList::new::<ALL>(pos).is_empty()
            {
                best_value = VALUE_DRAW;
            }
        }

        if !is_decisive(best_value) && best_value > beta {
            best_value = (462 * best_value + 562 * beta) / 1024;
        }

        // Step 10. Save gathered info in transposition table. The static
        // evaluation is saved as it was before adjustment by correction
        // history.
        tt_writer.write(
            pos_key,
            value_to_tt(best_value, ply),
            pv_hit,
            if best_value >= beta {
                BOUND_LOWER
            } else {
                BOUND_UPPER
            },
            DEPTH_QS,
            best_move,
            unadjusted_static_eval,
            sh.tt.generation(),
        );

        // The search is now complete
        debug_assert!(-VALUE_INFINITE < best_value && best_value < VALUE_INFINITE);
        best_value
    }

    #[inline(always)]
    fn reduction(&self, i: bool, d: Depth, mn: i32) -> i32 {
        let reduction_scale = self.reductions[d as usize & (MOVE_LIST_SIZE - 1)]
            * self.reductions[mn as usize & (MOVE_LIST_SIZE - 1)];
        reduction_scale + (!i) as i32 * reduction_scale * 197 / 512 + 948
    }

    /// Evaluate the current position of the game tree, from the point of
    /// view of the side to move.
    #[inline(always)]
    fn evaluate(&mut self, sh: &SharedState, pos: &Position<V>) -> Value {
        V::evaluate(
            V::eval_file(&sh.networks).network(),
            pos,
            &mut self.accumulator_stack,
            &mut self.refresh_table,
            self.optimism[pos.side_to_move().idx()],
            self.options.use_nnue,
        )
    }

    /// The continuation correction history entry of a stack entry for a
    /// piece and square.
    #[inline(always)]
    fn continuation_correction(
        &self,
        ss: usize,
        pc: Piece,
        to: Square,
    ) -> &crate::history::StatsEntry<CORRECTION_HISTORY_LIMIT> {
        let index = self.stack[ss].continuation_correction_history as usize;
        &self.hist.continuation_correction_history[index & (PIECE_NB * SQUARE_NB - 1)][pc.idx()]
            [to.idx()]
    }

    #[inline(always)]
    fn correction_value(&self, sh: &SharedState, pos: &Position<V>, ss: usize) -> i32 {
        let us = pos.side_to_move().idx();
        let m = self.stack[ss - 1].current_move;
        let shared = &sh.histories;
        let pcv = shared.pawn_correction_entry(pos)[us].pawn.get();
        let micv = shared.minor_piece_correction_entry(pos)[us].minor.get();
        let wnpcv = shared.nonpawn_correction_entry(pos, WHITE)[us]
            .non_pawn_white
            .get();
        let bnpcv = shared.nonpawn_correction_entry(pos, BLACK)[us]
            .non_pawn_black
            .get();
        let cntcv = if m.is_ok() {
            let to = m.to_sq();
            let pc = pos.piece_on(to);
            7885 * (self.continuation_correction(ss - 2, pc, to).get()
                + self.continuation_correction(ss - 4, pc, to).get())
                + 6307 * self.continuation_correction(ss - 6, pc, to).get()
        } else {
            80695
        };

        13806 * pcv + 9512 * micv + 11615 * (wnpcv + bnpcv) + cntcv
    }

    fn update_correction_history(
        &mut self,
        sh: &SharedState,
        pos: &Position<V>,
        ss: usize,
        bonus: i32,
    ) {
        let m = self.stack[ss - 1].current_move;
        let us = pos.side_to_move().idx();

        const NON_PAWN_WEIGHT: i32 = 186;
        let shared = &sh.histories;

        shared.pawn_correction_entry(pos)[us].pawn.update(bonus);
        shared.minor_piece_correction_entry(pos)[us]
            .minor
            .update(bonus * 150 / 128);
        shared.nonpawn_correction_entry(pos, WHITE)[us]
            .non_pawn_white
            .update(bonus * NON_PAWN_WEIGHT / 128);
        shared.nonpawn_correction_entry(pos, BLACK)[us]
            .non_pawn_black
            .update(bonus * NON_PAWN_WEIGHT / 128);

        if m.is_ok() {
            let to = m.to_sq();
            let pc = pos.piece_on(to);
            for (i, weight) in [(2, 130), (4, 70), (6, 35)] {
                let index = self.stack[ss - i].continuation_correction_history as usize;
                self.hist.continuation_correction_history[index & (PIECE_NB * SQUARE_NB - 1)]
                    [pc.idx()][to.idx()]
                .update(bonus * weight / 128);
            }
        }
    }

    /// Detect shuffling moves in order to limit search explosions. This code
    /// was added in #6447 as non-regression, and so its parameters should
    /// not be tuned.
    fn is_shuffling(&self, mv: Move, ss: usize, pos: &Position<V>) -> bool {
        if pos.capture_stage(mv) || pos.rule50_count() < 10 {
            return false;
        }
        if pos.state().plies_from_null < 6 || ply_of(ss) < 20 {
            return false;
        }
        mv.from_sq() == self.stack[ss - 2].current_move.to_sq()
            && self.stack[ss - 2].current_move.from_sq() == self.stack[ss - 4].current_move.to_sq()
    }

    /// Updates stats at the end of search() when a bestMove is found
    #[allow(clippy::too_many_arguments)]
    fn update_all_stats(
        &mut self,
        sh: &SharedState,
        pos: &Position<V>,
        ss: usize,
        best_move: Move,
        prev_sq: Square,
        quiets_searched: &SearchedList,
        captures_searched: &SearchedList,
        depth: Depth,
        tt_move: Move,
        pv_node: bool,
    ) {
        let moved_piece = pos.moved_piece(best_move);

        let mut bonus = (133 * depth - 81).min(1487)
            + 364 * (best_move == tt_move) as i32
            + self.stack[ss - 1].stat_score / 28;
        let malus = (968 * depth - 235).min(2244);

        if !pv_node {
            // Stockfish multiplies in unsigned 64-bit arithmetic, which
            // amounts to rounding the quotient down.
            let searched = (quiets_searched.len() + captures_searched.len()) as i64;
            bonus += (i64::from(bonus) * searched).div_euclid(256) as i32;
        }

        if !pos.capture_stage(best_move) {
            self.update_quiet_histories(sh, pos, ss, best_move, bonus * 899 / 1024);

            // Decrease stats for all non-best quiet moves
            let mut actual_malus = malus * 1159 / 1024;
            for &mv in quiets_searched {
                actual_malus = actual_malus * 921 / 1024;
                self.update_quiet_histories(sh, pos, ss, mv, -actual_malus);
            }
        } else {
            // Increase stats for the best move in case it was a capture move
            let captured_piece = pos.piece_on(best_move.to_sq()).type_of();
            self.hist.capture_history[moved_piece.idx()][best_move.to_sq().idx()]
                [captured_piece.idx()]
            .update(bonus * 1427 / 1024);
        }

        // Extra penalty for a quiet early move that was not a TT move in
        // previous ply when it gets refuted.
        if prev_sq != Square::NONE
            && self.stack[ss - 1].move_count == 1 + self.stack[ss - 1].tt_hit as i32
            && pos.captured_piece().is_none()
        {
            self.update_continuation_histories(
                sh,
                ss - 1,
                pos.piece_on(prev_sq),
                prev_sq,
                -malus * 713 / 1024,
            );
        }

        // Decrease stats for all non-best capture moves
        for &mv in captures_searched {
            let moved_piece = pos.moved_piece(mv);
            let captured_piece = pos.piece_on(mv.to_sq()).type_of();
            self.hist.capture_history[moved_piece.idx()][mv.to_sq().idx()][captured_piece.idx()]
                .update(-malus * 1489 / 1024);
        }
    }

    /// Updates the continuation histories for the move pairs formed by the
    /// current move and the moves played in previous plies.
    fn update_continuation_histories(
        &self,
        sh: &SharedState,
        ss: usize,
        pc: Piece,
        to: Square,
        bonus: i32,
    ) {
        const CONTHIST_BONUSES: [(usize, i32); 5] =
            [(1, 520), (2, 390), (3, 145), (4, 251), (6, 209)];

        // Multipliers for positive history consistency
        const CMHC_MULTIPLIERS: [i32; 6] = [94, 103, 110, 106, 119, 126];
        let mut positive_count = 0;

        for (i, weight) in CONTHIST_BONUSES {
            // Only update the first 2 continuation histories if we are in
            // check
            if self.stack[ss].in_check && i > 2 {
                break;
            }

            if self.stack[ss - i].current_move.is_ok() {
                let history_entry = &sh
                    .histories
                    .continuation(self.stack[ss - i].continuation_history)[pc.idx()][to.idx()];
                if history_entry.get() > 0 {
                    positive_count += 1;
                }

                let multiplier = CMHC_MULTIPLIERS[positive_count];
                history_entry.update(
                    bonus.wrapping_mul(weight).wrapping_mul(multiplier) / 65536
                        + 73 * (i < 2) as i32,
                );
            }
        }
    }

    /// Updates move sorting heuristics
    fn update_quiet_histories(
        &mut self,
        sh: &SharedState,
        pos: &Position<V>,
        ss: usize,
        mv: Move,
        bonus: i32,
    ) {
        let us = pos.side_to_move();
        // Untuned to prevent duplicate effort
        self.hist.main_history[us.idx()][mv.raw() as usize].update(bonus);

        let ply = ply_of(ss);
        if (ply as usize) < LOW_PLY_HISTORY_SIZE {
            self.hist.low_ply_history[ply as usize][mv.raw() as usize].update(bonus * 712 / 1024);
        }

        self.update_continuation_histories(
            sh,
            ss,
            pos.moved_piece(mv),
            mv.to_sq(),
            bonus * 750 / 1024,
        );

        sh.histories.pawn_entry(pos)[pos.moved_piece(mv).idx()][mv.to_sq().idx()]
            .update(bonus * if bonus > -4 { 1104 } else { 459 } / 1024);
    }
}

impl Worker<Chess> {
    /// Searches the root to `depth` in the window (alpha, beta).
    pub(crate) fn chess_search_root(
        &mut self,
        sh: &SharedState,
        pos: &mut Position<Chess>,
        alpha: Value,
        beta: Value,
        depth: Depth,
    ) -> Value {
        self.search::<ROOT>(sh, pos, ROOT_SS, alpha, beta, depth, false)
    }
}

/// Reports a search's result from the thread picked to play it: its
/// principal variations unless they were just sent, and its best move.
fn report_best<V: Variant>(
    manager: &mut SearchManager<V>,
    sh: &SharedState,
    root_pos: &mut Position<V>,
    scratch: &mut V::Dirties,
    best: BestThread,
    best_is_main: bool,
    mut uci_pv_sent: bool,
) {
    manager.best_previous_score = best.root_moves[0].score;
    manager.best_previous_average_score = best.root_moves[0].average_score;

    if best.root_moves[0].pv.len() == 1
        && best.root_moves[0].extract_ponder_from_tt(&sh.tt, root_pos, scratch)
    {
        uci_pv_sent = false;
    }

    // Send PV info if it has changed since last output in
    // iterative_deepening(). The thread's root position is the same as the
    // main thread's.
    if !uci_pv_sent || !best_is_main {
        manager.output_pv(
            sh,
            best.root_moves,
            root_pos,
            best.options,
            best.limits,
            best.tb_config,
            best.root_depth,
        );
    }

    // In rare cases, output_pv() may change the ponder move through
    // syzygy_extend_pv()
    let chess960 = root_pos.is_chess960();
    let ponder = if best.root_moves[0].pv.len() > 1 {
        move_to_uci(best.root_moves[0].pv[1], chess960)
    } else {
        String::new()
    };

    let bestmove = move_to_uci(best.root_moves[0].pv[0], chess960);
    (manager.updates.on_bestmove)(&bestmove, &ponder);
}
