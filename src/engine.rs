//! The engine that the UCI loop drives: its options, networks and
//! transposition table, and the position and search threads of the variant
//! it plays.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `engine.h` and `engine.cpp`: Stockfish, a UCI chess playing engine
//! derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.
//!
//! varifish has no NUMA configuration nor debug log, so it has none of their
//! options.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

use crate::evaluate::EVAL_FILE_DEFAULT_NAME;
use crate::misc::{IStream, get_hardware_concurrency};
use crate::nnue::eval_file::{Networks, UseNnue};
use crate::options::{OptionsMap, UciOption};
use crate::position::{Position, PositionSetError, START_FEN};
use crate::search::{
    LimitsType, SearchOptions, SharedState, Skill, UpdateBestmove, UpdateContext, UpdateFull,
    UpdateInfoString, UpdateIter, UpdateShort, UpdateStart,
};
use crate::syzygy::{ProbeState, SyzygyOptions};
use crate::thread::ThreadPool;
use crate::types::*;
use crate::uci::to_move;
use crate::variant::{Atomic, Chess, Variant, VariantKind};

/// The largest transposition table, in MB
pub const MAX_HASH_MB: i32 = if cfg!(target_pointer_width = "64") {
    33554432
} else {
    2048
};

/// A command that failed, after which the engine exits.
#[derive(Debug)]
pub struct CriticalError;

/// What the engine keeps of the variant it plays: the position, and the
/// threads that search it.
struct Game<V: Variant> {
    pos: Position<V>,
    threads: ThreadPool<V>,
}

impl<V: Variant> Game<V> {
    /// The starting position, without threads.
    fn new(chess960: bool) -> Game<V> {
        Game {
            pos: Position::from_fen(START_FEN, chess960).expect("the starting position is valid"),
            threads: ThreadPool::default(),
        }
    }

    fn set_position(
        &mut self,
        fen: &str,
        chess960: bool,
        moves: &[&str],
    ) -> Result<(), PositionSetError> {
        self.pos.set(fen, chess960)?;

        let mut dirties = V::Dirties::default();
        for &mv in moves {
            let m = to_move(&self.pos, mv);

            if m == Move::none() {
                return Err(PositionSetError(format!("Illegal move: {mv}")));
            }

            self.pos.make_move(m, &mut dirties);
        }
        Ok(())
    }
}

/// The game of the variant that UCI_Variant names.
enum AnyGame {
    Chess(Game<Chess>),
    Atomic(Game<Atomic>),
}

/// Evaluates `$body` with `$g` bound to the game, whatever its variant, so
/// that the body compiles for each variant.
macro_rules! with_game {
    ($game:expr, $g:ident => $body:expr) => {
        match $game {
            AnyGame::Chess($g) => $body,
            AnyGame::Atomic($g) => $body,
        }
    };
}

impl AnyGame {
    fn new(variant: VariantKind, chess960: bool) -> AnyGame {
        match variant {
            VariantKind::Chess => AnyGame::Chess(Game::new(chess960)),
            VariantKind::Atomic => AnyGame::Atomic(Game::new(chess960)),
        }
    }

    fn variant(&self) -> VariantKind {
        match self {
            AnyGame::Chess(_) => VariantKind::Chess,
            AnyGame::Atomic(_) => VariantKind::Atomic,
        }
    }
}

pub struct Engine {
    binary_directory: PathBuf,
    embedded_network: &'static [u8],

    game: AnyGame,
    options: OptionsMap,
    shared: Arc<SharedState>,

    updates: UpdateContext,
    on_verify_network: UpdateInfoString,
}

impl Engine {
    /// An engine playing chess, with the default chess network embedded in
    /// the binary, `embedded_network`, or none if it is empty. Network files
    /// are looked for in the working directory and in `binary_directory`.
    pub fn new(binary_directory: PathBuf, embedded_network: &'static [u8]) -> Engine {
        let max_threads = 1024.max(4 * get_hardware_concurrency() as i32);

        let mut options = OptionsMap::default();
        options.add("Threads", UciOption::spin(1, 1, max_threads));
        options.add("Hash", UciOption::spin(16, 1, MAX_HASH_MB));
        options.add("Clear Hash", UciOption::button());
        options.add("Ponder", UciOption::check(false));
        options.add("MultiPV", UciOption::spin(1, 1, MAX_MOVES as i32));
        options.add("Skill Level", UciOption::spin(20, 0, 20));
        options.add("Move Overhead", UciOption::spin(10, 0, 5000));
        options.add("nodestime", UciOption::spin(0, 0, 10000));
        options.add("UCI_Chess960", UciOption::check(false));
        options.add(
            "UCI_Variant",
            UciOption::combo(
                VariantKind::default().name(),
                &VariantKind::ALL.map(VariantKind::name),
            ),
        );
        options.add("UCI_LimitStrength", UciOption::check(false));
        options.add(
            "UCI_Elo",
            UciOption::spin(Skill::LOWEST_ELO, Skill::LOWEST_ELO, Skill::HIGHEST_ELO),
        );
        options.add("UCI_ShowWDL", UciOption::check(false));
        options.add("SyzygyPath", UciOption::string(""));
        options.add("SyzygyProbeDepth", UciOption::spin(1, 1, 100));
        options.add("Syzygy50MoveRule", UciOption::check(true));
        options.add("SyzygyProbeLimit", UciOption::spin(7, 0, 7));
        options.add(
            "Use NNUE",
            UciOption::combo(UseNnue::default().name(), &UseNnue::ALL.map(UseNnue::name)),
        );
        options.add("EvalFile", UciOption::string(EVAL_FILE_DEFAULT_NAME));
        #[cfg(feature = "tune")]
        for tunable in crate::tune::ALL {
            options.add(
                tunable.name,
                UciOption::spin(tunable.default, tunable.min, tunable.max),
            );
        }

        let mut networks = Networks::default();
        networks.load(&binary_directory, Path::new(""), embedded_network);

        let mut engine = Engine {
            binary_directory,
            embedded_network,
            game: AnyGame::new(VariantKind::default(), false),
            options,
            shared: Arc::new(SharedState::new(networks)),
            updates: UpdateContext::default(),
            on_verify_network: Arc::new(|_| {}),
        };

        engine.resize_threads();
        engine
    }

    /// The variant played
    pub fn variant(&self) -> VariantKind {
        self.game.variant()
    }

    pub fn go(&mut self, limits: LimitsType) -> Result<(), CriticalError> {
        debug_assert!(limits.perft == 0);
        self.verify_network()?;

        let options = self.search_options();
        with_game!(&self.game, g => {
            g.threads
                .start_thinking(&self.shared, &g.pos, limits, options, &self.updates);
        });
        Ok(())
    }

    /// The options that a search reads, which do not change while it runs
    fn search_options(&self) -> SearchOptions {
        let o = &self.options;
        SearchOptions {
            multi_pv: o.get("MultiPV").as_int() as usize,
            skill_level: o.get("Skill Level").as_int(),
            limit_strength: o.get("UCI_LimitStrength").as_bool(),
            uci_elo: o.get("UCI_Elo").as_int(),
            show_wdl: o.get("UCI_ShowWDL").as_bool(),
            move_overhead: o.get("Move Overhead").as_int(),
            nodestime: o.get("nodestime").as_int(),
            ponder: o.get("Ponder").as_bool(),
            use_nnue: self.use_nnue(),
            syzygy: SyzygyOptions {
                probe_depth: o.get("SyzygyProbeDepth").as_int(),
                rule50: o.get("Syzygy50MoveRule").as_bool(),
                probe_limit: o.get("SyzygyProbeLimit").as_int(),
            },
        }
    }

    /// How Atomic is evaluated
    fn use_nnue(&self) -> UseNnue {
        UseNnue::from_name(self.options.get("Use NNUE").as_str()).unwrap_or_default()
    }

    pub fn stop(&self) {
        self.shared.stop.store(true, Relaxed);
    }

    pub fn search_clear(&mut self) {
        self.wait_for_search_finished();

        let threads = self.threads();
        self.shared_mut().tt.clear(threads);
        with_game!(&self.game, g => g.threads.clear(&self.shared));

        // Free mapped files
        self.init_tablebases();
    }

    /// Finds the tables of the variant played in SyzygyPath's directories,
    /// dropping those mapped before, and tells what it found.
    fn init_tablebases(&mut self) {
        let paths = self.options.get("SyzygyPath").as_str().to_string();
        let variant = self.variant();

        if let Some(info) = self.shared_mut().tablebases.init(variant, &paths) {
            (self.updates.on_info_string)(&info);
        }
    }

    pub fn set_on_update_no_moves(&mut self, f: UpdateShort) {
        self.updates.on_update_no_moves = f;
    }

    pub fn set_on_update_full(&mut self, f: UpdateFull) {
        self.updates.on_update_full = f;
    }

    pub fn set_on_iter(&mut self, f: UpdateIter) {
        self.updates.on_iter = f;
    }

    pub fn set_on_bestmove(&mut self, f: UpdateBestmove) {
        self.updates.on_bestmove = f;
    }

    pub fn set_on_start(&mut self, f: UpdateStart) {
        self.updates.on_start = f;
    }

    pub fn set_on_info_string(&mut self, f: UpdateInfoString) {
        self.updates.on_info_string = f;
    }

    pub fn set_on_verify_network(&mut self, f: UpdateInfoString) {
        self.on_verify_network = f;
    }

    pub fn wait_for_search_finished(&self) {
        with_game!(&self.game, g => g.threads.wait_for_search_finished());
    }

    pub fn set_position(&mut self, fen: &str, moves: &[&str]) -> Result<(), PositionSetError> {
        let chess960 = self.options.get("UCI_Chess960").as_bool();
        with_game!(&mut self.game, g => g.set_position(fen, chess960, moves))
    }

    /// Applies `setoption` read from `is`, after the search finished: Ok
    /// with what to tell about the change, if anything, or Err with the
    /// message for an unknown option.
    pub fn setoption(&mut self, is: &mut IStream) -> Result<Option<String>, String> {
        self.wait_for_search_finished();

        let Some(name) = self.options.setoption(is)? else {
            return Ok(None);
        };

        Ok(match name {
            "Threads" => {
                self.resize_threads();
                Some(self.thread_allocation_information_as_string())
            }
            "Hash" => {
                let mb = self.options.get("Hash").as_int() as usize;
                self.set_tt_size(mb);
                None
            }
            "Clear Hash" => {
                self.search_clear();
                None
            }
            "UCI_Variant" => {
                let variant = VariantKind::from_name(self.options.get("UCI_Variant").as_str())
                    .expect("the option names a variant");
                self.set_variant(variant);
                None
            }
            "SyzygyPath" => {
                self.init_tablebases();
                None
            }
            "EvalFile" => {
                let file = PathBuf::from(self.options.get("EvalFile").as_str());
                self.load_network(&file);
                None
            }
            _ => {
                #[cfg(feature = "tune")]
                if let Some(tunable) = crate::tune::find(name) {
                    tunable.set(self.options.get(tunable.name).as_int());
                }
                None
            }
        })
    }

    /// Plays `variant`, from its starting position, with new threads, an
    /// empty transposition table and its own tables, unless it is the
    /// variant played.
    fn set_variant(&mut self, variant: VariantKind) {
        if variant == self.variant() {
            return;
        }

        // The threads of the variant played end before the new ones start.
        let chess960 = self.options.get("UCI_Chess960").as_bool();
        self.game = AnyGame::new(variant, chess960);
        self.resize_threads();
        self.init_tablebases();
    }

    /// The shared state, which no search may be using.
    fn shared_mut(&mut self) -> &mut SharedState {
        Arc::get_mut(&mut self.shared).expect("no search is running")
    }

    /// The number of search threads
    fn threads(&self) -> usize {
        with_game!(&self.game, g => g.threads.len())
    }

    fn resize_threads(&mut self) {
        self.wait_for_search_finished();
        let requested = self.options.get("Threads").as_int() as usize;
        with_game!(&mut self.game, g => g.threads.set(&mut self.shared, requested));

        // Reallocate the hash with the new threadpool size
        let mb = self.options.get("Hash").as_int() as usize;
        self.set_tt_size(mb);
    }

    fn set_tt_size(&mut self, mb: usize) {
        self.wait_for_search_finished();
        let threads = self.threads();
        self.shared_mut().tt.resize(mb, threads);
    }

    pub fn set_ponderhit(&self, b: bool) {
        self.shared.ponder.store(b, Relaxed);
    }

    /// Reports the network that evaluates the variant played, or fails if
    /// it is not loaded, or not the one EvalFile names, after which the
    /// engine exits.
    pub fn verify_network(&self) -> Result<(), CriticalError> {
        with_game!(&self.game, g => self.verify_network_of(g))
    }

    fn verify_network_of<V: Variant>(&self, _game: &Game<V>) -> Result<(), CriticalError> {
        let file = PathBuf::from(self.options.get("EvalFile").as_str());
        let networks = &self.shared.networks;
        match networks.verify(V::eval_file(networks), &file) {
            Ok(info) => {
                (self.on_verify_network)(&info);
                // The network is not shared between processes.
                (self.on_verify_network)("Network replica 1: Local memory.");
                Ok(())
            }
            Err(message) => {
                (self.on_verify_network)(&message);
                Err(CriticalError)
            }
        }
    }

    /// Replaces the networks, as if loaded.
    #[cfg(test)]
    pub fn set_networks(&mut self, networks: Networks) {
        self.wait_for_search_finished();
        self.shared_mut().networks = networks;
        with_game!(&self.game, g => g.threads.clear(&self.shared));
    }

    fn load_network(&mut self, file: &Path) {
        let (directory, embedded) = (self.binary_directory.clone(), self.embedded_network);
        self.shared_mut().networks.load(&directory, file, embedded);
        with_game!(&self.game, g => g.threads.clear(&self.shared));
    }

    /// Writes the network of the variant played to `file`, and returns what
    /// to tell the user.
    pub fn save_network(&self, file: Option<&Path>) -> String {
        with_game!(&self.game, g => self.save_network_of(g, file))
    }

    fn save_network_of<V: Variant>(&self, _game: &Game<V>, file: Option<&Path>) -> String {
        V::eval_file(&self.shared.networks).save(file)
    }

    /// The trace of the evaluation of the position.
    pub fn trace_eval(&self) -> Result<String, CriticalError> {
        with_game!(&self.game, g => self.trace_eval_of(g))
    }

    fn trace_eval_of<V: Variant>(&self, game: &Game<V>) -> Result<String, CriticalError> {
        let chess960 = self.options.get("UCI_Chess960").as_bool();
        let p = Position::<V>::from_fen(&game.pos.fen(), chess960)
            .expect("the position's FEN sets it again");

        self.verify_network()?;

        let network = V::eval_file(&self.shared.networks).network();
        Ok(format!("\n{}", V::trace(network, &p, self.use_nnue())))
    }

    pub fn options(&self) -> &OptionsMap {
        &self.options
    }

    pub fn fen(&self) -> String {
        with_game!(&self.game, g => g.pos.fen())
    }

    pub fn flip(&mut self) -> Result<(), PositionSetError> {
        with_game!(&mut self.game, g => g.pos.flip())
    }

    /// The board, and its tablebase scores if they have its material.
    pub fn visualize(&self) -> String {
        with_game!(&self.game, g => self.visualize_of(g))
    }

    fn visualize_of<V: Variant>(&self, game: &Game<V>) -> String {
        let pos = &game.pos;
        let mut s = pos.to_string();
        let tables = V::tables(&self.shared.tablebases);

        if tables.max_cardinality() >= pos.count_both(ALL_PIECES) && !pos.can_castle(ANY_CASTLING) {
            let mut p = Position::<V>::from_fen(&pos.fen(), pos.is_chess960())
                .expect("the position's FEN sets it again");
            let (mut s1, mut s2) = (ProbeState::Ok, ProbeState::Ok);
            let wdl = tables.probe_wdl(&mut p, &mut s1);
            let dtz = tables.probe_dtz(&mut p, &mut s2);
            s += &format!(
                "\nTablebases WDL: {wdl:>4} ({})\nTablebases DTZ: {dtz:>4} ({})",
                s1 as i32, s2 as i32
            );
        }
        s
    }

    pub fn get_hashfull(&self, max_age: i32) -> i32 {
        self.shared.tt.hashfull(max_age)
    }

    pub fn numa_config_information_as_string(&self) -> String {
        format!("Available processors: {}", available_processors())
    }

    pub fn thread_allocation_information_as_string(&self) -> String {
        let threads = self.threads();
        format!(
            "Using {threads} {}",
            if threads > 1 { "threads" } else { "thread" }
        )
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.wait_for_search_finished();
    }
}

/// The processors the engine may run on, written as Stockfish writes those
/// of a NUMA node: ranges of consecutive indices, separated by commas.
pub(crate) fn available_processors() -> String {
    let cpus = allowed_cpus().unwrap_or_else(|| (0..get_hardware_concurrency()).collect());

    let mut ranges: Vec<String> = Vec::new();
    let mut i = 0;
    while i < cpus.len() {
        let first = cpus[i];
        while i + 1 < cpus.len() && cpus[i + 1] == cpus[i] + 1 {
            i += 1;
        }
        let last = cpus[i];
        ranges.push(if last != first {
            format!("{first}-{last}")
        } else {
            last.to_string()
        });
        i += 1;
    }
    ranges.join(",")
}

/// The processors this process may run on, in increasing order.
#[cfg(target_os = "linux")]
fn allowed_cpus() -> Option<Vec<usize>> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?
        .trim();

    let mut cpus = Vec::new();
    for part in list.split(',') {
        match part.split_once('-') {
            Some((first, last)) => cpus.extend(first.parse::<usize>().ok()?..=last.parse().ok()?),
            None => cpus.push(part.parse().ok()?),
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    (!cpus.is_empty()).then_some(cpus)
}

#[cfg(not(target_os = "linux"))]
fn allowed_cpus() -> Option<Vec<usize>> {
    None
}
