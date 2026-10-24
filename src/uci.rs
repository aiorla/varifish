//! The UCI loop: the commands varifish reads and how it answers them.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `uci.h`, `uci.cpp` and parts of `misc.cpp`: Stockfish, a UCI chess playing
//! engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};

use crate::benchmark::setup_bench;
use crate::engine::{CriticalError, Engine};
use crate::misc::{IStream, is_space, mul_add, now};
use crate::movegen::{ALL, MoveList};
use crate::perft::perft;
use crate::position::{Position, START_FEN, square_to_uci};
use crate::score::Score;
use crate::search::{InfoFull, InfoIteration, InfoShort, LimitsType};
use crate::thread::lock;
use crate::types::*;
use crate::variant::{Atomic, Chess, Variant, VariantKind};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The engine's name, version and authors, for `uci` and the greeting.
pub fn engine_info(to_uci: bool) -> String {
    format!(
        "varifish {VERSION}{}the varifish and Stockfish developers (see AUTHORS)",
        if to_uci { "\nid author " } else { " by " }
    )
}

const HELP: &str = "\nvarifish is a chess engine for chess and its variants, translated from Stockfish.\
\nIt is released as free software licensed under the GNU GPLv3 License.\
\nvarifish is normally used with a graphical user interface (GUI) and implements\
\nthe Universal Chess Interface (UCI) protocol to communicate with a GUI, an API, etc.\
\nFor any further information, read the README.md and LICENSE files distributed along with this program.\n";

/// Where the engine writes, which the search threads share with the loop.
pub type Output = Arc<Mutex<dyn Write + Send>>;

/// Writes a message and a newline, flushed so the GUI sees it at once.
fn sync_println(out: &Output, message: &str) {
    let mut out = lock(out);
    // A GUI that stops reading also closes the input, which ends the loop.
    let _ = writeln!(out, "{message}");
    let _ = out.flush();
}

/// Writes each line of `s` that is not blank as an info string.
fn print_info_string(out: &Output, s: &str) {
    let mut out = lock(out);
    for line in s.split('\n') {
        if !line.bytes().all(is_space) {
            let _ = writeln!(out, "info string {line}");
        }
    }
    let _ = out.flush();
}

pub struct UciEngine {
    engine: Engine,
    out: Output,
    err: Output,
    current_cmd: String,
}

impl UciEngine {
    /// An engine answering on `out`, and reporting the bench on `err`, with
    /// the default network embedded in the binary, `embedded_network`, or
    /// none if it is empty. Network files are looked for in the working
    /// directory and in `binary_directory`.
    pub fn new(
        out: Output,
        err: Output,
        binary_directory: PathBuf,
        embedded_network: &'static [u8],
    ) -> UciEngine {
        let mut uci = UciEngine {
            engine: Engine::new(binary_directory, embedded_network),
            out,
            err,
            current_cmd: String::new(),
        };
        uci.init_search_update_listeners();
        uci
    }

    fn init_search_update_listeners(&mut self) {
        let out = Arc::clone(&self.out);
        self.engine
            .set_on_iter(Arc::new(move |i: &InfoIteration| on_iter(&out, i)));
        let out = Arc::clone(&self.out);
        self.engine
            .set_on_update_no_moves(Arc::new(move |i: &InfoShort| on_update_no_moves(&out, i)));
        let out = Arc::clone(&self.out);
        self.engine
            .set_on_update_full(Arc::new(move |i: &InfoFull| on_update_full(&out, i)));
        self.engine.set_on_start(Arc::new(|| {}));
        let out = Arc::clone(&self.out);
        self.engine
            .set_on_bestmove(Arc::new(move |bm: &str, p: &str| on_bestmove(&out, bm, p)));
        let out = Arc::clone(&self.out);
        self.engine
            .set_on_verify_network(Arc::new(move |s: &str| print_info_string(&out, s)));
        let out = Arc::clone(&self.out);
        self.engine
            .set_on_info_string(Arc::new(move |s: &str| print_info_string(&out, s)));
    }

    /// Runs the command that `args` make, if any, or else the commands read
    /// from `input` until "quit" or its end, and returns the exit code once
    /// the search, if any, has finished.
    pub fn run(&mut self, args: &[String], mut input: impl BufRead) -> i32 {
        // The command-line arguments are one-shot
        let one_shot = !args.is_empty();
        let mut cmd: String = args.iter().map(|arg| format!("{arg} ")).collect();

        loop {
            if !one_shot {
                // Wait for an input or an end-of-file (EOF) indication
                cmd = read_line(&mut input).unwrap_or_else(|| "quit".to_string());
            }
            match self.execute(&cmd) {
                Err(CriticalError) => return 1,
                Ok(false) => break,
                Ok(true) if one_shot => break,
                Ok(true) => {}
            }
        }

        self.engine.wait_for_search_finished();
        0
    }

    /// Executes one command: Ok(false) after "quit".
    pub fn execute(&mut self, cmd: &str) -> Result<bool, CriticalError> {
        self.current_cmd = cmd.to_string();
        let mut is = IStream::new(cmd);
        let token = is.token().unwrap_or("");

        match token {
            "quit" => {
                self.engine.stop();
                return Ok(false);
            }
            "stop" => self.engine.stop(),

            // The GUI sends 'ponderhit' to tell that the user has played
            // the expected move. So, 'ponderhit' is sent if pondering was
            // done on the same move that the user has played. The search
            // should continue, but should also switch from pondering to the
            // normal search.
            "ponderhit" => self.engine.set_ponderhit(false),

            "uci" => {
                let info = format!("id name {}\n{}", engine_info(true), self.engine.options());
                self.send(&info);
                self.send("uciok");
            }
            "setoption" => self.setoption(&mut is),
            "go" => {
                // Send info strings after the go command is sent for old
                // GUIs and python-chess
                print_info_string(&self.out, &self.engine.numa_config_information_as_string());
                print_info_string(
                    &self.out,
                    &self.engine.thread_allocation_information_as_string(),
                );
                self.go(&mut is)?;
            }
            "position" => self.position(&mut is)?,
            "ucinewgame" => self.engine.search_clear(),
            "isready" => self.send("readyok"),

            // Add custom non-UCI commands, mainly for debugging purposes.
            "flip" => {
                if let Err(err) = self.engine.flip() {
                    return Err(self.critical_error(&err.0));
                }
            }
            "bench" => self.bench(&mut is)?,
            "d" => {
                let board = self.engine.visualize();
                self.send(&board);
            }
            "eval" => self.trace_eval()?,
            "export_net" => {
                let file = is.token().map(PathBuf::from);
                let message = self.engine.save_network(file.as_deref());
                self.send(&message);
            }
            "--help" | "help" | "--license" | "license" => self.send(HELP),
            _ if !token.is_empty() && !token.starts_with('#') => {
                self.send(&format!(
                    "Unknown command: '{cmd}'. Type help for more information."
                ));
            }
            _ => {}
        }
        Ok(true)
    }

    fn send(&self, message: &str) {
        sync_println(&self.out, message);
    }

    fn trace_eval(&mut self) -> Result<(), CriticalError> {
        let trace = self.engine.trace_eval()?;
        self.send(&trace);
        Ok(())
    }

    fn critical_error(&self, message: &str) -> CriticalError {
        let text = format!(
            "info string CRITICAL ERROR: Command `{}` failed. Reason: {message}\n",
            self.current_cmd
        );
        self.send(&text);
        CriticalError
    }

    fn parse_limits(&self, is: &mut IStream) -> Result<LimitsType, CriticalError> {
        let mut limits = LimitsType {
            start_time: now(), // The search starts as early as possible
            ..LimitsType::default()
        };

        while let Some(token) = is.token() {
            match token {
                // Needs to be the last command on the line
                "searchmoves" => {
                    while let Some(m) = is.token() {
                        limits.searchmoves.push(m.to_ascii_lowercase());
                    }
                    break;
                }
                "wtime" => read_into(is.read_i64(), &mut limits.time[WHITE.idx()]),
                "btime" => read_into(is.read_i64(), &mut limits.time[BLACK.idx()]),
                "winc" => read_into(is.read_i64(), &mut limits.inc[WHITE.idx()]),
                "binc" => read_into(is.read_i64(), &mut limits.inc[BLACK.idx()]),
                "movestogo" => read_into(is.read_i32(), &mut limits.movestogo),
                "depth" => read_into(is.read_i32(), &mut limits.depth),
                "nodes" => read_into(is.read_u64(), &mut limits.nodes),
                "movetime" => read_into(is.read_i64(), &mut limits.movetime),
                "mate" => read_into(is.read_i32(), &mut limits.mate),
                "perft" => read_into(is.read_i32(), &mut limits.perft),
                "infinite" => limits.infinite = 1,
                "ponder" => limits.ponder_mode = true,
                _ => {}
            }

            if is.fail() {
                return Err(self.critical_error(&format!("Invalid argument for '{token}'")));
            }
        }

        Ok(limits)
    }

    fn go(&mut self, is: &mut IStream) -> Result<(), CriticalError> {
        let limits = self.parse_limits(is)?;

        if limits.perft != 0 {
            self.perft(&limits)?;
        } else {
            self.engine.go(limits)?;
        }
        Ok(())
    }

    fn bench(&mut self, args: &mut IStream) -> Result<(), CriticalError> {
        let mut nodes: u64 = 0;
        let mut cnt: u64 = 1;
        let nodes_searched = Arc::new(AtomicU64::new(0));

        {
            let out = Arc::clone(&self.out);
            let nodes_searched = Arc::clone(&nodes_searched);
            self.engine
                .set_on_update_full(Arc::new(move |i: &InfoFull| {
                    nodes_searched.store(i.nodes as u64, Relaxed);
                    on_update_full(&out, i);
                }));
        }

        let defaults = self.engine.variant().bench();
        let list = match setup_bench(defaults, &self.engine.fen(), args) {
            Ok(list) => list,
            Err(message) => {
                sync_println(&self.err, &message);
                return Err(CriticalError);
            }
        };

        let num = list
            .iter()
            .filter(|s| s.starts_with("go ") || s.starts_with("eval"))
            .count();

        let mut elapsed = now();

        for cmd in &list {
            let mut is = IStream::new(cmd);
            let token = is.token().unwrap_or("");

            if token == "go" || token == "eval" {
                let fen = self.engine.fen();
                sync_println(&self.err, &format!("\nPosition: {cnt}/{num} ({fen})"));
                cnt += 1;
                if token == "go" {
                    let limits = self.parse_limits(&mut is)?;

                    if limits.perft != 0 {
                        nodes_searched.store(self.perft(&limits)?, Relaxed);
                    } else {
                        self.engine.go(limits)?;
                        self.engine.wait_for_search_finished();
                    }

                    nodes += nodes_searched.swap(0, Relaxed);
                } else {
                    self.trace_eval()?;
                }
            } else if token == "setoption" {
                self.setoption(&mut is);
            } else if token == "position" {
                self.position(&mut is)?;
            } else if token == "ucinewgame" {
                self.engine.search_clear(); // search_clear may take a while
                elapsed = now();
            }
        }

        // Ensure positivity to avoid a 'divide by zero'
        let elapsed = (now() - elapsed + 1) as u64;

        sync_println(
            &self.err,
            &format!(
                "\n===========================\
                 \nTotal time (ms) : {elapsed}\
                 \nNodes searched  : {nodes}\
                 \nNodes/second    : {}",
                1000 * nodes / elapsed
            ),
        );

        // Reset the callback, which reports nodes to this bench
        let out = Arc::clone(&self.out);
        self.engine
            .set_on_update_full(Arc::new(move |i: &InfoFull| on_update_full(&out, i)));
        Ok(())
    }

    fn setoption(&mut self, is: &mut IStream) {
        match self.engine.setoption(is) {
            Ok(Some(info)) => print_info_string(&self.out, &info),
            Ok(None) => {}
            Err(message) => self.send(&message),
        }
    }

    /// Counts the leaves of the tree of legal moves of the variant played,
    /// as Stockfish does for chess, after checking its network, and as
    /// Atomic-Stockfish does for Atomic, without a network.
    fn perft(&mut self, limits: &LimitsType) -> Result<u64, CriticalError> {
        let variant = self.engine.variant();
        if variant == VariantKind::Chess {
            self.engine.verify_network()?;
        }

        let chess960 = self.engine.options().get("UCI_Chess960").as_bool();
        let fen = self.engine.fen();
        let out = &self.out;
        let on_root_move = |m, cnt| {
            sync_println(out, &format!("{}: {cnt}", move_to_uci(m, chess960)));
        };

        let result = match variant {
            VariantKind::Chess => perft::<Chess>(&fen, limits.perft, chess960, on_root_move),
            VariantKind::Atomic => perft::<Atomic>(&fen, limits.perft, chess960, on_root_move),
        };

        match result {
            Ok(nodes) => {
                self.send(&format!("\nNodes searched: {nodes}\n"));
                Ok(nodes)
            }
            Err(err) => Err(self.critical_error(&err.0)),
        }
    }

    fn position(&mut self, is: &mut IStream) -> Result<(), CriticalError> {
        let mut fen = String::new();

        match is.token() {
            Some("startpos") => {
                fen += START_FEN;
                is.token(); // Consume the "moves" token, if any
            }
            Some("fen") => {
                while let Some(token) = is.token() {
                    if token == "moves" {
                        break;
                    }
                    fen += token;
                    fen.push(' ');
                }
            }
            _ => return Ok(()),
        }

        let moves: Vec<&str> = std::iter::from_fn(|| is.token()).collect();

        match self.engine.set_position(&fen, &moves) {
            Ok(()) => Ok(()),
            Err(err) => Err(self.critical_error(&err.0)),
        }
    }
}

/// Stores a number read from a stream, unless reading could not start, as
/// `>>` does.
fn read_into<T>(value: Option<T>, field: &mut T) {
    if let Some(value) = value {
        *field = value;
    }
}

/// The next line of `input` without its newline, or None at its end.
fn read_line(input: &mut impl BufRead) -> Option<String> {
    let mut line = Vec::new();
    match input.read_until(b'\n', &mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => {
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            Some(String::from_utf8_lossy(&line).into_owned())
        }
    }
}

/// The parameters a and b of the win rate model for the material on the
/// board.
fn win_rate_params(pos: &Position<impl Variant>) -> (f64, f64) {
    let material = pos.count_both(PAWN)
        + 3 * pos.count_both(KNIGHT)
        + 3 * pos.count_both(BISHOP)
        + 5 * pos.count_both(ROOK)
        + 9 * pos.count_both(QUEEN);

    // The fitted model only uses data for material counts in [17, 78], and
    // is anchored at count 58.
    let m = f64::from(material.clamp(17, 78)) / 58.0;

    // Return a = p_a(material) and b = p_b(material), see
    // github.com/official-stockfish/WDL_model
    const AS: [f64; 4] = [-142.72052667, 372.35176398, -340.71073572, 415.23490212];
    const BS: [f64; 4] = [5.93832785, 15.61267078, -30.57816876, 69.63866711];

    let p = |c: [f64; 4]| mul_add(mul_add(mul_add(c[0], m, c[1]), m, c[2]), m, c[3]);

    (p(AS), p(BS))
}

/// The win rate model is 1 / (1 + exp((a - eval) / b)), where a =
/// p_a(material) and b = p_b(material). It fits the LTC fishtest statistics
/// rather accurately.
fn win_rate_model(v: Value, pos: &Position<impl Variant>) -> i32 {
    let (a, b) = win_rate_params(pos);

    // Return the win rate in per mille units, rounded to the nearest
    // integer.
    (0.5 + 1000.0 / (1.0 + ((a - f64::from(v)) / b).exp())) as i32
}

/// A score as UCI reports it: "mate" and the moves to mate, or "cp" and
/// centipawns, where tablebase wins and losses are 20000 less their plies.
pub fn format_score(s: &Score) -> String {
    const TB_CP: i32 = 20000;
    match *s {
        Score::Mate { plies } => {
            let m = (if plies > 0 { plies + 1 } else { plies }) / 2;
            format!("mate {m}")
        }
        Score::Tablebase { plies, win } => {
            format!("cp {}", (if win { TB_CP } else { -TB_CP }) - plies)
        }
        Score::InternalUnits { value } => format!("cp {value}"),
    }
}

/// Turns a Value to an integer centipawn number, without treatment of mate
/// and similar special scores.
pub fn to_cp<V: Variant>(v: Value, pos: &Position<V>) -> i32 {
    if V::ATOMIC {
        // Atomic-Stockfish's pawn, Fairy-Stockfish's fixed normalization
        return (100 * i64::from(v) / i64::from(PAWN_VALUE)) as i32;
    }

    // In general, the score can be defined via the WDL as
    // (log(1/L - 1) - log(1/W - 1)) / (log(1/L - 1) + log(1/W - 1)).
    // Based on our win_rate_model, this simply yields v / a.

    let (a, _) = win_rate_params(pos);

    (f64::from(100 * v) / a).round() as i32
}

/// The win, draw and loss chances of a value, in per mille, by the win rate
/// model of chess. Other variants have none.
pub fn wdl<V: Variant>(v: Value, pos: &Position<V>) -> Option<String> {
    if V::ATOMIC {
        return None;
    }

    let wdl_w = win_rate_model(v, pos);
    let wdl_l = win_rate_model(-v, pos);
    let wdl_d = 1000 - wdl_w - wdl_l;
    Some(format!("{wdl_w} {wdl_d} {wdl_l}"))
}

/// A move in UCI notation, where castling is the king's move to its square,
/// except in Chess960, where it captures its own rook.
pub fn move_to_uci(m: Move, chess960: bool) -> String {
    if m == Move::none() {
        return "(none)".to_string();
    }

    if m == Move::null() {
        return "0000".to_string();
    }

    let from = m.from_sq();
    let mut to = m.to_sq();

    if m.type_of() == CASTLING && !chess960 {
        to = Square::new(if to > from { FILE_G } else { FILE_C }, from.rank());
    }

    let mut s = square_to_uci(from) + &square_to_uci(to);

    if m.type_of() == PROMOTION {
        s.push(b" pnbrqk"[m.promotion_type().idx()] as char);
    }

    s
}

/// The legal move written `s` in UCI notation, in any case, or
/// `Move::none()`.
pub fn to_move<V: Variant>(pos: &Position<V>, s: &str) -> Move {
    let s = s.to_ascii_lowercase();

    MoveList::new::<ALL>(pos)
        .iter()
        .copied()
        .find(|&m| move_to_uci(m, pos.is_chess960()) == s)
        .unwrap_or(Move::none())
}

fn on_update_no_moves(out: &Output, info: &InfoShort) {
    sync_println(
        out,
        &format!(
            "info depth {} score {}",
            info.depth,
            format_score(&info.score)
        ),
    );
}

fn on_update_full(out: &Output, info: &InfoFull) {
    let mut s = format!(
        "info depth {} seldepth {} multipv {} score {}",
        info.depth,
        info.sel_depth,
        info.multi_pv,
        format_score(&info.score)
    );

    if !info.bound.is_empty() {
        s += " ";
        s += info.bound;
    }

    // The search reports the WDL only when UCI_ShowWDL is set.
    if !info.wdl.is_empty() {
        s += " wdl ";
        s += info.wdl;
    }

    s += &format!(
        " nodes {} nps {} hashfull {} tbhits {} time {} pv {}",
        info.nodes, info.nps, info.hashfull, info.tb_hits, info.time_ms, info.pv
    );

    sync_println(out, &s);
}

fn on_iter(out: &Output, info: &InfoIteration) {
    sync_println(
        out,
        &format!(
            "info depth {} currmove {} currmovenumber {}",
            info.depth, info.currmove, info.currmovenumber
        ),
    );
}

fn on_bestmove(out: &Output, bestmove: &str, ponder: &str) {
    if ponder.is_empty() {
        sync_println(out, &format!("bestmove {bestmove}"));
    } else {
        sync_println(out, &format!("bestmove {bestmove} ponder {ponder}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{MAX_HASH_MB, available_processors};
    use crate::nnue::atomic::network::Network as AtomicNetwork;
    use crate::nnue::eval_file::{EvalFile, Networks};
    use crate::nnue::network::Network as ChessNetwork;

    type Buffer = Arc<Mutex<Vec<u8>>>;

    fn text(buffer: &Buffer) -> String {
        String::from_utf8(lock(buffer).clone()).unwrap()
    }

    /// An engine without a network, and its output and error output.
    fn engine() -> (UciEngine, Buffer, Buffer) {
        let out = Buffer::default();
        let err = Buffer::default();
        let engine = UciEngine::new(
            out.clone(),
            err.clone(),
            PathBuf::from("no such directory"),
            &[],
        );
        (engine, out, err)
    }

    /// An engine with networks of zeros, and its output and error output.
    fn engine_with_network() -> (UciEngine, Buffer, Buffer) {
        let (mut engine, out, err) = engine();
        engine.engine.set_networks(Networks {
            chess: EvalFile::with_network(ChessNetwork::zeroed()),
            atomic: EvalFile::with_network(AtomicNetwork::zeroed()),
        });
        (engine, out, err)
    }

    /// The exit code and the output of an engine with a network of zeros.
    fn run(input: &str) -> (i32, String) {
        let (mut engine, out, _) = engine_with_network();
        let code = engine.run(&[], input.as_bytes());
        drop(engine);
        (code, text(&out))
    }

    /// The output of commands run one at a time, each search to its end.
    fn search(commands: &[&str]) -> String {
        let (mut engine, out, _) = engine_with_network();
        for cmd in commands {
            assert!(matches!(engine.execute(cmd), Ok(true)), "{cmd}");
            engine.engine.wait_for_search_finished();
        }
        drop(engine);
        text(&out)
    }

    /// What `go` reports before it searches.
    fn go_info(threads: usize) -> String {
        format!(
            "info string Available processors: {}\ninfo string Using {threads} thread{}\n",
            available_processors(),
            if threads > 1 { "s" } else { "" }
        )
    }

    const NETWORK_INFO: &str = "info string NNUE evaluation using nn-c9e480e92425.nnue \
                                (107MiB, (109424, 1024, 32, 32, 1))\n\
                                info string Network replica 1: Local memory.\n";

    #[test]
    fn uci_lists_the_options() {
        let (code, out) = run("uci\nisready\n");
        assert_eq!(code, 0);
        let max_threads = 1024.max(4 * crate::misc::get_hardware_concurrency());
        assert_eq!(
            out,
            format!(
                "id name varifish {VERSION}\nid author the varifish and Stockfish developers \
                 (see AUTHORS)\n\n\
                 option name Threads type spin default 1 min 1 max {max_threads}\n\
                 option name Hash type spin default 16 min 1 max {MAX_HASH_MB}\n\
                 option name Clear Hash type button\n\
                 option name Ponder type check default false\n\
                 option name MultiPV type spin default 1 min 1 max 256\n\
                 option name Skill Level type spin default 20 min 0 max 20\n\
                 option name Move Overhead type spin default 10 min 0 max 5000\n\
                 option name nodestime type spin default 0 min 0 max 10000\n\
                 option name UCI_Chess960 type check default false\n\
                 option name UCI_Variant type combo default chess var chess var atomic\n\
                 option name UCI_LimitStrength type check default false\n\
                 option name UCI_Elo type spin default 1320 min 1320 max 3190\n\
                 option name UCI_ShowWDL type check default false\n\
                 option name SyzygyPath type string default <empty>\n\
                 option name SyzygyProbeDepth type spin default 1 min 1 max 100\n\
                 option name Syzygy50MoveRule type check default true\n\
                 option name SyzygyProbeLimit type spin default 7 min 0 max 7\n\
                 option name Use NNUE type combo default true var true var pure\n\
                 option name EvalFile type string default nn-c9e480e92425.nnue\nuciok\n\
                 readyok\n"
            )
        );
    }

    #[test]
    fn perft_divides_the_root_moves() {
        let (code, out) = run("position startpos moves e2e4 e7e5\ngo perft 1\nquit\nisready\n");
        assert_eq!(code, 0);
        assert!(
            out.starts_with(&format!("{}{NETWORK_INFO}a2a3: 1\nb2b3: 1\n", go_info(1))),
            "{out}"
        );
        assert!(out.ends_with("e1e2: 1\n\nNodes searched: 29\n\n"), "{out}");
    }

    #[test]
    fn castling_is_written_as_the_king_moves_except_in_chess960() {
        let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        let (_, out) = run(&format!("position fen {fen}\ngo perft 1\n"));
        assert!(out.contains("e1g1: 1\ne1c1: 1\n"), "{out}");

        // White castles kingside, after which its rook on f1 stops Black
        // castling kingside.
        let (_, out) = run(&format!(
            "setoption name UCI_Chess960 value true\nposition fen {fen}\ngo perft 1\n\
             position fen {fen} moves E1H1\ngo perft 1\n"
        ));
        assert!(out.contains("e1h1: 1\ne1a1: 1\n"), "{out}");
        assert!(out.contains("e8a8: 1\n") && !out.contains("e8h8"), "{out}");
    }

    #[test]
    fn critical_errors_end_the_loop() {
        let (code, out) = run("position startpos moves e2e5\nisready\n");
        assert_eq!(code, 1);
        assert_eq!(
            out,
            "info string CRITICAL ERROR: Command `position startpos moves e2e5` failed. \
             Reason: Illegal move: e2e5\n\n"
        );

        let (code, out) = run("go perft\n");
        assert_eq!(code, 1);
        assert!(
            out.contains("Reason: Invalid argument for 'perft'"),
            "{out}"
        );

        let (code, out) = run("position fen 8/8/8/8/8/8/8/8 w - - 0 1\n");
        assert_eq!(code, 1);
        assert!(out.contains("Reason: Unsupported position. Incorrect number of kings."));
    }

    #[test]
    fn unknown_commands_and_comments() {
        let (_, out) = run("hello world\n# a comment\n\nsetoption name Foo value 1\n");
        assert_eq!(
            out,
            "Unknown command: 'hello world'. Type help for more information.\n\
             No such option: Foo\n"
        );
    }

    #[test]
    fn command_line_arguments_run_one_command() {
        let (mut engine, out, _) = engine_with_network();
        let args = ["go".to_string(), "perft".to_string(), "2".to_string()];
        assert_eq!(engine.run(&args, "isready\n".as_bytes()), 0);
        let out = text(&out);
        assert!(out.ends_with("\nNodes searched: 400\n\n"), "{out}");
    }

    #[test]
    fn searches_check_the_network_first() {
        let (code, out) = run("go perft 1\n");
        assert_eq!(code, 0);
        assert!(
            out.starts_with(&format!("{}{NETWORK_INFO}a2a3: 1\n", go_info(1))),
            "{out}"
        );

        let (mut engine, out, _) = engine();
        let code = engine.run(&[], "isready\ngo depth 1\nisready\n".as_bytes());
        assert_eq!(code, 1);
        assert_eq!(
            text(&out),
            format!(
                "readyok\n{}\
                 info string ERROR: Network evaluation parameters compatible with the engine \
                 must be available.\n\
                 info string ERROR: The network file nn-c9e480e92425.nnue was not loaded \
                 successfully.\n\
                 info string ERROR: The UCI option EvalFile might need to specify the full \
                 path, including the directory name, to the network file.\n\
                 info string ERROR: The default net can be downloaded from: \
                 https://tests.stockfishchess.org/api/nn/nn-c9e480e92425.nnue\n\
                 info string ERROR: The engine will be terminated now.\n",
                go_info(1)
            )
        );
    }

    #[test]
    fn a_network_that_does_not_load_leaves_the_one_loaded() {
        let (code, out) = run("setoption name EvalFile value no-such.nnue\neval\n");
        assert_eq!(code, 1);
        assert!(
            out.contains("ERROR: The network file no-such.nnue was not loaded successfully."),
            "{out}"
        );

        let (code, out) = run("setoption name EvalFile value no-such.nnue\n\
             setoption name EvalFile value <empty>\ngo perft 1\n");
        assert_eq!(code, 0);
        assert!(
            out.starts_with(&format!("{}{NETWORK_INFO}", go_info(1))),
            "{out}"
        );
    }

    #[test]
    fn eval_traces_the_evaluation() {
        let (code, out) = run("eval\n");
        assert_eq!(code, 0);
        let buckets: String = (0..32)
            .map(|b| {
                let used = if b == 31 {
                    " <-- this bucket is used"
                } else {
                    ""
                };
                format!("|  {b:<10}|     0.00   |{used}\n")
            })
            .collect();
        assert_eq!(
            out,
            format!(
                "{NETWORK_INFO}\n\nNNUE network contributions (Normalized, White to move)\n\
                 +------------+------------+\n\
                 |   Bucket   | Evaluation |\n\
                 +------------+------------+\n\
                 {buckets}\
                 +------------+------------+\n\n\
                 NNUE evaluation          +0 (side to move, internal units)\n\
                 NNUE evaluation        +0.00 (white side)\n\
                 Final evaluation      +0.00 (white side) [with scaled NNUE, ...]\n\n"
            )
        );

        let (_, out) = run("position startpos moves f2f3 e7e5 g2g4 d8h4\neval\n");
        assert_eq!(
            out,
            format!("{NETWORK_INFO}\nFinal evaluation: none (in check)\n")
        );
    }

    #[test]
    fn export_net_needs_a_network() {
        let (mut engine, out, _) = engine();
        assert_eq!(engine.run(&[], "export_net\n".as_bytes()), 0);
        assert_eq!(
            text(&out),
            "Failed to export a net. No network file is currently loaded. \
             Please load a network file first.\n"
        );
    }

    /// The info lines and the best move of a search's output.
    fn output_lines(out: &str) -> Vec<&str> {
        out.lines()
            .filter(|line| !line.starts_with("info string"))
            .collect()
    }

    #[test]
    fn searches_report_each_depth_and_the_best_move() {
        let out = search(&["position startpos", "go depth 4"]);
        let lines = output_lines(&out);
        assert_eq!(lines.len(), 5, "{out}");
        for (depth, line) in lines[..4].iter().enumerate() {
            let prefix = format!("info depth {} seldepth ", depth + 1);
            assert!(line.starts_with(&prefix), "{out}");
            assert!(line.contains(" multipv 1 score cp "), "{out}");
            assert!(line.contains(" hashfull "), "{out}");
            assert!(line.contains(" tbhits 0 time "), "{out}");
        }
        let pv = lines[3].split(" pv ").nth(1).unwrap();
        let bestmove = lines[4].strip_prefix("bestmove ").unwrap();
        assert!(bestmove.starts_with(pv.split(' ').next().unwrap()), "{out}");
    }

    #[test]
    fn searches_find_mates() {
        // Scholar's mate, which Qxf7 gives.
        let fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4";
        let out = search(&[&format!("position fen {fen}"), "go depth 3"]);
        let lines = output_lines(&out);
        assert!(lines[2].contains(" score mate 1 "), "{out}");
        assert!(lines[2].ends_with(" pv h5f7"), "{out}");
        assert_eq!(*lines.last().unwrap(), "bestmove h5f7", "{out}");

        // Black to move is mated, or stalemated.
        let out = search(&["position startpos moves f2f3 e7e5 g2g4 d8h4", "go depth 3"]);
        assert_eq!(
            output_lines(&out),
            ["info depth 0 score mate 0", "bestmove (none)"]
        );
        let out = search(&["position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "go depth 3"]);
        assert_eq!(
            output_lines(&out),
            ["info depth 0 score cp 0", "bestmove (none)"]
        );
    }

    #[test]
    fn searchmoves_restrict_the_root_moves() {
        let out = search(&["position startpos", "go depth 3 searchmoves a2a3 H2H4"]);
        let bestmove = *output_lines(&out).last().unwrap();
        assert!(
            bestmove.starts_with("bestmove a2a3") || bestmove.starts_with("bestmove h2h4"),
            "{out}"
        );
    }

    #[test]
    fn multipv_reports_each_line() {
        let out = search(&[
            "setoption name MultiPV value 3",
            "setoption name UCI_ShowWDL value true",
            "go depth 2",
        ]);
        let lines = output_lines(&out);
        assert_eq!(lines.len(), 7, "{out}");
        for (i, line) in lines[3..6].iter().enumerate() {
            assert!(
                line.starts_with("info depth 2 seldepth ")
                    && line.contains(&format!(" multipv {} score cp ", i + 1))
                    && line.contains(" wdl "),
                "{out}"
            );
        }
    }

    #[test]
    fn helper_threads_search_too() {
        let out = search(&["setoption name Threads value 3", "go depth 5"]);
        assert!(out.starts_with("info string Using 3 threads\n"), "{out}");
        assert!(out.contains(&go_info(3)), "{out}");
        assert!(
            output_lines(&out).last().unwrap().starts_with("bestmove "),
            "{out}"
        );
    }

    #[test]
    fn stop_ends_an_infinite_search() {
        let (mut engine, out, _) = engine_with_network();
        assert!(matches!(engine.execute("go infinite"), Ok(true)));
        while !text(&out).contains("info depth 2 ") {
            std::thread::yield_now();
        }
        assert!(!text(&out).contains("bestmove"));
        assert!(matches!(engine.execute("stop"), Ok(true)));
        engine.engine.wait_for_search_finished();
        assert!(text(&out).ends_with('\n') && text(&out).contains("\nbestmove "));
    }

    #[test]
    fn quit_stops_the_search_and_waits_for_its_best_move() {
        let (code, out) = run("go infinite\nquit\n");
        assert_eq!(code, 0);
        assert!(
            output_lines(&out).last().unwrap().starts_with("bestmove "),
            "{out}"
        );
    }

    #[test]
    fn bench_reports_the_positions_and_the_nodes() {
        let (mut engine, out, err) = engine_with_network();
        assert!(matches!(engine.execute("bench 16 1 3"), Ok(true)));
        let err = text(&err);
        assert!(err.starts_with(
            "\nPosition: 1/51 (rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1)\n"
        ));
        assert!(err.contains(
            "\nPosition: 51/51 (nqbnrkrb/pppppppp/8/8/8/8/PPPPPPPP/NQBNRKRB w GEge - 0 1)\n"
        ));
        assert_eq!(text(&out).matches("bestmove").count(), 51);

        // The nodes are those of each search's last line.
        let nodes: u64 = text(&out)
            .split("\nbestmove")
            .filter_map(|search| {
                let last = search.lines().rfind(|l| l.starts_with("info depth"))?;
                last.split(" nodes ")
                    .nth(1)?
                    .split(' ')
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
            .sum();
        assert!(
            err.contains(&format!("\nNodes searched  : {nodes}\n")),
            "{err}"
        );
    }

    const ATOMIC: &str = "setoption name UCI_Variant value atomic\n";

    const ATOMIC_NETWORK_INFO: &str = "info string NNUE evaluation using Legacy Atomic V1 \
                                       atomic_run3b_e202_l05.nnue (45MiB, (45056, 1024, 16, 32, 1))\n\
                                       info string Network replica 1: Local memory.\n";

    #[test]
    fn atomic_perft_needs_no_network() {
        let (mut engine, out, _) = engine();
        let input = format!("{ATOMIC}position startpos moves e2e4 d7d5\ngo perft 1\n");
        assert_eq!(engine.run(&[], input.as_bytes()), 0);
        let out = text(&out);
        // Captures come first.
        assert!(
            out.starts_with(&format!("{}e4d5: 1\na2a3: 1\n", go_info(1))),
            "{out}"
        );
        assert!(
            out.ends_with("\ne1e2: 1\n\nNodes searched: 31\n\n"),
            "{out}"
        );
    }

    #[test]
    fn atomic_searches_check_its_network_first() {
        let (mut uci, out, _) = engine();
        let input = format!("{ATOMIC}go depth 1\n");
        assert_eq!(uci.run(&[], input.as_bytes()), 1);
        assert_eq!(
            text(&out),
            format!(
                "{}\
                 info string ERROR: Network evaluation parameters compatible with the engine \
                 must be available.\n\
                 info string ERROR: The network file atomic_run3b_e202_l05.nnue was not loaded \
                 successfully.\n\
                 info string ERROR: The UCI option EvalFile might need to specify the full \
                 path, including the directory name, to the network file.\n\
                 info string ERROR: The default net can be downloaded from: \
                 https://github.com/Belzedar94/Atomic-Stockfish/releases/download/v1.0/atomic-run3b.nnue\n\
                 info string ERROR: Save it as atomic_run3b_e202_l05.nnue next to the engine.\n\
                 info string ERROR: The engine will be terminated now.\n",
                go_info(1)
            )
        );
    }

    #[test]
    fn atomic_eval_traces_the_material_and_the_layers() {
        let (code, out) = run(&format!("{ATOMIC}eval\n"));
        assert_eq!(code, 0);
        let buckets: String = (0..8)
            .map(|b| {
                let used = if b == 7 {
                    " <-- this bucket is used"
                } else {
                    ""
                };
                format!("|  {b}         |     0.00   |     0.00   |     0.00   |{used}\n")
            })
            .collect();
        assert_eq!(
            out,
            format!(
                "{ATOMIC_NETWORK_INFO}\n\nNNUE network contributions (Normalized, White to move)\n\
                 +------------+------------+------------+------------+\n\
                 |   Bucket   |  Material  | Positional |   Total    |\n\
                 |            |   (PSQT)   |  (Layers)  |            |\n\
                 +------------+------------+------------+------------+\n\
                 {buckets}\
                 +------------+------------+------------+------------+\n\n\
                 NNUE evaluation          +0 (side to move, internal units)\n\
                 NNUE evaluation        +0.00 (white side)\n\
                 Final evaluation      +0.00 (white side) [Use NNUE=true]\n\n"
            )
        );

        // Atomic-Stockfish evaluates positions in check too, but none
        // without a king.
        let (_, out) = run(&format!(
            "{ATOMIC}position startpos moves e2e4 d7d5 f1b5\neval\n"
        ));
        assert!(out.contains("\nFinal evaluation      +0.00 (white side) [Use NNUE=true]\n"));
        let (_, out) = run(&format!(
            "{ATOMIC}position fen k7/1p6/8/8/8/8/8/4K2Q w - - 0 1 moves h1b7\neval\n"
        ));
        assert_eq!(
            out,
            format!("{ATOMIC_NETWORK_INFO}\nFinal evaluation: none (Atomic terminal)\n")
        );
    }

    #[test]
    fn atomic_searches_explode_the_king() {
        // The queen explodes the king by taking the pawn next to it.
        let fen = "k7/1p6/8/8/8/8/8/4K2Q w - - 0 1";
        let out = search(&[ATOMIC.trim(), &format!("position fen {fen}"), "go depth 3"]);
        let lines = output_lines(&out);
        assert!(lines[2].contains(" score mate 1 "), "{out}");
        assert!(lines[2].ends_with(" pv h1b7"), "{out}");
        assert_eq!(*lines.last().unwrap(), "bestmove h1b7", "{out}");

        // Black, whose king has exploded, has lost, and no search starts.
        let out = search(&[
            ATOMIC.trim(),
            &format!("position fen {fen} moves h1b7"),
            "go depth 3",
        ]);
        assert_eq!(
            output_lines(&out),
            ["info depth 0 score mate 0", "bestmove (none)"]
        );

        // Atomic scores are in Atomic-Stockfish's pawns, and without WDL.
        let out = search(&[
            ATOMIC.trim(),
            "setoption name UCI_ShowWDL value true",
            "go depth 3",
        ]);
        assert!(!out.contains(" wdl "), "{out}");
    }

    #[test]
    fn atomic_principal_variations_end_with_the_explosion() {
        // White's reply explodes the black king below the root: in qsearch
        // at depth 1, in the main search at depth 2. Atomic-Stockfish's
        // cases: en passant, a capture that promotes, and a pinned bishop.
        let cases = [
            ("8/2kp4/8/4P3/8/8/8/K7 b - - 0 1", "d7d5", "e5d6"),
            ("6k1/6Pr/8/8/8/8/8/K7 b - - 0 1", "h7h8", "g7h8q"),
            ("1r4k1/7r/8/8/8/8/1B6/1K6 b - - 0 1", "h7h8", "b2h8"),
        ];
        for (fen, m, explosion) in cases {
            for depth in [1, 2] {
                let out = search(&[
                    ATOMIC.trim(),
                    &format!("position fen {fen}"),
                    &format!("go depth {depth} searchmoves {m}"),
                ]);
                let lines = output_lines(&out);
                let last = lines[lines.len() - 2];
                assert!(last.contains(" score mate -1 "), "{out}");
                assert!(last.ends_with(&format!(" pv {m} {explosion}")), "{out}");
            }
        }
    }

    #[test]
    fn the_variant_starts_from_its_starting_position() {
        let out = search(&[
            "position startpos moves e2e4",
            ATOMIC.trim(),
            "d",
            "setoption name UCI_Variant value Chess",
            "position startpos moves e2e4",
            "setoption name UCI_Variant value chess",
            "d",
        ]);
        let fens: Vec<&str> = out.lines().filter(|l| l.starts_with("Fen: ")).collect();
        assert_eq!(
            fens,
            [
                "Fen: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "Fen: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            ]
        );
    }

    #[test]
    fn the_atomic_bench_searches_atomic_stockfish_positions() {
        let (mut engine, out, err) = engine_with_network();
        assert!(matches!(engine.execute(ATOMIC.trim()), Ok(true)));
        assert!(matches!(engine.execute("bench 16 1 2"), Ok(true)));
        let err = text(&err);
        assert!(
            err.contains("\nPosition: 13/13 (1R4kr/4K3/8/8/8/8/8/8 b h - 0 1)\n"),
            "{err}"
        );
        assert_eq!(text(&out).matches("bestmove").count(), 13);
    }

    #[test]
    fn scores_are_formatted_for_uci() {
        assert_eq!(format_score(&Score::Mate { plies: 1 }), "mate 1");
        assert_eq!(format_score(&Score::Mate { plies: 4 }), "mate 2");
        assert_eq!(format_score(&Score::Mate { plies: -2 }), "mate -1");
        assert_eq!(format_score(&Score::Mate { plies: 0 }), "mate 0");
        assert_eq!(
            format_score(&Score::Tablebase {
                plies: 3,
                win: true
            }),
            "cp 19997"
        );
        assert_eq!(
            format_score(&Score::Tablebase {
                plies: -3,
                win: false
            }),
            "cp -19997"
        );
        assert_eq!(format_score(&Score::InternalUnits { value: -42 }), "cp -42");
    }

    #[test]
    fn wdl_splits_a_thousand() {
        let pos = Position::<Chess>::new();
        assert_eq!(wdl(0, &pos).unwrap(), "18 964 18");
        assert_eq!(wdl(-300, &pos).unwrap(), "0 442 558");
        assert_eq!(wdl(VALUE_MATE, &pos).unwrap(), "1000 0 0");
        assert_eq!(wdl(0, &Position::<Atomic>::new()), None);
    }
}
