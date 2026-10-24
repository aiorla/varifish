//! Probes varifish's Syzygy tables for the positions given: each line of the
//! input is a FEN.
//!
//! For each position it prints the WDL and DTZ probes with their states, and
//! the root moves as rank_root_moves ranks them, first as the search does,
//! then by DTZ, each move with its rank and score, sorted by move, so that
//! the order in which moves are generated does not show. The search's probe
//! limit is 7 pieces for chess and 6 for Atomic, whose tables go no further.
//!
//! With "oracle", it prints instead the WDL and DTZ probes of shakmaty-syzygy,
//! an independent reading of the same tables, as "FEN | WDL DTZ", or "FEN |
//! -" for a position it cannot probe.
//!
//! With "generate", it prints random positions with the material of each
//! WDL table found, COUNT for each side the stronger and each side to move,
//! with random move counters, and the positions that one to three random
//! moves lead to from them. With EXTRA, each has that many more random
//! pieces, which take it out of the tables.
//!
//!     tablebases chess|atomic SYZYGYPATH < positions
//!     tablebases chess|atomic SYZYGYPATH oracle < positions
//!     tablebases chess|atomic SYZYGYPATH generate COUNT SEED [EXTRA]

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use varifish::misc::Prng;
use varifish::movegen::{ALL, MoveList};
use varifish::position::Position;
use varifish::search::RootMove;
use varifish::syzygy::{ProbeState, SyzygyOptions, Tables};
use varifish::uci::move_to_uci;
use varifish::variant::{Atomic, Chess, Variant};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(variant), Some(paths)) = (args.get(1), args.get(2)) else {
        eprintln!(
            "usage: tablebases chess|atomic SYZYGYPATH [oracle | generate COUNT SEED [EXTRA]]"
        );
        return ExitCode::FAILURE;
    };
    let number = |i: usize| args.get(i).and_then(|a| a.parse::<u64>().ok());
    let extra = args.get(6).map_or(Some(0), |a| a.parse::<usize>().ok());

    let result = match (variant.as_str(), args.get(3).map(String::as_str)) {
        ("chess", None) => probe::<Chess>(paths),
        ("atomic", None) => probe::<Atomic>(paths),
        ("chess", Some("oracle")) => oracle::<shakmaty::Chess>(paths),
        ("atomic", Some("oracle")) => oracle::<shakmaty::variant::Atomic>(paths),
        ("chess", Some("generate")) => {
            generate_positions::<Chess>(paths, number(4), number(5), extra)
        }
        ("atomic", Some("generate")) => {
            generate_positions::<Atomic>(paths, number(4), number(5), extra)
        }
        (_, Some(mode)) if !["oracle", "generate"].contains(&mode) => {
            Err(format!("unknown mode {mode}"))
        }
        _ => Err(format!("unknown variant {variant}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn probe<V: Variant>(paths: &str) -> Result<(), String> {
    let tables = Tables::<V>::new(paths);
    let options = SyzygyOptions {
        probe_depth: 1,
        rule50: true,
        probe_limit: if V::ATOMIC { 6 } else { 7 },
    };

    let mut out = io::BufWriter::new(io::stdout().lock());
    if let Some(info) = tables.info() {
        writeln!(out, "info string {info}").map_err(|e| e.to_string())?;
    }

    for line in io::stdin().lock().lines() {
        let fen = line.map_err(|e| e.to_string())?;
        if fen.is_empty() {
            continue;
        }

        let mut pos = Position::<V>::from_fen(&fen, false).map_err(|e| format!("{fen}: {e}"))?;
        let (mut s1, mut s2) = (ProbeState::Ok, ProbeState::Ok);
        let wdl = tables.probe_wdl(&mut pos, &mut s1);
        let dtz = tables.probe_dtz(&mut pos, &mut s2);

        writeln!(
            out,
            "{fen} | {wdl} {} {dtz} {} | {} | {}",
            s1 as i32,
            s2 as i32,
            ranking(&tables, &options, &mut pos, false),
            ranking(&tables, &options, &mut pos, true)
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn ranking<V: Variant>(
    tables: &Tables<V>,
    options: &SyzygyOptions,
    pos: &mut Position<V>,
    rank_dtz: bool,
) -> String {
    let mut root_moves: Vec<RootMove> = MoveList::new::<ALL>(pos)
        .iter()
        .map(|&m| RootMove::new(m))
        .collect();

    let config = tables.rank_root_moves(options, pos, &mut root_moves, rank_dtz, &|| false);

    let mut s = format!(
        "{} {} {} {}",
        u8::from(config.root_in_tb),
        config.cardinality,
        config.probe_depth,
        u8::from(config.use_rule50)
    );
    if !config.root_in_tb {
        return s;
    }

    let mut moves: Vec<String> = root_moves
        .iter()
        .map(|rm| {
            format!(
                "{} {} {}",
                move_to_uci(rm.pv[0], false),
                rm.tb_rank,
                rm.tb_score
            )
        })
        .collect();
    moves.sort();
    for m in moves {
        s += ", ";
        s += &m;
    }
    s
}

/// The directories of a SyzygyPath
fn directories(paths: &str) -> impl Iterator<Item = &str> {
    let sep = if cfg!(windows) { ';' } else { ':' };
    paths.split(sep).filter(|p| !p.is_empty())
}

fn oracle<S>(paths: &str) -> Result<(), String>
where
    S: shakmaty::Position + shakmaty::FromSetup + shakmaty_syzygy::Syzygy + Clone,
{
    let mut tables = shakmaty_syzygy::Tablebase::<S>::new();
    for path in directories(paths) {
        tables
            .add_directory(path)
            .map_err(|e| format!("{path}: {e}"))?;
    }

    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let fen = line.map_err(|e| e.to_string())?;
        if fen.is_empty() {
            continue;
        }

        // Positions with a check no move can give are legal for the tables.
        let pos: Option<S> = fen.parse::<shakmaty::fen::Fen>().ok().and_then(|f| {
            f.into_position(shakmaty::CastlingMode::Standard)
                .or_else(shakmaty::PositionError::ignore_impossible_check)
                .ok()
        });
        // Stockfish's WDL probe takes the halfmove clock as zero, and its
        // DTZ probe gives the distance as stored, rounded or not.
        let probes = pos.and_then(|pos| {
            let wdl = tables.probe_wdl_after_zeroing(&pos).ok()?;
            let dtz = tables.probe_dtz(&pos).ok()?;
            Some((wdl as i32, dtz.ignore_rounding().0))
        });
        match probes {
            Some((wdl, dtz)) => writeln!(out, "{fen} | {wdl} {dtz}"),
            None => writeln!(out, "{fen} | -"),
        }
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The codes of the WDL tables in the directories, such as "KRvK"
fn table_codes<V: Variant>(paths: &str) -> Vec<String> {
    let suffix = V::SYZYGY.suffixes[0];
    let mut codes: Vec<String> = directories(paths)
        .filter_map(|path| std::fs::read_dir(path).ok())
        .flatten()
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            Some(name.strip_suffix(suffix)?.to_string())
        })
        .collect();
    codes.sort();
    codes.dedup();
    codes
}

fn generate_positions<V: Variant>(
    paths: &str,
    count: Option<u64>,
    seed: Option<u64>,
    extra: Option<usize>,
) -> Result<(), String> {
    let (Some(count), Some(seed), Some(extra)) = (count, seed, extra) else {
        return Err("generate needs COUNT and SEED, and EXTRA a number".to_string());
    };
    let mut rng = Prng::new(seed.max(1));
    let mut out = io::BufWriter::new(io::stdout().lock());
    let mut scratch = V::Dirties::default();

    for code in table_codes::<V>(paths) {
        let (strong, weak) = code.split_once('v').ok_or(format!("bad table {code}"))?;
        for white_strong in [true, false] {
            let (white, black) = if white_strong {
                (strong, weak)
            } else {
                (weak, strong)
            };
            for stm in ['w', 'b'] {
                let mut found = 0;
                let mut attempts = 0;
                while found < count && attempts < 1000 * count {
                    attempts += 1;
                    let (mut white, mut black) = (white.to_string(), black.to_string());
                    for _ in 0..extra {
                        let piece = b"PNBRQ"[(rng.rand64() % 5) as usize] as char;
                        if rng.rand64().is_multiple_of(2) {
                            white.push(piece);
                        } else {
                            black.push(piece);
                        }
                    }
                    let Some(fen) = random_fen(&mut rng, &white, &black, stm) else {
                        continue;
                    };
                    let Ok(mut pos) = Position::<V>::from_fen(&fen, false) else {
                        continue;
                    };
                    found += 1;
                    writeln!(out, "{fen}").map_err(|e| e.to_string())?;

                    // A few random moves on, which may capture, push a pawn
                    // two squares or end the game
                    for _ in 0..1 + rng.rand64() % 3 {
                        let moves = MoveList::new::<ALL>(&pos);
                        if moves.is_empty() {
                            break;
                        }
                        let m = moves
                            .iter()
                            .nth((rng.rand64() % moves.len() as u64) as usize);
                        pos.make_move(*m.expect("a move"), &mut scratch);
                        writeln!(out, "{}", pos.fen()).map_err(|e| e.to_string())?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// A FEN with White's pieces `white` and Black's `black`, as "KRP", on random
/// squares, pawns on the second to seventh ranks, and a random halfmove
/// clock, or None if two pieces fell on the same square.
fn random_fen(rng: &mut Prng, white: &str, black: &str, stm: char) -> Option<String> {
    let mut board = [None; 64];
    let pieces = white
        .chars()
        .map(|c| c.to_ascii_uppercase())
        .chain(black.chars().map(|c| c.to_ascii_lowercase()));
    for piece in pieces {
        let s = if piece.eq_ignore_ascii_case(&'p') {
            8 + rng.rand64() % 48
        } else {
            rng.rand64() % 64
        } as usize;
        if board[s].is_some() {
            return None;
        }
        board[s] = Some(piece);
    }

    let mut fen = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            match board[8 * rank + file] {
                Some(piece) => {
                    if empty > 0 {
                        fen += &empty.to_string();
                        empty = 0;
                    }
                    fen.push(piece);
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            fen += &empty.to_string();
        }
        if rank > 0 {
            fen.push('/');
        }
    }

    let rule50 = match rng.rand64() % 4 {
        0 => rng.rand64() % 100,
        _ => 0,
    };
    Some(format!("{fen} {stm} - - {rule50} 1"))
}
