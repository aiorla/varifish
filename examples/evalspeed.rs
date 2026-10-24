//! Times varifish's evaluation along random walks from the positions given:
//! each line of the input is "0 FEN" or, for Chess960, "1 FEN".
//!
//! From each position, for the plies given, the walk evaluates the position
//! on every second ply, as the random generator picks, and each position a
//! legal move leads to, as a search does with the accumulators kept along
//! the line, then plays one of those moves at random, Atomic's in the order
//! of their encodings. Chess positions in check are not evaluated, nor Atomic
//! positions without both kings. It prints the number of evaluations and a
//! hash of them in order, which the same walks always repeat, and the seconds
//! the walks took.
//!
//!     cargo run --release --example evalspeed [PLIES [SEED [EVALFILE]]] < positions
//!
//! The walks are chess's, with the default network from nets/, where `make
//! net` downloads it, unless EVALFILE names a network, of chess or of
//! Atomic, whose variant they then walk.

use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use varifish::evaluate::EVAL_FILE_DEFAULT_NAME;
use varifish::misc::Prng;
use varifish::movegen::{ALL, MoveList};
use varifish::nnue::atomic::network::Network as AtomicNetwork;
use varifish::nnue::eval_file::UseNnue;
use varifish::nnue::network::Network as ChessNetwork;
use varifish::nnue::{Accumulators, Caches};
use varifish::position::Position;
use varifish::types::{BLACK, Move, WHITE};
use varifish::variant::{Atomic, Chess, Variant};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let plies: usize = args.get(1).map_or(100, |a| a.parse().expect("PLIES"));
    let seed: u64 = args.get(2).map_or(1, |a| a.parse().expect("SEED"));

    let path = match args.get(3) {
        Some(file) => PathBuf::from(file),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("nets")
            .join(EVAL_FILE_DEFAULT_NAME),
    };
    let bytes = std::fs::read(&path).unwrap_or_default();

    let result = if let Some((network, _)) = ChessNetwork::read(&bytes) {
        walk::<Chess>(&network, plies, seed)
    } else if let Some((network, _)) = AtomicNetwork::read(&bytes) {
        walk::<Atomic>(&network, plies, seed)
    } else {
        Err(format!(
            "{} is not a network; make net downloads the default one",
            path.display()
        ))
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// Whether the walks evaluate a position
fn evaluable<V: Variant>(pos: &Position<V>) -> bool {
    if V::ATOMIC {
        pos.has_king(WHITE) && pos.has_king(BLACK)
    } else {
        pos.checkers().is_empty()
    }
}

fn walk<V: Variant>(network: &V::Network, plies: usize, seed: u64) -> Result<(), String> {
    let mut caches = V::Caches::new(Some(network));
    let mut accumulators = V::Accumulators::new();

    let mut rng = Prng::new(seed);
    let mut evaluations: u64 = 0;
    let mut hash: u64 = 0;
    let mut seconds = 0.0;

    for line in io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.len() < 3 {
            continue;
        }

        let mut pos =
            Position::<V>::from_fen(&line[2..], line.starts_with('1')).map_err(|e| e.0)?;
        accumulators.reset();

        let mut evaluate = |pos: &Position<V>, accumulators: &mut V::Accumulators| {
            let value = V::evaluate(
                Some(network),
                pos,
                accumulators,
                &mut caches,
                0,
                UseNnue::True,
            );
            hash = hash.wrapping_mul(1000003).wrapping_add(value as u64);
            evaluations += 1;
        };

        let start = Instant::now();

        for _ in 0..plies {
            let legal = MoveList::new::<ALL>(&pos);
            let sorted: Vec<Move>;
            let moves = if V::ATOMIC {
                // Atomic-Stockfish generates the moves in another order.
                let mut moves = legal.as_slice().to_vec();
                moves.sort_unstable_by_key(|m| m.raw());
                sorted = moves;
                &sorted
            } else {
                legal.as_slice()
            };
            if moves.is_empty() {
                break;
            }

            if evaluable(&pos) && !rng.rand64().is_multiple_of(2) {
                evaluate(&pos, &mut accumulators);
            }

            for &m in moves {
                let gives_check = pos.gives_check(m);
                pos.do_move(m, gives_check, accumulators.push());
                if evaluable(&pos) {
                    evaluate(&pos, &mut accumulators);
                }
                pos.undo_move(m);
                accumulators.pop();
            }

            let m = moves[(rng.rand64() % moves.len() as u64) as usize];
            let gives_check = pos.gives_check(m);
            pos.do_move(m, gives_check, accumulators.push());
        }

        seconds += start.elapsed().as_secs_f64();
    }

    println!("evaluations {evaluations} hash {hash:016x} seconds {seconds:.3}");
    Ok(())
}
