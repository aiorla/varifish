//! The positions and commands of `bench`, which searches a fixed set of
//! positions to measure the engine's speed and check its determinism:
//! Stockfish's for chess, and Atomic-Stockfish's for Atomic.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `benchmark.h` and `benchmark.cpp`, and of Atomic-Stockfish's positions
//! (https://github.com/Belzedar94/Atomic-Stockfish): Stockfish, a UCI chess
//! playing engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The
//! Stockfish developers (see AUTHORS), under the GNU General Public License,
//! version 3 or later.

use std::fs;

use crate::misc::IStream;

#[rustfmt::skip]
pub const DEFAULTS: &[&str] = &[
    "setoption name UCI_Chess960 value false",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14 moves d4e6",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14 moves g2g4",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
    "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
    "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 0 1 moves g5g6 f3e3 g6g5 e3f3",
    "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
    "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
    "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
    "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
    "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
    "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
    "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
    "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
    "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
    "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
    "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
    "5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
    "4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
    "r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
    "3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40",
    "4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1",
    "1r6/1P4bk/3qr1p1/N6p/3pp2P/6R1/3Q1PP1/1R4K1 w - - 1 42",

    // Positions with high numbers of changed threats
    "k7/2n1n3/1nbNbn2/2NbRBn1/1nbRQR2/2NBRBN1/3N1N2/7K w - - 0 1",
    "K7/8/8/BNQNQNB1/N5N1/R1Q1q2r/n5n1/bnqnqnbk w - - 0 1",

    // 5-man positions
    "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1", // Kc2 - mate
    "8/8/8/5N2/8/p7/8/2NK3k w - - 0 1", // Na2 - mate
    "8/3k4/8/8/8/4B3/4KB2/2B5 w - - 0 1", // draw

    // 6-man positions
    "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1", // Re5 - mate
    "8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1", // Ka2 - mate
    "8/8/3P3k/8/1p6/8/1P6/1K3n2 b - - 0 1", // Nd2 - draw

    // 7-man positions
    "8/R7/2q5/8/6k1/8/1P5p/K6R w - - 0 124", // Draw

    // Mate and stalemate positions
    "6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1",
    "r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1",
    "8/8/8/8/8/6k1/6p1/6K1 w - -",
    "7k/7P/6K1/8/3B4/8/8/8 b - -",

    // Chess 960
    "setoption name UCI_Chess960 value true",
    "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1 moves g2g3 d7d5 d2d4 c8h3 c1g5 e8d6 g5e7 f7f6",
    "nqbnrkrb/pppppppp/8/8/8/8/PPPPPPPP/NQBNRKRB w KQkq - 0 1",
    "setoption name UCI_Chess960 value false",

];

#[rustfmt::skip]
pub const ATOMIC_DEFAULTS: &[&str] = &[
    "setoption name UCI_Chess960 value false",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rn2kb1r/1pp1p2p/p2q1pp1/3P4/2P3b1/4PN2/PP3PPP/R2QKB1R b KQkq - 0 1",
    "rn1qkb1r/p5pp/2p5/3p4/N3P3/5P2/PPP4P/R1BQK3 w Qkq - 0 1",
    "r4b1r/2kb1N2/p2Bpnp1/8/2Pp3p/1P1PPP2/P5PP/R3K2R b KQ - 0 1",
    "4K1R1/RR6/8/8/8/8/qqq3p1/7k b - - 0 1",
    "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
    "6kr/6P1/8/8/8/8/8/K7 w - - 0 1",
    "6rk/8/8/8/8/8/8/K5R1 w - - 0 1",
    "5BBB/8/8/8/8/8/6k1/7K w - - 0 1",
    "8/8/8/8/8/8/NkN5/1K6 w - - 0 1",
    "setoption name UCI_Chess960 value true",
    "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1",
    "Rr2k1rR/3K4/3p4/8/8/8/7P/8 w kq - 0 1",
    "1R4kr/4K3/8/8/8/8/8/8 b k - 0 1",
    "setoption name UCI_Chess960 value false",
];

/// Builds a list of UCI commands to be run by bench, whose default
/// positions are `defaults`. There are five parameters: TT size in MB,
/// number of search threads that should be used, the limit value spent for
/// each position, a file name where to look for positions in FEN format, and
/// the type of the limit: depth, perft, nodes and movetime (in
/// milliseconds). Examples:
///
/// bench                            : search default positions up to depth 13
/// bench 64 1 15                    : search default positions up to depth 15 (TT = 64MB)
/// bench 64 1 100000 default nodes  : search default positions for 100K nodes each
/// bench 64 4 5000 current movetime : search current position with 4 threads for 5 sec
/// bench 16 1 5 blah perft          : run a perft 5 on positions in file "blah"
///
/// Err names the file of positions that could not be read.
pub fn setup_bench(
    defaults: &[&str],
    current_fen: &str,
    is: &mut IStream,
) -> Result<Vec<String>, String> {
    // Assign default values to missing arguments
    let tt_size = is.token().unwrap_or("16");
    let threads = is.token().unwrap_or("1");
    let limit = is.token().unwrap_or("13");
    let fen_file = is.token().unwrap_or("default");
    let limit_type = is.token().unwrap_or("depth");

    let go = if limit_type == "eval" {
        "eval".to_string()
    } else {
        format!("go {limit_type} {limit}")
    };

    let fens: Vec<String> = match fen_file {
        "default" => defaults.iter().map(|fen| fen.to_string()).collect(),
        "current" => vec![current_fen.to_string()],
        _ => {
            let Ok(bytes) = fs::read(fen_file) else {
                return Err(format!("Unable to open file {fen_file}"));
            };
            String::from_utf8_lossy(&bytes)
                .split('\n')
                .filter(|fen| !fen.is_empty())
                .map(str::to_string)
                .collect()
        }
    };

    let mut list = vec![
        format!("setoption name Threads value {threads}"),
        format!("setoption name Hash value {tt_size}"),
        "ucinewgame".to_string(),
    ];

    for fen in fens {
        if fen.contains("setoption") {
            list.push(fen);
        } else {
            list.push(format!("position fen {fen}"));
            list.push(go.clone());
        }
    }

    Ok(list)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_bench_searches_51_positions_to_depth_13() {
        let list = setup_bench(DEFAULTS, "", &mut IStream::new("")).unwrap();
        assert_eq!(
            &list[..3],
            [
                "setoption name Threads value 1",
                "setoption name Hash value 16",
                "ucinewgame"
            ]
        );
        assert_eq!(list.iter().filter(|cmd| *cmd == "go depth 13").count(), 51);
        assert_eq!(list[3], "setoption name UCI_Chess960 value false");
        assert_eq!(
            list[4],
            "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
        assert_eq!(
            list.last().unwrap(),
            "setoption name UCI_Chess960 value false"
        );
    }

    #[test]
    fn the_atomic_bench_searches_13_positions() {
        let list = setup_bench(ATOMIC_DEFAULTS, "", &mut IStream::new("")).unwrap();
        assert_eq!(list.iter().filter(|cmd| *cmd == "go depth 13").count(), 13);
        assert_eq!(
            list.iter()
                .filter(|cmd| cmd.starts_with("setoption"))
                .count(),
            2 + 3
        );
    }

    #[test]
    fn arguments_set_the_hash_threads_limit_and_positions() {
        let fen = "8/8/8/8/8/6k1/6p1/6K1 w - - 0 1";
        let list = setup_bench(
            DEFAULTS,
            fen,
            &mut IStream::new("64 4 5000 current movetime"),
        )
        .unwrap();
        assert_eq!(
            list,
            [
                "setoption name Threads value 4",
                "setoption name Hash value 64",
                "ucinewgame",
                &format!("position fen {fen}"),
                "go movetime 5000",
            ]
        );

        let list = setup_bench(DEFAULTS, fen, &mut IStream::new("16 1 1 current eval")).unwrap();
        assert_eq!(list.last().unwrap(), "eval");

        assert_eq!(
            setup_bench(
                DEFAULTS,
                fen,
                &mut IStream::new("16 1 5 no-such-file perft")
            ),
            Err("Unable to open file no-such-file".to_string())
        );
    }
}
