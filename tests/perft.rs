//! Perft counts from Stockfish's suite (its tests/perft.sh) and the
//! chessprogramming wiki. The quick ones run by default; the whole suite takes
//! `cargo test --release -- --ignored`.

use varifish::perft::perft;
use varifish::position::START_FEN;
use varifish::variant::Chess;

const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
const POSITION_3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const POSITION_4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
const POSITION_5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
const POSITION_6: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

fn count(fen: &str, depth: i32, chess960: bool) -> u64 {
    perft::<Chess>(fen, depth, chess960, |_, _| {}).expect("a valid FEN")
}

#[test]
fn quick_counts() {
    let cases: [(&str, bool, &[u64]); 8] = [
        (START_FEN, false, &[20, 400, 8902, 197281]),
        (KIWIPETE, false, &[48, 2039, 97862]),
        (POSITION_3, false, &[14, 191, 2812, 43238, 674624]),
        (POSITION_4, false, &[6, 264, 9467, 422333]),
        (POSITION_5, false, &[44, 1486, 62379]),
        (POSITION_6, false, &[46, 2079, 89890]),
        (
            "rr6/2kpp3/1ppn2p1/p2b1q1p/P4P1P/1PNN2P1/2PP4/1K2R2R b E - 1 20",
            true,
            &[0, 1438],
        ),
        (
            "rr6/2kpp3/1ppn2p1/p2b1q1p/P4P1P/1PNN2P1/2PP4/1K2RR2 w E - 0 20",
            true,
            &[0, 0, 37340],
        ),
    ];
    for (fen, chess960, counts) in cases {
        for (depth, &nodes) in counts.iter().enumerate() {
            if nodes != 0 {
                assert_eq!(
                    count(fen, depth as i32 + 1, chess960),
                    nodes,
                    "{fen} at depth {}",
                    depth + 1
                );
            }
        }
    }
}

#[test]
fn depths_below_one_count_the_moves() {
    assert_eq!(count(START_FEN, 0, false), 20);
    assert_eq!(count(START_FEN, -3, false), 20);

    let mut divide = Vec::new();
    perft::<Chess>(KIWIPETE, 2, false, |m, nodes| divide.push((m, nodes))).unwrap();
    assert_eq!(divide.len(), 48);
    assert_eq!(divide.iter().map(|&(_, nodes)| nodes).sum::<u64>(), 2039);
}

/// Stockfish's tests/perft.sh, at the commit varifish follows.
#[test]
#[ignore = "billions of nodes: run with --release"]
fn stockfish_suite() {
    let cases = [
        (START_FEN, 7, 3195901860, false),
        (KIWIPETE, 5, 193690690, false),
        (POSITION_3, 7, 178633661, false),
        (POSITION_4, 6, 706045033, false),
        (POSITION_5, 5, 89941194, false),
        (POSITION_6, 5, 164075551, false),
        (
            "r7/4p3/5p1q/3P4/4pQ2/4pP2/6pp/R3K1kr w Q - 1 3",
            5,
            11609488,
            false,
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w AHah - 0 1",
            6,
            119060324,
            true,
        ),
        (
            "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 9",
            6,
            191762235,
            true,
        ),
        (
            "rbbqn1kr/pp2p1pp/6n1/2pp1p2/2P4P/P7/BP1PPPP1/R1BQNNKR w HAha - 0 9",
            6,
            924181432,
            true,
        ),
        (
            "rqbbknr1/1ppp2pp/p5n1/4pp2/P7/1PP5/1Q1PPPPP/R1BBKNRN w GAga - 0 9",
            6,
            308553169,
            true,
        ),
        (
            "4rrb1/1kp3b1/1p1p4/pP1Pn2p/5p2/1PR2P2/2P1NB1P/2KR1B2 w D - 0 21",
            6,
            872323796,
            true,
        ),
        (
            "1rkr3b/1ppn3p/3pB1n1/6q1/R2P4/4N1P1/1P5P/2KRQ1B1 b Dbd - 0 14",
            6,
            2678022813,
            true,
        ),
        (
            "qbbnrkr1/p1pppppp/1p4n1/8/2P5/6N1/PPNPPPPP/1BRKBRQ1 b FCge - 1 3",
            6,
            521301336,
            true,
        ),
        (
            "rr6/2kpp3/1ppn2p1/p2b1q1p/P4P1P/1PNN2P1/2PP4/1K2R2R b E - 1 20",
            2,
            1438,
            true,
        ),
        (
            "rr6/2kpp3/1ppn2p1/p2b1q1p/P4P1P/1PNN2P1/2PP4/1K2RR2 w E - 0 20",
            3,
            37340,
            true,
        ),
        (
            "rr6/2kpp3/1ppnb1p1/p2Q1q1p/P4P1P/1PNN2P1/2PP4/1K2RR2 b E - 2 19",
            4,
            2237725,
            true,
        ),
        (
            "rr6/2kpp3/1ppnb1p1/p4q1p/P4P1P/1PNN2P1/2PP2Q1/1K2RR2 w E - 1 19",
            4,
            2098209,
            true,
        ),
        (
            "rr6/2kpp3/1ppnb1p1/p4q1p/P4P1P/1PNN2P1/2PP2Q1/1K2RR2 w E - 1 19",
            5,
            79014522,
            true,
        ),
        (
            "rr6/2kpp3/1ppnb1p1/p4q1p/P4P1P/1PNN2P1/2PP2Q1/1K2RR2 w E - 1 19",
            6,
            2998685421,
            true,
        ),
    ];
    for (fen, depth, nodes, chess960) in cases {
        assert_eq!(count(fen, depth, chess960), nodes, "{fen} at depth {depth}");
    }
}
