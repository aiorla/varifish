//! varifish's Antichess positions checked along random games against
//! shakmaty's Antichess: legal moves (mandatory captures, en passant, 5-way
//! promotions including King, 0-king and multi-king positions), FENs,
//! incremental state (keys, threats including kings, undoing), perft counts,
//! which 16 bits `legal` accepts, the bootstrap evaluation's color-flip
//! symmetry, and rule edge cases (stalemate over 50-move draw, upcoming repetition under
//! mandatory captures).

use std::collections::BTreeMap;

use shakmaty::fen::Fen;
use shakmaty::variant::Antichess as SkAntichess;
use shakmaty::{CastlingMode, EnPassantMode, Position as _, perft as sk_perft};
use varifish::attacks::piece_attacks_bb;
use varifish::misc::Prng;
use varifish::movegen::{ALL, CAPTURES, MoveList, QUIETS};
use varifish::nnue::antichess::{AntichessAccumulators, AntichessCaches};
use varifish::nnue::eval_file::{Networks, UseNnue};
use varifish::nnue::{Accumulators, Caches};
use varifish::perft::perft;
use varifish::types::*;
use varifish::uci::{move_to_uci, to_move};
use varifish::variant::{Antichess, Variant};

type Position = varifish::position::Position<Antichess>;

const STARTS: [&str; 10] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    "rnbqkbnr/p1pppppp/8/1p6/8/4P3/PPPP1PPP/RNBQKBNR w - - 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/8/4P3/PPPP1PPP/RNBQKBNR w - - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/8/4P3/PPPP1PPP/RNBQKBNR w - - 0 2",
    // En passant capture available (mandatory!)
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
    // Promotions (including king promotion) and capture-promotions
    "3r1n2/4P3/8/8/8/8/3p4/2B1R3 w - - 0 1",
    // Kingless position (both kings already captured)
    "2b2b2/pp2pppp/8/2n1N3/8/8/PP2PPPP/2B2B2 w - - 0 10",
    // Multi-king position (from king promotions)
    "2k1k3/1p6/8/3K4/8/8/6P1/3K1K2 w - - 0 25",
    // Dense tactical position with many threats and captures
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w - - 0 4",
];

fn shakmaty_position(fen: &str) -> SkAntichess {
    fen.parse::<Fen>()
        .unwrap_or_else(|e| panic!("{fen}: {e}"))
        .into_position(CastlingMode::Standard)
        .unwrap_or_else(|e| panic!("{fen}: {e}"))
}

/// Computes every all-pieces threat in `pos` from scratch (matching
/// `position/antichess.rs`'s incremental `DirtyThreats`).
fn all_threats(pos: &Position) -> BTreeMap<u32, i32> {
    let occupied = pos.pieces();
    let mut out = BTreeMap::new();
    for s in occupied {
        let pc = pos.piece_on(s);
        for t in piece_attacks_bb(pc, s, occupied) & occupied {
            out.insert(DirtyThreat::new(pc, pos.piece_on(t), s, t, false).raw(), 1);
        }
    }
    out
}

fn check_position(pos: &mut Position, sk: &SkAntichess) {
    let fen = pos.fen();

    // Legal moves match shakmaty
    let moves = MoveList::new::<ALL>(pos);
    let mut ours: Vec<String> = moves.iter().map(|&mv| move_to_uci(mv, false)).collect();
    let mut theirs: Vec<String> = sk
        .legal_moves()
        .iter()
        .map(|mv| mv.to_uci(CastlingMode::Standard).to_string())
        .collect();
    ours.sort();
    theirs.sort();
    assert_eq!(ours, theirs, "{fen}");

    // CAPTURES / QUIETS partition matches ALL
    let caps = MoveList::new::<CAPTURES>(pos);
    let quiets = MoveList::new::<QUIETS>(pos);
    assert_eq!(pos.antichess_has_captures(), !caps.is_empty(), "{fen}");
    if !caps.is_empty() {
        assert!(
            quiets.is_empty(),
            "quiets generated despite captures: {fen}"
        );
        assert_eq!(caps.len(), moves.len(), "{fen}");
    } else {
        assert_eq!(quiets.len(), moves.len(), "{fen}");
        assert_eq!(pos.antichess_has_quiet_move(), !quiets.is_empty(), "{fen}");
    }
    assert_eq!(pos.antichess_has_legal_move(), !moves.is_empty(), "{fen}");

    // Every generated move passes `legal` and `to_move` round-trips
    for &mv in &moves {
        assert!(pos.legal(mv), "generated illegal move {mv:?} in {fen}");
        let s = move_to_uci(mv, false);
        assert_eq!(to_move(pos, &s), mv, "{fen} {s}");
    }

    // `legal` accepts no other 16 bits: a move read from a transposition
    // table entry of another position must not pass for one of this one's
    for raw in 0..=u16::MAX {
        let mv = Move(raw);
        if mv.is_ok() && pos.legal(mv) {
            assert!(moves.contains(mv), "legal accepts {raw:#06x} in {fen}");
        }
    }

    // FEN matches shakmaty
    let sk_fen = Fen::from_position(sk, EnPassantMode::Legal).to_string();
    assert_eq!(fen, sk_fen);
    assert!(pos.checkers().is_empty(), "{fen}");

    // Incremental state matches fresh parse
    let fresh = Position::from_fen(&fen, false).unwrap();
    let (st, new) = (pos.state(), fresh.state());
    assert_eq!(
        (
            st.key,
            st.pawn_key,
            st.minor_piece_key,
            st.non_pawn_key,
            st.material_key,
            st.non_pawn_material,
            st.piece_placement,
        ),
        (
            new.key,
            new.pawn_key,
            new.minor_piece_key,
            new.non_pawn_key,
            new.material_key,
            new.non_pawn_material,
            new.piece_placement,
        ),
        "{fen}"
    );
    assert_eq!(pos.check(), Ok(()), "{fen}");
}

#[test]
fn random_antichess_games_match_shakmaty_and_incremental_threats() {
    let mut rng = Prng::new(0xA471_C4E5_5123_4567);

    for (start_idx, &start_fen) in STARTS.iter().enumerate() {
        for game in 0..8 {
            let mut pos = Position::from_fen(start_fen, false).unwrap();
            let mut sk = shakmaty_position(start_fen);
            let mut cur_threats = all_threats(&pos);
            let mut history: Vec<(Move, String, u64)> = Vec::new();

            check_position(&mut pos, &sk);

            for _ply in 0..80 {
                let moves = MoveList::new::<ALL>(&pos);
                if moves.is_empty() {
                    assert!(sk.is_game_over(), "start {start_idx} game {game}");
                    break;
                }

                // Pick a random legal move
                let idx = (rng.rand64() as usize) % moves.len();
                let mv = moves.as_slice()[idx];
                let uci_str = move_to_uci(mv, false);
                let prev_fen = pos.fen();
                let prev_key = pos.key();

                // Apply on shakmaty
                let sk_mv = sk
                    .legal_moves()
                    .into_iter()
                    .find(|m| m.to_uci(CastlingMode::Standard).to_string() == uci_str)
                    .unwrap_or_else(|| panic!("shakmaty missing {uci_str} in {prev_fen}"));
                sk = sk.play(sk_mv).unwrap();

                // Apply on varifish and verify incremental DirtyThreats
                let mut dirties = Dirties::default();
                pos.make_move(mv, &mut dirties);

                for &dt in dirties.dirty_threats.iter() {
                    let key = DirtyThreat::new(
                        dt.pc(),
                        dt.threatened_pc(),
                        dt.pc_sq(),
                        dt.threatened_sq(),
                        false,
                    )
                    .raw();
                    let delta = if dt.add() { 1 } else { -1 };
                    let entry = cur_threats.entry(key).or_insert(0);
                    *entry += delta;
                    if *entry == 0 {
                        cur_threats.remove(&key);
                    }
                }

                let expected_threats = all_threats(&pos);
                assert_eq!(
                    cur_threats, expected_threats,
                    "DirtyThreats mismatch after {uci_str} from {prev_fen}"
                );

                history.push((mv, prev_fen, prev_key));
                check_position(&mut pos, &sk);
            }

            // Unwind the entire game and verify undo_move restores exact FEN & key
            while let Some((mv, prev_fen, prev_key)) = history.pop() {
                pos.undo_move(mv);
                assert_eq!(pos.fen(), prev_fen);
                assert_eq!(pos.key(), prev_key);
                assert_eq!(pos.check(), Ok(()));
            }
        }
    }
}

#[test]
fn antichess_perft_matches_shakmaty() {
    let cases = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
            4,
            153_299,
        ),
        (
            "rnbqkbnr/p1pppppp/8/1p6/8/4P3/PPPP1PPP/RNBQKBNR w - - 0 2",
            4,
            74,
        ),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", 4, 109),
        ("3r1n2/4P3/8/8/8/8/3p4/2B1R3 w - - 0 1", 3, 470),
        ("2k1k3/1p6/8/3K4/8/8/6P1/3K1K2 w - - 0 25", 4, 39_982),
        ("2b2b2/pp2pppp/8/2n1N3/8/8/PP2PPPP/2B2B2 w - - 0 10", 3, 373),
    ];

    for (fen, depth, expected_nodes) in cases {
        let sk = shakmaty_position(fen);
        let expected = sk_perft(&sk, depth);
        assert_eq!(expected, expected_nodes, "shakmaty perft({depth}) on {fen}");
        let got = perft::<Antichess>(fen, depth as i32, false, |_, _| {}).unwrap();
        assert_eq!(got, expected, "perft({depth}) mismatch on {fen}");
    }
}

/// The bootstrap classical evaluation is the same for both colours. (The
/// network's incremental updates are tested with random weights in
/// `nnue::antichess`'s unit tests: a zeroed network evaluates every position
/// alike.)
#[test]
fn antichess_bootstrap_evaluation_is_color_symmetric() {
    let empty_nets = Networks::default();
    let mut dummy_acc = AntichessAccumulators::new();
    let mut dummy_caches = AntichessCaches::new(None);
    for &start_fen in &STARTS {
        let mut pos = Position::from_fen(start_fen, false).unwrap();
        let v1 = Antichess::evaluate(
            &empty_nets,
            &pos,
            &mut dummy_acc,
            &mut dummy_caches,
            UseNnue::True,
        );
        pos.flip().unwrap();
        let v2 = Antichess::evaluate(
            &empty_nets,
            &pos,
            &mut dummy_acc,
            &mut dummy_caches,
            UseNnue::True,
        );
        // Same side-to-move score after flipping colors and ranks!
        assert_eq!(v1, v2, "bootstrap eval asymmetric for {start_fen}");
    }
}

#[test]
fn antichess_rule_edge_cases() {
    // 1. Stalemate overrides 50-move draw (`is_draw` must return false when stalemated).
    let stalemated = Position::from_fen("8/8/8/8/8/p7/P7/8 w - - 100 60", false).unwrap();
    assert!(!stalemated.antichess_has_legal_move());
    assert!(!stalemated.is_draw(1));

    // Non-stalemated at rule50 == 100 IS a draw.
    let drawn = Position::from_fen("8/8/8/8/p7/8/P7/8 w - - 100 60", false).unwrap();
    assert!(drawn.antichess_has_legal_move());
    assert!(drawn.is_draw(1));

    // 2. Mandatory capture blocks quiet repeating moves (`upcoming_repetition` must be false).
    let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1", false).unwrap();
    let mut dirties = Dirties::default();
    // Play a reversible loop without captures: Ke1-d1, Ke8-d8, Kd1-e1, Kd8-e8
    for uci in ["e1d1", "e8d8", "d1e1", "d8e8"] {
        let m = to_move(&pos, uci);
        pos.make_move(m, &mut dirties);
    }
    assert!(pos.is_repetition(5));

    // Now place a mandatory capture on a position that could otherwise repeat:
    let mut cap_pos = Position::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", false).unwrap();
    assert!(cap_pos.antichess_has_captures());
    assert!(!cap_pos.upcoming_repetition(1));
    // King move is illegal because exd5 is mandatory
    assert!(!cap_pos.legal(Move::new(Square::E1, Square::D1)));
    assert!(cap_pos.legal(Move::new(Square::E4, Square::D5)));
    let _ = &mut cap_pos;

    // 3. Reject corrupt / illegal moves in `legal()`.
    let start = Position::from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
        false,
    )
    .unwrap();
    assert!(!start.legal(Move::none()));
    assert!(!start.legal(Move::null()));
    assert!(!start.legal(Move::make_drop(Square::E4, PAWN)));
    assert!(!start.legal(Move::make(CASTLING, Square::E1, Square::H1, KNIGHT)));
    assert!(!start.legal(Move::make_king_promotion(Square::E2, Square::E4)));
}
