//! varifish's Three-check positions checked along random games against
//! shakmaty's ThreeCheck: legal moves, FENs, remaining checks, checkers,
//! incremental state (keys, threats, undoing, null moves), perft counts,
//! rule edge cases, and dedicated Three-check network evaluation.

use std::collections::BTreeMap;

use shakmaty::fen::Fen;
use shakmaty::variant::ThreeCheck as SkThreeCheck;
use shakmaty::{CastlingMode, Color as SkColor, EnPassantMode, Position as _, perft as sk_perft};
use varifish::attacks::piece_attacks_bb;
use varifish::misc::Prng;
use varifish::movegen::{ALL, CAPTURES, EVASIONS, MoveList, QUIETS};
use varifish::perft::perft;
use varifish::position::START_FEN;
use varifish::types::*;
use varifish::uci::move_to_uci;
use varifish::variant::ThreeCheck;

type Position = varifish::position::Position<ThreeCheck>;

const STARTS: [(&str, bool); 12] = [
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1",
        false,
    ),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 2+2 0 10",
        false,
    ),
    (
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1+2 4 4",
        false,
    ),
    (
        "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1+1 1 16",
        false,
    ),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 2+1 0 11", false),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1+3 1 8",
        false,
    ),
    ("r7/4p3/5p1q/3P4/4pQ2/4pP2/6pp/R3K1kr w Q - 2+1 1 3", false),
    // En passant along the king's rank and diagonals
    ("8/8/3p4/KPp4r/1R3p1k/8/4P1P1/8 w - c6 1+2 0 2", false),
    ("8/2p5/8/KP1p3r/1R2Pp1k/8/6P1/8 b - e3 2+1 0 2", false),
    // Chess960 Three-check
    (
        "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 2+3 0 9",
        true,
    ),
    (
        "rbbqn1kr/pp2p1pp/6n1/2pp1p2/2P4P/P7/BP1PPPP1/R1BQNNKR w HAha - 1+2 0 9",
        true,
    ),
    (
        "4rrb1/1kp3b1/1p1p4/pP1Pn2p/5p2/1PR2P2/2P1NB1P/2KR1B2 w D - 1+1 0 21",
        true,
    ),
];

fn mode(chess960: bool) -> CastlingMode {
    if chess960 {
        CastlingMode::Chess960
    } else {
        CastlingMode::Standard
    }
}

fn shakmaty_position(fen: &str, chess960: bool) -> SkThreeCheck {
    fen.parse::<Fen>()
        .unwrap()
        .into_position(mode(chess960))
        .unwrap()
}

fn threats(pos: &Position) -> BTreeMap<u32, i32> {
    let occupied = pos.pieces();
    let not_kings = occupied & !pos.pieces_pt(KING);
    let mut out = BTreeMap::new();
    for s in not_kings {
        let pc = pos.piece_on(s);
        let targets = match pc.type_of() {
            PAWN => pos.pieces_pt2(KNIGHT, ROOK),
            BISHOP | ROOK => pos.pieces_pt2(PAWN, KNIGHT) | pos.pieces_pt2(BISHOP, ROOK),
            _ => not_kings,
        };
        for t in piece_attacks_bb(pc, s, occupied) & targets {
            out.insert(DirtyThreat::new(pc, pos.piece_on(t), s, t, false).raw(), 1);
        }
    }
    out
}

fn check_position(pos: &mut Position, sk: &SkThreeCheck, chess960: bool) {
    let fen = pos.fen();
    let m = mode(chess960);

    // Remaining checks match shakmaty
    let sk_rem = sk.remaining_checks().expect("ThreeCheck has checks");
    assert_eq!(
        pos.checks_remaining(WHITE),
        u8::from(sk_rem[SkColor::White]),
        "{fen}"
    );
    assert_eq!(
        pos.checks_remaining(BLACK),
        u8::from(sk_rem[SkColor::Black]),
        "{fen}"
    );
    assert_eq!(pos.is_three_check_over(), sk.is_variant_end(), "{fen}");

    // Legal moves match shakmaty
    let moves = MoveList::new::<ALL>(pos);
    let mut ours: Vec<String> = moves.iter().map(|&mv| move_to_uci(mv, chess960)).collect();
    let mut theirs: Vec<String> = sk
        .legal_moves()
        .iter()
        .map(|mv| mv.to_uci(m).to_string())
        .collect();
    ours.sort();
    theirs.sort();
    assert_eq!(ours, theirs, "{fen}");

    // FEN matches shakmaty (ignoring Chess960 castling file spelling)
    let sk_fen = Fen::from_position(sk, EnPassantMode::Legal).to_string();
    if chess960 {
        let fields = |f: &str| -> Vec<String> {
            let mut v: Vec<String> = f.split(' ').map(str::to_string).collect();
            v.remove(2);
            v
        };
        assert_eq!(fields(&fen), fields(&sk_fen), "{fen} vs {sk_fen}");
    } else {
        assert_eq!(fen, sk_fen);
    }
    assert_eq!(pos.checkers().0, sk.checkers().0, "{fen}");

    // Incremental state matches fresh parse
    let fresh = Position::from_fen(&fen, chess960).unwrap();
    let (st, new) = (pos.state(), fresh.state());
    assert_eq!(
        (
            st.key,
            st.pawn_key,
            st.material_key,
            st.minor_piece_key,
            st.non_pawn_key,
            st.extra,
        ),
        (
            new.key,
            new.pawn_key,
            new.material_key,
            new.minor_piece_key,
            new.non_pawn_key,
            new.extra,
        ),
        "{fen}"
    );
    assert_eq!(
        (st.rule50, st.ep_square, st.checkers_bb, st.castling_rights),
        (
            new.rule50,
            new.ep_square,
            new.checkers_bb,
            new.castling_rights
        ),
        "{fen}"
    );
    assert_eq!(st.piece_placement, new.piece_placement, "{fen}");
    assert_eq!((pos.key(), pos.game_ply()), (fresh.key(), fresh.game_ply()));

    // Move generation kinds equal ALL in the same order
    let mut kinds = Vec::new();
    if pos.checkers().any() {
        kinds.extend(MoveList::new::<EVASIONS>(pos).iter().copied());
    } else {
        kinds.extend(MoveList::new::<CAPTURES>(pos).iter().copied());
        kinds.extend(MoveList::new::<QUIETS>(pos).iter().copied());
    }
    assert_eq!(kinds.as_slice(), moves.as_slice(), "{fen}");

    // Null move preserves remaining checks and restores cleanly
    if !pos.in_check() {
        let before_key = st.key;
        let before_extra = st.extra;
        pos.do_null_move();
        assert_eq!(pos.state().extra, before_extra, "{fen}");
        assert_ne!(pos.state().key, before_key, "{fen}");
        pos.undo_null_move();
        assert_eq!(pos.state().key, before_key, "{fen}");
        assert_eq!(pos.state().extra, before_extra, "{fen}");
    }
}

#[test]
fn random_games_match_shakmaty_and_incremental_state() {
    let mut rng = Prng::new(0x3C4E_C43B);
    let mut dirties = Dirties::default();
    let mut saw_third_check_end = false;

    for (start_fen, chess960) in STARTS {
        for _game in 0..8 {
            let mut pos = Position::from_fen(start_fen, chess960).unwrap();
            let mut sk = shakmaty_position(start_fen, chess960);

            for _ply in 0..60 {
                check_position(&mut pos, &sk, chess960);

                let moves = MoveList::new::<ALL>(&pos);
                if moves.is_empty() {
                    if pos.is_three_check_over() {
                        saw_third_check_end = true;
                        assert!(pos.is_three_check_loss());
                    }
                    break;
                }

                // Verify gives_check, legal, threats, and undo on a sample move
                let idx = (rng.rand64() as usize) % moves.len();
                let mv = moves.as_slice()[idx];
                assert!(pos.legal(mv));

                let before_fen = pos.fen();
                let before_key = pos.state().key;
                let before_threats = threats(&pos);
                let gives_check = pos.gives_check(mv);

                let uci = move_to_uci(mv, chess960);
                let sk_mv = sk
                    .legal_moves()
                    .into_iter()
                    .find(|m| m.to_uci(mode(chess960)).to_string() == uci)
                    .unwrap();
                let sk_next = sk.clone().play(sk_mv).unwrap();

                pos.make_move(mv, &mut dirties);
                assert_eq!(pos.in_check(), gives_check, "{before_fen} {uci}");
                assert_eq!(pos.in_check(), sk_next.is_check(), "{before_fen} {uci}");

                // Threats updated incrementally match threats recomputed from scratch
                let mut applied = before_threats;
                for &dt in dirties.dirty_threats.iter() {
                    let raw = DirtyThreat::new(
                        dt.pc(),
                        dt.threatened_pc(),
                        dt.pc_sq(),
                        dt.threatened_sq(),
                        false,
                    )
                    .raw();
                    let delta = if dt.add() { 1 } else { -1 };
                    let entry = applied.entry(raw).or_insert(0);
                    *entry += delta;
                    if *entry == 0 {
                        applied.remove(&raw);
                    }
                }
                assert_eq!(applied, threats(&pos), "{before_fen} {uci}");

                // Undo and redo to check exact restoration
                pos.undo_move(mv);
                assert_eq!(pos.fen(), before_fen);
                assert_eq!(pos.state().key, before_key);

                pos.make_move(mv, &mut dirties);
                sk = sk_next;
            }
        }
    }

    assert!(
        saw_third_check_end,
        "random games should hit 3rd-check terminations"
    );
}

#[test]
fn perft_matches_shakmaty() {
    let cases = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1",
            false,
            4,
        ),
        (
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1+1 4 4",
            false,
            3,
        ),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 1+2 0 10",
            false,
            3,
        ),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1+1 1 8",
            false,
            3,
        ),
        (
            "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 1+1 0 9",
            true,
            3,
        ),
    ];

    for (fen, chess960, depth) in cases {
        let ours = perft::<ThreeCheck>(fen, depth, chess960, |_, _| {}).unwrap();
        let sk = shakmaty_position(fen, chess960);
        let theirs = sk_perft(&sk, depth as u32);
        assert_eq!(ours, theirs, "{fen} depth {depth}");
    }
}

#[test]
fn double_check_counts_as_one_check() {
    // Nc7 checks the king on e8 with the knight and uncovers the rook on
    // e1: a double check, which counts once.
    let fen = "4k3/8/4N3/8/8/8/8/4R1K1 w - - 2+3 0 1";
    let mut pos = Position::from_fen(fen, false).unwrap();
    let mv = Move::new(Square::E6, Square::C7);
    assert!(pos.legal(mv));
    assert!(pos.gives_check(mv));
    let mut dirties = Dirties::default();
    pos.make_move(mv, &mut dirties);
    assert!(
        pos.checkers().more_than_one(),
        "Nc7+ must be a double check"
    );
    assert_eq!(
        pos.checks_remaining(WHITE),
        1,
        "double check counts as 1 check"
    );
    assert_eq!(pos.checks_remaining(BLACK), 3);
    assert!(!pos.is_three_check_over());
}

#[test]
fn fen_formats_and_flip() {
    // Default without check field parses as 3+3
    let start = Position::from_fen(START_FEN, false).unwrap();
    assert_eq!(start.checks_remaining(WHITE), 3);
    assert_eq!(start.checks_remaining(BLACK), 3);
    assert_eq!(
        start.fen(),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1"
    );

    // XBoard +W+B checks-given notation parses identically to Lichess W+B
    let xboard = Position::from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 +2+1",
        false,
    )
    .unwrap();
    assert_eq!(xboard.checks_remaining(WHITE), 1);
    assert_eq!(xboard.checks_remaining(BLACK), 2);
    assert_eq!(
        xboard.fen(),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 1+2 0 1"
    );

    // Flip swaps remaining checks
    let mut flipped = xboard.clone();
    flipped.flip().unwrap();
    assert_eq!(flipped.checks_remaining(WHITE), 2);
    assert_eq!(flipped.checks_remaining(BLACK), 1);

    // Invalid check counts rejected
    assert!(Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0+0 0 1", false).is_err());
    assert!(Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 4+3 0 1", false).is_err());
    assert!(Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 3+x 0 1", false).is_err());
}

#[test]
fn third_check_on_move_50_is_not_a_draw() {
    // Halfmove clock at 99; White gives its 3rd check -> halfmove becomes 100,
    // but Black has 0 legal moves and has already lost by 3rd check.
    let fen = "4k3/8/8/8/8/8/3R4/4K3 w - - 1+3 99 50";
    let mut pos = Position::from_fen(fen, false).unwrap();
    let mv = Move::new(Square::D2, Square::E2);
    assert!(pos.gives_check(mv));
    let mut dirties = Dirties::default();
    pos.make_move(mv, &mut dirties);
    assert_eq!(pos.rule50_count(), 100);
    assert!(pos.is_three_check_loss());
    assert!(MoveList::new::<ALL>(&pos).is_empty());
    assert!(!pos.is_draw(1));
}

#[test]
fn double_pawn_push_giving_third_check_clears_ep_square() {
    // Black plays c7-c5+, delivering its 3rd check while a White pawn stands
    // on b5: since the game is over, no legal en passant capture exists and
    // the FEN must write `-`, not `c6`.
    let fen = "7r/2p5/3p4/1P6/1K3p1k/8/1R2P1P1/8 b - - 2+1 0 12";
    let mut pos = Position::from_fen(fen, false).unwrap();
    let mv = Move::new(Square::C7, Square::C5);
    assert!(pos.legal(mv));
    assert!(pos.gives_check(mv));
    let mut dirties = Dirties::default();
    pos.make_move(mv, &mut dirties);
    assert_eq!(pos.checks_remaining(BLACK), 0);
    assert_eq!(pos.ep_square(), Square::NONE);
    assert_eq!(pos.fen(), "7r/8/3p4/1Pp5/1K3p1k/8/1R2P1P1/8 w - - 2+0 0 13");

    // Parsing a terminal FEN that includes an ep square must clear it too.
    let parsed =
        Position::from_fen("7r/8/3p4/1Pp5/1K3p1k/8/1R2P1P1/8 w - c6 2+0 0 13", false).unwrap();
    assert_eq!(parsed.ep_square(), Square::NONE);
}

#[test]
fn upcoming_repetition_respects_three_check_counts() {
    // Non-checking cycle repeats cleanly.
    let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w - - 3+3 0 1", false).unwrap();
    let mut dirties = Dirties::default();
    let play = |pos: &mut Position, from: Square, to: Square, dirties: &mut Dirties| {
        let mv = Move::new(from, to);
        assert!(pos.legal(mv));
        pos.make_move(mv, dirties);
    };
    play(&mut pos, Square::H1, Square::G1, &mut dirties);
    play(&mut pos, Square::E8, Square::D8, &mut dirties);
    play(&mut pos, Square::G1, Square::H1, &mut dirties);
    assert!(pos.upcoming_repetition(4));
    play(&mut pos, Square::D8, Square::E8, &mut dirties);
    assert!(pos.is_repetition(5));

    // In-check position can never be repeated because entering check decrements
    // checks_remaining.
    let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/3R2K1 w - - 3+3 0 1", false).unwrap();
    // 1. Re1+ Kd8 2. Rd1 Ke8 — now Re1+ gives check again (2+3 -> 1+3), NOT a repetition!
    play(&mut pos, Square::D1, Square::E1, &mut dirties); // 2+3, Black in check
    play(&mut pos, Square::E8, Square::D8, &mut dirties);
    play(&mut pos, Square::E1, Square::D1, &mut dirties);
    play(&mut pos, Square::D8, Square::E8, &mut dirties);
    assert!(!pos.upcoming_repetition(5));
}
