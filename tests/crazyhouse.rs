//! varifish's Crazyhouse positions checked along random games against
//! shakmaty's Crazyhouse: legal moves (including drops and drop evasions),
//! FENs (including promoted `~` markers and pockets), pockets, checkers,
//! incremental state (keys, threats, undoing, null moves), perft counts,
//! rule edge cases (repetitions across captures/drops, no 50-move draw,
//! promoted piece demotion, drop SEE), and UCI search with drops.

use std::collections::BTreeMap;

use shakmaty::fen::Fen;
use shakmaty::variant::Crazyhouse as SkCrazyhouse;
use shakmaty::{
    CastlingMode, Color as SkColor, EnPassantMode, Position as _, Role as SkRole, perft as sk_perft,
};
use varifish::attacks::piece_attacks_bb;
use varifish::misc::Prng;
use varifish::movegen::{ALL, CAPTURES, EVASIONS, MoveList, QUIETS};
use varifish::perft::perft;
use varifish::position::START_FEN;
use varifish::types::*;
use varifish::uci::{move_to_uci, to_move};
use varifish::variant::Crazyhouse;

type Position = varifish::position::Position<Crazyhouse>;

const STARTS: [(&str, bool); 12] = [
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
        false,
    ),
    (
        "r1bqk2r/pppp1ppp/2n1p3/4P3/1b1Pn3/2NB1N2/PPP2PPP/R1BQK2R[] b KQkq - 0 1",
        false,
    ),
    ("2k5/8/8/8/8/8/8/4K3[QRBNPqrbnp] w - - 0 1", false),
    (
        "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1[NPp] w - - 1 16",
        false,
    ),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R[p] w KQ - 1 8",
        false,
    ),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/4P3/2N2Q~2/PPPBBPPP/R3K2R[Q] w KQkq - 0 10",
        false,
    ),
    // En passant with active pockets
    ("8/8/3p4/KPp4r/1R3p1k/8/4P1P1/8[Nn] w - c6 0 2", false),
    ("8/2p5/8/KP1p3r/1R2Pp1k/8/6P1/8[Qp] b - e3 0 2", false),
    // Promoted pieces on both sides
    (
        "3rk2r/1p3ppp/p1n5/2b1q~3/4P3/2N2N2/PP3PPP/R1BQ~1RK1[Pp] w k - 0 12",
        false,
    ),
    // Chess960 Crazyhouse
    (
        "1rqbkrbn/1ppppp1p/1n6/2N3p1/8/2P5/PP1PPPP1/1RQBKRB1[Np] w FBfb - 0 9",
        true,
    ),
    (
        "rbb1n1kr/pp2p1pp/6n1/2pp1p2/2P5/8/BP1PPPP1/R1BQNNKR[Qp] w HAha - 0 9",
        true,
    ),
    (
        "4rrb1/1kp3b1/1p1p4/pP1Pn2p/5p2/1PR2P2/2P1NB1P/2KR1B2[Qnp] w D - 0 21",
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

fn shakmaty_position(fen: &str, chess960: bool) -> SkCrazyhouse {
    fen.parse::<Fen>()
        .unwrap_or_else(|e| panic!("{fen}: {e}"))
        .into_position(mode(chess960))
        .unwrap_or_else(|e| panic!("{fen}: {e}"))
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

/// Normalizes pocket letter order inside `[...]` so varifish's Fairy-Stockfish
/// order (`QRBNPqrbnp`) can be compared field-for-field against shakmaty's
/// (`PNBRQpnbrq`).
fn normalize_pocket_order(fen: &str) -> String {
    let Some(open) = fen.find('[') else {
        return fen.to_string();
    };
    let Some(close) = fen[open..].find(']').map(|i| open + i) else {
        return fen.to_string();
    };
    let mut pocket: Vec<u8> = fen.as_bytes()[open + 1..close].to_vec();
    pocket.sort_unstable();
    format!(
        "{}[{}]{}",
        &fen[..open],
        std::str::from_utf8(&pocket).unwrap(),
        &fen[close + 1..]
    )
}

fn check_position(pos: &mut Position, sk: &SkCrazyhouse, chess960: bool) {
    let fen = pos.fen();
    let m = mode(chess960);

    // Pockets match shakmaty
    let sk_pockets = sk.pockets().expect("Crazyhouse has pockets");
    for (c, sk_c) in [(WHITE, SkColor::White), (BLACK, SkColor::Black)] {
        for (pt, sk_role) in [
            (PAWN, SkRole::Pawn),
            (KNIGHT, SkRole::Knight),
            (BISHOP, SkRole::Bishop),
            (ROOK, SkRole::Rook),
            (QUEEN, SkRole::Queen),
        ] {
            assert_eq!(
                pos.hand_count(c, pt),
                sk_pockets[sk_c][sk_role],
                "{fen} {c:?} {pt:?}"
            );
        }
    }

    // Promoted bitboard matches shakmaty
    assert_eq!(pos.state().extra.promoted.0, sk.promoted().0, "{fen}");

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

    // FEN matches shakmaty (ignoring Chess960 castling file spelling and pocket sort order)
    let norm_ours = normalize_pocket_order(&fen);
    let norm_sk = normalize_pocket_order(&Fen::from_position(sk, EnPassantMode::Legal).to_string());
    if chess960 {
        let fields = |f: &str| -> Vec<String> {
            let mut v: Vec<String> = f.split(' ').map(str::to_string).collect();
            v.remove(2);
            v
        };
        assert_eq!(fields(&norm_ours), fields(&norm_sk), "{fen} vs {norm_sk}");
    } else {
        assert_eq!(norm_ours, norm_sk);
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

    // Null move preserves pockets and promoted squares and restores cleanly
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
    let mut rng = Prng::new(0x5A68_C43B);
    let mut dirties = Dirties::default();
    let mut saw_drop = false;
    let mut saw_drop_evasion = false;
    let mut saw_promoted_capture = false;

    for (start_fen, chess960) in STARTS {
        for _game in 0..8 {
            let mut pos = Position::from_fen(start_fen, chess960).unwrap();
            let mut sk = shakmaty_position(start_fen, chess960);

            for _ply in 0..50 {
                check_position(&mut pos, &sk, chess960);

                let moves = MoveList::new::<ALL>(&pos);
                if moves.is_empty() {
                    break;
                }

                let idx = (rng.rand64() as usize) % moves.len();
                let mv = moves.as_slice()[idx];
                assert!(pos.legal(mv));

                if mv.is_drop() {
                    saw_drop = true;
                    if pos.in_check() {
                        saw_drop_evasion = true;
                    }
                } else if pos.is_promoted(mv.to_sq()) {
                    saw_promoted_capture = true;
                }

                let before_fen = pos.fen();
                let before_key = pos.state().key;
                let before_extra = pos.state().extra;
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
                assert_eq!(pos.state().extra, before_extra);

                pos.make_move(mv, &mut dirties);
                sk = sk_next;
            }
        }
    }

    assert!(saw_drop, "random games should exercise drops");
    assert!(
        saw_drop_evasion,
        "random games should exercise drop evasions"
    );
    assert!(
        saw_promoted_capture,
        "random games should exercise captures of promoted pieces"
    );
}

#[test]
fn perft_matches_expected_and_shakmaty() {
    // Exact values, as Fairy-Stockfish counts them.
    let expected = [
        (START_FEN, false, 4, 197_281u64),
        (
            "2k5/8/8/8/8/8/8/4K3[QRBNPqrbnp] w - - 0 1",
            false,
            2,
            75_353,
        ),
        ("2k5/8/8/8/8/8/8/4K3[Qn] w - - 0 1", false, 3, 88_634),
        (
            "r1bqk2r/pppp1ppp/2n1p3/4P3/1b1Pn3/2NB1N2/PPP2PPP/R1BQK2R[] b KQkq - 0 1",
            false,
            3,
            58_057,
        ),
    ];

    for (fen, chess960, depth, nodes) in expected {
        let ours = perft::<Crazyhouse>(fen, depth, chess960, |_, _| {}).unwrap();
        assert_eq!(ours, nodes, "{fen} depth {depth}");
        let sk = shakmaty_position(fen, chess960);
        assert_eq!(ours, sk_perft(&sk, depth as u32), "{fen} depth {depth}");
    }

    // Extra promoted, drop-evasion, and Chess960 positions vs shakmaty.
    let extra = [
        ("4k3/8/8/8/4r3/8/8/4K3[PNBRQ] w - - 0 1", false, 3),
        (
            "3rk2r/1p3ppp/p1n5/2b1q~3/4P3/2N2N2/PP3PPP/R1BQ~1RK1[Pp] w k - 0 12",
            false,
            2,
        ),
        (
            "1rqbkrbn/1ppppp1p/1n6/2N3p1/8/2P5/PP1PPPP1/1RQBKRB1[Np] w FBfb - 0 9",
            true,
            2,
        ),
    ];

    for (fen, chess960, depth) in extra {
        let ours = perft::<Crazyhouse>(fen, depth, chess960, |_, _| {}).unwrap();
        let sk = shakmaty_position(fen, chess960);
        assert_eq!(ours, sk_perft(&sk, depth as u32), "{fen} depth {depth}");
    }
}

#[test]
fn fen_formats_promoted_and_flip() {
    // Plain chess FEN parses with empty pockets and serializes with [].
    let start = Position::from_fen(START_FEN, false).unwrap();
    assert_eq!(
        start.fen(),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
    );

    // Lichess 9th-slash pocket notation matches bracket notation.
    let bracket = Position::from_fen("4k3/8/8/8/8/8/8/4K3[QNRpp] w - - 0 1", false).unwrap();
    let slash = Position::from_fen("4k3/8/8/8/8/8/8/4K3/QNRpp w - - 0 1", false).unwrap();
    assert_eq!(bracket.fen(), slash.fen());
    assert_eq!(bracket.state().key, slash.state().key);
    assert_eq!(bracket.state().extra, slash.state().extra);

    // Flip swaps pocket colors and mirrors promoted squares symmetrically.
    let orig = Position::from_fen("4k3/8/1Q~6/8/8/5n~2/8/4K3[QBp] w - - 0 1", false).unwrap();
    let mut flipped = orig.clone();
    flipped.flip().unwrap();
    assert_eq!(flipped.hand_count(WHITE, PAWN), 1);
    assert_eq!(flipped.hand_count(BLACK, QUEEN), 1);
    assert_eq!(flipped.hand_count(BLACK, BISHOP), 1);
    assert!(flipped.is_promoted(Square::B3));
    assert!(flipped.is_promoted(Square::F6));
    assert!(!flipped.is_promoted(Square::B6));
    flipped.flip().unwrap();
    assert_eq!(flipped.fen(), orig.fen());
    assert_eq!(flipped.state().key, orig.state().key);

    // More than 8 pawns or 16 pieces for one side is legal up to 32 board pieces.
    let many_pawns =
        Position::from_fen("4k3/8/PPPPPPPP/PPPP4/8/8/8/4K3[] w - - 0 1", false).unwrap();
    assert_eq!(many_pawns.count(WHITE, PAWN), 12);

    // Invalid Crazyhouse FENs rejected.
    assert!(Position::from_fen("4k3/8/8/8/8/8/8/4K3[K] w - - 0 1", false).is_err());
    assert!(Position::from_fen("4k3/8/8/8/8/8/P~7/4K3[] w - - 0 1", false).is_err());
    assert!(Position::from_fen("4k~3/8/8/8/8/8/8/4K3[] w - - 0 1", false).is_err());
    assert!(Position::from_fen("4k3/8/8/8/8/8/8/4K3[Q w - - 0 1", false).is_err());
}

/// Drops give a side more than 8 pawns, and so more than 8 pushes or
/// captures in one direction, which Ice Lake's AVX-512 build once stored in
/// 8 lanes, its ninth move garbage: `d1a1` for `c5c6` in the first position.
#[test]
fn more_than_eight_pawns_generate_every_pawn_move() {
    for fen in [
        // 9 pawns, 9 single pushes
        "r2qkbnr/pp2nppp/4p3/1PPp4/5Pb1/4PN2/PPP3PP/RNBQKB1R[] w KQkq - 1 7",
        // Two more from games that met it: d6d7 and g7g6
        "r1Nbb2r/1p2Nppk/3Ppn1p/2Pp3B/5B2/4P3/PPP2PPP/R2Q1RK1[Qnp] w - - 2 23",
        "r1bqkb1r/ppp1ppp1/7p/3pP3/6p1/8/PPP1BKPP/RNBQN2R[NNp] b kq - 0 9",
        // 10 pawns, 10 captures towards the h-file
        "4k3/8/1nbrqp2/PPPPP3/1ppppp2/PPPPP3/8/4K3[] w - - 0 1",
        // 14 pawns, 14 single pushes
        "4k3/8/8/8/PPPPPP2/8/PPPPPPPP/4K3[pp] w - - 0 1",
        // 16 pawns, 16 single pushes, a store's every lane
        "4k3/8/8/8/PPPPPPPP/8/PPPPPPPP/4K3[] w - - 0 1",
    ] {
        // And with the colours flipped, for Black's pawns
        let mut flipped = Position::from_fen(fen, false).unwrap();
        flipped.flip().unwrap();
        for mut pos in [Position::from_fen(fen, false).unwrap(), flipped] {
            let sk = shakmaty_position(&pos.fen(), false);
            check_position(&mut pos, &sk, false);
        }
    }

    // No game has more than 16 pawns, counting those in hand and the
    // promoted pieces, which captures turn back into pawns.
    for fen in [
        "4k3/8/PPPPPPPP/PPPPPPPP/P7/8/8/4K3[] w - - 0 1",
        "4k3/8/8/8/PPPPPPPP/8/PPPPPPPP/4K3[p] w - - 0 1",
        "4k3/8/8/8/PPPPPPPP/8/PPPPPPPP/3Q~K3[] w - - 0 1",
    ] {
        assert!(Position::from_fen(fen, false).is_err(), "{fen}");
    }
}

#[test]
fn crazyhouse_rules_edge_cases() {
    let mut dirties = Dirties::default();

    // 1. En passant captures enemy pawn into mover's pocket.
    let mut pos = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3[] w - d6 0 1", false).unwrap();
    let ep = Move::make(EN_PASSANT, Square::E5, Square::D6, KNIGHT);
    assert!(pos.legal(ep));
    pos.make_move(ep, &mut dirties);
    assert_eq!(pos.hand_count(WHITE, PAWN), 1);
    assert_eq!(pos.hand_count(BLACK, PAWN), 0);

    // 2. Capturing a promoted queen puts a pawn in pocket; real queen puts a queen.
    let mut pos_prom = Position::from_fen("4k3/8/8/3q~4/3R4/8/8/4K3[] w - - 0 1", false).unwrap();
    let mut pos_real = Position::from_fen("4k3/8/8/3q4/3R4/8/8/4K3[] w - - 0 1", false).unwrap();
    assert_ne!(pos_prom.state().key, pos_real.state().key);
    let cap = Move::new(Square::D4, Square::D5);
    pos_prom.make_move(cap, &mut dirties);
    pos_real.make_move(cap, &mut dirties);
    assert_eq!(pos_prom.hand_count(WHITE, PAWN), 1);
    assert_eq!(pos_prom.hand_count(WHITE, QUEEN), 0);
    assert!(!pos_prom.is_promoted(Square::D5));
    assert_eq!(pos_real.hand_count(WHITE, QUEEN), 1);
    assert_eq!(pos_real.hand_count(WHITE, PAWN), 0);

    // 3. Pawn drops forbidden on rank 1 and rank 8; invalid/corrupt drops rejected.
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3[P] w - - 0 1", false).unwrap();
    assert!(!pos.legal(Move::make_drop(Square::A1, PAWN)));
    assert!(!pos.legal(Move::make_drop(Square::H8, PAWN)));
    assert!(pos.legal(Move::make_drop(Square::E4, PAWN)));
    assert!(!pos.legal(Move::make_drop(Square::E4, QUEEN))); // not in hand
    assert!(!pos.legal(Move::make_drop(Square::E1, PAWN))); // occupied
    let corrupt_king_drop = Move((1 << 12) | ((KING.0 as u16) << 6) | Square::E4.0 as u16);
    assert!(!pos.legal(corrupt_king_drop)); // invalid piece type

    // 4. Double check allows no drop evasions.
    let pos = Position::from_fen("4k3/8/8/8/4r3/5n2/8/4K3[QRBNP] w - - 0 1", false).unwrap();
    assert!(pos.checkers().more_than_one());
    let evasions = MoveList::new::<ALL>(&pos);
    assert!(evasions.iter().all(|m| !m.is_drop()));
    assert!(!pos.legal(Move::make_drop(Square::E2, QUEEN)));

    // 5. No 50-move draw in Crazyhouse.
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K2R[] w - - 100 60", false).unwrap();
    assert!(!pos.is_draw(1));

    // 6. Repetition across a capture + drop cycle (rule50 resets to 0 mid-cycle!).
    let mut pos = Position::from_fen("4k3/5r2/8/5N2/3n4/8/3R4/4K3[] w - - 0 1", false).unwrap();
    let start_key = pos.state().key;
    // 1. Rxd4 (W gets N, rule50=0) Rxf5 (B gets N)
    pos.make_move(Move::new(Square::D2, Square::D4), &mut dirties);
    assert_eq!(pos.rule50_count(), 0);
    pos.make_move(Move::new(Square::F7, Square::F5), &mut dirties);
    // 2. Rd2 Rf7
    pos.make_move(Move::new(Square::D4, Square::D2), &mut dirties);
    pos.make_move(Move::new(Square::F5, Square::F7), &mut dirties);
    // 3. N@f5 N@d4 -> exact repeat of start position across rule50 reset!
    pos.make_move(Move::make_drop(Square::F5, KNIGHT), &mut dirties);
    pos.make_move(Move::make_drop(Square::D4, KNIGHT), &mut dirties);
    assert_eq!(pos.state().key, start_key);
    assert!(pos.is_repetition(10));
    assert!(pos.is_draw(10));

    // 7. SEE on drops: undefended drop passes see_ge(0); dropping queen onto pawn-defended square fails.
    let pos = Position::from_fen("4k3/3p4/8/8/8/8/8/4K3[Q] w - - 0 1", false).unwrap();
    assert!(pos.see_ge(Move::make_drop(Square::E4, QUEEN), 0));
    assert!(!pos.see_ge(Move::make_drop(Square::E6, QUEEN), 0));
}

#[test]
fn uci_drop_parsing_is_case_insensitive() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3[PQ] w - - 0 1", false).unwrap();
    let upper = to_move(&pos, "P@e4");
    let lower = to_move(&pos, "p@e4");
    assert_eq!(upper, Move::make_drop(Square::E4, PAWN));
    assert_eq!(lower, upper);
    assert_eq!(move_to_uci(upper, false), "P@e4");
}
