//! varifish's positions checked along random games: legal moves, FENs and
//! checkers against shakmaty, and the incrementally updated state (keys,
//! threats for the network, undoing) against the same computed afresh.

use std::collections::BTreeMap;

use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, EnPassantMode, Position as _};
use varifish::attacks::piece_attacks_bb;
use varifish::misc::Prng;
use varifish::movegen::{ALL, CAPTURES, EVASIONS, MoveList, QUIETS};
use varifish::position::START_FEN;
use varifish::types::*;
use varifish::uci::move_to_uci;

type Position = varifish::position::Position<varifish::variant::Chess>;

const STARTS: [(&str, bool); 14] = [
    (START_FEN, false),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        false,
    ),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", false),
    (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        false,
    ),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        false,
    ),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        false,
    ),
    ("r7/4p3/5p1q/3P4/4pQ2/4pP2/6pp/R3K1kr w Q - 1 3", false),
    // En passant along the king's rank and diagonals
    ("8/8/3p4/KPp4r/1R3p1k/8/4P1P1/8 w - c6 0 2", false),
    ("8/2p5/8/KP1p3r/1R2Pp1k/8/6P1/8 b - e3 0 2", false),
    (
        "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 9",
        true,
    ),
    (
        "rbbqn1kr/pp2p1pp/6n1/2pp1p2/2P4P/P7/BP1PPPP1/R1BQNNKR w HAha - 0 9",
        true,
    ),
    (
        "4rrb1/1kp3b1/1p1p4/pP1Pn2p/5p2/1PR2P2/2P1NB1P/2KR1B2 w D - 0 21",
        true,
    ),
    (
        "1rkr3b/1ppn3p/3pB1n1/6q1/R2P4/4N1P1/1P5P/2KRQ1B1 b Dbd - 0 14",
        true,
    ),
    (
        "rr6/2kpp3/1ppnb1p1/p4q1p/P4P1P/1PNN2P1/2PP2Q1/1K2RR2 w E - 1 19",
        true,
    ),
];

fn shakmaty_position(fen: &str, chess960: bool) -> Chess {
    let mode = if chess960 {
        CastlingMode::Chess960
    } else {
        CastlingMode::Standard
    };
    fen.parse::<Fen>().unwrap().into_position(mode).unwrap()
}

/// The threats the network follows: of each piece but a king on each piece
/// but a king it attacks, where pawns threaten knights and rooks, bishops
/// and rooks all but queens, and knights and queens all.
fn threats(pos: &Position) -> BTreeMap<u32, i32> {
    let occupied = pos.pieces();
    let not_kings = occupied & !pos.pieces_pt(KING);
    let mut threats = BTreeMap::new();
    for s in not_kings {
        let pc = pos.piece_on(s);
        let targets = match pc.type_of() {
            PAWN => pos.pieces_pt2(KNIGHT, ROOK),
            BISHOP | ROOK => pos.pieces_pt2(PAWN, KNIGHT) | pos.pieces_pt2(BISHOP, ROOK),
            _ => not_kings,
        };
        for t in piece_attacks_bb(pc, s, occupied) & targets {
            threats.insert(DirtyThreat::new(pc, pos.piece_on(t), s, t, false).raw(), 1);
        }
    }
    threats
}

/// Checks a position and each of its moves, against shakmaty and against
/// the same position set afresh from its FEN.
fn check_position(pos: &mut Position, sk: &Chess, chess960: bool) {
    let fen = pos.fen();
    let mode = if chess960 {
        CastlingMode::Chess960
    } else {
        CastlingMode::Standard
    };

    // The legal moves, in any order
    let moves = MoveList::new::<ALL>(pos);
    let mut ours: Vec<String> = moves.iter().map(|&m| move_to_uci(m, chess960)).collect();
    let mut theirs: Vec<String> = sk
        .legal_moves()
        .iter()
        .map(|m| m.to_uci(mode).to_string())
        .collect();
    ours.sort();
    theirs.sort();
    assert_eq!(ours, theirs, "{fen}");

    // The FEN, but for castling in Chess960, where Stockfish writes the
    // files of the rooks and shakmaty only those it must
    let sk_fen = Fen::from_position(sk, EnPassantMode::Legal).to_string();
    if chess960 {
        let fields = |fen: &str| -> Vec<String> {
            let mut fields: Vec<String> = fen.split(' ').map(str::to_string).collect();
            fields.remove(2);
            fields
        };
        assert_eq!(fields(&fen), fields(&sk_fen));
        let mut rooks = 0u64;
        for cr in [WHITE_OO, WHITE_OOO, BLACK_OO, BLACK_OOO] {
            if pos.can_castle(cr) {
                rooks |= 1 << pos.castling_rook_square(cr).0;
            }
        }
        assert_eq!(rooks, sk.castles().castling_rights().0, "{fen}");
    } else {
        assert_eq!(fen, sk_fen);
    }
    assert_eq!(pos.checkers().0, sk.checkers().0, "{fen}");

    // The state kept incrementally, against the same computed afresh
    let fresh = Position::from_fen(&fen, chess960).unwrap();
    let (st, new) = (pos.state(), fresh.state());
    assert_eq!(
        (
            st.key,
            st.pawn_key,
            st.material_key,
            st.minor_piece_key,
            st.non_pawn_key
        ),
        (
            new.key,
            new.pawn_key,
            new.material_key,
            new.minor_piece_key,
            new.non_pawn_key
        ),
        "{fen}"
    );
    assert_eq!(
        (st.non_pawn_material, st.castling_rights),
        (new.non_pawn_material, new.castling_rights)
    );
    assert_eq!(
        (st.rule50, st.ep_square, st.checkers_bb),
        (new.rule50, new.ep_square, new.checkers_bb)
    );
    assert_eq!(
        (st.blockers_for_king, st.pinners),
        (new.blockers_for_king, new.pinners),
        "{fen}"
    );
    assert_eq!(st.check_squares, new.check_squares, "{fen}");
    let threats_by = |pos: &Position| {
        [ALL_PIECES, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING].map(|pt| pos.threats_by(pt))
    };
    assert_eq!(threats_by(pos), threats_by(&fresh), "{fen}");
    assert_eq!((pos.key(), pos.game_ply()), (fresh.key(), fresh.game_ply()));

    // The kinds of moves, which make up all the moves in the same order
    let mut kinds: Vec<Move> = Vec::new();
    if pos.checkers().any() {
        kinds.extend(MoveList::new::<EVASIONS>(pos).iter());
    } else {
        let captures = MoveList::new::<CAPTURES>(pos);
        let quiets = MoveList::new::<QUIETS>(pos);
        assert!(captures.iter().all(|&m| pos.capture_stage(m)), "{fen}");
        assert!(quiets.iter().all(|&m| !pos.capture_stage(m)), "{fen}");
        kinds.extend(captures.iter().chain(quiets.iter()));
    }
    assert_eq!(kinds, moves.as_slice(), "{fen}");

    // legal() for every move between two squares, the generator's moves for
    // the uncommon ones
    for from in 0..64 {
        for to in 0..64 {
            let m = Move::new(Square(from), Square(to));
            if from != to {
                assert_eq!(
                    pos.legal(m),
                    moves.contains(m),
                    "{fen} {}",
                    move_to_uci(m, chess960)
                );
            }
        }
    }
    assert!(moves.iter().all(|&m| pos.legal(m)));

    // Each move: whether it gives check, what it changes for the network,
    // and undoing it
    let state = pos.state().clone();
    let depth = pos.states().len();
    let before = threats(pos);
    let pawns = [pos.pieces_cpt(WHITE, PAWN), pos.pieces_cpt(BLACK, PAWN)];
    let mut dirties = Dirties::default();
    for &m in &moves {
        let sk_move = sk
            .legal_moves()
            .into_iter()
            .find(|n| n.to_uci(mode).to_string() == move_to_uci(m, chess960));
        assert_eq!(
            pos.capture(m),
            sk_move.unwrap().is_capture(),
            "{fen} {}",
            move_to_uci(m, chess960)
        );

        let gives_check = pos.gives_check(m);
        pos.make_move(m, &mut dirties);
        let name = move_to_uci(m, chess960);
        assert_eq!(gives_check, pos.checkers().any(), "{fen} {name}");

        let mut changes: BTreeMap<u32, i32> = BTreeMap::new();
        for dt in dirties.dirty_threats.iter() {
            *changes.entry(dt.raw() & !(1 << 31)).or_default() += if dt.add() { 1 } else { -1 };
        }
        changes.retain(|_, change| *change != 0);
        let after = threats(pos);
        let mut expected = BTreeMap::new();
        expected.extend(
            before
                .keys()
                .filter(|k| !after.contains_key(k))
                .map(|&k| (k, -1)),
        );
        expected.extend(
            after
                .keys()
                .filter(|k| !before.contains_key(k))
                .map(|&k| (k, 1)),
        );
        assert_eq!(changes, expected, "{fen} {name}");
        assert_eq!(dirties.dirty_pawn_pairs.before, pawns);
        assert_eq!(
            dirties.dirty_pawn_pairs.after,
            [pos.pieces_cpt(WHITE, PAWN), pos.pieces_cpt(BLACK, PAWN)]
        );
        assert_eq!(dirties.dirty_piece.from, m.from_sq());

        pos.undo_move(m);
        assert_eq!(pos.fen(), fen, "{name}");
        assert_eq!(
            (pos.state().clone(), pos.states().len()),
            (state.clone(), depth),
            "{fen} {name}"
        );
    }

    // A null move, where legal
    if pos.checkers().is_empty() {
        pos.do_null_move();
        let fresh = Position::from_fen(&pos.fen(), chess960).unwrap();
        assert_eq!(pos.state().key, fresh.state().key, "{fen}");
        pos.undo_null_move();
        assert_eq!((pos.fen(), pos.state().clone()), (fen.clone(), state));
    }
}

fn play_games(
    games: usize,
    plies: usize,
    seed: u64,
    check: impl Fn(&mut Position, &Chess, bool),
) -> usize {
    let mut rng = Prng::new(seed);
    let mut positions = 0;
    for game in 0..games {
        let (fen, chess960) = STARTS[game % STARTS.len()];
        let mode = if chess960 {
            CastlingMode::Chess960
        } else {
            CastlingMode::Standard
        };
        let mut pos = Position::from_fen(fen, chess960).unwrap();
        let mut sk = shakmaty_position(fen, chess960);
        let mut dirties = Dirties::default();
        for _ in 0..plies {
            check(&mut pos, &sk, chess960);
            positions += 1;
            let moves = MoveList::new::<ALL>(&pos);
            if moves.is_empty() {
                break;
            }
            let m = moves.as_slice()[(rng.rand64() % moves.len() as u64) as usize];
            let name = move_to_uci(m, chess960);
            let sk_move = sk
                .legal_moves()
                .into_iter()
                .find(|n| n.to_uci(mode).to_string() == name)
                .unwrap();
            pos.make_move(m, &mut dirties);
            sk.play_unchecked(sk_move);
        }
    }
    positions
}

#[test]
fn random_games_agree_with_shakmaty_and_with_fresh_positions() {
    let positions = play_games(140, 100, 20261012, check_position);
    assert!(positions > 5000, "{positions}");
}

/// Whether some legal move makes the position a repetition, by playing them
/// all: what `upcoming_repetition` answers without doing so.
fn repetition_after_a_move(pos: &mut Position, ply: i32) -> bool {
    let mut dirties = Dirties::default();
    MoveList::new::<ALL>(pos).iter().any(|&m| {
        pos.make_move(m, &mut dirties);
        let repeats = pos.is_repetition(ply + 1);
        pos.undo_move(m);
        repeats
    })
}

#[test]
fn upcoming_repetitions_are_those_after_a_move() {
    let mut rng = Prng::new(7);
    let mut found = 0;
    for game in 0..60 {
        let (fen, chess960) = STARTS[game % STARTS.len()];
        let mut pos = Position::from_fen(fen, chess960).unwrap();
        let mut dirties = Dirties::default();
        for _ in 0..80 {
            for ply in [0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 100] {
                let expected = repetition_after_a_move(&mut pos, ply);
                assert_eq!(
                    pos.upcoming_repetition(ply),
                    expected,
                    "{} at ply {ply}",
                    pos.fen()
                );
                found += expected as usize;
            }
            // Mostly reversible moves, so that positions repeat
            let moves = MoveList::new::<ALL>(&pos);
            let reversible: Vec<Move> = moves
                .iter()
                .copied()
                .filter(|&m| {
                    !pos.capture(m) && pos.moved_piece(m).type_of() != PAWN && m.type_of() == NORMAL
                })
                .collect();
            let choices = if reversible.is_empty() || rng.rand64().is_multiple_of(16) {
                moves.as_slice()
            } else {
                &reversible
            };
            if choices.is_empty() {
                break;
            }
            let m = choices[(rng.rand64() % choices.len() as u64) as usize];
            pos.make_move(m, &mut dirties);
        }
    }
    assert!(found > 1000, "{found}");
}

#[test]
fn repetitions_count_from_the_root() {
    let mut pos = Position::new();
    let mut dirties = Dirties::default();
    let play = |pos: &mut Position, dirties: &mut Dirties, moves: &str| {
        for name in moves.split(' ') {
            let m = varifish::uci::to_move(pos, name);
            assert_ne!(m, Move::none(), "{name}");
            pos.make_move(m, dirties);
        }
    };

    play(&mut pos, &mut dirties, "g1f3 g8f6 f3g1");
    assert!(!pos.upcoming_repetition(3));
    assert!(pos.upcoming_repetition(4));

    // The start repeats once: a draw if the root came after the first time
    play(&mut pos, &mut dirties, "f6g8");
    assert_eq!(pos.state().repetition, 4);
    assert!(pos.has_repeated());
    assert!(!pos.is_repetition(4) && pos.is_repetition(5));
    assert!(!pos.is_draw(4) && pos.is_draw(5));

    // and twice: always a draw
    play(&mut pos, &mut dirties, "g1f3 g8f6 f3g1 f6g8");
    assert_eq!(pos.state().repetition, -4);
    assert!(pos.is_repetition(0));

    // A pawn move ends the stretch in which positions can repeat.
    play(&mut pos, &mut dirties, "e2e4");
    assert!(!pos.has_repeated());
    assert!(!pos.upcoming_repetition(100));
}

#[test]
fn flipped_positions_have_the_same_perft() {
    for (fen, chess960) in STARTS {
        let mut flipped = Position::from_fen(fen, chess960).unwrap();
        flipped.flip().unwrap();
        let mut back = flipped.clone();
        back.flip().unwrap();
        assert_eq!(back.fen(), Position::from_fen(fen, chess960).unwrap().fen());
        let nodes = |fen: &str| {
            varifish::perft::perft::<varifish::variant::Chess>(fen, 3, chess960, |_, _| {}).unwrap()
        };
        assert_eq!(nodes(&flipped.fen()), nodes(fen), "{fen}");
    }
}

#[test]
fn static_exchange_evaluation() {
    // Rxe5 wins a pawn.
    let pos = Position::from_fen("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", false).unwrap();
    let m = varifish::uci::to_move(&pos, "e1e5");
    assert!(pos.see_ge(m, PAWN_VALUE) && !pos.see_ge(m, PAWN_VALUE + 1));

    // Nxe5 loses a knight for a pawn, as long exchanges on e5 show.
    let pos = Position::from_fen(
        "1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1",
        false,
    )
    .unwrap();
    let m = varifish::uci::to_move(&pos, "d3e5");
    let value = PAWN_VALUE - KNIGHT_VALUE;
    assert!(pos.see_ge(m, value) && !pos.see_ge(m, value + 1));

    // Nxd6 wins a pawn while the rook pins its only defender to the king...
    let pos = Position::from_fen("4k3/4b3/3p4/1N6/8/8/8/4R1K1 w - - 0 1", false).unwrap();
    let m = varifish::uci::to_move(&pos, "b5d6");
    assert!(pos.see_ge(m, PAWN_VALUE) && !pos.see_ge(m, PAWN_VALUE + 1));

    // ...and loses the knight for it without the pin.
    let pos = Position::from_fen("4k3/4b3/3p4/1N6/8/8/8/R5K1 w - - 0 1", false).unwrap();
    let m = varifish::uci::to_move(&pos, "b5d6");
    let value = PAWN_VALUE - KNIGHT_VALUE;
    assert!(pos.see_ge(m, value) && !pos.see_ge(m, value + 1));
}
