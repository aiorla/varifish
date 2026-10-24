//! varifish's Atomic positions checked along random games against shakmaty's
//! Atomic: legal moves, FENs and checks, and the incrementally updated state
//! (keys, what moves change for the network, undoing) against the same
//! computed afresh; then Atomic-Stockfish's perft suite and rule fixtures.

use std::collections::BTreeMap;

use shakmaty::fen::Fen;
use shakmaty::variant::Atomic as SkAtomic;
use shakmaty::{CastlingMode, EnPassantMode, Position as _};
use varifish::attacks::pseudo_attacks;
use varifish::misc::Prng;
use varifish::movegen::{ALL, CAPTURES, EVASIONS, MoveList, QUIETS};
use varifish::perft::perft;
use varifish::position::START_FEN;
use varifish::types::*;
use varifish::uci::move_to_uci;
use varifish::variant::Atomic;

type Position = varifish::position::Position<Atomic>;

/// Atomic-Stockfish's normative perft suite, from Fairy-Stockfish's, with
/// whether each is Chess960.
const PERFT_SUITE: [(&str, bool, i32, u64); 8] = [
    (START_FEN, false, 4, 197326),
    (
        "rn2kb1r/1pp1p2p/p2q1pp1/3P4/2P3b1/4PN2/PP3PPP/R2QKB1R b KQkq - 0 1",
        false,
        4,
        1434825,
    ),
    (
        "rn1qkb1r/p5pp/2p5/3p4/N3P3/5P2/PPP4P/R1BQK3 w Qkq - 0 1",
        false,
        4,
        714499,
    ),
    (
        "r4b1r/2kb1N2/p2Bpnp1/8/2Pp3p/1P1PPP2/P5PP/R3K2R b KQ - 0 1",
        false,
        2,
        148,
    ),
    ("8/8/8/8/8/8/2k5/rR4KR w KQ - 0 1", true, 4, 61401),
    ("r3k1rR/5K2/8/8/8/8/8/8 b kq - 0 1", true, 4, 98729),
    ("Rr2k1rR/3K4/3p4/8/8/8/7P/8 w kq - 0 1", true, 4, 241478),
    ("1R4kr/4K3/8/8/8/8/8/8 b k - 0 1", true, 4, 17915),
];

/// Starts for random games: the suite's positions and others where kings
/// meet, explode, castle and take en passant.
const STARTS: [(&str, bool); 16] = [
    (START_FEN, false),
    (
        "rn2kb1r/1pp1p2p/p2q1pp1/3P4/2P3b1/4PN2/PP3PPP/R2QKB1R b KQkq - 0 1",
        false,
    ),
    (
        "rn1qkb1r/p5pp/2p5/3p4/N3P3/5P2/PPP4P/R1BQK3 w Qkq - 0 1",
        false,
    ),
    (
        "r4b1r/2kb1N2/p2Bpnp1/8/2Pp3p/1P1PPP2/P5PP/R3K2R b KQ - 0 1",
        false,
    ),
    ("8/8/8/8/8/8/2k5/rR4KR w KQ - 0 1", true),
    ("r3k1rR/5K2/8/8/8/8/8/8 b kq - 0 1", true),
    ("Rr2k1rR/3K4/3p4/8/8/8/7P/8 w kq - 0 1", true),
    ("1R4kr/4K3/8/8/8/8/8/8 b k - 0 1", true),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        false,
    ),
    ("7k/8/8/2pBn3/3r4/2PQN3/8/K7 w - - 0 1", false),
    ("7k/8/2N1b3/2ppP3/8/8/8/K7 w - d6 0 2", false),
    ("6k1/8/8/8/8/8/1r6/RN2K2R b KQ - 0 1", false),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", false),
    ("4k3/3ppp2/8/8/8/8/3PPP2/4K3 w - - 0 1", false),
    (
        "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 9",
        true,
    ),
    ("4k3/8/8/3pP3/8/8/8/4K2R w K d6 0 2", false),
];

fn mode(chess960: bool) -> CastlingMode {
    if chess960 {
        CastlingMode::Chess960
    } else {
        CastlingMode::Standard
    }
}

fn shakmaty_position(fen: &str, chess960: bool) -> SkAtomic {
    fen.parse::<Fen>()
        .unwrap()
        .into_position(mode(chess960))
        .unwrap()
}

/// The pieces a move removes from and adds to squares, by its record for the
/// network, with what cancels out left out.
fn recorded_changes(dirties: &AtomicDirties) -> BTreeMap<(u8, u8), i32> {
    let dp = &dirties.dirty_piece;
    let mut changes = BTreeMap::new();
    let mut change = |pc: Piece, s: Square, by: i32| {
        *changes.entry((pc.0, s.0)).or_insert(0) += by;
    };
    change(dp.pc, dp.from, -1);
    if dp.to != Square::NONE {
        change(dp.pc, dp.to, 1);
    }
    if dp.remove_sq != Square::NONE {
        change(dp.remove_pc, dp.remove_sq, -1);
    }
    if dp.add_sq != Square::NONE {
        change(dp.add_pc, dp.add_sq, 1);
    }
    for (pc, s) in dirties.blast.iter() {
        change(pc, s, -1);
    }
    changes.retain(|_, by| *by != 0);
    changes
}

/// The pieces that differ between two boards.
fn board_changes(before: &Position, after: &Position) -> BTreeMap<(u8, u8), i32> {
    let mut changes = BTreeMap::new();
    for s in 0..64 {
        let (old, new) = (before.piece_on(Square(s)), after.piece_on(Square(s)));
        if old != new {
            if old != NO_PIECE {
                changes.insert((old.0, s), -1);
            }
            if new != NO_PIECE {
                changes.insert((new.0, s), 1);
            }
        }
    }
    changes
}

/// How often the games met what Atomic changes, so that a change of seed or
/// of rules cannot silently leave it untested
#[derive(Debug, Default)]
struct Coverage {
    in_check: usize,
    kings_side_by_side: usize,
    blasts: usize,
    exploded_kings: usize,
    winning_captures: usize,
    castlings: usize,
    en_passants: usize,
    promotions: usize,
}

/// Checks a position and each of its moves, against shakmaty and against
/// the same position set afresh from its FEN.
fn check_position(pos: &mut Position, sk: &SkAtomic, chess960: bool, seen: &mut Coverage) {
    let fen = pos.fen();
    let mode = mode(chess960);

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
    assert_eq!(pos.check(), Ok(()), "{fen}");
    seen.in_check += pos.in_check() as usize;
    seen.kings_side_by_side += (pos.has_king(WHITE)
        && pos.has_king(BLACK)
        && pseudo_attacks(KING, pos.king_square(WHITE)).has(pos.king_square(BLACK)))
        as usize;

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
        (new.non_pawn_material, new.castling_rights),
        "{fen}"
    );
    assert_eq!(
        (st.rule50, st.ep_square, st.checkers_bb),
        (new.rule50, new.ep_square, new.checkers_bb),
        "{fen}"
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

    // The kinds of moves, which make up all the moves: in the same order out
    // of check, and in check as the evasions do, in another order.
    let captures = MoveList::new::<CAPTURES>(pos);
    let quiets = MoveList::new::<QUIETS>(pos);
    assert!(captures.iter().all(|&m| pos.capture_stage(m)), "{fen}");
    assert!(quiets.iter().all(|&m| !pos.capture_stage(m)), "{fen}");
    let mut kinds: Vec<Move> = captures.iter().chain(quiets.iter()).copied().collect();
    if pos.checkers().any() {
        assert_eq!(
            MoveList::new::<EVASIONS>(pos).as_slice(),
            moves.as_slice(),
            "{fen}"
        );
        let mut all = moves.as_slice().to_vec();
        kinds.sort_unstable_by_key(|m| m.raw());
        all.sort_unstable_by_key(|m| m.raw());
        assert_eq!(kinds, all, "{fen}");
    } else {
        assert_eq!(kinds, moves.as_slice(), "{fen}");
    }

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
    let before = pos.clone();
    let mut dirties = AtomicDirties::default();
    let mut wins = Vec::new();
    for &m in &moves {
        let name = move_to_uci(m, chess960);
        let sk_move = sk
            .legal_moves()
            .into_iter()
            .find(|n| n.to_uci(mode).to_string() == name)
            .unwrap();
        assert_eq!(pos.capture(m), sk_move.is_capture(), "{fen} {name}");

        let gives_check = pos.gives_check(m);
        pos.make_move(m, &mut dirties);
        assert_eq!(gives_check, pos.in_check(), "{fen} {name}");
        assert_eq!(pos.check(), Ok(()), "{fen} {name}");

        let mut after = sk.clone();
        after.play_unchecked(sk_move);
        assert_eq!(
            pos.fen().split(' ').next(),
            Fen::from_position(&after, EnPassantMode::Legal)
                .to_string()
                .split(' ')
                .next(),
            "{fen} {name}"
        );
        assert_eq!(
            recorded_changes(&dirties),
            board_changes(&before, pos),
            "{fen} {name}"
        );
        seen.blasts += dirties.blast.squares().any() as usize;
        seen.exploded_kings += !pos.has_king(pos.side_to_move()) as usize;
        if !pos.has_king(pos.side_to_move()) {
            wins.push(m);
        }
        seen.castlings += (m.type_of() == CASTLING) as usize;
        seen.en_passants += (m.type_of() == EN_PASSANT) as usize;
        seen.promotions += (m.type_of() == PROMOTION) as usize;

        pos.undo_move(m);
        assert_eq!(pos.fen(), fen, "{name}");
        assert_eq!(
            (pos.state().clone(), pos.states().len()),
            (state.clone(), depth),
            "{fen} {name}"
        );
    }

    // A capture that explodes the other king, found without generating the
    // moves, exactly when one of them does
    if pos.has_king(pos.side_to_move()) {
        let win = pos.atomic_winning_capture();
        assert_eq!(win != Move::none(), !wins.is_empty(), "{fen}");
        assert!(win == Move::none() || wins.contains(&win), "{fen}");
        seen.winning_captures += !wins.is_empty() as usize;
    }

    // A null move, where legal
    if !pos.in_check() && pos.has_king(pos.side_to_move()) {
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
    mut check: impl FnMut(&mut Position, &SkAtomic, bool),
) -> usize {
    let mut rng = Prng::new(seed);
    let mut positions = 0;
    for game in 0..games {
        let (fen, chess960) = STARTS[game % STARTS.len()];
        let mode = mode(chess960);
        let mut pos = Position::from_fen(fen, chess960).unwrap();
        let mut sk = shakmaty_position(fen, chess960);
        let mut dirties = AtomicDirties::default();
        for _ in 0..plies {
            check(&mut pos, &sk, chess960);
            positions += 1;
            let moves = MoveList::new::<ALL>(&pos);
            if moves.is_empty() {
                break;
            }
            // Mostly moves other than captures, so that games last
            let quiet: Vec<Move> = moves.iter().copied().filter(|&m| !pos.capture(m)).collect();
            let choices = if quiet.is_empty() || rng.rand64().is_multiple_of(4) {
                moves.as_slice()
            } else {
                &quiet
            };
            let m = choices[(rng.rand64() % choices.len() as u64) as usize];
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
    let mut seen = Coverage::default();
    let positions = play_games(400, 120, 20261012, |pos, sk, chess960| {
        check_position(pos, sk, chess960, &mut seen)
    });
    assert!(positions > 10000, "{positions}");
    let Coverage {
        in_check,
        kings_side_by_side,
        blasts,
        exploded_kings,
        winning_captures,
        castlings,
        en_passants,
        promotions,
    } = seen;
    assert!(
        [
            in_check,
            kings_side_by_side,
            blasts,
            exploded_kings,
            winning_captures,
            castlings,
            en_passants,
            promotions
        ]
        .iter()
        .all(|&n| n >= 50),
        "{seen:?}"
    );
}

#[test]
fn perft_suite() {
    for (fen, chess960, depth, nodes) in PERFT_SUITE {
        // The deepest counts take a while in the dev profile.
        let depth = if cfg!(debug_assertions) && nodes > 300_000 {
            depth - 1
        } else {
            depth
        };
        let count = perft::<Atomic>(fen, depth, chess960, |_, _| {}).unwrap();
        if cfg!(debug_assertions) && nodes > 300_000 {
            let sk = shakmaty::perft(&shakmaty_position(fen, chess960), depth as u32);
            assert_eq!(count, sk, "{fen}");
        } else {
            assert_eq!(count, nodes, "{fen}");
        }
    }
}

/// The root moves and counts of `go perft`, as "move=count" sorted.
fn divide(fen: &str, depth: i32, chess960: bool) -> Vec<String> {
    let mut moves = Vec::new();
    perft::<Atomic>(fen, depth, chess960, |m, n| {
        moves.push(format!("{}={n}", move_to_uci(m, chess960)))
    })
    .unwrap();
    moves.sort();
    moves
}

fn sorted(counts: &str) -> Vec<String> {
    let mut moves: Vec<String> = counts.split_whitespace().map(str::to_string).collect();
    moves.sort();
    moves
}

#[test]
fn atomic_stockfish_rule_fixtures() {
    // Qxd4 explodes the capturer, the rook and the pieces next to d4 but the
    // pawns on c3 and c5.
    assert_eq!(
        divide("7k/8/8/2pBn3/3r4/2PQN3/8/K7 w - - 0 1", 2, false),
        sorted(
            "c3c4=16 c3d4=4 e3d1=19 e3f1=19 e3c2=19 e3g2=19
             e3c4=16 e3g4=18 e3f5=19 d5h1=23 d5a2=22 d5g2=23
             d5b3=22 d5f3=23 d5c4=19 d5e4=20 d5c6=23 d5e6=22
             d5b7=23 d5f7=22 d5a8=23 d5g8=22 d3b1=21 d3d1=22
             d3f1=22 d3c2=21 d3d2=21 d3e2=22 d3c4=19 d3d4=4
             d3e4=18 d3b5=22 d3f5=21 d3a6=22 d3g6=20 d3h7=0
             a1b1=19 a1a2=19 a1b2=19"
        )
    );

    // Rxd2 would explode the white king.
    assert_eq!(
        divide("7k/8/8/8/8/8/3r4/2KR4 w - - 0 1", 2, false),
        sorted("d1e1=17 d1f1=17 d1g1=15 d1h1=3 c1b1=17")
    );

    // Adjacent kings cannot capture each other.
    assert_eq!(
        divide("8/8/8/8/8/8/4k3/4K3 w - - 0 1", 2, false),
        sorted("e1d1=7 e1f1=7 e1d2=7 e1f2=7")
    );

    // En passant explodes both pawns and the pieces next to d6.
    assert_eq!(
        divide("7k/8/2N1b3/2ppP3/8/8/8/K7 w - d6 0 2", 2, false),
        sorted(
            "e5d6=4 c6b4=13 c6d4=12 c6a5=12 c6a7=12 c6e7=11
             c6b8=12 c6d8=12 a1b1=12 a1a2=12 a1b2=12"
        )
    );

    // An exploded rook loses its castling right.
    assert_eq!(
        divide("6k1/8/8/8/8/8/1r6/RN2K2R b KQ - 0 1", 2, false),
        sorted(
            "b2b1=15 b2a2=16 b2c2=22 b2d2=20 b2e2=2 b2f2=20
             b2g2=21 b2h2=16 b2b3=25 b2b4=25 b2b5=25 b2b6=25
             b2b7=25 b2b8=25 g8f7=22 g8g7=22 g8f8=22"
        )
    );

    // Promoting with a capture explodes the new piece.
    assert_eq!(
        divide("k5br/6P1/8/8/8/8/8/K7 w - - 0 1", 2, false),
        sorted("g7h8q=3 g7h8r=3 g7h8b=3 g7h8n=3 a1b1=17 a1b2=17")
    );

    // Exploding the king ends the game.
    assert_eq!(
        divide("6rk/8/8/8/8/8/8/K5R1 w - - 0 1", 2, false),
        sorted(
            "g1b1=15 g1c1=15 g1d1=15 g1e1=15 g1f1=15 g1h1=1
             g1g2=13 g1g3=12 g1g4=11 g1g5=10 g1g6=9 g1g7=6
             g1g8=0 a1b1=14 a1a2=14 a1b2=14"
        )
    );

    // Mate and stalemate
    for fen in [
        "BQ6/Rk6/8/8/8/8/8/4K3 b - - 0 1",
        "KQ6/Rk6/2B5/8/8/8/8/8 b - - 0 1",
    ] {
        assert!(divide(fen, 1, false).is_empty(), "{fen}");
    }
    let mated = Position::from_fen("BQ6/Rk6/8/8/8/8/8/4K3 b - - 0 1", false).unwrap();
    assert!(mated.in_check());
    let stalemated = Position::from_fen("KQ6/Rk6/2B5/8/8/8/8/8 b - - 0 1", false).unwrap();
    assert!(!stalemated.in_check());

    // Chess960 castling, legal and through an attacked square
    assert_eq!(
        divide("7k/8/8/8/8/8/8/1RK5 w Q - 0 1", 2, true),
        sorted(
            "b1a1=3 b1b2=3 b1b3=3 b1b4=3 b1b5=3 b1b6=3 b1b7=1
             b1b8=2 c1d1=3 c1b2=3 c1c2=3 c1d2=3 c1b1=3"
        )
    );
    assert_eq!(
        divide("7k/8/8/8/8/8/2PP4/1RK4q w Q - 0 1", 2, true),
        sorted("c1b2=22")
    );
}

#[test]
fn explosions_reach_the_expected_positions() {
    for (fen, m, after) in [
        (
            "7k/8/8/2pBn3/3r4/2PQN3/8/K7 w - - 0 1",
            "d3d4",
            "7k/8/8/2p5/8/2P5/8/K7 b - - 0 1",
        ),
        (
            "7k/8/2N1b3/2ppP3/8/8/8/K7 w - d6 0 2",
            "e5d6",
            "7k/8/8/2p5/8/8/8/K7 b - - 0 2",
        ),
        (
            "6k1/8/8/8/8/8/1r6/RN2K2R b KQ - 0 1",
            "b2b1",
            "6k1/8/8/8/8/8/8/4K2R w K - 0 2",
        ),
        (
            "k5br/6P1/8/8/8/8/8/K7 w - - 0 1",
            "g7h8q",
            "k7/8/8/8/8/8/8/K7 b - - 0 1",
        ),
        (
            "6rk/8/8/8/8/8/8/K5R1 w - - 0 1",
            "g1g8",
            "8/8/8/8/8/8/8/K7 b - - 0 1",
        ),
    ] {
        let mut pos = Position::from_fen(fen, false).unwrap();
        let mv = varifish::uci::to_move(&pos, m);
        assert_ne!(mv, Move::none(), "{fen} {m}");
        pos.make_move(mv, &mut AtomicDirties::default());
        assert_eq!(pos.fen(), after, "{fen} {m}");
        assert!(!MoveList::new::<ALL>(&pos).is_empty() || !pos.has_king(pos.side_to_move()));
    }
}

#[test]
fn positions_lichess_refuses_are_refused() {
    // The side not to move in check, or without its king
    for fen in [
        "7k/7R/8/8/8/8/8/K7 w - - 0 1",
        "8/8/8/8/8/8/8/K7 w - - 0 1",
        "8/8/8/8/8/8/8/8 w - - 0 1",
    ] {
        assert!(Position::from_fen(fen, false).is_err(), "{fen}");
    }

    // The side to move without its king has lost, and has no moves.
    let pos = Position::from_fen("8/8/8/8/8/8/8/K7 b - - 0 1", false).unwrap();
    assert!(MoveList::new::<ALL>(&pos).is_empty());

    // Kings side by side are not in check.
    let pos = Position::from_fen("8/8/8/8/8/8/3qk3/4K3 w - - 0 1", false).unwrap();
    assert!(!pos.in_check());
}

#[test]
fn en_passant_is_recorded_only_when_legal() {
    // After ...d5, taking on d6 would explode the white king on c7, so there
    // is no en passant.
    let mut pos = Position::from_fen("4k3/2Kp4/8/4P3/8/8/8/8 b - - 0 1", false).unwrap();
    let m = varifish::uci::to_move(&pos, "d7d5");
    pos.make_move(m, &mut AtomicDirties::default());
    assert_eq!(pos.ep_square(), Square::NONE, "{}", pos.fen());

    // Without the king there, it is recorded.
    let mut pos = Position::from_fen("4k3/3p4/8/4P3/8/8/8/K7 b - - 0 1", false).unwrap();
    let m = varifish::uci::to_move(&pos, "d7d5");
    pos.make_move(m, &mut AtomicDirties::default());
    assert_eq!(
        pos.ep_square(),
        Square::new(FILE_D, RANK_6),
        "{}",
        pos.fen()
    );
}

#[test]
fn winning_captures() {
    let win = |fen: &str| {
        let pos = Position::from_fen(fen, false).unwrap();
        let m = pos.atomic_winning_capture();
        (m != Move::none()).then(|| move_to_uci(m, false))
    };

    // Next to the other king: promoting, from check, and en passant
    let cases = [
        ("4kb2/6P1/8/8/8/8/8/4K3 w - - 0 1", "g7f8q"),
        ("4r1k1/5p2/8/8/8/1B6/8/4K3 w - - 0 1", "b3f7"),
        ("8/8/2k5/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
    ];
    for (fen, m) in cases {
        assert_eq!(win(fen).as_deref(), Some(m), "{fen}");
    }

    // Not when the capture is next to one's own king too
    assert_eq!(win("8/8/4k3/4p3/2N1K3/8/8/8 w - - 0 1"), None);
}

#[test]
fn static_exchange_evaluation() {
    let see = |fen: &str, m: &str| {
        let pos = Position::from_fen(fen, false).unwrap();
        let m = varifish::uci::to_move(&pos, m);
        assert_ne!(m, Move::none());
        // The largest threshold the move passes
        let (mut lo, mut hi) = (-VALUE_MATE, VALUE_MATE);
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if pos.see_ge(m, mid) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    };

    // Qxd4 explodes a rook, a bishop and a knight for the queen and a knight.
    assert_eq!(
        see("7k/8/8/2pBn3/3r4/2PQN3/8/K7 w - - 0 1", "d3d4"),
        1079 + 702 - 1862 - 702 - 738 - 1
    );
    // Exploding the king wins.
    assert_eq!(see("6rk/8/8/8/8/8/8/K5R1 w - - 0 1", "g1g8"), VALUE_MATE);
    // A quiet move next to the enemy king cannot be taken.
    assert_eq!(see("7k/8/8/8/8/8/8/K5R1 w - - 0 1", "g1g7"), 0);
}
