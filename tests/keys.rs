//! The keys and the variant's own state of every variant's positions when
//! their states reuse slots, as a search's do: each position of a small tree,
//! reached by playing a move where another was undone, has the state of its
//! FEN. A move copies only the first fields of the state before it into the
//! next slot, and what the slot held of the others belongs to the position
//! last undone there.

use varifish::movegen::{ALL, MoveList};
use varifish::position::Position;
use varifish::variant::{Antichess, Atomic, Chess, Crazyhouse, ThreeCheck, Variant};

fn check_below<V: Variant>(pos: &mut Position<V>, depth: u32, dirties: &mut V::Dirties) {
    let fen = pos.fen();
    let fresh = Position::<V>::from_fen(&fen, false).unwrap();
    let (st, new) = (pos.state(), fresh.state());
    assert_eq!(
        (
            st.key,
            st.pawn_key,
            st.minor_piece_key,
            st.non_pawn_key,
            st.material_key,
            st.non_pawn_material,
            st.checkers_bb,
            st.piece_placement,
        ),
        (
            new.key,
            new.pawn_key,
            new.minor_piece_key,
            new.non_pawn_key,
            new.material_key,
            new.non_pawn_material,
            new.checkers_bb,
            new.piece_placement,
        ),
        "{} {fen}",
        V::NAME
    );
    assert_eq!(pos.key(), fresh.key(), "{} {fen}", V::NAME);
    if depth == 0 {
        return;
    }
    let moves = MoveList::new::<ALL>(pos);
    for &mv in &moves {
        if !pos.legal(mv) {
            continue;
        }
        pos.make_move(mv, dirties);
        check_below(pos, depth - 1, dirties);
        pos.undo_move(mv);
    }
}

fn check<V: Variant>(fens: &[&str])
where
    V::Dirties: Default,
{
    for fen in fens {
        let mut pos = Position::<V>::from_fen(fen, false)
            .unwrap_or_else(|e| panic!("{} {fen}: {e:?}", V::NAME));
        let mut dirties = V::Dirties::default();
        check_below(&mut pos, 3, &mut dirties);
    }
}

const CHESS: [&str; 4] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
];

#[test]
fn chess_states_hold_after_moves_are_undone() {
    check::<Chess>(&CHESS);
}

#[test]
fn atomic_states_hold_after_moves_are_undone() {
    check::<Atomic>(&CHESS);
}

#[test]
fn threecheck_states_hold_after_moves_are_undone() {
    check::<ThreeCheck>(&[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 2+2 0 10",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1+2 4 4",
    ]);
}

#[test]
fn crazyhouse_states_hold_after_moves_are_undone() {
    check::<Crazyhouse>(&[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n1p3/4P3/1b1Pn3/2NB1N2/PPP2PPP/R1BQK2R[] b KQkq - 0 1",
        "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1[NPp] w - - 1 16",
    ]);
}

#[test]
fn antichess_states_hold_after_moves_are_undone() {
    check::<Antichess>(&[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
        "8/Q7/5N2/8/P5P1/8/r1PP1P2/8 w - - 0 19",
        "3r1n2/4P3/8/8/8/8/3p4/2B1R3 w - - 0 1",
    ]);
}
