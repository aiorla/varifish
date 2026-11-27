//! Three-check's state and rules: the checks each side still needs to give
//! to win (three at the start, down to zero on the winning third check), the
//! Zobrist keys for those counts, and Three-check's move generation and
//! legality on top of chess's.

use crate::misc::Prng;
use crate::movegen::{ALL, GenType, MoveList, Moves};
use crate::position::Position;
use crate::types::*;
use crate::variant::{ThreeCheck, Variant};

/// The number of checks a side gives to win in Three-check.
pub const MAX_CHECKS: u8 = 3;

/// The number of non-terminal check-remaining states (`{1, 2, 3} × {1, 2, 3}`).
pub const CHECK_STATES: usize = 9;

/// The checks each side still needs to give to win, indexed by `Color::idx()`.
/// Starts at `[3, 3]`; a side wins immediately when its count reaches `0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChecksRemaining(pub [u8; COLOR_NB]);

impl Default for ChecksRemaining {
    #[inline(always)]
    fn default() -> ChecksRemaining {
        ChecksRemaining::START
    }
}

impl ChecksRemaining {
    pub const START: ChecksRemaining = ChecksRemaining([MAX_CHECKS, MAX_CHECKS]);

    #[inline(always)]
    pub const fn new(white: u8, black: u8) -> ChecksRemaining {
        ChecksRemaining([white, black])
    }

    #[inline(always)]
    pub const fn get(self, c: Color) -> u8 {
        self.0[c.idx()]
    }

    #[inline(always)]
    pub fn set(&mut self, c: Color, n: u8) {
        self.0[c.idx()] = n;
    }

    #[inline(always)]
    pub fn decrement(&mut self, c: Color) {
        let slot = &mut self.0[c.idx()];
        *slot = slot.saturating_sub(1);
    }

    #[inline(always)]
    pub const fn is_over(self) -> bool {
        self.0[WHITE.idx()] == 0 || self.0[BLACK.idx()] == 0
    }

    /// Index in `0..9` of the non-terminal check state from `perspective`'s
    /// point of view: `(our_remaining - 1) * 3 + (their_remaining - 1)`.
    #[inline(always)]
    pub const fn state_index(self, perspective: Color) -> usize {
        let ours = self.get(perspective);
        let theirs = self.get(Color(perspective.0 ^ 1));
        debug_assert!(ours >= 1 && ours <= MAX_CHECKS);
        debug_assert!(theirs >= 1 && theirs <= MAX_CHECKS);
        (ours - 1) as usize * 3 + (theirs - 1) as usize
    }

    /// Swaps White's and Black's remaining checks, for `Position::flip()`.
    #[inline(always)]
    pub const fn swapped(self) -> ChecksRemaining {
        ChecksRemaining([self.0[BLACK.idx()], self.0[WHITE.idx()]])
    }
}

/// Zobrist keys for each color's remaining checks (`0..=3`), with the lowest
/// bit clear as every other key in `ZOBRIST` keeps it, and zero for `3` so
/// that the starting state's key needs no adjustment.
pub static ZOBRIST_CHECKS: [[Key; 4]; COLOR_NB] = {
    let mut rng = Prng::new(0x3C4E_C43B_5A7E_19D3);
    let mut keys = [[0; 4]; COLOR_NB];
    let mut c = 0;
    while c < COLOR_NB {
        let mut rem = 0;
        while rem < MAX_CHECKS as usize {
            keys[c][rem] = rng.rand64() << 1;
            rem += 1;
        }
        c += 1;
    }
    keys
};

const _: () = {
    let mut any = 0;
    let mut c = 0;
    while c < COLOR_NB {
        assert!(ZOBRIST_CHECKS[c][MAX_CHECKS as usize] == 0);
        let mut rem = 0;
        while rem < MAX_CHECKS as usize {
            assert!(ZOBRIST_CHECKS[c][rem] != 0);
            any |= ZOBRIST_CHECKS[c][rem];
            rem += 1;
        }
        c += 1;
    }
    assert!(
        any & 1 == 0,
        "Least significant bit of check-count Zobrist keys must be zero"
    );
};

/// Parses a Lichess remaining-checks field such as `"3+3"` or `"2+1"`.
pub(super) fn parse_remaining_checks(token: &str) -> Option<ChecksRemaining> {
    let bytes = token.as_bytes();
    if bytes.len() != 3 || bytes[1] != b'+' {
        return None;
    }
    let w = parse_check_digit(bytes[0])?;
    let b = parse_check_digit(bytes[2])?;
    Some(ChecksRemaining::new(w, b))
}

/// Parses an XBoard checks-given field such as `"+0+0"` or `"+2+1"`.
pub(super) fn parse_given_checks(token: &str) -> Option<ChecksRemaining> {
    let bytes = token.as_bytes();
    if bytes.len() != 4 || bytes[0] != b'+' || bytes[2] != b'+' {
        return None;
    }
    let w = parse_check_digit(bytes[1])?;
    let b = parse_check_digit(bytes[3])?;
    Some(ChecksRemaining::new(MAX_CHECKS - w, MAX_CHECKS - b))
}

#[inline(always)]
fn parse_check_digit(b: u8) -> Option<u8> {
    if (b'0'..=b'0' + MAX_CHECKS).contains(&b) {
        Some(b - b'0')
    } else {
        None
    }
}

impl<V: Variant> Position<V> {
    /// Checks `c` still needs to give to win (meaningful when `V::THREE_CHECK`).
    #[inline(always)]
    pub fn checks_remaining(&self, c: Color) -> u8 {
        V::checks_remaining(&self.st().extra, c)
    }

    /// Whether the side not to move has already given its third check.
    #[inline(always)]
    pub fn is_three_check_loss(&self) -> bool {
        V::THREE_CHECK && self.checks_remaining(!self.side_to_move) == 0
    }

    /// Whether either side has already given its third check.
    #[inline(always)]
    pub fn is_three_check_over(&self) -> bool {
        V::THREE_CHECK && (self.checks_remaining(WHITE) == 0 || self.checks_remaining(BLACK) == 0)
    }

    /// Returns a legal move other than `excluded` that gives check, or
    /// `Move::none()` if none exists. Used when the side to move needs only
    /// one more check to win.
    pub fn find_checking_move(&self, excluded: Move) -> Move {
        for &m in MoveList::new::<ALL>(self).iter() {
            if m != excluded && self.gives_check(m) {
                return m;
            }
        }
        Move::none()
    }
}

impl Position<ThreeCheck> {
    /// Appends the legal Three-check moves of kind `T` to `moves`, or none if
    /// either side has already given three checks.
    #[inline]
    pub(crate) fn three_check_generate<const T: GenType>(&self, moves: &mut Moves) {
        if self.st().extra.is_over() {
            return;
        }
        crate::movegen::generate::<T, ThreeCheck>(self, moves);
    }

    /// Tests whether `m` is a legal Three-check move.
    #[inline]
    pub(crate) fn three_check_legal(&self, m: Move) -> bool {
        !self.st().extra.is_over() && self.chess_legal(m)
    }
}
