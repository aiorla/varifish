//! The input features PP_3Wide: each pair of pawns on the same or adjacent
//! files, by their colors and squares, the board mirrored as for
//! FullThreats. Their indices follow the threats', in the same list.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/features/pp_3wide.h` and `.cpp`: Stockfish, a UCI chess playing
//! engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

#[cfg(use_avx512icl)]
use std::arch::x86_64::*;

use super::full_threats::{FullThreats, ThreatBoard};
use crate::bitboard::{Bitboard, pawn_pair_bb};
use crate::misc::ValueList;
#[cfg(not(use_avx512icl))]
use crate::misc::prefetch_low;
use crate::nnue::common::ORIENT_TBL;
use crate::position::Position;
use crate::types::*;

pub use super::full_threats::IndexList;

pub struct Pp3Wide;

impl Pp3Wide {
    pub const HASH_VALUE: u32 = 0x5c80_fa69;

    pub const PAWN_IDS: u32 = COLOR_NB as u32 * 48;
    pub const DIMENSIONS: u32 = Self::PAWN_IDS * (Self::PAWN_IDS - 1) / 2;

    /// Pawn pair feature indices are concatenated to threats, so this must
    /// equal FullThreats::DIMENSIONS; see the feature transformer.
    pub const INDEX_BASE: u32 = 59808;
}

const _: () = assert!(Pp3Wide::INDEX_BASE == FullThreats::DIMENSIONS);

/// The end of the indices, below which every pair of pawns off the first
/// and last ranks falls.
const END: u16 = (Pp3Wide::INDEX_BASE + Pp3Wide::DIMENSIONS) as u16;

#[inline(always)]
fn make_pawn_id(color: Color, square: usize) -> u32 {
    debug_assert!((Square::A2.idx()..=Square::H7.idx()).contains(&square));
    (48 * color.idx() + square).wrapping_sub(Square::A2.idx()) as u32
}

/// The index of the pair of the pawns of ids `a` and `b`, either way round.
#[inline(always)]
fn pp_idx(a: u32, b: u32) -> u32 {
    let hi = a.max(b);
    let lo = a.min(b);
    (hi.wrapping_mul(hi.wrapping_sub(1)) / 2)
        .wrapping_add(lo)
        .wrapping_add(Pp3Wide::INDEX_BASE)
}

/// `pp_idx` of 16 pairs of ids at once, in 16 bits.
#[cfg(use_avx512icl)]
#[inline(always)]
fn pp_idx_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: the build targets Ice Lake's AVX-512.
    unsafe {
        let hi = _mm256_max_epu16(a, b);
        let lo = _mm256_min_epu16(a, b);
        let prod = _mm256_mullo_epi16(hi, _mm256_sub_epi16(hi, _mm256_set1_epi16(1)));
        _mm256_add_epi16(
            _mm256_add_epi16(_mm256_srli_epi16::<1>(prod), lo),
            _mm256_set1_epi16(Pp3Wide::INDEX_BASE as i16),
        )
    }
}

#[inline(always)]
pub fn make_index(
    perspective: Color,
    color: Color,
    from: Square,
    to: Square,
    paired_color: Color,
    ksq: Square,
) -> u32 {
    let orientation = usize::from(ORIENT_TBL[ksq.idx()]) ^ (56 * perspective.idx());
    let from_oriented = from.idx() ^ orientation;
    let to_oriented = to.idx() ^ orientation;

    let color_oriented = Color(color.0 ^ perspective.0);
    let paired_color_oriented = Color(paired_color.0 ^ perspective.0);

    let id_a = make_pawn_id(color_oriented, from_oriented);
    let id_b = make_pawn_id(paired_color_oriented, to_oriented);
    pp_idx(id_a, id_b)
}

/// Pushes a pawn pair's index, as Stockfish does, if it is below END. Only
/// pawns on the first or last rank, which positions never hold, can give
/// one beyond, so the bound changes no evaluation and keeps every index
/// within the weights. Atomic's moves, whose blasts change more threats than
/// a chess move, list the pairs they change in longer lists.
#[inline(always)]
fn push<const M: usize>(list: &mut ValueList<u16, M>, index: u32) {
    list.push_back_if_lt(index as u16, END);
}

/// The indices, from `perspective` and, if `N` is 2, from the other color's
/// too, of the pairs that the pawns `updated`, among the pawns `pawns_w` and
/// `pawns_b`, make: each pawn of `updated` in turn with its partners among
/// the others but the pawns of `updated` already taken, each pair once, its
/// weights' rows prefetched unless `prefetch_base` is null. The pawns and
/// their partners are enumerated once for every perspective asked for.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn append_indices_impl<const N: usize, const M: usize>(
    perspective: Color,
    ksq: [Square; N],
    updated: Bitboard,
    pawns_w: Bitboard,
    pawns_b: Bitboard,
    out: [&mut ValueList<u16, M>; N],
    prefetch_base: *const i8,
    prefetch_stride: usize,
) {
    const { assert!(N == 1 || N == 2) };

    if updated.is_empty() {
        return;
    }

    let orientation: [u8; N] =
        std::array::from_fn(|i| ORIENT_TBL[ksq[i].idx()] ^ (56 * (perspective.0 ^ i as u8)));
    #[cfg(use_avx512icl)]
    let enemy: [Bitboard; N] = std::array::from_fn(|i| {
        if perspective.0 ^ i as u8 == WHITE.0 {
            pawns_b
        } else {
            pawns_w
        }
    });

    // Each square's pawn id, as make_pawn_id gives it on the oriented board,
    // for a pawn of the perspective's color or of the other
    #[cfg(use_avx512icl)]
    let ids: [__m512i; N] = std::array::from_fn(|i| {
        use crate::bitboard::all_squares;
        // SAFETY: the build targets Ice Lake's AVX-512.
        unsafe {
            let adjusted = _mm512_sub_epi8(
                _mm512_xor_si512(all_squares(), _mm512_set1_epi8(orientation[i] as i8)),
                _mm512_set1_epi8(8),
            );
            _mm512_mask_add_epi8(adjusted, enemy[i].0, adjusted, _mm512_set1_epi8(48))
        }
    });

    // Enumerate changed pawns and their partners once for all requested
    // perspectives.
    let unchanged = (pawns_w | pawns_b) & !updated;
    let mut u = updated;
    while u.any() {
        let a = u.pop_lsb();
        let partners = pawn_pair_bb(a) & (unchanged | u);
        if partners.is_empty() {
            continue;
        }

        // With Ice Lake's AVX-512, as in Stockfish's avx512icl builds, the
        // indices of the pawn's pairs in one vector store, their weights not
        // prefetched.
        #[cfg(use_avx512icl)]
        {
            let _ = (prefetch_base, prefetch_stride);
            let n = partners.popcount() as usize;
            debug_assert!(n <= 16);
            for i in 0..N {
                let a_id =
                    u16::from((a.0 ^ orientation[i]).wrapping_sub(8)) + 48 * enemy[i].has(a) as u16;
                // SAFETY: the build targets Ice Lake's AVX-512, and the store
                // writes 16 indices, the first those of the pawn's pairs: at
                // most 15, as the other pawns are.
                unsafe {
                    let pids = _mm256_cvtepu8_epi16(_mm512_castsi512_si128(
                        _mm512_maskz_compress_epi8(partners.0, ids[i]),
                    ));
                    let indices = pp_idx_epi16(_mm256_set1_epi16(a_id as i16), pids);
                    // Below END, as positions hold no pawns on the first or
                    // last rank; the bound keeps any index within the
                    // weights, as push's does.
                    let indices = _mm256_min_epu16(indices, _mm256_set1_epi16((END - 1) as i16));
                    out[i].extend_by_store(16, n, |p| _mm256_storeu_si256(p.cast(), indices));
                }
            }
        }

        #[cfg(not(use_avx512icl))]
        {
            let a_color = if pawns_b.has(a) { BLACK } else { WHITE };
            let a_id: [u32; N] = std::array::from_fn(|i| {
                make_pawn_id(
                    Color(a_color.0 ^ perspective.0 ^ i as u8),
                    usize::from(a.0 ^ orientation[i]),
                )
            });

            let mut push_pair = |to: Square, paired_color: Color| {
                for i in 0..N {
                    let b_id = make_pawn_id(
                        Color(paired_color.0 ^ perspective.0 ^ i as u8),
                        usize::from(to.0 ^ orientation[i]),
                    );
                    let index = pp_idx(a_id[i], b_id);
                    if !prefetch_base.is_null() {
                        prefetch_low(prefetch_base.wrapping_add(index as usize * prefetch_stride));
                    }
                    push(out[i], index);
                }
            };

            for to in pawns_b & partners {
                push_pair(to, BLACK);
            }
            for to in pawns_w & partners {
                push_pair(to, WHITE);
            }
        }
    }
}

/// The pairs that a change of pawns removes and adds, from each of the
/// perspectives of `append_indices_impl`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn append_changed_indices_impl<const N: usize, const M: usize>(
    perspective: Color,
    ksq: [Square; N],
    diff: &DirtyPawnPairs,
    removed: [&mut ValueList<u16, M>; N],
    added: [&mut ValueList<u16, M>; N],
    prefetch_base: *const i8,
    prefetch_stride: usize,
) {
    let [white_before, black_before] = diff.before;
    let [white_after, black_after] = diff.after;

    if white_before == white_after && black_before == black_after {
        return;
    }

    append_indices_impl(
        perspective,
        ksq,
        (white_after & !white_before) | (black_after & !black_before),
        white_after,
        black_after,
        added,
        prefetch_base,
        prefetch_stride,
    );
    append_indices_impl(
        perspective,
        ksq,
        (white_before & !white_after) | (black_before & !black_after),
        white_before,
        black_before,
        removed,
        prefetch_base,
        prefetch_stride,
    );
}

pub fn append_active_indices(perspective: Color, pos: &Position, active: &mut IndexList) {
    let white = pos.pieces_cpt(WHITE, PAWN);
    let black = pos.pieces_cpt(BLACK, PAWN);
    append_indices_impl(
        perspective,
        [pos.king_square(perspective)],
        white | black,
        white,
        black,
        [active],
        std::ptr::null(),
        0,
    );
}

/// The pairs of `append_active_indices` on any board, as Atomic's
/// positions, the white pawns' pairs first, as Stockfish listed them before
/// it enumerated every pawn in turn: the same pairs in another order. Chess
/// keeps its own, whose machine code a generic function would change.
pub fn append_active_indices_of<B: ThreatBoard>(
    perspective: Color,
    pos: &B,
    active: &mut IndexList,
) {
    let ksq = pos.king_square(perspective);
    let white = pos.pieces_cpt(WHITE, PAWN);
    let black = pos.pieces_cpt(BLACK, PAWN);

    let mut bb = white;
    while bb.any() {
        let from = bb.pop_lsb();
        let band = pawn_pair_bb(from);
        for to in band & bb {
            push(active, make_index(perspective, WHITE, from, to, WHITE, ksq));
        }
        for to in band & black {
            push(active, make_index(perspective, WHITE, from, to, BLACK, ksq));
        }
    }

    let mut bb = black;
    while bb.any() {
        let from = bb.pop_lsb();
        let band = pawn_pair_bb(from);
        for to in band & bb {
            push(active, make_index(perspective, BLACK, from, to, BLACK, ksq));
        }
    }
}

#[inline(always)]
pub fn append_changed_indices<const M: usize>(
    perspective: Color,
    ksq: Square,
    diff: &DirtyPawnPairs,
    removed: &mut ValueList<u16, M>,
    added: &mut ValueList<u16, M>,
    prefetch_base: *const i8,
    prefetch_stride: usize,
) {
    append_changed_indices_impl(
        perspective,
        [ksq],
        diff,
        [removed],
        [added],
        prefetch_base,
        prefetch_stride,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn append_changed_indices_both<const M: usize>(
    white_ksq: Square,
    black_ksq: Square,
    diff: &DirtyPawnPairs,
    white_removed: &mut ValueList<u16, M>,
    white_added: &mut ValueList<u16, M>,
    black_removed: &mut ValueList<u16, M>,
    black_added: &mut ValueList<u16, M>,
    prefetch_base: *const i8,
    prefetch_stride: usize,
) {
    append_changed_indices_impl(
        WHITE,
        [white_ksq, black_ksq],
        diff,
        [white_removed, black_removed],
        [white_added, black_added],
        prefetch_base,
        prefetch_stride,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_pair_has_its_own_feature() {
        let mut seen = vec![false; Pp3Wide::DIMENSIONS as usize];
        for (c1, c2) in [(WHITE, WHITE), (WHITE, BLACK), (BLACK, BLACK)] {
            for s1 in 8..56 {
                for s2 in 8..56 {
                    // Pawns of either color on one square have a feature
                    // too, though no position holds them.
                    if c1 == c2 && s2 <= s1 {
                        continue;
                    }
                    let index = make_index(WHITE, c1, Square(s1), Square(s2), c2, Square::E1);
                    let index = (index - Pp3Wide::INDEX_BASE) as usize;
                    assert!(!seen[index]);
                    seen[index] = true;
                    // The pair is the same both ways round.
                    assert_eq!(
                        make_index(WHITE, c2, Square(s2), Square(s1), c1, Square::E1),
                        index as u32 + Pp3Wide::INDEX_BASE
                    );
                }
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// Every pair of pawns of a board, from a perspective, by make_index,
    /// sorted.
    fn pairs(perspective: Color, ksq: Square, pawns: [Bitboard; COLOR_NB]) -> Vec<u16> {
        let mut pairs = Vec::new();
        for c1 in [WHITE, BLACK] {
            for s1 in pawns[c1.idx()] {
                for c2 in [WHITE, BLACK] {
                    for s2 in pawns[c2.idx()] & pawn_pair_bb(s1) {
                        assert!(pawn_pair_bb(s2).has(s1));
                        if (c1.0, s1.0) < (c2.0, s2.0) {
                            pairs.push(make_index(perspective, c1, s1, s2, c2, ksq) as u16);
                        }
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }

    fn sorted(list: &IndexList) -> Vec<u16> {
        let mut values = list.as_slice().to_vec();
        values.sort_unstable();
        values
    }

    /// Up to 8 pawns of each color, on ranks 2 to 7, on squares of their own.
    fn random_pawns(rng: &mut crate::misc::Prng) -> [Bitboard; COLOR_NB] {
        let mut pawns = [Bitboard::EMPTY; COLOR_NB];
        for c in [WHITE, BLACK] {
            for _ in 0..rng.rand64() % 9 {
                let s = Square(8 + (rng.rand64() % 48) as u8);
                if !(pawns[0] | pawns[1]).has(s) {
                    pawns[c.idx()] |= s;
                }
            }
        }
        pawns
    }

    /// The pairs a board's pawns make, and those a change of pawns removes
    /// and adds, from either perspective or both at once, are those of
    /// every pair of pawns, as make_index gives them.
    #[test]
    fn pairs_listed_are_those_of_the_boards() {
        let mut rng = crate::misc::Prng::new(20261107);
        for _ in 0..2000 {
            let before = random_pawns(&mut rng);
            let mut after = before;
            for _ in 0..1 + rng.rand64() % 3 {
                let c = (rng.rand64() % 2) as usize;
                let s = Square(8 + (rng.rand64() % 48) as u8);
                if after[c].has(s) {
                    after[c] ^= s;
                } else if !after[1 - c].has(s) && after[c].popcount() < 8 {
                    after[c] |= s;
                }
            }
            let ksq = [WHITE, BLACK].map(|_| Square((rng.rand64() % 64) as u8));
            let diff = DirtyPawnPairs { before, after };

            let mut both = [(); 4].map(|_| IndexList::new());
            let [white_removed, white_added, black_removed, black_added] = &mut both;
            append_changed_indices_both(
                ksq[0],
                ksq[1],
                &diff,
                white_removed,
                white_added,
                black_removed,
                black_added,
                std::ptr::null(),
                0,
            );

            for perspective in [WHITE, BLACK] {
                let p = perspective.idx();
                let (pairs_before, pairs_after) = (
                    pairs(perspective, ksq[p], before),
                    pairs(perspective, ksq[p], after),
                );

                let mut active = IndexList::new();
                append_indices_impl(
                    perspective,
                    [ksq[p]],
                    before[0] | before[1],
                    before[0],
                    before[1],
                    [&mut active],
                    std::ptr::null(),
                    0,
                );
                assert_eq!(sorted(&active), pairs_before);

                let (mut removed, mut added) = (IndexList::new(), IndexList::new());
                append_changed_indices(
                    perspective,
                    ksq[p],
                    &diff,
                    &mut removed,
                    &mut added,
                    std::ptr::null(),
                    0,
                );
                let only = |a: &[u16], b: &[u16]| -> Vec<u16> {
                    a.iter().copied().filter(|x| !b.contains(x)).collect()
                };
                assert_eq!(sorted(&removed), only(&pairs_before, &pairs_after));
                assert_eq!(sorted(&added), only(&pairs_after, &pairs_before));

                // Both perspectives at once list each one's pairs in the
                // same order.
                assert_eq!(both[2 * p].as_slice(), removed.as_slice());
                assert_eq!(both[2 * p + 1].as_slice(), added.as_slice());
            }
        }
    }
}
