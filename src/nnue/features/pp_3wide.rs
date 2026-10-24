//! The input features PP_3Wide: each pair of pawns on the same or adjacent
//! files, by their colors and squares, the board mirrored as for
//! FullThreats. Their indices follow the threats', in the same list.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/features/pp_3wide.h` and `.cpp`: Stockfish, a UCI chess playing
//! engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish
//! developers (see AUTHORS), under the GNU General Public License, version 3
//! or later.

use super::full_threats::{self, FullThreats};
use crate::bitboard::{Bitboard, pawn_pair_bb};
#[cfg(not(use_avx512icl))]
use crate::misc::prefetch_low;
use crate::position::Position;
use crate::types::*;

pub use super::full_threats::IndexList;

pub struct Pp3Wide;

impl Pp3Wide {
    pub const HASH_VALUE: u32 = 0x86f2_b1dd;

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

#[inline(always)]
pub fn make_index(
    perspective: Color,
    color: Color,
    from: Square,
    to: Square,
    paired_color: Color,
    ksq: Square,
) -> u32 {
    let orientation = usize::from(full_threats::ORIENT_TBL[ksq.idx()]) ^ (56 * perspective.idx());
    let from_oriented = from.idx() ^ orientation;
    let to_oriented = to.idx() ^ orientation;

    let color_oriented = Color(color.0 ^ perspective.0);
    let paired_color_oriented = Color(paired_color.0 ^ perspective.0);

    let id_a = make_pawn_id(color_oriented, from_oriented);
    let id_b = make_pawn_id(paired_color_oriented, to_oriented);
    let hi = id_a.max(id_b);
    let lo = id_a.min(id_b);

    (hi.wrapping_mul(hi.wrapping_sub(1)) / 2)
        .wrapping_add(lo)
        .wrapping_add(Pp3Wide::INDEX_BASE)
}

/// Pushes a pawn pair's index, as Stockfish does, if it is below END. Only
/// pawns on the first or last rank, which positions never hold, can give
/// one beyond, so the bound changes no evaluation and keeps every index
/// within the weights.
#[inline(always)]
fn push(list: &mut IndexList, index: u32) {
    list.push_back_if_lt(index as u16, END);
}

pub fn append_active_indices(perspective: Color, pos: &Position, active: &mut IndexList) {
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

/// The pairs that the pawns `updated_w` and `updated_b`, among the pawns
/// `pawns_w` and `pawns_b`, make, each passed to `push` once with the
/// color, the squares and the paired color.
#[cfg(not(use_avx512icl))]
#[inline(always)]
fn generate(
    updated_w: Bitboard,
    updated_b: Bitboard,
    pawns_w: Bitboard,
    pawns_b: Bitboard,
    mut push: impl FnMut(Color, Square, Square, Color),
) {
    let unchanged = (pawns_w | pawns_b) & !(updated_w | updated_b);
    let mut updated = updated_w | updated_b;
    while updated.any() {
        let from = updated.pop_lsb();
        let mask = pawn_pair_bb(from) & (unchanged | updated);
        let color = if pawns_b.has(from) { BLACK } else { WHITE };

        for paired in pawns_b & mask {
            push(color, from, paired, BLACK);
        }
        for paired in pawns_w & mask {
            push(color, from, paired, WHITE);
        }
    }
}

#[inline(always)]
pub fn append_changed_indices(
    perspective: Color,
    ksq: Square,
    diff: &DirtyPawnPairs,
    removed: &mut IndexList,
    added: &mut IndexList,
    prefetch_base: *const i8,
    prefetch_stride: usize,
) {
    let [white_before, black_before] = diff.before;
    let [white_after, black_after] = diff.after;

    if white_before == white_after && black_before == black_after {
        return;
    }

    // With Ice Lake's AVX-512, as in Stockfish's avx512icl builds, the
    // indices of each pawn's pairs in one vector store, their weights not
    // prefetched.
    #[cfg(use_avx512icl)]
    {
        let _ = (prefetch_base, prefetch_stride);
        write_indices(
            perspective,
            ksq,
            white_after & !white_before,
            black_after & !black_before,
            white_after,
            black_after,
            added,
        );
        write_indices(
            perspective,
            ksq,
            white_before & !white_after,
            black_before & !black_after,
            white_before,
            black_before,
            removed,
        );
    }

    #[cfg(not(use_avx512icl))]
    {
        let index_of = |color: Color, from: Square, to: Square, paired_color: Color| {
            let index = make_index(perspective, color, from, to, paired_color, ksq);
            prefetch_low(prefetch_base.wrapping_add(index as usize * prefetch_stride));
            index
        };

        generate(
            white_after & !white_before,
            black_after & !black_before,
            white_after,
            black_after,
            |c, from, to, paired| push(added, index_of(c, from, to, paired)),
        );
        generate(
            white_before & !white_after,
            black_before & !black_after,
            white_before,
            black_before,
            |c, from, to, paired| push(removed, index_of(c, from, to, paired)),
        );
    }
}

/// The indices from `perspective` of the pairs that the pawns `updated_w`
/// and `updated_b`, among the pawns `pawns_w` and `pawns_b`, make, written
/// to `out` in a vector store for each of those pawns: Stockfish's
/// `generate` for Ice Lake's AVX-512.
#[cfg(use_avx512icl)]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn write_indices(
    perspective: Color,
    ksq: Square,
    updated_w: Bitboard,
    updated_b: Bitboard,
    pawns_w: Bitboard,
    pawns_b: Bitboard,
    out: &mut IndexList,
) {
    use crate::bitboard::all_squares;
    use std::arch::x86_64::*;

    let orientation = full_threats::ORIENT_TBL[ksq.idx()] ^ (56 * perspective.0);
    let (friendly, enemy) = if perspective == WHITE {
        (pawns_w, pawns_b)
    } else {
        (pawns_b, pawns_w)
    };
    let unchanged = (pawns_w | pawns_b) & !(updated_w | updated_b);

    // SAFETY: the build targets Ice Lake's AVX-512, and each store writes 16
    // indices, the first those of the pawn's pairs: at most 15, as the
    // other pawns are.
    unsafe {
        // Each square's pawn id, as make_pawn_id gives it on the oriented
        // board, for a pawn of the perspective's color or of the other
        let adjusted = _mm512_sub_epi8(
            _mm512_xor_si512(all_squares(), _mm512_set1_epi8(orientation as i8)),
            _mm512_set1_epi8(8),
        );
        let ids = _mm512_mask_blend_epi8(
            friendly.0,
            _mm512_add_epi8(adjusted, _mm512_set1_epi8(48)),
            adjusted,
        );

        let mut updated = updated_w | updated_b;
        while updated.any() {
            let a = updated.pop_lsb();
            let partners = pawn_pair_bb(a) & (unchanged | updated);
            let n = partners.popcount() as usize;
            if n == 0 {
                continue;
            }

            let color_off = if enemy.has(a) { 48 } else { 0 };
            let a_id = u16::from((a.0 ^ orientation).wrapping_sub(8)) + color_off;
            let pids = _mm256_cvtepu8_epi16(_mm512_castsi512_si128(_mm512_maskz_compress_epi8(
                partners.0, ids,
            )));

            // make_index's hi * (hi - 1) / 2 + lo + INDEX_BASE, in 16 bits
            let a = _mm256_set1_epi16(a_id as i16);
            let hi = _mm256_max_epu16(a, pids);
            let lo = _mm256_min_epu16(a, pids);
            let prod = _mm256_mullo_epi16(hi, _mm256_sub_epi16(hi, _mm256_set1_epi16(1)));
            let indices = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_srli_epi16::<1>(prod), lo),
                _mm256_set1_epi16(Pp3Wide::INDEX_BASE as i16),
            );
            // Below END, as positions hold no pawns on the first or last
            // rank; the bound keeps any index within the weights, as push's
            // does.
            let indices = _mm256_min_epu16(indices, _mm256_set1_epi16((END - 1) as i16));

            out.extend_by_store(16, n, |p| _mm256_storeu_si256(p.cast(), indices));
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn append_changed_indices_both(
    white_ksq: Square,
    black_ksq: Square,
    diff: &DirtyPawnPairs,
    white_removed: &mut IndexList,
    white_added: &mut IndexList,
    black_removed: &mut IndexList,
    black_added: &mut IndexList,
    prefetch_base: *const i8,
    prefetch_stride: usize,
) {
    // With Ice Lake's AVX-512, as in Stockfish's avx512icl builds, each
    // perspective's indices in vector stores
    #[cfg(use_avx512icl)]
    {
        append_changed_indices(
            WHITE,
            white_ksq,
            diff,
            white_removed,
            white_added,
            prefetch_base,
            prefetch_stride,
        );
        append_changed_indices(
            BLACK,
            black_ksq,
            diff,
            black_removed,
            black_added,
            prefetch_base,
            prefetch_stride,
        );
    }

    #[cfg(not(use_avx512icl))]
    {
        let [white_before, black_before] = diff.before;
        let [white_after, black_after] = diff.after;

        if white_before == white_after && black_before == black_after {
            return;
        }

        let indices_of = |color: Color, from: Square, to: Square, paired_color: Color| {
            let white_index = make_index(WHITE, color, from, to, paired_color, white_ksq);
            let black_index = make_index(BLACK, color, from, to, paired_color, black_ksq);

            prefetch_low(prefetch_base.wrapping_add(white_index as usize * prefetch_stride));
            prefetch_low(prefetch_base.wrapping_add(black_index as usize * prefetch_stride));

            (white_index, black_index)
        };

        generate(
            white_after & !white_before,
            black_after & !black_before,
            white_after,
            black_after,
            |c, from, to, paired| {
                let (white_index, black_index) = indices_of(c, from, to, paired);
                push(white_added, white_index);
                push(black_added, black_index);
            },
        );
        generate(
            white_before & !white_after,
            black_before & !black_after,
            white_before,
            black_before,
            |c, from, to, paired| {
                let (white_index, black_index) = indices_of(c, from, to, paired);
                push(white_removed, white_index);
                push(black_removed, black_index);
            },
        );
    }
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
}
