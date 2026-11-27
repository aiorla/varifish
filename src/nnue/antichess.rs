//! Antichess's networks in Stockfish's design, SFNNv20, adapted for kingless
//! play (0, 1, or many kings per side, no horizontal king-mirroring):
//!
//! - **AntichessThreats** (60,032 features): every piece (including kings)
//!   attacking every enemy piece (all 6 × 6 pairs, covering every forced
//!   capture) plus friendly attacks distinguished by pawn vs. non-pawn target
//!   (covering forced recapture links), vertically oriented per perspective.
//! - **PP_3Wide** (4,560 features): pawn pairs on the same or adjacent files,
//!   vertically oriented per perspective (indices `60_032..64_592`, fitting in
//!   `u16` alongside threats to reuse Stockfish's SIMD paired-widening-add
//!   kernels).
//! - **AntichessPsq** (4 × 768 = 3,072 features): all 12 piece types on all 64
//!   squares in 4 total-piece-count blocks (`1..=8`, `9..=16`, `17..=24`,
//!   `25..=32`), with a compact 4-block Finny cache per perspective.
//! - **Layer stacks** (144 raw stacks): 6 own piece-count bands × 6 opponent
//!   piece-count bands × 4 pawn-presence states, reusing `NetworkArchitecture`
//!   (`1024 -> 32 -> 32 -> 1`).
//!
//! Its header hash XORs `"Anti"` (`0x416E_7469`) into the architecture hash so
//! no chess, Atomic, Three-check, or Crazyhouse network reads as Antichess.

#![allow(unused_unsafe)]

use std::mem::MaybeUninit;

use crate::attacks::{PSEUDO_ATTACKS, attacks_bb};
use crate::bitboard::{Bitboard, pawn_pair_bb, square_bb};
use crate::memory::LargePageBox;
use crate::misc::ValueList;
use crate::nnue::accumulator::{
    DIMENSIONS, Tile, apply_changed_features, apply_features, apply_features_but_odd,
    apply_odd_features, load_tile, store_tile,
};
use crate::nnue::architecture::{L1, L2, L3, NetworkArchitecture};
use crate::nnue::common::{
    Aligned, OUTPUT_SCALE, Reader, VERSION, write_i8s, write_i16s, write_u32,
};
use crate::nnue::eval_file::NetworkFile;
#[cfg(use_paired_widening_add)]
use crate::nnue::feature_transformer::WIDENING_PAIR_ORDER;
use crate::nnue::feature_transformer::{
    FeatureTransformer as ChessFeatureTransformer, INVERSE_PACKUS_EPI16_ORDER, PACKUS_EPI16_ORDER,
    combine_hash, permute, unpermuted_weights,
};
use crate::nnue::features::pp_3wide::Pp3Wide;
use crate::nnue::network::{
    LAYER_STACK_TABLE_HASH, read_hash, read_header, read_layer_stack_table_n,
    write_layer_stack_table_n,
};
use crate::nnue::nnz_helper::NnzInfo;
use crate::nnue::simd::TILE_HEIGHT;
use crate::nnue::{Accumulators, Caches};
use crate::position::Position;
use crate::types::*;
use crate::variant::Antichess;

const _: () = assert!(DIMENSIONS == L1);

/// Default filename for Antichess's dedicated SFNNv20 network, which the
/// Makefile's `ANTICHESS_NET` names too.
pub const EVAL_FILE_DEFAULT_NAME: &str = "antichess-64def68f336c.nnue";

/// What sets Antichess's SFNNv20 networks apart from other variants: `"Anti"`.
pub const ANTICHESS_HASH: u32 = 0x416E_7469;

/// Number of output dimensions for one side.
const HALF_DIMENSIONS: usize = L1;

/// Number of piece-square blocks by total board piece count (`1..=8`,
/// `9..=16`, `17..=24`, `25..=32`).
pub const PSQ_NUM_BLOCKS: usize = 4;

/// Piece-square features per block (`12` piece types × `64` squares).
pub const PSQ_BLOCK_DIMENSIONS: usize = 12 * SQUARE_NB;

/// Total piece-square features (`4 × 768 = 3,072`).
pub const PSQ_DIMENSIONS: usize = PSQ_NUM_BLOCKS * PSQ_BLOCK_DIMENSIONS;

/// Target buckets per attacker: 6 enemy piece types (`0..6`) + friendly pawn
/// (`6`) + friendly non-pawn (`7`).
const THREAT_TARGETS_PER_ATTACKER: usize = 8;

/// Number of Antichess threat features: `7,504 × 8 = 60,032`.
pub const THREAT_DIMENSIONS: usize = 60_032;

/// Number of pawn-pair features (`PP_3Wide`, 4,560), placed directly after
/// threats in the same `u16` index space.
pub const PAIR_DIMENSIONS: usize = Pp3Wide::DIMENSIONS as usize;

/// Total threat + pawn-pair feature rows (`64,592 <= 65,535`).
pub const THREAT_AND_PAIR_DIMENSIONS: usize = THREAT_DIMENSIONS + PAIR_DIMENSIONS;

const _: () = assert!(THREAT_AND_PAIR_DIMENSIONS <= u16::MAX as usize);

/// Number of piece-count bands per side for layer stack selection.
pub const PIECE_BANDS: usize = 6;

/// Number of pawn-presence states (`2 × has_own_pawns + has_their_pawns`).
pub const PAWN_STATES: usize = 4;

/// Raw layer stacks for Antichess: `6 × 6 × 4 = 144`.
pub const RAW_LAYER_STACKS: usize = PIECE_BANDS * PIECE_BANDS * PAWN_STATES;
pub const LAYER_STACKS: usize = RAW_LAYER_STACKS;

/// Piece-square index list (at most 32 pieces on the board).
pub type PsqIndexList = ValueList<u16, 32>;
/// Active threat + pawn-pair index list.
pub type ActiveThreatList = ValueList<u16, 512>;
/// Changed threat + pawn-pair index list for a single move.
type ChangedThreatList = ValueList<u16, 256>;

/// Piece-square block of a position by total piece count (`0..4`).
#[inline(always)]
pub fn psq_block_of(total_pieces: i32) -> usize {
    ((total_pieces.saturating_sub(1)) / 8).clamp(0, (PSQ_NUM_BLOCKS - 1) as i32) as usize
}

/// Piece-square feature index within a block (`0..768`) for `pc` on `sq` from
/// `perspective` (vertically flipped for Black, no horizontal king mirroring).
#[inline(always)]
pub const fn psq_index(perspective: Color, sq: Square, pc: Piece) -> u16 {
    let oriented_sq = (sq.0 ^ (56 * perspective.0)) as u16;
    let rel_color = (pc.color().0 ^ perspective.0) as u16;
    let pt_idx = (pc.type_of().0 - PAWN.0) as u16;
    (rel_color * 6 + pt_idx) * 64 + oriented_sq
}

/// Maps a side's piece count (`0..=16`) to one of `PIECE_BANDS = 6` bands:
/// `0: <=1`, `1: 2`, `2: 3..=4`, `3: 5..=7`, `4: 8..=11`, `5: 12..=16`.
#[inline(always)]
pub const fn piece_band(count: i32) -> usize {
    match count {
        ..=1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=7 => 3,
        8..=11 => 4,
        _ => 5,
    }
}

/// Raw layer stack index (`0..144`) for an Antichess position.
#[inline(always)]
pub fn raw_layer_stack(
    own_pieces: i32,
    their_pieces: i32,
    own_pawns: bool,
    their_pawns: bool,
) -> usize {
    let own_b = piece_band(own_pieces);
    let their_b = piece_band(their_pieces);
    let pawn_state = 2 * usize::from(own_pawns) + usize::from(their_pawns);
    (own_b * PIECE_BANDS + their_b) * PAWN_STATES + pawn_state
}

#[inline(always)]
fn pos_raw_layer_stack(pos: &Position<Antichess>) -> usize {
    let stm = pos.side_to_move();
    raw_layer_stack(
        pos.count(stm, ALL_PIECES),
        pos.count(!stm, ALL_PIECES),
        pos.count(stm, PAWN) > 0,
        pos.count(!stm, PAWN) > 0,
    )
}

/// Compile-time lookup tables for `AntichessThreats`.
#[repr(C, align(64))]
struct ThreatLuts {
    /// Base feature index for `(oriented_attacker, oriented_attacked)`.
    lut1: [[u32; PIECE_NB]; PIECE_NB],
    /// Attack slot index `(pt, from_oriented, to_oriented)` within an
    /// attacker's attack block.
    lut2: [[[u16; SQUARE_NB]; SQUARE_NB]; PIECE_TYPE_NB],
}

const fn init_threat_luts() -> ThreatLuts {
    let mut tables = ThreatLuts {
        lut1: [[THREAT_DIMENSIONS as u32; PIECE_NB]; PIECE_NB],
        lut2: [[[0; SQUARE_NB]; SQUARE_NB]; PIECE_TYPE_NB],
    };
    let mut offsets = [0u16; PIECE_TYPE_NB];

    let mut pt = PAWN.idx();
    while pt <= KING.idx() {
        let mut offset = 0u16;
        let mut from = 0usize;
        while from < SQUARE_NB {
            let mut to = 0usize;
            while to < SQUARE_NB {
                let attacks = if pt != PAWN.idx() {
                    PSEUDO_ATTACKS[pt][from].0
                } else if to < from {
                    PSEUDO_ATTACKS[BLACK.idx()][from].0
                } else {
                    PSEUDO_ATTACKS[WHITE.idx()][from].0
                };
                tables.lut2[pt][from][to] =
                    offset + (((1u64 << to) - 1) & attacks).count_ones() as u16;
                to += 1;
            }
            if pt != PAWN.idx() {
                offset += PSEUDO_ATTACKS[pt][from].0.count_ones() as u16;
            } else if from >= Square::A2.0 as usize && from <= Square::H7.0 as usize {
                offset += PSEUDO_ATTACKS[WHITE.idx()][from].0.count_ones() as u16;
            }
            from += 1;
        }
        offsets[pt] = offset;
        pt += 1;
    }

    let mut cumulative_offset = 0u32;
    let mut i = 0;
    while i < PIECES.len() {
        let attacker = PIECES[i];
        let attacker_type = attacker.type_of().0 as usize;
        let piece_span = offsets[attacker_type] as u32;

        let mut j = 0;
        while j < PIECES.len() {
            let attacked = PIECES[j];
            let enemy = (attacker.0 ^ attacked.0) == 8;
            let attacked_type = attacked.type_of().0 as usize;
            let bucket = if enemy {
                (attacked_type - PAWN.0 as usize) as u32
            } else if attacked_type == PAWN.0 as usize {
                6
            } else {
                7
            };
            tables.lut1[attacker.idx()][attacked.idx()] = cumulative_offset + bucket * piece_span;
            j += 1;
        }

        cumulative_offset += THREAT_TARGETS_PER_ATTACKER as u32 * piece_span;
        i += 1;
    }

    assert!(cumulative_offset as usize == THREAT_DIMENSIONS);
    tables
}

static THREAT_LUTS: ThreatLuts = init_threat_luts();

/// Threat feature index (`0..THREAT_DIMENSIONS`) for `attacker` on `from`
/// attacking `attacked` on `to`, vertically oriented for `perspective`.
#[inline(always)]
pub fn threat_index(
    perspective: Color,
    attacker: Piece,
    from: Square,
    to: Square,
    attacked: Piece,
) -> u16 {
    let flip = 56 * perspective.idx();
    let from_oriented = from.idx() ^ flip;
    let to_oriented = to.idx() ^ flip;
    let swap = 8 * perspective.idx();
    let attacker_oriented = attacker.idx() ^ swap;
    let attacked_oriented = attacked.idx() ^ swap;

    let idx = THREAT_LUTS.lut1[attacker_oriented][attacked_oriented]
        + u32::from(THREAT_LUTS.lut2[attacker.type_of().idx()][from_oriented][to_oriented]);
    debug_assert!((idx as usize) < THREAT_DIMENSIONS);
    idx as u16
}

/// Pawn-pair feature index (`THREAT_DIMENSIONS..THREAT_AND_PAIR_DIMENSIONS`)
/// for two pawns `(c1, sq1)` and `(c2, sq2)` on the same or adjacent files,
/// vertically oriented for `perspective` (no horizontal king mirroring).
#[inline(always)]
pub fn pawn_pair_index(perspective: Color, c1: Color, sq1: Square, c2: Color, sq2: Square) -> u16 {
    let flip = 56 * perspective.idx();
    let s1 = sq1.idx() ^ flip;
    let s2 = sq2.idx() ^ flip;
    let rc1 = c1.idx() ^ perspective.idx();
    let rc2 = c2.idx() ^ perspective.idx();
    debug_assert!((Square::A2.idx()..=Square::H7.idx()).contains(&s1));
    debug_assert!((Square::A2.idx()..=Square::H7.idx()).contains(&s2));
    let id1 = (48 * rc1 + s1 - Square::A2.idx()) as u32;
    let id2 = (48 * rc2 + s2 - Square::A2.idx()) as u32;
    let hi = id1.max(id2);
    let lo = id1.min(id2);
    let idx = (hi * (hi - 1) / 2) + lo + THREAT_DIMENSIONS as u32;
    debug_assert!((idx as usize) < THREAT_AND_PAIR_DIMENSIONS);
    idx as u16
}

/// Appends all active `AntichessPsq` indices (within the current block,
/// `0..768`) for `perspective`.
pub fn append_active_psq(
    perspective: Color,
    board: &[Piece; SQUARE_NB],
    occupied: Bitboard,
    active: &mut PsqIndexList,
) {
    for sq in occupied {
        active.push(psq_index(perspective, sq, board[sq.idx()]));
    }
}

/// Appends all active `AntichessThreats` indices (`0..THREAT_DIMENSIONS`) for
/// `perspective` from a board snapshot.
pub fn append_active_threats_from_board<const N: usize>(
    perspective: Color,
    board: &[Piece; SQUARE_NB],
    occupied: Bitboard,
    active: &mut ValueList<u16, N>,
) {
    for from in occupied {
        let pc = board[from.idx()];
        let pt = pc.type_of();
        let attacks = if pt == PAWN {
            crate::attacks::pawn_attacks(pc.color(), from)
        } else {
            attacks_bb(pt, from, occupied)
        };
        let mut targets = attacks & occupied;
        while targets.any() {
            let to = targets.pop_lsb();
            active.push(threat_index(perspective, pc, from, to, board[to.idx()]));
        }
    }
}

/// Appends all active `AntichessThreats` and `PP_3Wide` indices for
/// `perspective` from a board snapshot (`board`, `occupied`, `pawns`).
pub fn append_active_threats_and_pairs_from_board(
    perspective: Color,
    board: &[Piece; SQUARE_NB],
    occupied: Bitboard,
    pawns: [Bitboard; COLOR_NB],
    active: &mut ActiveThreatList,
) {
    append_active_threats_from_board(perspective, board, occupied, active);
    append_active_pawn_pairs(perspective, pawns, active);
}

/// Appends all active `PP_3Wide` pawn-pair indices
/// (`THREAT_DIMENSIONS..THREAT_AND_PAIR_DIMENSIONS`) for `perspective`.
pub fn append_active_pawn_pairs<const N: usize>(
    perspective: Color,
    pawns: [Bitboard; COLOR_NB],
    active: &mut ValueList<u16, N>,
) {
    let wp = pawns[WHITE.idx()];
    let bp = pawns[BLACK.idx()];

    // White-White pairs (sq1 < sq2)
    let mut bb = wp;
    while bb.any() {
        let sq1 = bb.pop_lsb();
        let mut partners = pawn_pair_bb(sq1) & bb;
        while partners.any() {
            let sq2 = partners.pop_lsb();
            active.push(pawn_pair_index(perspective, WHITE, sq1, WHITE, sq2));
        }
    }

    // Black-Black pairs (sq1 < sq2)
    let mut bb = bp;
    while bb.any() {
        let sq1 = bb.pop_lsb();
        let mut partners = pawn_pair_bb(sq1) & bb;
        while partners.any() {
            let sq2 = partners.pop_lsb();
            active.push(pawn_pair_index(perspective, BLACK, sq1, BLACK, sq2));
        }
    }

    // White-Black pairs
    let mut bb = wp;
    while bb.any() {
        let sq1 = bb.pop_lsb();
        let mut partners = pawn_pair_bb(sq1) & bp;
        while partners.any() {
            let sq2 = partners.pop_lsb();
            active.push(pawn_pair_index(perspective, WHITE, sq1, BLACK, sq2));
        }
    }
}

/// Appends changed pawn-pair indices between `dpps.before` and `dpps.after`.
fn append_changed_pawn_pairs<const N: usize>(
    perspective: Color,
    dpps: &DirtyPawnPairs,
    removed: &mut ValueList<u16, N>,
    added: &mut ValueList<u16, N>,
) {
    let wb = dpps.before[WHITE.idx()];
    let bb = dpps.before[BLACK.idx()];
    let wa = dpps.after[WHITE.idx()];
    let ba = dpps.after[BLACK.idx()];
    if wb == wa && bb == ba {
        return;
    }

    let rem_w = wb & !wa;
    let add_w = wa & !wb;
    let rem_b = bb & !ba;
    let add_b = ba & !bb;

    let mut cur_w = wb;
    let mut cur_b = bb;

    for sq in rem_w {
        cur_w ^= square_bb(sq);
        let mask = pawn_pair_bb(sq);
        for p in mask & cur_w {
            removed.push(pawn_pair_index(perspective, WHITE, sq, WHITE, p));
        }
        for p in mask & cur_b {
            removed.push(pawn_pair_index(perspective, WHITE, sq, BLACK, p));
        }
    }
    for sq in rem_b {
        cur_b ^= square_bb(sq);
        let mask = pawn_pair_bb(sq);
        for p in mask & cur_w {
            removed.push(pawn_pair_index(perspective, BLACK, sq, WHITE, p));
        }
        for p in mask & cur_b {
            removed.push(pawn_pair_index(perspective, BLACK, sq, BLACK, p));
        }
    }
    for sq in add_w {
        let mask = pawn_pair_bb(sq);
        for p in mask & cur_w {
            added.push(pawn_pair_index(perspective, WHITE, sq, WHITE, p));
        }
        for p in mask & cur_b {
            added.push(pawn_pair_index(perspective, WHITE, sq, BLACK, p));
        }
        cur_w |= square_bb(sq);
    }
    for sq in add_b {
        let mask = pawn_pair_bb(sq);
        for p in mask & cur_w {
            added.push(pawn_pair_index(perspective, BLACK, sq, WHITE, p));
        }
        for p in mask & cur_b {
            added.push(pawn_pair_index(perspective, BLACK, sq, BLACK, p));
        }
        cur_b |= square_bb(sq);
    }
}

/// Appends changed piece-square indices (`0..768`) for a single move.
fn append_changed_psq(
    perspective: Color,
    dp: &DirtyPiece,
    removed: &mut PsqIndexList,
    added: &mut PsqIndexList,
) {
    if dp.pc != NO_PIECE {
        removed.push(psq_index(perspective, dp.from, dp.pc));
        if dp.to != Square::NONE {
            added.push(psq_index(perspective, dp.to, dp.pc));
        }
    }
    if dp.remove_sq != Square::NONE {
        removed.push(psq_index(perspective, dp.remove_sq, dp.remove_pc));
    }
    if dp.add_sq != Square::NONE {
        added.push(psq_index(perspective, dp.add_sq, dp.add_pc));
    }
}

/// Appends changed threat and pawn-pair indices for a single move.
fn append_changed_threats_and_pairs(
    perspective: Color,
    dirties: &Dirties,
    removed: &mut ChangedThreatList,
    added: &mut ChangedThreatList,
) {
    for &dt in dirties.dirty_threats.iter() {
        let idx = threat_index(
            perspective,
            dt.pc(),
            dt.pc_sq(),
            dt.threatened_sq(),
            dt.threatened_pc(),
        );
        if dt.add() {
            added.push(idx);
        } else {
            removed.push(idx);
        }
    }
    append_changed_pawn_pairs(perspective, &dirties.dirty_pawn_pairs, removed, added);
}

/// The feature transformer of `AntichessNetwork`.
pub struct AntichessFeatureTransformer {
    pub(crate) biases: Aligned<[i16; HALF_DIMENSIONS]>,
    pub(crate) threat_weights: Aligned<[i8; THREAT_AND_PAIR_DIMENSIONS * HALF_DIMENSIONS]>,
    pub(crate) psq_weights: Aligned<[i8; PSQ_DIMENSIONS * HALF_DIMENSIONS]>,
}

impl AntichessFeatureTransformer {
    pub const fn get_hash_value() -> u32 {
        combine_hash(&[
            ANTICHESS_HASH,
            THREAT_AND_PAIR_DIMENSIONS as u32,
            PSQ_DIMENSIONS as u32,
        ]) ^ (HALF_DIMENSIONS as u32)
    }

    /// Reads feature transformer parameters from a network file and permutes
    /// weights for the target's SIMD layout.
    pub fn read_parameters(&mut self, reader: &mut Reader) -> bool {
        reader.read_i16s(&mut self.biases.0);
        reader.read_i8s(&mut self.threat_weights.0);
        reader.read_i8s(&mut self.psq_weights.0);
        if !reader.good() {
            return false;
        }
        self.permute_weights();
        true
    }

    /// Writes feature transformer parameters in canonical (unpermuted) order.
    pub fn write_parameters(&self, out: &mut Vec<u8>) {
        let mut biases = self.biases.0;
        permute(&mut biases, &INVERSE_PACKUS_EPI16_ORDER);
        write_i16s(out, &biases);

        let threats = unpermuted_weights(&self.threat_weights.0);
        write_i8s(out, &threats);

        let psq = unpermuted_weights(&self.psq_weights.0);
        write_i8s(out, &psq);
    }

    pub(crate) fn permute_weights(&mut self) {
        permute(&mut self.biases.0, &PACKUS_EPI16_ORDER);
        permute(&mut self.threat_weights.0, &PACKUS_EPI16_ORDER);
        permute(&mut self.psq_weights.0, &PACKUS_EPI16_ORDER);
        #[cfg(use_paired_widening_add)]
        {
            permute(&mut self.threat_weights.0, &WIDENING_PAIR_ORDER);
            permute(&mut self.psq_weights.0, &WIDENING_PAIR_ORDER);
        }
    }

    #[inline(always)]
    fn psq_block_weights(&self, block: usize) -> *const i8 {
        // SAFETY: `block < PSQ_NUM_BLOCKS`.
        unsafe {
            self.psq_weights
                .0
                .as_ptr()
                .add(block * PSQ_BLOCK_DIMENSIONS * HALF_DIMENSIONS)
        }
    }
}

/// Accumulator state for one ply in an Antichess search.
#[repr(C, align(64))]
pub struct AntichessAccumulatorState {
    pub accumulation: [[i16; L1]; COLOR_NB],
    pub computed: [bool; COLOR_NB],
    pub dirties: Dirties,
    pub block_before: u8,
}

/// Compact 4-block Finny cache per perspective for Antichess PSQ refreshes.
pub struct AntichessCaches {
    entries: [[FinnyEntry; COLOR_NB]; PSQ_NUM_BLOCKS],
}

#[repr(C, align(64))]
struct FinnyEntry {
    accumulation: [i16; L1],
    pieces: [Piece; SQUARE_NB],
    piece_bb: Bitboard,
}

impl FinnyEntry {
    fn clear(&mut self, biases: &[i16; L1]) {
        self.accumulation = *biases;
        self.pieces = [NO_PIECE; SQUARE_NB];
        self.piece_bb = Bitboard::EMPTY;
    }
}

impl Caches for AntichessCaches {
    type Network = AntichessNetwork;

    fn new(network: Option<&AntichessNetwork>) -> LargePageBox<AntichessCaches> {
        // SAFETY: zero bytes make valid entries; cleared below when network is Some.
        let mut caches = unsafe { LargePageBox::<AntichessCaches>::zeroed() };
        if let Some(network) = network {
            caches.clear(network);
        }
        caches
    }

    fn clear(&mut self, network: &AntichessNetwork) {
        let biases = &network.feature_transformer.biases.0;
        for block_entries in &mut self.entries {
            for entry in block_entries {
                entry.clear(biases);
            }
        }
    }
}

/// Accumulator stack along an Antichess search line.
pub struct AntichessAccumulators {
    accumulators: [AntichessAccumulatorState; Self::MAX_SIZE],
    size: usize,
}

impl AntichessAccumulators {
    pub const MAX_SIZE: usize = MAX_PLY as usize + 1;

    pub fn new_stack() -> LargePageBox<Self> {
        // SAFETY: zero bytes make a valid stack; reset initializes root.
        let mut stack = unsafe { LargePageBox::<Self>::zeroed() };
        stack.reset_root();
        stack
    }

    fn reset_root(&mut self) {
        let root = &mut self.accumulators[0];
        root.dirties.dirty_piece = DirtyPiece::default();
        root.dirties.dirty_threats.clear();
        root.dirties.dirty_pawn_pairs = DirtyPawnPairs::default();
        root.computed = [false; COLOR_NB];
        root.block_before = 0;
        self.size = 1;
    }

    #[inline(always)]
    fn latest(&self) -> &AntichessAccumulatorState {
        &self.accumulators[self.size - 1]
    }

    /// Ensures the latest accumulator state is computed for `perspective`.
    fn evaluate_perspective(
        &mut self,
        ft: &AntichessFeatureTransformer,
        pos: &Position<Antichess>,
        caches: &mut AntichessCaches,
        perspective: Color,
    ) {
        let p = perspective.idx();
        let current = self.size - 1;
        if self.accumulators[current].computed[p] {
            return;
        }

        let current_block = psq_block_of(pos.count_both(ALL_PIECES)) as u8;

        // Walk back to find an ancestor with a computed accumulator and the
        // same PSQ block.
        let mut idx = current;
        let mut can_update = false;
        while idx > 0 {
            if self.accumulators[idx].block_before != current_block {
                break;
            }
            idx -= 1;
            if self.accumulators[idx].computed[p] {
                can_update = true;
                break;
            }
        }

        if !can_update {
            self.refresh(ft, pos, caches, perspective, current_block as usize);
            return;
        }

        // Incrementally step forward from `idx` to `current`.
        let psq_weights = ft.psq_block_weights(current_block as usize);
        let thr_weights = ft.threat_weights.0.as_ptr();
        while idx < current {
            let next = idx + 1;
            let (left, right) = self.accumulators.split_at_mut(next);
            let prev_acc = &left[idx].accumulation[p];
            let next_st = &mut right[0];

            let mut rem_psq = PsqIndexList::new();
            let mut add_psq = PsqIndexList::new();
            append_changed_psq(
                perspective,
                &next_st.dirties.dirty_piece,
                &mut rem_psq,
                &mut add_psq,
            );

            let mut rem_thr = ChangedThreatList::new();
            let mut add_thr = ChangedThreatList::new();
            append_changed_threats_and_pairs(
                perspective,
                &next_st.dirties,
                &mut rem_thr,
                &mut add_thr,
            );

            update_tiles(
                prev_acc,
                &mut next_st.accumulation[p],
                rem_psq.as_slice(),
                add_psq.as_slice(),
                rem_thr.as_slice(),
                add_thr.as_slice(),
                psq_weights,
                thr_weights,
            );
            next_st.computed[p] = true;
            idx = next;
        }
    }

    /// Refreshes `perspective` at the top of the stack using the 4-block Finny
    /// cache for PSQ and computing active threats + pawn pairs fresh.
    fn refresh(
        &mut self,
        ft: &AntichessFeatureTransformer,
        pos: &Position<Antichess>,
        caches: &mut AntichessCaches,
        perspective: Color,
        block: usize,
    ) {
        let p = perspective.idx();
        let entry = &mut caches.entries[block][p];
        let psq_weights = ft.psq_block_weights(block);

        let mut removed = PsqIndexList::new();
        let mut added = PsqIndexList::new();

        let occupied = pos.pieces();
        let changed = entry.piece_bb ^ occupied;
        // Pieces removed from previously occupied squares
        for sq in changed & entry.piece_bb {
            removed.push(psq_index(perspective, sq, entry.pieces[sq.idx()]));
            entry.pieces[sq.idx()] = NO_PIECE;
        }
        // Pieces added to previously empty squares
        for sq in changed & occupied {
            let pc = pos.piece_on(sq);
            added.push(psq_index(perspective, sq, pc));
            entry.pieces[sq.idx()] = pc;
        }
        // Squares occupied in both whose piece changed
        for sq in entry.piece_bb & occupied {
            let new_pc = pos.piece_on(sq);
            let old_pc = entry.pieces[sq.idx()];
            if old_pc != new_pc {
                removed.push(psq_index(perspective, sq, old_pc));
                added.push(psq_index(perspective, sq, new_pc));
                entry.pieces[sq.idx()] = new_pc;
            }
        }
        entry.piece_bb = occupied;

        if !removed.is_empty() || !added.is_empty() {
            update_psq_cache_tiles(
                &mut entry.accumulation,
                removed.as_slice(),
                added.as_slice(),
                psq_weights,
            );
        }

        // Compute active threats + pawn pairs and add onto the cached PSQ sum.
        let mut threats = ActiveThreatList::new();
        let mut board = [NO_PIECE; SQUARE_NB];
        for sq in occupied {
            board[sq.idx()] = pos.piece_on(sq);
        }
        append_active_threats_and_pairs_from_board(
            perspective,
            &board,
            occupied,
            [pos.pieces_cpt(WHITE, PAWN), pos.pieces_cpt(BLACK, PAWN)],
            &mut threats,
        );

        let current = self.size - 1;
        let dst = &mut self.accumulators[current].accumulation[p];
        let thr_weights = ft.threat_weights.0.as_ptr();
        add_threats_onto_psq(&entry.accumulation, dst, threats.as_slice(), thr_weights);
        self.accumulators[current].computed[p] = true;
    }
}

impl Accumulators<Antichess> for AntichessAccumulators {
    type Dirties = Dirties;

    fn new() -> LargePageBox<Self> {
        Self::new_stack()
    }

    fn reset(&mut self, _network: Option<&AntichessNetwork>) {
        self.reset_root();
    }

    #[inline(always)]
    fn push(&mut self, pos: &Position<Antichess>) -> &mut Dirties {
        debug_assert!(self.size < Self::MAX_SIZE);
        let st = &mut self.accumulators[self.size];
        st.computed = [false; COLOR_NB];
        st.dirties.dirty_piece = DirtyPiece::default();
        st.dirties.dirty_threats.clear();
        st.dirties.dirty_pawn_pairs = DirtyPawnPairs::default();
        st.block_before = psq_block_of(pos.count_both(ALL_PIECES)) as u8;
        self.size += 1;
        &mut st.dirties
    }

    #[inline(always)]
    fn pop(&mut self) {
        debug_assert!(self.size > 1);
        self.size -= 1;
    }
}

/// Applies changed PSQ and threat/pair indices from `src` into `dst`, tile by
/// tile, reusing `nnue::accumulator`'s SIMD kernels.
#[allow(clippy::too_many_arguments)]
fn update_tiles(
    src: &[i16; L1],
    dst: &mut [i16; L1],
    rem_psq: &[u16],
    add_psq: &[u16],
    rem_thr: &[u16],
    add_thr: &[u16],
    psq_weights: *const i8,
    thr_weights: *const i8,
) {
    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: `j + TILE_HEIGHT <= DIMENSIONS`, weights have full rows.
        unsafe {
            let mut acc: Tile = load_tile(j, src.as_ptr());
            let psq_tile = psq_weights.add(j);
            let thr_tile = thr_weights.add(j);
            let odd_rem = apply_features_but_odd::<-1, [u16]>(
                j, &mut acc, rem_psq, rem_thr, psq_tile, thr_tile,
            );
            let odd_add = apply_features_but_odd::<1, [u16]>(
                j, &mut acc, add_psq, add_thr, psq_tile, thr_tile,
            );
            apply_odd_features(&mut acc, odd_rem, odd_add);
            store_tile(j, dst.as_mut_ptr(), &acc);
        }
    }
}

/// Updates a Finny cache entry's PSQ accumulation in place.
fn update_psq_cache_tiles(
    acc_buf: &mut [i16; L1],
    removed: &[u16],
    added: &[u16],
    psq_weights: *const i8,
) {
    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: `j + TILE_HEIGHT <= DIMENSIONS`.
        unsafe {
            let mut acc: Tile = load_tile(j, acc_buf.as_ptr());
            apply_changed_features(&mut acc, removed, added, psq_weights.add(j));
            store_tile(j, acc_buf.as_mut_ptr(), &acc);
        }
    }
}

/// Copies `psq_acc` into `dst` while adding active threat + pawn-pair weights.
fn add_threats_onto_psq(
    psq_acc: &[i16; L1],
    dst: &mut [i16; L1],
    threats: &[u16],
    thr_weights: *const i8,
) {
    for j in (0..DIMENSIONS).step_by(TILE_HEIGHT) {
        // SAFETY: `j + TILE_HEIGHT <= DIMENSIONS`.
        unsafe {
            let mut acc: Tile = load_tile(j, psq_acc.as_ptr());
            apply_features::<1>(&mut acc, threats, thr_weights.add(j));
            store_tile(j, dst.as_mut_ptr(), &acc);
        }
    }
}

/// Evaluation by each stored layer stack of an Antichess network, for `eval`.
pub struct AntichessEvalTrace {
    pub positional: [Value; LAYER_STACKS],
    pub correct_bucket: usize,
    pub layer_stacks: usize,
}

/// Antichess's dedicated SFNNv20 network.
pub struct AntichessNetwork {
    pub(crate) feature_transformer: AntichessFeatureTransformer,
    network: [NetworkArchitecture; LAYER_STACKS],
    layer_stack_fallback: [u16; RAW_LAYER_STACKS],
    layer_stack_index: [u16; RAW_LAYER_STACKS],
    stored_layer_stacks: usize,
}

impl AntichessNetwork {
    pub const HASH_VALUE: u32 = AntichessFeatureTransformer::get_hash_value()
        ^ NetworkArchitecture::get_hash_value()
        ^ LAYER_STACK_TABLE_HASH;
    pub const HASH: u32 = Self::HASH_VALUE;

    /// A network of zeros with `stored_stacks` stored layer stacks, for tests.
    pub fn zeroed_with_stacks(stored_stacks: usize) -> LargePageBox<AntichessNetwork> {
        assert!((1..=LAYER_STACKS).contains(&stored_stacks));
        // SAFETY: zero bytes make a valid network before initializing stack maps.
        let mut net = unsafe { LargePageBox::<AntichessNetwork>::zeroed() };
        net.stored_layer_stacks = stored_stacks;
        for i in 0..RAW_LAYER_STACKS {
            let mapped = (i % stored_stacks) as u16;
            net.layer_stack_fallback[i] = mapped;
            net.layer_stack_index[i] = mapped;
        }
        net
    }

    /// Evaluates `pos` from the side to move's point of view in internal units.
    pub fn evaluate(
        &self,
        pos: &Position<Antichess>,
        accumulators: &mut AntichessAccumulators,
        caches: &mut AntichessCaches,
    ) -> Value {
        let stm = pos.side_to_move();
        accumulators.evaluate_perspective(&self.feature_transformer, pos, caches, WHITE);
        accumulators.evaluate_perspective(&self.feature_transformer, pos, caches, BLACK);

        let accumulation = &accumulators.latest().accumulation;
        let mut transformed = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::default();
        let perspectives = [stm, !stm];
        for (p, perspective) in perspectives.into_iter().enumerate() {
            ChessFeatureTransformer::transform_perspective(
                &accumulation[perspective.idx()],
                &mut transformed,
                p,
                &mut nnz_info,
            );
        }
        // SAFETY: both halves of `transformed` were written above.
        let features = unsafe { &transformed.assume_init_ref().0 };

        let raw = pos_raw_layer_stack(pos);
        let bucket = self.layer_stack_index[raw] as usize;
        let output = self.network[bucket].propagate(features, &nnz_info);
        output / OUTPUT_SCALE
    }

    /// Evaluates `pos` across all stored layer stacks for `eval` trace output.
    pub fn trace_evaluate(
        &self,
        pos: &Position<Antichess>,
        accumulators: &mut AntichessAccumulators,
        caches: &mut AntichessCaches,
    ) -> AntichessEvalTrace {
        let stm = pos.side_to_move();
        accumulators.evaluate_perspective(&self.feature_transformer, pos, caches, WHITE);
        accumulators.evaluate_perspective(&self.feature_transformer, pos, caches, BLACK);

        let accumulation = &accumulators.latest().accumulation;
        let mut transformed = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::default();
        for (p, perspective) in [stm, !stm].into_iter().enumerate() {
            ChessFeatureTransformer::transform_perspective(
                &accumulation[perspective.idx()],
                &mut transformed,
                p,
                &mut nnz_info,
            );
        }
        // SAFETY: both halves of `transformed` were written above.
        let features = unsafe { &transformed.assume_init_ref().0 };

        let mut positional = [VALUE_ZERO; LAYER_STACKS];
        for (i, slot) in positional[..self.stored_layer_stacks]
            .iter_mut()
            .enumerate()
        {
            *slot = self.network[i].propagate(features, &nnz_info) / OUTPUT_SCALE;
        }
        let correct_bucket = self.layer_stack_index[pos_raw_layer_stack(pos)] as usize;
        AntichessEvalTrace {
            positional,
            correct_bucket,
            layer_stacks: self.stored_layer_stacks,
        }
    }

    pub fn stored_layer_stacks(&self) -> usize {
        self.stored_layer_stacks
    }
}

impl NetworkFile for AntichessNetwork {
    const DEFAULT_NAME: &'static str = EVAL_FILE_DEFAULT_NAME;

    fn read(bytes: &[u8]) -> Option<(LargePageBox<Self>, Vec<u8>)> {
        let mut reader = Reader::new(bytes);
        let (hash, description) = read_header(&mut reader)?;
        if hash != Self::HASH_VALUE {
            return None;
        }

        let mut network = Self::zeroed_with_stacks(1);
        let net = &mut *network;
        let stored = read_layer_stack_table_n(
            &mut reader,
            &mut net.layer_stack_fallback,
            &mut net.layer_stack_index,
            LAYER_STACKS,
        )?;
        net.stored_layer_stacks = stored;

        if !read_hash(&mut reader, AntichessFeatureTransformer::get_hash_value())
            || !net.feature_transformer.read_parameters(&mut reader)
        {
            return None;
        }

        for stack in &mut net.network[..stored] {
            if !read_hash(&mut reader, NetworkArchitecture::get_hash_value())
                || !stack.read_parameters(&mut reader)
            {
                return None;
            }
        }

        (reader.good() && reader.at_end()).then_some((network, description))
    }

    fn write(&self, description: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        write_u32(&mut out, VERSION);
        write_u32(&mut out, Self::HASH_VALUE);
        write_u32(&mut out, description.len() as u32);
        out.extend_from_slice(description);

        write_layer_stack_table_n(
            &mut out,
            self.stored_layer_stacks,
            &self.layer_stack_fallback,
        );

        write_u32(&mut out, AntichessFeatureTransformer::get_hash_value());
        self.feature_transformer.write_parameters(&mut out);

        for stack in &self.network[..self.stored_layer_stacks] {
            write_u32(&mut out, NetworkArchitecture::get_hash_value());
            stack.write_parameters(&mut out);
        }

        out
    }

    fn kind(&self) -> &'static str {
        "Antichess Threats+PP_3Wide+Psq SFNNv20 "
    }

    fn download_url() -> String {
        format!(
            "https://github.com/aiorla/varifish/releases/download/networks/{}",
            Self::DEFAULT_NAME
        )
    }

    fn size_and_dimensions(&self) -> String {
        let mib = size_of::<Self>() as f64 / (1024.0 * 1024.0);
        format!(
            "({mib:.2} MiB, ({THREAT_AND_PAIR_DIMENSIONS} + {PSQ_DIMENSIONS}, {L1}, {L2}, {L3}, 1), {} stacks)",
            self.stored_layer_stacks
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn antichess_network_round_trips_and_separates_hashes() {
        let net = AntichessNetwork::zeroed_with_stacks(4);
        let bytes = net.write(b"antichess-test");
        let (loaded, desc) = AntichessNetwork::read(&bytes).expect("valid antichess network");
        assert_eq!(desc, b"antichess-test");
        assert_eq!(loaded.stored_layer_stacks(), 4);
        assert_eq!(loaded.write(b"antichess-test"), bytes);

        // Must not parse as chess, Three-check, or Crazyhouse.
        assert!(crate::nnue::network::Network::read(&bytes).is_none());
        assert!(crate::nnue::threecheck::ThreeCheckNetwork::read(&bytes).is_none());
        assert!(crate::nnue::crazyhouse::CrazyhouseNetwork::read(&bytes).is_none());
    }

    #[test]
    fn threat_and_psq_indices_are_in_bounds_and_color_symmetric() {
        for attacker in PIECES {
            for attacked in PIECES {
                for from in 0..64u8 {
                    let sq_from = Square(from);
                    let pt = attacker.type_of();
                    if pt == PAWN && !(Square::A2.0..=Square::H7.0).contains(&from) {
                        continue;
                    }
                    let attacks = if pt == PAWN {
                        crate::attacks::pawn_attacks(attacker.color(), sq_from)
                    } else {
                        attacks_bb(pt, sq_from, Bitboard::EMPTY)
                    };
                    for to in attacks {
                        let w_idx = threat_index(WHITE, attacker, sq_from, to, attacked);
                        assert!((w_idx as usize) < THREAT_DIMENSIONS);

                        // Flip board vertically and swap colors: Black's view
                        // must yield the exact same feature index.
                        let f_attacker = Piece(attacker.0 ^ 8);
                        let f_attacked = Piece(attacked.0 ^ 8);
                        let f_from = Square(sq_from.0 ^ 56);
                        let f_to = Square(to.0 ^ 56);
                        let b_idx = threat_index(BLACK, f_attacker, f_from, f_to, f_attacked);
                        assert_eq!(w_idx, b_idx);
                    }
                }
            }
        }
    }

    #[test]
    fn incremental_eval_matches_from_scratch_and_is_color_symmetric() {
        use crate::misc::Prng;
        use crate::movegen::{ALL, MoveList};

        let mut rng = Prng::new(0xA471_C4E5_9876_5432);
        let mut net = AntichessNetwork::zeroed_with_stacks(4);

        for b in &mut net.feature_transformer.biases.0 {
            *b = ((rng.rand64() % 33) as i16) - 16;
        }
        for w in &mut net.feature_transformer.threat_weights.0 {
            *w = ((rng.rand64() % 17) as i8) - 8;
        }
        for w in &mut net.feature_transformer.psq_weights.0 {
            *w = ((rng.rand64() % 17) as i8) - 8;
        }
        net.feature_transformer.permute_weights();

        // Write + read round-trip exercises full file serialization with non-zero weights
        // and verifies the non-zero network produces non-trivial accumulator sums.
        let bytes = net.write(b"nonzero");
        let (loaded, _) = AntichessNetwork::read(&bytes).unwrap();

        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
            "3r1n2/4P3/8/8/8/8/3p4/2B1R3 w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "2k1k3/1p6/8/3K4/8/8/6P1/3K1K2 w - - 0 25",
            "2b2b2/pp2pppp/8/2n1N3/8/8/PP2PPPP/2B2B2 w - - 0 10",
        ];

        let mut inc_acc = AntichessAccumulators::new_stack();
        let mut inc_caches = AntichessCaches::new(Some(&loaded));

        for &fen in &starts {
            let mut pos = Position::<Antichess>::from_fen(fen, false).unwrap();
            inc_acc.reset(Some(&loaded));

            for _ in 0..40 {
                // Compare full accumulation vectors (both perspectives) against fresh refresh.
                inc_acc.evaluate_perspective(
                    &loaded.feature_transformer,
                    &pos,
                    &mut inc_caches,
                    WHITE,
                );
                inc_acc.evaluate_perspective(
                    &loaded.feature_transformer,
                    &pos,
                    &mut inc_caches,
                    BLACK,
                );

                let mut fresh_acc = AntichessAccumulators::new_stack();
                let mut fresh_caches = AntichessCaches::new(Some(&loaded));
                fresh_acc.evaluate_perspective(
                    &loaded.feature_transformer,
                    &pos,
                    &mut fresh_caches,
                    WHITE,
                );
                fresh_acc.evaluate_perspective(
                    &loaded.feature_transformer,
                    &pos,
                    &mut fresh_caches,
                    BLACK,
                );

                assert_eq!(
                    inc_acc.latest().accumulation,
                    fresh_acc.latest().accumulation,
                    "accumulator mismatch at {}",
                    pos.fen()
                );

                // Verify color-flip symmetry of accumulations (White <-> Black swap).
                let mut flipped = Position::<Antichess>::from_fen(&pos.fen(), false).unwrap();
                flipped.flip().unwrap();
                let mut flip_acc = AntichessAccumulators::new_stack();
                let mut flip_caches = AntichessCaches::new(Some(&loaded));
                flip_acc.evaluate_perspective(
                    &loaded.feature_transformer,
                    &flipped,
                    &mut flip_caches,
                    WHITE,
                );
                flip_acc.evaluate_perspective(
                    &loaded.feature_transformer,
                    &flipped,
                    &mut flip_caches,
                    BLACK,
                );
                assert_eq!(
                    fresh_acc.latest().accumulation[WHITE.idx()],
                    flip_acc.latest().accumulation[BLACK.idx()],
                    "white/black flip asymmetry at {}",
                    pos.fen()
                );
                assert_eq!(
                    fresh_acc.latest().accumulation[BLACK.idx()],
                    flip_acc.latest().accumulation[WHITE.idx()],
                    "black/white flip asymmetry at {}",
                    pos.fen()
                );

                let moves = MoveList::new::<ALL>(&pos);
                if moves.is_empty() {
                    break;
                }
                let mv = moves.as_slice()[(rng.rand64() as usize) % moves.len()];
                let dirties = inc_acc.push(&pos);
                pos.make_move(mv, dirties);
            }
        }
    }

    #[test]
    fn accumulators_match_fresh_ones_along_a_search_like_walk() {
        // A search evaluates only some of its nodes, undoes moves and plays
        // others in their place: the accumulators it then updates from an
        // ancestor's, over the plies in between, must be those of the
        // position.
        use crate::misc::Prng;
        use crate::movegen::{ALL, MoveList};

        let mut rng = Prng::new(0x5EA2_C4E5_0BAD_C0DE);
        let mut net = AntichessNetwork::zeroed_with_stacks(4);
        for b in &mut net.feature_transformer.biases.0 {
            *b = ((rng.rand64() % 33) as i16) - 16;
        }
        for w in &mut net.feature_transformer.threat_weights.0 {
            *w = ((rng.rand64() % 17) as i8) - 8;
        }
        for w in &mut net.feature_transformer.psq_weights.0 {
            *w = ((rng.rand64() % 17) as i8) - 8;
        }
        let ft = &net.feature_transformer;

        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w - - 0 4",
            "3r1n2/4P3/8/8/8/8/3p4/2B1R3 w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "2b2b2/pp2pppp/8/2n1N3/8/8/PP2PPPP/2B2B2 w - - 0 10",
            // 17 pieces: a capture changes the piece-square block
            "4k3/pppppppp/8/8/8/8/PPPPPPPP/8 w - - 0 1",
        ];

        let mut acc = AntichessAccumulators::new_stack();
        let mut caches = AntichessCaches::new(Some(&net));
        let mut checked = 0;

        for &fen in &starts {
            let mut pos = Position::<Antichess>::from_fen(fen, false).unwrap();
            acc.reset(Some(&net));
            let mut line: Vec<Move> = Vec::new();

            for _ in 0..4000 {
                if rng.rand64().is_multiple_of(3) {
                    acc.evaluate_perspective(ft, &pos, &mut caches, WHITE);
                    acc.evaluate_perspective(ft, &pos, &mut caches, BLACK);
                    let mut fresh_acc = AntichessAccumulators::new_stack();
                    let mut fresh_caches = AntichessCaches::new(Some(&net));
                    fresh_acc.evaluate_perspective(ft, &pos, &mut fresh_caches, WHITE);
                    fresh_acc.evaluate_perspective(ft, &pos, &mut fresh_caches, BLACK);
                    assert_eq!(
                        acc.latest().accumulation,
                        fresh_acc.latest().accumulation,
                        "after {} moves at {}",
                        line.len(),
                        pos.fen()
                    );
                    checked += 1;
                }

                let moves = MoveList::new::<ALL>(&pos);
                let back = !line.is_empty()
                    && (moves.is_empty() || line.len() >= 60 || rng.rand64() % 5 < 2);
                if back {
                    pos.undo_move(line.pop().unwrap());
                    acc.pop();
                } else if !moves.is_empty() {
                    let mv = moves.as_slice()[(rng.rand64() as usize) % moves.len()];
                    let dirties = acc.push(&pos);
                    pos.make_move(mv, dirties);
                    line.push(mv);
                }
            }
        }
        assert!(checked > 5000, "{checked}");
    }
}
