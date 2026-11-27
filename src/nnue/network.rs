//! The network: its feature transformer and layer stacks, read from and
//! written to its file, and its evaluation of positions.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/network.h` and `.cpp` and of the network's parts of `engine.cpp`:
//! Stockfish, a UCI chess playing engine derived from Glaurung 2.1,
//! Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under the
//! GNU General Public License, version 3 or later.

use std::mem::MaybeUninit;

use super::accumulator::{AccumulatorCaches, AccumulatorStack};
use super::architecture::{L1, L2, L3, LAYER_STACKS, NetworkArchitecture, RAW_LAYER_STACKS};
use super::common::{Aligned, OUTPUT_SCALE, Reader, VERSION, write_u16, write_u32};
use super::eval_file::NetworkFile;
use super::feature_transformer::{FeatureTransformer, INPUT_DIMENSIONS};
use super::nnz_helper::NnzInfo;
use crate::bitboard::*;
use crate::evaluate::EVAL_FILE_DEFAULT_NAME;
use crate::memory::LargePageBox;
use crate::position::Position;
use crate::types::*;

pub struct Network {
    // Input feature converter
    pub(super) feature_transformer: FeatureTransformer,

    // Evaluation function
    network: [NetworkArchitecture; LAYER_STACKS],
    // The stored stack that each raw stack falls back to, and where in the
    // stored stacks each raw stack's is
    layer_stack_fallback: [u16; RAW_LAYER_STACKS],
    layer_stack_index: [u16; RAW_LAYER_STACKS],
    stored_layer_stacks: usize,
}

/// The evaluation of each stored layer stack, and the stack a position uses.
pub struct NnueEvalTrace {
    pub positional: [Value; LAYER_STACKS],
    pub correct_bucket: usize,
    pub layer_stacks: usize,
}

/// What a file's table of layer stacks starts with, also in its network's
/// hash
pub(super) const LAYER_STACK_TABLE_HASH: u32 = 0x5A7A_B1E5;

/// Reads a table of layer stacks: how many are stored, and the stored stack
/// each raw one falls back to, which must fall back to itself, into
/// `fallback`, and into `index` where in the stored stacks each raw stack's
/// is. Returns how many are stored, at most `max_stored`.
pub(super) fn read_layer_stack_table_n<const RAW: usize>(
    reader: &mut Reader,
    fallback: &mut [u16; RAW],
    index: &mut [u16; RAW],
    max_stored: usize,
) -> Option<usize> {
    if reader.read_u32() != LAYER_STACK_TABLE_HASH || reader.read_u32() != RAW as u32 {
        return None;
    }
    let stored = reader.read_u32() as usize;
    if stored == 0 || stored > max_stored {
        return None;
    }
    for f in fallback.iter_mut() {
        *f = reader.read_u16();
        if usize::from(*f) >= RAW {
            return None;
        }
    }
    let mut count = 0;
    for s in 0..RAW {
        if usize::from(fallback[s]) == s {
            index[s] = count;
            count += 1;
        }
    }
    if usize::from(count) != stored {
        return None;
    }
    for s in 0..RAW {
        let f = usize::from(fallback[s]);
        if usize::from(fallback[f]) != f {
            return None;
        }
        index[s] = index[f];
    }
    reader.good().then_some(stored)
}

pub(super) fn read_layer_stack_table(
    reader: &mut Reader,
    fallback: &mut [u16; RAW_LAYER_STACKS],
    index: &mut [u16; RAW_LAYER_STACKS],
    max_stored: usize,
) -> Option<usize> {
    read_layer_stack_table_n(reader, fallback, index, max_stored)
}

/// Writes a table of layer stacks, of `stored` stacks and each raw one's
/// `fallback`.
pub(super) fn write_layer_stack_table_n<const RAW: usize>(
    out: &mut Vec<u8>,
    stored: usize,
    fallback: &[u16; RAW],
) {
    write_u32(out, LAYER_STACK_TABLE_HASH);
    write_u32(out, RAW as u32);
    write_u32(out, stored as u32);
    for &f in fallback {
        write_u16(out, f);
    }
}

pub(super) fn write_layer_stack_table(
    out: &mut Vec<u8>,
    stored: usize,
    fallback: &[u16; RAW_LAYER_STACKS],
) {
    write_layer_stack_table_n(out, stored, fallback);
}

impl Network {
    /// Hash value of evaluation function structure
    const HASH: u32 = FeatureTransformer::get_hash_value()
        ^ NetworkArchitecture::get_hash_value()
        ^ LAYER_STACK_TABLE_HASH;

    /// A network whose parameters are all zero.
    pub fn zeroed() -> LargePageBox<Network> {
        // SAFETY: the network's parameters are integers.
        unsafe { LargePageBox::zeroed() }
    }

    /// A network whose parameters are all zero, which stores `stored` of
    /// the layer stacks, the others falling back to the first: a file that
    /// reads back.
    #[cfg(test)]
    pub fn zeroed_with_stacks(stored: usize) -> LargePageBox<Network> {
        let mut network = Network::zeroed();
        for s in 0..RAW_LAYER_STACKS {
            let f = if s < stored { s } else { 0 };
            network.layer_stack_fallback[s] = f as u16;
            network.layer_stack_index[s] = f as u16;
        }
        network.stored_layer_stacks = stored;
        network
    }

    /// The network in the bytes of a network file, and the file's
    /// description, or None if the bytes are not a network of this
    /// architecture.
    pub fn read(bytes: &[u8]) -> Option<(LargePageBox<Network>, Vec<u8>)> {
        let mut reader = Reader::new(bytes);
        let (hash_value, description) = read_header(&mut reader)?;
        if hash_value != Self::HASH {
            return None;
        }

        let mut network = Self::zeroed();
        if !network.read_layer_stack_table(&mut reader) {
            return None;
        }
        if !read_hash(&mut reader, FeatureTransformer::get_hash_value())
            || !network.feature_transformer.read_parameters(&mut reader)
        {
            return None;
        }
        let stored = network.stored_layer_stacks;
        for stack in &mut network.network[..stored] {
            if !read_hash(&mut reader, NetworkArchitecture::get_hash_value())
                || !stack.read_parameters(&mut reader)
            {
                return None;
            }
        }

        (reader.good() && reader.at_end()).then_some((network, description))
    }

    /// Reads the table of layer stacks: how many are stored, and the stored
    /// stack each raw one falls back to, which must fall back to itself.
    fn read_layer_stack_table(&mut self, reader: &mut Reader) -> bool {
        match read_layer_stack_table(
            reader,
            &mut self.layer_stack_fallback,
            &mut self.layer_stack_index,
            LAYER_STACKS,
        ) {
            Some(stored) => {
                self.stored_layer_stacks = stored;
                true
            }
            None => false,
        }
    }

    /// The network's file, with the description given.
    pub fn write(&self, description: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        write_u32(&mut out, VERSION);
        write_u32(&mut out, Self::HASH);
        write_u32(&mut out, description.len() as u32);
        out.extend_from_slice(description);

        write_layer_stack_table(
            &mut out,
            self.stored_layer_stacks,
            &self.layer_stack_fallback,
        );

        write_u32(&mut out, FeatureTransformer::get_hash_value());
        self.feature_transformer.write_parameters(&mut out);
        for stack in &self.network[..self.stored_layer_stacks] {
            write_u32(&mut out, NetworkArchitecture::get_hash_value());
            stack.write_parameters(&mut out);
        }
        out
    }

    /// The evaluation of the position, from the side to move's point of
    /// view, whose state the stack's latest state holds.
    pub fn evaluate<V: crate::variant::Variant>(
        &self,
        pos: &Position<V>,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> Value {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::new();

        let bucket = self.layer_stack(pos);
        let transformed_features = self.feature_transformer.transform(
            pos,
            accumulator_stack,
            cache,
            &mut transformed_features,
            &mut nnz_info,
        );
        let positional = self.network[bucket].propagate(transformed_features, &nnz_info);
        positional / OUTPUT_SCALE
    }

    /// The evaluation by each layer stack.
    pub fn trace_evaluate<V: crate::variant::Variant>(
        &self,
        pos: &Position<V>,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> NnueEvalTrace {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::new();

        let correct_bucket = self.layer_stack(pos);
        let transformed_features = self.feature_transformer.transform(
            pos,
            accumulator_stack,
            cache,
            &mut transformed_features,
            &mut nnz_info,
        );

        let layer_stacks = self.stored_layer_stacks;
        NnueEvalTrace {
            positional: std::array::from_fn(|bucket| {
                if bucket < layer_stacks {
                    self.network[bucket].propagate(transformed_features, &nnz_info) / OUTPUT_SCALE
                } else {
                    VALUE_ZERO
                }
            }),
            correct_bucket,
            layer_stacks,
        }
    }

    /// The stored layer stack of a position: by its number of pieces, by
    /// which sides have a queen, by whether its pawns stand on both halves
    /// of the board, one or none, and by whether a side has two officers,
    /// pieces other than pawns and kings.
    #[inline(always)]
    fn layer_stack<V: crate::variant::Variant>(&self, pos: &Position<V>) -> usize {
        let stm = pos.side_to_move();
        let band = (pos.count_both(ALL_PIECES) - 1) / 4;
        let queens = 2 * i32::from(pos.pieces_cpt(stm, QUEEN).any())
            + i32::from(pos.pieces_cpt(!stm, QUEEN).any());

        // Pawn layout: 0 if pawns stand on both board halves, 1 if on one
        // half only, 2 if there are no pawns
        const QUEEN_SIDE: Bitboard =
            Bitboard(FILE_A_BB.0 | FILE_B_BB.0 | FILE_C_BB.0 | FILE_D_BB.0);
        let pawns = pos.pieces_pt(PAWN);
        let layout = if !pawns.any() {
            2
        } else if (pawns & QUEEN_SIDE).any() && (pawns & !QUEEN_SIDE).any() {
            0
        } else {
            1
        };

        // Officers are the pieces other than pawns and kings
        let officers = |c: Color| (pos.pieces_c(c) & !pos.pieces_pt2(PAWN, KING)).popcount();
        let many = i32::from(officers(WHITE).max(officers(BLACK)) >= 2);

        let raw = ((band * 4 + queens) * 3 + layout) * 2 + many;
        usize::from(self.layer_stack_index[raw as usize])
    }
}

/// Reads the version, the hash value and the description.
pub(super) fn read_header(reader: &mut Reader) -> Option<(u32, Vec<u8>)> {
    let version = reader.read_u32();
    let hash_value = reader.read_u32();
    let size = reader.read_u32();
    if !reader.good() || version != VERSION {
        return None;
    }
    let description = reader.take(size as usize)?;
    Some((hash_value, description.to_vec()))
}

/// Reads the hash value that precedes a part's parameters.
pub(super) fn read_hash(reader: &mut Reader, hash_value: u32) -> bool {
    reader.read_u32() == hash_value && reader.good()
}

impl NetworkFile for Network {
    const DEFAULT_NAME: &'static str = EVAL_FILE_DEFAULT_NAME;

    fn read(bytes: &[u8]) -> Option<(LargePageBox<Network>, Vec<u8>)> {
        Network::read(bytes)
    }

    fn write(&self, description: &[u8]) -> Vec<u8> {
        Network::write(self, description)
    }

    fn kind(&self) -> &'static str {
        ""
    }

    fn download_url() -> String {
        format!("https://tests.stockfishchess.org/api/nn/{EVAL_FILE_DEFAULT_NAME}")
    }

    fn size_and_dimensions(&self) -> String {
        let size = size_of::<FeatureTransformer>()
            + size_of::<NetworkArchitecture>() * self.stored_layer_stacks;
        format!(
            "({}MiB, ({INPUT_DIMENSIONS}, {L1}, {L2}, {L3}, 1))",
            size / (1024 * 1024)
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;

    #[test]
    fn files_are_read_whole_or_refused() {
        let file = Network::zeroed_with_stacks(RAW_LAYER_STACKS).write(b"a description");
        let (network, description) = Network::read(&file).expect("a network");
        assert_eq!(description, b"a description");
        assert!(network.write(&description) == file);

        assert!(Network::read(&file[..file.len() - 1]).is_none());
        assert!(Network::read(&[&file[..], &[0]].concat()).is_none());
        let mut other_hash = file.clone();
        other_hash[4] ^= 1;
        assert!(Network::read(&other_hash).is_none());
    }

    #[test]
    fn layer_stacks_fall_back_to_stored_ones() {
        let file = Network::zeroed_with_stacks(5).write(b"");
        let (network, _) = Network::read(&file).expect("a network");
        assert!(network.write(b"") == file);
        assert_eq!(network.stored_layer_stacks, 5);
        assert_eq!(network.layer_stack_index[3], 3);
        assert_eq!(network.layer_stack_index[100], 0);

        // The table follows the header: its hash, the raw stacks, the
        // stored ones, then a fallback per raw stack.
        let table = 12;
        let fallback = |s: usize| table + 12 + 2 * s;
        let mut none_stored = file.clone();
        none_stored[table + 8..table + 12].copy_from_slice(&0u32.to_le_bytes());
        assert!(Network::read(&none_stored).is_none());
        // A raw stack falling back to one that falls back elsewhere
        let mut chained = file.clone();
        chained[fallback(4)..fallback(4) + 2].copy_from_slice(&7u16.to_le_bytes());
        assert!(Network::read(&chained).is_none());
        // A raw stack beyond the table
        let mut beyond = file.clone();
        beyond[fallback(9)..fallback(9) + 2].copy_from_slice(&192u16.to_le_bytes());
        assert!(Network::read(&beyond).is_none());
    }

    #[test]
    #[ignore = "needs the default network, which `make net` downloads into nets/"]
    fn the_default_network_is_written_back_as_read() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("nets")
            .join(EVAL_FILE_DEFAULT_NAME);
        let file = fs::read(&path).expect("the default network, which `make net` downloads");
        let (network, description) = Network::read(&file).expect("a network");
        assert!(network.write(&description) == file);
    }
}
