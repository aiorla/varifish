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
use super::architecture::{L1, L2, L3, LAYER_STACKS, NetworkArchitecture};
use super::common::{Aligned, OUTPUT_SCALE, Reader, VERSION, write_u32};
use super::eval_file::NetworkFile;
use super::feature_transformer::{FeatureTransformer, INPUT_DIMENSIONS};
use super::nnz_helper::NnzInfo;
use crate::evaluate::EVAL_FILE_DEFAULT_NAME;
use crate::memory::LargePageBox;
use crate::position::Position;
use crate::types::*;

pub struct Network {
    // Input feature converter
    pub(super) feature_transformer: FeatureTransformer,

    // Evaluation function
    network: [NetworkArchitecture; LAYER_STACKS],
}

/// The evaluation of each layer stack, and the stack a position uses.
pub struct NnueEvalTrace {
    pub positional: [Value; LAYER_STACKS],
    pub correct_bucket: usize,
}

impl Network {
    /// Hash value of evaluation function structure
    const HASH: u32 = FeatureTransformer::get_hash_value() ^ NetworkArchitecture::get_hash_value();

    /// A network whose parameters are all zero.
    pub fn zeroed() -> LargePageBox<Network> {
        // SAFETY: the network's parameters are integers.
        unsafe { LargePageBox::zeroed() }
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
        if !read_hash(&mut reader, FeatureTransformer::get_hash_value())
            || !network.feature_transformer.read_parameters(&mut reader)
        {
            return None;
        }
        for stack in &mut network.network {
            if !read_hash(&mut reader, NetworkArchitecture::get_hash_value())
                || !stack.read_parameters(&mut reader)
            {
                return None;
            }
        }

        (reader.good() && reader.at_end()).then_some((network, description))
    }

    /// The network's file, with the description given.
    pub fn write(&self, description: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        write_u32(&mut out, VERSION);
        write_u32(&mut out, Self::HASH);
        write_u32(&mut out, description.len() as u32);
        out.extend_from_slice(description);

        write_u32(&mut out, FeatureTransformer::get_hash_value());
        self.feature_transformer.write_parameters(&mut out);
        for stack in &self.network {
            write_u32(&mut out, NetworkArchitecture::get_hash_value());
            stack.write_parameters(&mut out);
        }
        out
    }

    /// The evaluation of the position, from the side to move's point of
    /// view, whose state the stack's latest state holds.
    pub fn evaluate(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> Value {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::new();

        let bucket = bucket(pos);
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
    pub fn trace_evaluate(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> NnueEvalTrace {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::new();

        let correct_bucket = bucket(pos);
        let transformed_features = self.feature_transformer.transform(
            pos,
            accumulator_stack,
            cache,
            &mut transformed_features,
            &mut nnz_info,
        );

        NnueEvalTrace {
            positional: std::array::from_fn(|bucket| {
                self.network[bucket].propagate(transformed_features, &nnz_info) / OUTPUT_SCALE
            }),
            correct_bucket,
        }
    }
}

/// The layer stack of a position, by its number of pieces and by which
/// sides have a queen.
#[inline(always)]
fn bucket(pos: &Position) -> usize {
    let pc_bucket = (pos.count_both(ALL_PIECES) - 1) / 4;
    let stm = pos.side_to_move();
    let queen_bucket = 2 * i32::from(pos.pieces_cpt(stm, QUEEN).any())
        + i32::from(pos.pieces_cpt(!stm, QUEEN).any());
    (pc_bucket * 4 + queen_bucket) as usize
}

/// Reads the version, the hash value and the description.
fn read_header(reader: &mut Reader) -> Option<(u32, Vec<u8>)> {
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
fn read_hash(reader: &mut Reader, hash_value: u32) -> bool {
    reader.read_u32() == hash_value && reader.good()
}

impl NetworkFile for Network {
    const DEFAULT_NAME: &'static str = EVAL_FILE_DEFAULT_NAME;
    const KIND: &'static str = "";

    fn read(bytes: &[u8]) -> Option<(LargePageBox<Network>, Vec<u8>)> {
        Network::read(bytes)
    }

    fn write(&self, description: &[u8]) -> Vec<u8> {
        Network::write(self, description)
    }

    fn download_url() -> String {
        format!("https://tests.stockfishchess.org/api/nn/{EVAL_FILE_DEFAULT_NAME}")
    }

    fn size_and_dimensions() -> String {
        let size =
            size_of::<FeatureTransformer>() + size_of::<NetworkArchitecture>() * LAYER_STACKS;
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
        let file = Network::zeroed().write(b"a description");
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
