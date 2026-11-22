//! Atomic's networks in Stockfish's design, SFNNv20: Stockfish's FullThreats
//! input features, each piece attacking another, by the types, colors and
//! squares of both, its PP_3Wide features after them, each pair of pawns on
//! the same or adjacent files, by their colors and squares, and its K32H3
//! piece-square features, summed into 1024 accumulators a side, then
//! SFNNv20's layer stacks of 1024, 32, 32 and 1 neurons, by the pieces, the
//! queens, the pawns' halves of the board and the officers, of which the
//! file stores those its table says, in Stockfish's file layout, as chess's.
//! Their features are chess's, under Atomic's rules: kings, which never
//! capture, attack nothing and are attacked by nothing, as in chess's
//! features; a pawn's pushes are not threats; a capture a pin or a blast
//! next to its own king makes illegal is still a threat; and pawns, which
//! blasts spare, leave the board only when they capture or are captured.
//! Their hash is chess's with its feature sets' xored with "Atom", so that
//! no chess network reads as an Atomic one.
//!
//! Derived, with changes for varifish in 2026, from Stockfish's
//! `nnue/network.h` and `.cpp` and `nnue/nnue_feature_transformer.h`:
//! Stockfish, a UCI chess playing engine derived from Glaurung 2.1,
//! Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under the
//! GNU General Public License, version 3 or later. The network is chess's.

pub mod accumulator;

use std::mem::MaybeUninit;

use accumulator::{AccumulatorCaches, AccumulatorStack};

use crate::bitboard::*;
use crate::memory::LargePageBox;
use crate::nnue::architecture::{L1, L2, L3, NetworkArchitecture, RAW_LAYER_STACKS};
use crate::nnue::common::{
    Aligned, OUTPUT_SCALE, Reader, VERSION, write_i8s, write_i16s, write_u32,
};
#[cfg(use_paired_widening_add)]
use crate::nnue::feature_transformer::WIDENING_PAIR_ORDER;
use crate::nnue::feature_transformer::{
    FeatureTransformer as ChessFeatureTransformer, INVERSE_PACKUS_EPI16_ORDER, PACKUS_EPI16_ORDER,
    combine_hash, permute, unpermuted_weights,
};
use crate::nnue::features::full_threats::FullThreats;
use crate::nnue::features::k32h3::K32H3;
use crate::nnue::features::pp_3wide::Pp3Wide;
use crate::nnue::network::{
    LAYER_STACK_TABLE_HASH, read_hash, read_header, read_layer_stack_table, write_layer_stack_table,
};
use crate::nnue::nnz_helper::NnzInfo;
use crate::position::Position;
use crate::types::*;
use crate::variant::Atomic;

/// Number of output dimensions for one side
const HALF_DIMENSIONS: usize = L1;

/// Number of threat features: FullThreats's 59,808
pub const THREAT_DIMENSIONS: usize = FullThreats::DIMENSIONS as usize;

/// Number of pawn-pair features: PP_3Wide's 4,560, whose indices follow the
/// threats'
pub const PAIR_DIMENSIONS: usize = Pp3Wide::DIMENSIONS as usize;

/// Number of threat features and pawn pairs, the rows of the threat weights
pub const THREAT_AND_PAIR_DIMENSIONS: usize = THREAT_DIMENSIONS + PAIR_DIMENSIONS;

const _: () = assert!(Pp3Wide::INDEX_BASE as usize == THREAT_DIMENSIONS);

/// Number of piece-square features: K32H3's 67,584
pub const PSQ_DIMENSIONS: usize = K32H3::DIMENSIONS;

/// What sets the features of Atomic's networks apart from chess's: "Atom"
pub const ATOMIC_HASH: u32 = 0x4174_6F6D;

/// SFNNv20's raw layer stack of a position, as chess's: by its number of
/// pieces, by which sides have a queen, by whether its pawns stand on both
/// halves of the board, one or none, and by whether a side has two
/// officers, pieces other than pawns and kings.
#[inline(always)]
fn raw_layer_stack(pos: &Position<Atomic>) -> usize {
    let pc_bucket = (pos.count_both(ALL_PIECES) - 1) / 4;
    let stm = pos.side_to_move();
    let queen_bucket = 2 * i32::from(pos.pieces_cpt(stm, QUEEN).any())
        + i32::from(pos.pieces_cpt(!stm, QUEEN).any());

    // Pawn layout: 0 if pawns stand on both board halves, 1 if on one
    // half only, 2 if there are no pawns
    const QUEEN_SIDE: Bitboard = Bitboard(FILE_A_BB.0 | FILE_B_BB.0 | FILE_C_BB.0 | FILE_D_BB.0);
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
    let many = usize::from(officers(WHITE).max(officers(BLACK)) >= 2);

    (((pc_bucket * 4 + queen_bucket) as usize) * 3 + layout) * 2 + many
}

/// Input feature converter
pub struct FeatureTransformer {
    pub biases: Aligned<[i16; HALF_DIMENSIONS]>,
    /// The weights of the threat features and pawn pairs, by feature, then
    /// by accumulator: the threats' rows, then the pairs'
    pub threat_weights: Aligned<[[i8; HALF_DIMENSIONS]; THREAT_AND_PAIR_DIMENSIONS]>,
    /// The weights of the piece-square features, by feature, then by
    /// accumulator
    pub weights: Aligned<[i8; PSQ_DIMENSIONS * HALF_DIMENSIONS]>,
}

impl FeatureTransformer {
    /// Number of input feature dimensions: the threats' and pairs', then
    /// the piece-square features'
    pub const INPUT_DIMENSIONS: usize = THREAT_AND_PAIR_DIMENSIONS + PSQ_DIMENSIONS;

    /// Hash value embedded in the evaluation file: chess's, with "Atom"
    pub const fn get_hash_value() -> u32 {
        combine_hash(&[
            FullThreats::HASH_VALUE,
            Pp3Wide::HASH_VALUE,
            K32H3::HASH_VALUE,
        ]) ^ ATOMIC_HASH
            ^ (HALF_DIMENSIONS * 2) as u32
    }

    /// The threat weights, row after row
    #[inline(always)]
    pub fn threat_weights(&self) -> &[i8] {
        self.threat_weights.0.as_flattened()
    }

    /// The weights of the piece-square features of a block
    #[inline(always)]
    pub fn psq_weights(&self, block: usize) -> &[i8] {
        &self.weights.0[K32H3::block_offset(block) * HALF_DIMENSIONS..]
            [..K32H3::BLOCK_DIMENSIONS * HALF_DIMENSIONS]
    }

    /// Read network parameters, as chess's: the biases, the threat weights
    /// (with the pawn pairs'), then the piece-square weights. The biases and
    /// weights are permuted, as chess's are, for the packs that convert the
    /// accumulations and for the widening of the weights.
    pub fn read_parameters(&mut self, reader: &mut Reader) -> bool {
        let threat_weights = self.threat_weights.0.as_flattened_mut();
        reader.read_i16s(&mut self.biases.0);
        reader.read_i8s(threat_weights);
        reader.read_i8s(&mut self.weights.0);

        permute(&mut self.biases.0, &PACKUS_EPI16_ORDER);
        permute(threat_weights, &PACKUS_EPI16_ORDER);
        permute(&mut self.weights.0, &PACKUS_EPI16_ORDER);
        #[cfg(use_paired_widening_add)]
        {
            permute(threat_weights, &WIDENING_PAIR_ORDER);
            permute(&mut self.weights.0, &WIDENING_PAIR_ORDER);
        }

        reader.good()
    }

    /// Write network parameters
    pub fn write_parameters(&self, out: &mut Vec<u8>) {
        let mut biases = self.biases.0;
        permute(&mut biases, &INVERSE_PACKUS_EPI16_ORDER);
        write_i16s(out, &biases);
        write_i8s(out, &unpermuted_weights(self.threat_weights()));
        write_i8s(out, &unpermuted_weights(&self.weights.0));
    }

    /// Convert input features: writes `output`, the side to move's half
    /// then the other's, and returns it.
    #[inline(always)]
    pub fn transform<'a>(
        &self,
        pos: &Position<Atomic>,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
        output: &'a mut MaybeUninit<Aligned<[u8; L1]>>,
        nnz_info: &mut NnzInfo,
    ) -> &'a [u8; L1] {
        accumulator_stack.evaluate(pos, self, cache);
        let accumulation = &accumulator_stack.latest().accumulation;

        let perspectives = [pos.side_to_move(), !pos.side_to_move()];
        for (p, perspective) in perspectives.into_iter().enumerate() {
            ChessFeatureTransformer::transform_perspective(
                &accumulation[perspective.idx()],
                output,
                p,
                nnz_info,
            );
        }

        // SAFETY: each perspective wrote its half.
        unsafe { &output.assume_init_ref().0 }
    }
}

/// The evaluation of each stored layer stack, and the stack a position uses.
pub struct NnueEvalTrace {
    pub positional: Vec<Value>,
    pub correct_bucket: usize,
}

pub struct Network {
    // Input feature converter
    pub(super) feature_transformer: FeatureTransformer,

    // Evaluation function: SFNNv20's stored layer stacks
    network: [NetworkArchitecture; RAW_LAYER_STACKS],
    // The stored stack that each raw stack falls back to, and where in the
    // stored stacks each raw stack's is
    layer_stack_fallback: [u16; RAW_LAYER_STACKS],
    layer_stack_index: [u16; RAW_LAYER_STACKS],
    stored_layer_stacks: usize,
}

impl Network {
    /// Hash value of evaluation function structure, as chess's: with the
    /// table's
    pub const HASH: u32 = FeatureTransformer::get_hash_value()
        ^ NetworkArchitecture::get_hash_value()
        ^ LAYER_STACK_TABLE_HASH;

    /// A network whose parameters are all zero, which stores `stored` layer
    /// stacks, the others falling back to the first: a file that reads
    /// back.
    pub fn zeroed(stored: usize) -> LargePageBox<Network> {
        assert!((1..=RAW_LAYER_STACKS).contains(&stored));
        // SAFETY: the network's parameters are integers.
        let mut network: LargePageBox<Network> = unsafe { LargePageBox::zeroed() };
        for s in 0..RAW_LAYER_STACKS {
            let f = if s < stored { s } else { 0 };
            network.layer_stack_fallback[s] = f as u16;
            network.layer_stack_index[s] = f as u16;
        }
        network.stored_layer_stacks = stored;
        network
    }

    /// The stored layer stack of a position, as the table stores it.
    #[inline(always)]
    fn layer_stack(&self, pos: &Position<Atomic>) -> usize {
        usize::from(self.layer_stack_index[raw_layer_stack(pos)])
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

        // SAFETY: the network's parameters are integers.
        let mut network: LargePageBox<Network> = unsafe { LargePageBox::zeroed() };
        let n = &mut *network;
        n.stored_layer_stacks = read_layer_stack_table(
            &mut reader,
            &mut n.layer_stack_fallback,
            &mut n.layer_stack_index,
            RAW_LAYER_STACKS,
        )?;
        if !read_hash(&mut reader, FeatureTransformer::get_hash_value())
            || !n.feature_transformer.read_parameters(&mut reader)
        {
            return None;
        }
        for stack in &mut n.network[..n.stored_layer_stacks] {
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

    /// The network's output for a position with both kings, from the side
    /// to move's point of view, whose state the stack's latest state holds.
    pub fn evaluate(
        &self,
        pos: &Position<Atomic>,
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
    pub fn trace_evaluate(
        &self,
        pos: &Position<Atomic>,
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

        NnueEvalTrace {
            positional: (0..self.stored_layer_stacks)
                .map(|bucket| {
                    self.network[bucket].propagate(transformed_features, &nnz_info) / OUTPUT_SCALE
                })
                .collect(),
            correct_bucket,
        }
    }

    /// The size in MiB and the dimensions, as verify reports them: of the
    /// stored layer stacks.
    pub fn size_and_dimensions(&self) -> String {
        let size = size_of::<FeatureTransformer>()
            + size_of::<NetworkArchitecture>() * self.stored_layer_stacks;
        format!(
            "({}MiB, ({}, {L1}, {L2}, {L3}, 1))",
            size / (1024 * 1024),
            FeatureTransformer::INPUT_DIMENSIONS
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::atomic::AtomicNetwork;
    use crate::nnue::eval_file::NetworkFile;

    /// A network written reads back whole, with the stacks its table says,
    /// and no other format reads it.
    #[test]
    fn files_are_read_whole_or_refused() {
        let mut network = Network::zeroed(40);
        // Values in every permuted block of the biases and of both kinds of
        // weights, to see that writing undoes the permutations, in the last
        // threat rows, the pawn pairs', and in each stored stack.
        let ft = &mut network.feature_transformer;
        for (i, value) in ft.biases.0.iter_mut().enumerate() {
            *value = i as i16;
        }
        let threat_weights = ft.threat_weights.0.as_flattened_mut();
        for (i, value) in threat_weights[..8192].iter_mut().enumerate() {
            *value = i as i8;
        }
        let last = threat_weights.len() - 8192;
        for (i, value) in threat_weights[last..].iter_mut().enumerate() {
            *value = ((i % 255) as u8 + 1) as i8;
        }
        for (i, value) in ft.weights.0[..8192].iter_mut().enumerate() {
            *value = (i as i8).wrapping_neg();
        }
        for (s, stack) in network.network[..40].iter_mut().enumerate() {
            let mut bytes = Vec::new();
            stack.write_parameters(&mut bytes);
            bytes[4 * L2] = s as u8;
            assert!(stack.read_parameters(&mut Reader::new(&bytes)));
        }

        let file = network.write(b"a description");
        let (read, description) = Network::read(&file).expect("a network");
        assert_eq!(description, b"a description");
        assert_eq!(read.stored_layer_stacks, 40);
        assert!(read.write(&description) == file);

        // Stockfish's version; the hash of chess's feature sets, combined as
        // chess's are, with "Atom", of chess's layer stacks and of the
        // table; then the table: its hash, the raw stacks, the stored ones.
        let ft_hash = ((0x2c41_eaea_u32.rotate_left(1) ^ 0x5c80_fa69).rotate_left(1) ^ 0xAB00_5CEF)
            ^ 0x4174_6F6D
            ^ 2048;
        assert_eq!(FeatureTransformer::get_hash_value(), ft_hash);
        assert_eq!(
            crate::nnue::feature_transformer::FeatureTransformer::get_hash_value() ^ ATOMIC_HASH,
            ft_hash
        );
        let word = |i: usize| u32::from_le_bytes(file[i..i + 4].try_into().unwrap());
        assert_eq!(word(0), VERSION);
        assert_eq!(
            word(4),
            ft_hash ^ NetworkArchitecture::get_hash_value() ^ LAYER_STACK_TABLE_HASH
        );
        let table = 12 + 13;
        assert_eq!(word(table), LAYER_STACK_TABLE_HASH);
        assert_eq!(word(table + 4), RAW_LAYER_STACKS as u32);
        assert_eq!(word(table + 8), 40);
        // The biases come first after the table and the transformer's hash,
        // in their order, then the threat weights, by feature, the pairs'
        // last, then the piece-square weights, then the 40 stacks.
        let biases = table + 12 + 2 * RAW_LAYER_STACKS + 4;
        assert_eq!(&file[biases..][..6], &[0, 0, 1, 0, 2, 0]);
        let threats = biases + 2 * L1;
        assert_eq!(&file[threats..][..3], &[0, 1, 2]);
        let psq = threats + THREAT_AND_PAIR_DIMENSIONS * L1;
        assert!(file[psq - 8192..psq].iter().all(|&b| b != 0));
        assert_eq!(&file[psq..][..3], &[0, 255, 254]);
        let mut stack = Vec::new();
        read.network[0].write_parameters(&mut stack);
        assert_eq!(
            file.len(),
            psq + PSQ_DIMENSIONS * L1 + 40 * (4 + stack.len())
        );

        assert!(Network::read(&file[..file.len() - 1]).is_none());
        assert!(Network::read(&[&file[..], &[0]].concat()).is_none());
        let mut other_hash = file.clone();
        other_hash[4] ^= 1;
        assert!(Network::read(&other_hash).is_none());
        // A raw stack falling back to one that is not stored
        let mut bad = file.clone();
        bad[table + 12 + 2 * 50] = 45;
        assert!(Network::read(&bad).is_none());
        // Stored stacks other than the table's
        let mut bad = file.clone();
        bad[table + 8] = 41;
        assert!(Network::read(&bad).is_none());
        // The hash of the networks before SFNNv20, without the table's
        let mut bad = file.clone();
        bad[4..8].copy_from_slice(&(Network::HASH ^ LAYER_STACK_TABLE_HASH).to_le_bytes());
        assert!(Network::read(&bad).is_none());
        assert!(AtomicNetwork::read(&bad).is_none());

        // Atomic's networks read both formats, and neither is chess's or
        // the other's.
        let (atomic, _) = AtomicNetwork::read(&file).expect("an Atomic network");
        assert!(matches!(*atomic, AtomicNetwork::Sfnnv20(_)));
        assert!(crate::nnue::network::Network::read(&file).is_none());
        assert!(crate::nnue::atomic::network::Network::read(&file).is_none());
        let legacy = crate::nnue::atomic::network::Network::zeroed().write(b"");
        let (atomic, _) = AtomicNetwork::read(&legacy).expect("an Atomic network");
        assert!(matches!(*atomic, AtomicNetwork::Legacy(_)));
        assert!(Network::read(&legacy).is_none());
        let chess = crate::nnue::network::Network::zeroed_with_stacks(1).write(b"");
        assert!(Network::read(&chess).is_none());
        assert!(AtomicNetwork::read(&chess).is_none());
    }
}
