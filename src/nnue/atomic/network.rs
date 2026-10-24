//! The Atomic network: its feature transformer and layer stacks, read from
//! and written to its file, and its evaluation of positions.
//!
//! Derived, with changes for varifish in 2026, from Atomic-Stockfish's
//! `nnue/network.h` and `.cpp`, `nnue/nnue_feature_transformer.h` and
//! `nnue/nnue_architecture.h` (https://github.com/Belzedar94/Atomic-Stockfish)
//! and from Stockfish's, Copyright (C) 2004-2026 The Stockfish developers
//! (see AUTHORS), under the GNU General Public License, version 3 or later.

#![allow(unused_unsafe)]

#[cfg(use_neon)]
use std::arch::aarch64::*;
use std::mem::MaybeUninit;

use super::accumulator::{AccumulatorCaches, AccumulatorStack, AccumulatorState};
use super::features;
use crate::memory::LargePageBox;
use crate::nnue::architecture::L1;
use crate::nnue::common::{
    Aligned, MAX_SIMD_WIDTH, OUTPUT_SCALE, Reader, WEIGHT_SCALE_BITS, ceil_to_multiple, write_i16s,
    write_i32s, write_u32,
};
use crate::nnue::eval_file::NetworkFile;
use crate::nnue::feature_transformer::{INVERSE_PACKUS_EPI16_ORDER, PACKUS_EPI16_ORDER, permute};
use crate::nnue::layers::affine_transform::AffineTransform;
use crate::nnue::layers::affine_transform_sparse_input::AffineTransformSparseInput;
use crate::nnue::layers::clipped_relu::ClippedReLU;
use crate::nnue::nnz_helper::NnzInfo;
#[cfg(vector)]
use crate::nnue::simd::*;
use crate::position::Position;
use crate::types::*;
use crate::variant::Atomic;

/// Version of the evaluation file, Stockfish's of 2021
pub const VERSION: u32 = 0x7AF3_2F20;

/// The network Atomic-Stockfish evaluates with by default, run3b, which is
/// not embedded
pub const EVAL_FILE_DEFAULT_NAME: &str = "atomic_run3b_e202_l05.nnue";

/// Number of accumulators of one perspective
pub const HALF_DIMENSIONS: usize = 512;
/// Number of input feature dimensions after conversion: both perspectives'
pub const OUTPUT_DIMENSIONS: usize = 2 * HALF_DIMENSIONS;
pub const INPUT_DIMENSIONS: usize = features::DIMENSIONS;

pub const PSQT_BUCKETS: usize = 8;
pub const LAYER_STACKS: usize = 8;

const FC_0_OUTPUTS: usize = 16;
const FC_1_OUTPUTS: usize = 32;

// The first layer reads only the non-zero chunks of its input, which the
// feature transformer records as chess's does for the same number of inputs.
const _: () = assert!(OUTPUT_DIMENSIONS == L1);

/// The longest description a file may have
const MAX_DESCRIPTION_SIZE: u32 = 1 << 20;

/// Input feature converter
pub struct FeatureTransformer {
    pub biases: Aligned<[i16; HALF_DIMENSIONS]>,
    pub weights: Aligned<[i16; HALF_DIMENSIONS * INPUT_DIMENSIONS]>,
    pub psqt_weights: Aligned<[i32; INPUT_DIMENSIONS * PSQT_BUCKETS]>,
}

impl FeatureTransformer {
    /// Hash value embedded in the evaluation file
    pub const fn get_hash_value() -> u32 {
        features::HASH_VALUE ^ OUTPUT_DIMENSIONS as u32
    }

    /// The PSQT weights of a feature, a value per bucket.
    #[inline(always)]
    pub fn psqt_row(&self, index: u32) -> &[i32; PSQT_BUCKETS] {
        self.psqt_weights.0[index as usize * PSQT_BUCKETS..]
            .first_chunk()
            .expect("a feature's PSQT weights")
    }

    /// Read network parameters. The biases and weights are permuted, as
    /// chess's are, for the packs that convert the accumulations.
    pub fn read_parameters(&mut self, reader: &mut Reader) -> bool {
        reader.read_i16s(&mut self.biases.0);
        reader.read_i16s(&mut self.weights.0);
        reader.read_i32s(&mut self.psqt_weights.0);

        permute(&mut self.biases.0, &PACKUS_EPI16_ORDER);
        permute(&mut self.weights.0, &PACKUS_EPI16_ORDER);

        reader.good()
    }

    /// Write network parameters
    pub fn write_parameters(&self, out: &mut Vec<u8>) {
        let mut biases = self.biases.0;
        permute(&mut biases, &INVERSE_PACKUS_EPI16_ORDER);
        write_i16s(out, &biases);

        let mut weights = self.weights.0.to_vec();
        permute(&mut weights, &INVERSE_PACKUS_EPI16_ORDER);
        write_i16s(out, &weights);
        drop(weights);

        write_i32s(out, &self.psqt_weights.0);
    }

    /// The PSQT term of a bucket for the side to move, from the sums of its
    /// perspective and of the other.
    #[inline(always)]
    pub fn psqt(state: &AccumulatorState, side_to_move: Color, bucket: usize) -> i32 {
        let psqt = &state.psqt_accumulation;
        psqt[side_to_move.idx()][bucket].wrapping_sub(psqt[(!side_to_move).idx()][bucket]) / 2
    }

    /// Convert input features: writes `output`, the side to move's
    /// accumulations then the other's, clamped to 0..=127, and returns it.
    #[inline(always)]
    pub fn transform<'a>(
        &self,
        pos: &Position<Atomic>,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
        output: &'a mut MaybeUninit<Aligned<[u8; OUTPUT_DIMENSIONS]>>,
        nnz_info: &mut NnzInfo,
    ) -> &'a [u8; OUTPUT_DIMENSIONS] {
        accumulator_stack.evaluate(pos, self, cache);
        let accumulation = &accumulator_stack.latest().accumulation;

        let perspectives = [pos.side_to_move(), !pos.side_to_move()];
        for (p, perspective) in perspectives.into_iter().enumerate() {
            Self::transform_perspective(&accumulation[perspective.idx()], output, p, nnz_info);
        }

        // SAFETY: each perspective wrote its half.
        unsafe { &output.assume_init_ref().0 }
    }

    /// Writes the half of `output` for perspective `p`: its accumulation
    /// clamped to 0..=127.
    #[inline(always)]
    fn transform_perspective(
        accumulation: &[i16; HALF_DIMENSIONS],
        output: &mut MaybeUninit<Aligned<[u8; OUTPUT_DIMENSIONS]>>,
        perspective: usize,
        nnz_info: &mut NnzInfo,
    ) {
        let offset = HALF_DIMENSIONS * perspective;

        #[cfg(vector)]
        // SAFETY: the loads read vectors of the accumulation, the stores
        // write vectors of the perspective's half of the output.
        unsafe {
            let mut cursor = nnz_info.make_cursor(perspective);

            // An output vector of bytes packs two vectors of accumulations.
            const OUTPUT_CHUNK_SIZE: usize = MAX_CHUNK_SIZE;
            const { assert!(HALF_DIMENSIONS.is_multiple_of(OUTPUT_CHUNK_SIZE)) };
            const NUM_OUTPUT_CHUNKS: usize = HALF_DIMENSIONS / OUTPUT_CHUNK_SIZE;
            // The chunks go in pairs.
            const { assert!(NUM_OUTPUT_CHUNKS.is_multiple_of(2)) };

            #[cfg(not(use_neon))]
            let max = vec_set_16(127);

            let input = |i: usize| vec_load(accumulation.as_ptr().add(i * LANES));
            let out = output.as_mut_ptr().cast::<u8>().add(offset);

            // Packing with unsigned saturation clamps below at zero.
            let mut j = 0;
            while j < NUM_OUTPUT_CHUNKS {
                let mut packed = [input(0); 2];
                for (k, packed) in packed.iter_mut().enumerate() {
                    let i = (j + k) * 2;

                    #[cfg(use_neon)]
                    let result = {
                        let max = vdupq_n_s16(127);
                        vreinterpretq_s16_u8(vcombine_u8(
                            vqmovun_s16(vminq_s16(input(i), max)),
                            vqmovun_s16(vminq_s16(input(i + 1), max)),
                        ))
                    };

                    #[cfg(not(use_neon))]
                    let result =
                        vec_packus_16(vec_min_16(input(i), max), vec_min_16(input(i + 1), max));

                    *packed = result;
                    vec_store(out.add((j + k) * OUTPUT_CHUNK_SIZE).cast(), result);
                }

                cursor.record2(packed[0], packed[1]);
                j += 2;
            }
        }

        #[cfg(not(vector))]
        {
            let _ = nnz_info;
            let output = output.as_mut_ptr().cast::<u8>();
            for (j, &value) in accumulation.iter().enumerate() {
                // SAFETY: the perspective's half of the output.
                unsafe { output.add(offset + j).write(value.clamp(0, 127) as u8) };
            }
        }
    }
}

const WSB: i32 = WEIGHT_SCALE_BITS as i32;

/// The weights of an affine layer: a row of padded inputs per output.
const fn weights(inputs: usize, outputs: usize) -> usize {
    outputs * ceil_to_multiple(inputs, MAX_SIMD_WIDTH)
}

type Fc0 = AffineTransformSparseInput<
    OUTPUT_DIMENSIONS,
    FC_0_OUTPUTS,
    { weights(OUTPUT_DIMENSIONS, FC_0_OUTPUTS) },
>;
type Ac0 = ClippedReLU<FC_0_OUTPUTS, WSB>;
type Fc1 =
    AffineTransform<FC_0_OUTPUTS, FC_1_OUTPUTS, { weights(FC_0_OUTPUTS, FC_1_OUTPUTS) }, false>;
type Ac1 = ClippedReLU<FC_1_OUTPUTS, WSB>;
type Fc2 = AffineTransform<FC_1_OUTPUTS, 1, { weights(FC_1_OUTPUTS, 1) }, false>;

pub struct NetworkArchitecture {
    fc_0: Fc0,
    ac_0: Ac0,
    fc_1: Fc1,
    ac_1: Ac1,
    fc_2: Fc2,
}

impl NetworkArchitecture {
    /// Hash value embedded in the evaluation file
    pub const fn get_hash_value() -> u32 {
        // input slice hash
        let mut hash_value = 0xEC42_E90Du32;
        hash_value ^= OUTPUT_DIMENSIONS as u32;

        hash_value = Fc0::get_hash_value(hash_value);
        hash_value = Ac0::get_hash_value(hash_value);
        hash_value = Fc1::get_hash_value(hash_value);
        hash_value = Ac1::get_hash_value(hash_value);
        hash_value = Fc2::get_hash_value(hash_value);

        hash_value
    }

    /// Read network parameters
    pub fn read_parameters(&mut self, reader: &mut Reader) -> bool {
        self.fc_0.read_parameters(reader)
            && self.ac_0.read_parameters(reader)
            && self.fc_1.read_parameters(reader)
            && self.ac_1.read_parameters(reader)
            && self.fc_2.read_parameters(reader)
    }

    /// Write network parameters
    pub fn write_parameters(&self, out: &mut Vec<u8>) {
        self.fc_0.write_parameters(out);
        self.ac_0.write_parameters(out);
        self.fc_1.write_parameters(out);
        self.ac_1.write_parameters(out);
        self.fc_2.write_parameters(out);
    }

    /// The layer stack's output, before dividing by OUTPUT_SCALE.
    #[inline(always)]
    pub fn propagate(
        &self,
        transformed_features: &[u8; OUTPUT_DIMENSIONS],
        nnz_info: &NnzInfo,
    ) -> i32 {
        let mut fc_0_out = Aligned([0; FC_0_OUTPUTS]);
        let mut ac_0_out = Aligned([0; ceil_to_multiple(FC_0_OUTPUTS, 32)]);
        let mut fc_1_out = Aligned([0; FC_1_OUTPUTS]);
        let mut ac_1_out = Aligned([0; ceil_to_multiple(FC_1_OUTPUTS, 32)]);
        let mut fc_2_out = Aligned([0; 1]);

        self.fc_0
            .propagate(transformed_features, &mut fc_0_out.0, nnz_info);
        self.ac_0
            .propagate(&fc_0_out.0, ac_0_out.0.first_chunk_mut().unwrap());
        self.fc_1.propagate(&ac_0_out.0, &mut fc_1_out.0);
        self.ac_1
            .propagate(&fc_1_out.0, ac_1_out.0.first_chunk_mut().unwrap());
        self.fc_2.propagate(&ac_1_out.0, &mut fc_2_out.0);

        fc_2_out.0[0]
    }
}

const _: () = assert!(FeatureTransformer::get_hash_value() == 0x5F23_48B8);
const _: () = assert!(NetworkArchitecture::get_hash_value() == 0x6333_76CA);

pub struct Network {
    // Input feature converter
    pub(super) feature_transformer: FeatureTransformer,

    // Evaluation function
    network: [NetworkArchitecture; LAYER_STACKS],
}

/// The evaluation of each layer stack, and the stack a position uses: the
/// PSQT and layer terms, each divided by OUTPUT_SCALE, and their sum.
pub struct NnueEvalTrace {
    pub psqt: [Value; LAYER_STACKS],
    pub positional: [Value; LAYER_STACKS],
    pub total: [Value; LAYER_STACKS],
    pub correct_bucket: usize,
}

/// The layer stack and PSQT bucket of a position, by its number of pieces.
#[inline(always)]
fn bucket(pos: &Position<Atomic>) -> usize {
    ((pos.count_both(ALL_PIECES) - 1) / 4).clamp(0, LAYER_STACKS as i32 - 1) as usize
}

/// A value within the evaluations, outside the tablebase range.
pub fn value_from_scaled(scaled: i64) -> Value {
    scaled.clamp(
        i64::from(VALUE_TB_LOSS_IN_MAX_PLY + 1),
        i64::from(VALUE_TB_WIN_IN_MAX_PLY - 1),
    ) as Value
}

impl Network {
    /// Hash value of evaluation function structure
    pub const HASH: u32 =
        FeatureTransformer::get_hash_value() ^ NetworkArchitecture::get_hash_value();

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
        let version = reader.read_u32();
        let hash_value = reader.read_u32();
        let size = reader.read_u32();
        if !reader.good()
            || version != VERSION
            || hash_value != Self::HASH
            || size > MAX_DESCRIPTION_SIZE
        {
            return None;
        }
        let description = reader.take(size as usize)?.to_vec();

        let mut network = Self::zeroed();
        if reader.read_u32() != FeatureTransformer::get_hash_value()
            || !network.feature_transformer.read_parameters(&mut reader)
        {
            return None;
        }
        for stack in &mut network.network {
            if reader.read_u32() != NetworkArchitecture::get_hash_value()
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

    /// The PSQT and layer stack terms of the position, from the side to
    /// move's point of view and before dividing by OUTPUT_SCALE, whose
    /// state the stack's latest state holds.
    pub fn evaluate_raw(
        &self,
        pos: &Position<Atomic>,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> (i32, i32) {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; OUTPUT_DIMENSIONS]>>::uninit();
        let mut nnz_info = NnzInfo::new();

        let bucket = bucket(pos);
        let transformed_features = self.feature_transformer.transform(
            pos,
            accumulator_stack,
            cache,
            &mut transformed_features,
            &mut nnz_info,
        );
        let psqt = FeatureTransformer::psqt(accumulator_stack.latest(), pos.side_to_move(), bucket);
        let positional = self.network[bucket].propagate(transformed_features, &nnz_info);
        (psqt, positional)
    }

    /// The evaluation by each layer stack.
    pub fn trace_evaluate(
        &self,
        pos: &Position<Atomic>,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> NnueEvalTrace {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; OUTPUT_DIMENSIONS]>>::uninit();
        let mut nnz_info = NnzInfo::new();

        let transformed_features = self.feature_transformer.transform(
            pos,
            accumulator_stack,
            cache,
            &mut transformed_features,
            &mut nnz_info,
        );
        let state = accumulator_stack.latest();

        let psqt = std::array::from_fn(|bucket| {
            FeatureTransformer::psqt(state, pos.side_to_move(), bucket) / OUTPUT_SCALE
        });
        let positional = std::array::from_fn(|bucket| {
            self.network[bucket].propagate(transformed_features, &nnz_info) / OUTPUT_SCALE
        });
        NnueEvalTrace {
            psqt,
            positional,
            total: std::array::from_fn(|bucket| {
                value_from_scaled(i64::from(psqt[bucket]) + i64::from(positional[bucket]))
            }),
            correct_bucket: bucket(pos),
        }
    }
}

impl NetworkFile for Network {
    const DEFAULT_NAME: &'static str = EVAL_FILE_DEFAULT_NAME;
    const KIND: &'static str = "Legacy Atomic V1 ";

    fn read(bytes: &[u8]) -> Option<(LargePageBox<Network>, Vec<u8>)> {
        Network::read(bytes)
    }

    fn write(&self, description: &[u8]) -> Vec<u8> {
        Network::write(self, description)
    }

    fn download_url() -> String {
        "https://github.com/Belzedar94/Atomic-Stockfish/releases/download/v1.0/atomic-run3b.nnue"
            .to_string()
    }

    fn size_and_dimensions() -> String {
        let size =
            size_of::<FeatureTransformer>() + size_of::<NetworkArchitecture>() * LAYER_STACKS;
        format!(
            "({}MiB, ({INPUT_DIMENSIONS}, {OUTPUT_DIMENSIONS}, {FC_0_OUTPUTS}, {FC_1_OUTPUTS}, 1))",
            size / (1024 * 1024)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_are_read_whole_or_refused() {
        let mut network = Network::zeroed();
        // Values in every permuted block, to see that writing undoes the
        // permutation.
        for (i, value) in network.feature_transformer.biases.0.iter_mut().enumerate() {
            *value = i as i16;
        }
        for (i, value) in network.feature_transformer.weights.0[..4096]
            .iter_mut()
            .enumerate()
        {
            *value = -(i as i16);
        }
        network.feature_transformer.psqt_weights.0[7] = -7;

        let file = network.write(b"a description");
        let (read, description) = Network::read(&file).expect("a network");
        assert_eq!(description, b"a description");
        assert!(read.write(&description) == file);
        assert_eq!(
            &file[..12],
            &[0x20, 0x2F, 0xF3, 0x7A, 0x72, 0x3E, 0x10, 0x3C, 13, 0, 0, 0]
        );
        // The biases come first after the hash, in their order.
        assert_eq!(&file[12 + 13 + 4..][..6], &[0, 0, 1, 0, 2, 0]);

        assert!(Network::read(&file[..file.len() - 1]).is_none());
        assert!(Network::read(&[&file[..], &[0]].concat()).is_none());
        let mut other_hash = file.clone();
        other_hash[4] ^= 1;
        assert!(Network::read(&other_hash).is_none());
        let mut long_description = file.clone();
        long_description[8..12].copy_from_slice(&(MAX_DESCRIPTION_SIZE + 1).to_le_bytes());
        assert!(Network::read(&long_description).is_none());
    }
}
