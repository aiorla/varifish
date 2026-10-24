//! The network's input features and the layers of each of its layer
//! stacks.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/nnue_architecture.h`: Stockfish, a UCI chess playing engine derived
//! from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! AUTHORS), under the GNU General Public License, version 3 or later.

use super::common::{
    Aligned, HIDDEN_ONE_VAL, MAX_SIMD_WIDTH, OUTPUT_SCALE, Reader, WEIGHT_SCALE_BITS,
    ceil_to_multiple,
};
use super::layers::affine_transform::AffineTransform;
use super::layers::affine_transform_sparse_input::AffineTransformSparseInput;
use super::layers::clipped_relu::ClippedReLU;
use super::layers::sqr_clipped_relu::SqrClippedReLU;
use super::nnz_helper::NnzInfo;

// Input features used in evaluation function
pub use super::features::full_threats::FullThreats as ThreatFeatureSet;
pub use super::features::k32q2::K32Q2 as PsqFeatureSet;
pub use super::features::pp_3wide::Pp3Wide as PairFeatureSet;

/// Number of input feature dimensions after conversion
pub const L1: usize = 1024;
pub const L2: usize = 32;
pub const L3: usize = 32;

pub const LAYER_STACKS: usize = 32;

const TRANSFORMED_FEATURE_DIMENSIONS: usize = L1;
const FC_0_OUTPUTS: usize = L2;
const FC_1_OUTPUTS: usize = L3;

const WSB: i32 = WEIGHT_SCALE_BITS as i32;

/// The weights of an affine layer: a row of padded inputs per output.
const fn weights(inputs: usize, outputs: usize) -> usize {
    outputs * ceil_to_multiple(inputs, MAX_SIMD_WIDTH)
}

type Fc0 = AffineTransformSparseInput<
    TRANSFORMED_FEATURE_DIMENSIONS,
    FC_0_OUTPUTS,
    { weights(TRANSFORMED_FEATURE_DIMENSIONS, FC_0_OUTPUTS) },
>;
type AcSqr0 = SqrClippedReLU<FC_0_OUTPUTS, { WSB + 1 }>;
type Ac0 = ClippedReLU<FC_0_OUTPUTS, { WSB + 1 }>;
type Fc1 = AffineTransform<
    { FC_0_OUTPUTS * 2 },
    FC_1_OUTPUTS,
    { weights(FC_0_OUTPUTS * 2, FC_1_OUTPUTS) },
    true,
>;
type AcSqr1 = SqrClippedReLU<FC_1_OUTPUTS, WSB>;
type Ac1 = ClippedReLU<FC_1_OUTPUTS, WSB>;
type Fc2 = AffineTransform<
    { FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2 },
    1,
    { weights(FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 1) },
    true,
>;

pub struct NetworkArchitecture {
    fc_0: Fc0,
    ac_sqr_0: AcSqr0,
    ac_0: Ac0,
    fc_1: Fc1,
    ac_sqr_1: AcSqr1,
    ac_1: Ac1,
    fc_2: Fc2,
}

impl NetworkArchitecture {
    /// Hash value embedded in the evaluation file
    pub const fn get_hash_value() -> u32 {
        // input slice hash
        let mut hash_value = 0xEC42_E90Du32;
        hash_value ^= (TRANSFORMED_FEATURE_DIMENSIONS * 2) as u32;

        hash_value = Fc0::get_hash_value(hash_value);
        // Stockfish leaves ac_sqr_0 out on purpose, as the trainer does not
        // write its hash value yet.
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

    #[inline(always)]
    pub fn propagate(&self, transformed_features: &[u8; L1], nnz_info: &NnzInfo) -> i32 {
        let mut fc_0_out = Aligned([0; FC_0_OUTPUTS]);
        let mut concat_buffer =
            Aligned([0; ceil_to_multiple(FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 32)]);
        let mut fc_1_out = Aligned([0; FC_1_OUTPUTS]);
        let mut fc_2_out = Aligned([0; 1]);

        self.fc_0
            .propagate(transformed_features, &mut fc_0_out.0, nnz_info);
        {
            let (squared, rest) = concat_buffer.0.split_first_chunk_mut().unwrap();
            let clipped = rest.first_chunk_mut().unwrap();
            #[cfg(use_pair_activations)]
            self.ac_sqr_0.propagate_pair(&fc_0_out.0, squared, clipped);
            #[cfg(not(use_pair_activations))]
            {
                self.ac_sqr_0.propagate(&fc_0_out.0, squared);
                self.ac_0.propagate(&fc_0_out.0, clipped);
            }
        }

        self.fc_1.propagate(&concat_buffer.0, &mut fc_1_out.0);
        {
            let rest = &mut concat_buffer.0[FC_0_OUTPUTS * 2..];
            let (squared, rest) = rest.split_first_chunk_mut().unwrap();
            let clipped = rest.first_chunk_mut().unwrap();
            #[cfg(use_pair_activations)]
            self.ac_sqr_1.propagate_pair(&fc_1_out.0, squared, clipped);
            #[cfg(not(use_pair_activations))]
            {
                self.ac_sqr_1.propagate(&fc_1_out.0, squared);
                self.ac_1.propagate(&fc_1_out.0, clipped);
            }
        }

        self.fc_2.propagate(&concat_buffer.0, &mut fc_2_out.0);

        const { assert!(FC_0_OUTPUTS >= 2) };
        let mut fwd_out = fc_2_out.0[0];
        let skip_0 = fc_0_out.0[FC_0_OUTPUTS - 2] - fc_0_out.0[FC_0_OUTPUTS - 1];
        fwd_out += skip_0;

        // fwd_out is such that 1.0 is equal to HiddenOneVal*(1<<WeightScaleBits)*2
        // in quantized form, but we want 1.0 to be equal to 600*OutputScale.
        // To make overflow impossible we cast to i64.
        const MULTIPLIER: i64 = 600 * OUTPUT_SCALE as i64;
        const DENOMINATOR: i64 = HIDDEN_ONE_VAL as i64 * (1i64 << WEIGHT_SCALE_BITS) * 2;

        ((i64::from(fwd_out) * MULTIPLIER) / DENOMINATOR) as i32
    }
}
