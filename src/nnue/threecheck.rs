//! Three-check's networks in Stockfish's design, SFNNv20: Stockfish's
//! FullThreats, PP_3Wide and K32H3 feature transformer (reusing chess's
//! accumulators and caches verbatim), plus a per-perspective check-remaining
//! bias vector for each of the nine non-terminal check states (`{1..3} ×
//! {1..3}`), and 144 raw layer stacks conditioned on piece count (4 bands),
//! queens (4 states) and remaining checks (9 states).
//!
//! Its hash XORs Stockfish's feature transformer hash with `"3Chk"`, so no
//! chess or Atomic network reads as a Three-check network.

use std::mem::MaybeUninit;

use crate::memory::LargePageBox;
use crate::nnue::accumulator::{AccumulatorCaches, AccumulatorStack};
use crate::nnue::architecture::{L1, L2, L3, NetworkArchitecture};
use crate::nnue::common::{Aligned, OUTPUT_SCALE, Reader, VERSION, write_i16s, write_u32};
use crate::nnue::eval_file::NetworkFile;
use crate::nnue::feature_transformer::{
    FeatureTransformer, INPUT_DIMENSIONS, INVERSE_PACKUS_EPI16_ORDER, PACKUS_EPI16_ORDER, permute,
};
use crate::nnue::network::{
    LAYER_STACK_TABLE_HASH, read_hash, read_header, read_layer_stack_table_n,
    write_layer_stack_table_n,
};
use crate::nnue::nnz_helper::NnzInfo;
use crate::position::{CHECK_STATES, Position};
use crate::types::*;
use crate::variant::ThreeCheck;

/// Default filename for Three-check's dedicated SFNNv20 network, which the
/// Makefile's `THREE_CHECK_NET` names too.
pub const EVAL_FILE_DEFAULT_NAME: &str = "3check-ae396b7e8123.nnue";

/// What sets Three-check's SFNNv20 networks apart from chess's: `"3Chk"`.
pub const THREE_CHECK_HASH: u32 = 0x3343_686B;

/// Hash of the per-perspective check-state bias table.
const CHECK_BIASES_HASH: u32 = THREE_CHECK_HASH ^ (CHECK_STATES as u32 * L1 as u32);

/// Raw layer stacks for Three-check: 4 piece-count bands × 4 queen states × 9
/// remaining-check states.
pub const RAW_LAYER_STACKS: usize = 4 * 4 * CHECK_STATES;
pub const LAYER_STACKS: usize = RAW_LAYER_STACKS;

/// Evaluation by each stored layer stack of a Three-check network, for `eval`.
pub struct ThreeCheckEvalTrace {
    pub positional: [Value; LAYER_STACKS],
    pub correct_bucket: usize,
    pub layer_stacks: usize,
}

/// Three-check's dedicated SFNNv20 network.
pub struct ThreeCheckNetwork {
    pub(crate) feature_transformer: FeatureTransformer,
    /// Per-perspective additive bias on the L1 accumulator, indexed by
    /// `ChecksRemaining::state_index(perspective)` (`0..9`), permuted by
    /// `PACKUS_EPI16_ORDER` at load time to match the accumulators.
    check_biases: [Aligned<[i16; L1]>; CHECK_STATES],
    network: [NetworkArchitecture; LAYER_STACKS],
    layer_stack_fallback: [u16; RAW_LAYER_STACKS],
    layer_stack_index: [u16; RAW_LAYER_STACKS],
    stored_layer_stacks: usize,
}

impl ThreeCheckNetwork {
    /// Hash value of evaluation function structure.
    pub const HASH: u32 = FeatureTransformer::get_hash_value()
        ^ CHECK_BIASES_HASH
        ^ NetworkArchitecture::get_hash_value()
        ^ LAYER_STACK_TABLE_HASH;

    /// A network whose parameters are all zero.
    pub fn zeroed() -> LargePageBox<ThreeCheckNetwork> {
        // SAFETY: all parameters are integers.
        unsafe { LargePageBox::zeroed() }
    }

    /// A zeroed network storing `stored` layer stacks (1..=144), the others
    /// falling back to stack 0.
    #[cfg(test)]
    pub fn zeroed_with_stacks(stored: usize) -> LargePageBox<ThreeCheckNetwork> {
        let mut network = ThreeCheckNetwork::zeroed();
        for s in 0..RAW_LAYER_STACKS {
            let f = if s < stored { s } else { 0 };
            network.layer_stack_fallback[s] = f as u16;
            network.layer_stack_index[s] = f as u16;
        }
        network.stored_layer_stacks = stored;
        network
    }

    /// Reads a Three-check SFNNv20 network file and its description.
    pub fn read(bytes: &[u8]) -> Option<(LargePageBox<ThreeCheckNetwork>, Vec<u8>)> {
        let mut reader = Reader::new(bytes);
        let (hash_value, description) = read_header(&mut reader)?;
        if hash_value != Self::HASH {
            return None;
        }

        let mut network = Self::zeroed();
        let net = &mut *network;
        let stored = read_layer_stack_table_n(
            &mut reader,
            &mut net.layer_stack_fallback,
            &mut net.layer_stack_index,
            LAYER_STACKS,
        )?;
        net.stored_layer_stacks = stored;

        if !read_hash(&mut reader, FeatureTransformer::get_hash_value())
            || !network.feature_transformer.read_parameters(&mut reader)
        {
            return None;
        }
        if !read_hash(&mut reader, CHECK_BIASES_HASH) {
            return None;
        }
        for bias in &mut network.check_biases {
            reader.read_i16s(&mut bias.0);
            permute(&mut bias.0, &PACKUS_EPI16_ORDER);
        }
        if !reader.good() {
            return None;
        }

        for stack in &mut network.network[..stored] {
            if !read_hash(&mut reader, NetworkArchitecture::get_hash_value())
                || !stack.read_parameters(&mut reader)
            {
                return None;
            }
        }

        (reader.good() && reader.at_end()).then_some((network, description))
    }

    /// Writes the network in its file format with `description`.
    pub fn write(&self, description: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        write_u32(&mut out, VERSION);
        write_u32(&mut out, Self::HASH);
        write_u32(&mut out, description.len() as u32);
        out.extend_from_slice(description);

        write_layer_stack_table_n(
            &mut out,
            self.stored_layer_stacks,
            &self.layer_stack_fallback,
        );

        write_u32(&mut out, FeatureTransformer::get_hash_value());
        self.feature_transformer.write_parameters(&mut out);

        write_u32(&mut out, CHECK_BIASES_HASH);
        for bias in &self.check_biases {
            let mut unpermuted = bias.0;
            permute(&mut unpermuted, &INVERSE_PACKUS_EPI16_ORDER);
            write_i16s(&mut out, &unpermuted);
        }

        for stack in &self.network[..self.stored_layer_stacks] {
            write_u32(&mut out, NetworkArchitecture::get_hash_value());
            stack.write_parameters(&mut out);
        }
        out
    }

    /// Evaluates `pos` (which is neither in check nor terminal) from the side
    /// to move's point of view.
    pub fn evaluate(
        &self,
        pos: &Position<ThreeCheck>,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> Value {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::new();

        let bucket = self.layer_stack(pos);
        let checks = pos.state().extra;
        let stm = pos.side_to_move();
        let biases = [
            &self.check_biases[checks.state_index(stm)].0,
            &self.check_biases[checks.state_index(!stm)].0,
        ];
        let transformed_features = self.feature_transformer.transform_with_bias(
            pos,
            accumulator_stack,
            cache,
            biases,
            &mut transformed_features,
            &mut nnz_info,
        );
        let positional = self.network[bucket].propagate(transformed_features, &nnz_info);
        positional / OUTPUT_SCALE
    }

    /// Evaluates `pos` across every stored layer stack, for `eval`.
    pub fn trace_evaluate(
        &self,
        pos: &Position<ThreeCheck>,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> ThreeCheckEvalTrace {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::new();

        let correct_bucket = self.layer_stack(pos);
        let checks = pos.state().extra;
        let stm = pos.side_to_move();
        let biases = [
            &self.check_biases[checks.state_index(stm)].0,
            &self.check_biases[checks.state_index(!stm)].0,
        ];
        let transformed_features = self.feature_transformer.transform_with_bias(
            pos,
            accumulator_stack,
            cache,
            biases,
            &mut transformed_features,
            &mut nnz_info,
        );

        let layer_stacks = self.stored_layer_stacks;
        ThreeCheckEvalTrace {
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

    /// Raw layer stack: 4 piece-count bands × 4 queen states × 9 check states.
    #[inline(always)]
    pub fn raw_layer_stack(pos: &Position<ThreeCheck>) -> usize {
        let stm = pos.side_to_move();
        let band = ((pos.count_both(ALL_PIECES) - 1) / 8) as usize;
        let queens = 2 * usize::from(pos.pieces_cpt(stm, QUEEN).any())
            + usize::from(pos.pieces_cpt(!stm, QUEEN).any());
        let checks = pos.state().extra.state_index(stm);
        (band * 4 + queens) * CHECK_STATES + checks
    }

    #[inline(always)]
    fn layer_stack(&self, pos: &Position<ThreeCheck>) -> usize {
        usize::from(self.layer_stack_index[Self::raw_layer_stack(pos)])
    }
}

impl NetworkFile for ThreeCheckNetwork {
    const DEFAULT_NAME: &'static str = EVAL_FILE_DEFAULT_NAME;

    fn read(bytes: &[u8]) -> Option<(LargePageBox<ThreeCheckNetwork>, Vec<u8>)> {
        ThreeCheckNetwork::read(bytes)
    }

    fn write(&self, description: &[u8]) -> Vec<u8> {
        ThreeCheckNetwork::write(self, description)
    }

    fn kind(&self) -> &'static str {
        "Three-check FullThreats+PP_3Wide+K32H3+Checks SFNNv20 "
    }

    fn download_url() -> String {
        format!(
            "https://github.com/aiorla/varifish/releases/download/networks/{}",
            Self::DEFAULT_NAME
        )
    }

    fn size_and_dimensions(&self) -> String {
        let size = (size_of::<FeatureTransformer>()
            + size_of::<[Aligned<[i16; L1]>; CHECK_STATES]>()
            + self.stored_layer_stacks * size_of::<NetworkArchitecture>())
            / (1024 * 1024);
        let stacks = self.stored_layer_stacks;
        format!("({size}MiB, ({INPUT_DIMENSIONS}, {L1}, {L2}, {L3}, 1) x {stacks})")
    }
}

/// Per-thread accumulator caches for Three-check, wrapping chess's Finny
/// tables so both dedicated Three-check networks and the chess bootstrap
/// network share the same cache structure.
#[repr(transparent)]
pub struct ThreeCheckCaches(pub AccumulatorCaches);

impl crate::nnue::Caches for ThreeCheckCaches {
    type Network = ThreeCheckNetwork;

    fn new(network: Option<&ThreeCheckNetwork>) -> LargePageBox<Self> {
        // SAFETY: zero bytes make entries, which clear() initializes.
        let mut caches = unsafe { LargePageBox::<ThreeCheckCaches>::zeroed() };
        if let Some(network) = network {
            caches.clear(network);
        }
        caches
    }

    fn clear(&mut self, network: &ThreeCheckNetwork) {
        self.0
            .clear_with_biases(&network.feature_transformer.biases.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::network::Network as ChessNetwork;
    use crate::position::START_FEN;

    #[test]
    fn files_are_read_whole_or_refused() {
        let mut net = ThreeCheckNetwork::zeroed_with_stacks(4);
        net.check_biases[0].0[0] = 42;
        net.check_biases[8].0[L1 - 1] = -17;
        let bytes = net.write(b"three-check test");

        // Must not parse as a chess network.
        assert!(ChessNetwork::read(&bytes).is_none());

        let (read, desc) = ThreeCheckNetwork::read(&bytes).unwrap();
        assert_eq!(desc, b"three-check test");
        assert_eq!(read.stored_layer_stacks, 4);
        assert_eq!(read.check_biases[0].0, net.check_biases[0].0);
        assert_eq!(read.check_biases[8].0, net.check_biases[8].0);
        assert_eq!(read.write(&desc), bytes);

        // Truncated or trailing bytes are refused.
        assert!(ThreeCheckNetwork::read(&bytes[..bytes.len() - 1]).is_none());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(ThreeCheckNetwork::read(&extra).is_none());

        // A chess network file must not parse as a Three-check network.
        let chess_bytes = ChessNetwork::zeroed_with_stacks(1).write(b"chess");
        assert!(ThreeCheckNetwork::read(&chess_bytes).is_none());
    }

    #[test]
    fn raw_layer_stack_covers_all_check_states() {
        for w in 1..=3u8 {
            for b in 1..=3u8 {
                let fen = format!("4k3/8/8/8/8/8/8/4K3 w - - {w}+{b} 0 1");
                let pos = Position::<ThreeCheck>::from_fen(&fen, false).unwrap();
                let raw = ThreeCheckNetwork::raw_layer_stack(&pos);
                assert!(raw < RAW_LAYER_STACKS);
                assert_eq!(raw % CHECK_STATES, ((w - 1) * 3 + (b - 1)) as usize);
            }
        }
        let pos = Position::<ThreeCheck>::from_fen(START_FEN, false).unwrap();
        assert!(ThreeCheckNetwork::raw_layer_stack(&pos) < RAW_LAYER_STACKS);
    }

    #[test]
    fn incremental_eval_matches_from_scratch_and_uses_check_biases() {
        use crate::misc::Prng;
        use crate::movegen::{ALL, MoveList};
        use crate::nnue::eval_file::{EvalFile, Networks, UseNnue};
        use crate::variant::Variant;

        // Non-zero FT, layer stack and check biases so every SIMD path
        // (incremental accumulator updates, transform_with_bias, and FC
        // propagation) produces genuinely non-zero, check-sensitive outputs.
        let mut net = ThreeCheckNetwork::zeroed_with_stacks(1);
        for (i, b) in net.feature_transformer.biases.0.iter_mut().enumerate() {
            *b = ((i as i16 * 11 + 3) % 25) - 12;
        }
        for (i, w) in net.feature_transformer.weights.0.iter_mut().enumerate() {
            *w = ((i * 7 + 1) % 7) as i8 - 3;
        }
        for (i, w) in net
            .feature_transformer
            .threat_and_pp_weights
            .0
            .iter_mut()
            .enumerate()
        {
            *w = ((i * 5 + 2) % 5) as i8 - 2;
        }
        for (idx, bias) in net.check_biases.iter_mut().enumerate() {
            bias.0.fill((idx as i16 + 1) * 32);
        }
        let mut stack_bytes = Vec::new();
        net.network[0].write_parameters(&mut stack_bytes);
        for (i, byte) in stack_bytes.iter_mut().enumerate() {
            *byte = ((i * 13 + 7) % 5) as u8;
        }
        let mut reader = Reader::new(&stack_bytes);
        assert!(net.network[0].read_parameters(&mut reader) && reader.at_end());
        let bytes = net.write(b"test");
        let (loaded, _) = ThreeCheckNetwork::read(&bytes).unwrap();

        let networks = Networks {
            chess: EvalFile::with_network(ChessNetwork::zeroed_with_stacks(1)),
            atomic: EvalFile::default(),
            threecheck: EvalFile::with_network(loaded),
            crazyhouse: EvalFile::default(),
            antichess: EvalFile::default(),
        };

        let mut pos = Position::<ThreeCheck>::from_fen(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2+2 4 4",
            false,
        )
        .unwrap();
        let mut stack = AccumulatorStack::new();
        let mut caches = ThreeCheck::new_caches(&networks);

        let mut rng = Prng::new(12345);
        let mut saw_nonzero = false;
        for _ in 0..25 {
            let moves = MoveList::new::<ALL>(&pos);
            if moves.is_empty() {
                break;
            }
            let mv = moves.as_slice()[(rng.rand64() as usize) % moves.len()];
            let dirties = stack.push(&pos);
            pos.make_move(mv, dirties);
            if !pos.in_check() && !pos.is_three_check_over() {
                let inc =
                    ThreeCheck::evaluate(&networks, &pos, &mut stack, &mut caches, UseNnue::Pure);
                let mut fresh_stack = AccumulatorStack::new();
                let mut fresh_caches = ThreeCheck::new_caches(&networks);
                let scratch = ThreeCheck::evaluate(
                    &networks,
                    &pos,
                    &mut fresh_stack,
                    &mut fresh_caches,
                    UseNnue::Pure,
                );
                assert_eq!(inc, scratch, "{}", pos.fen());
                if inc != 0 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "expected non-zero ThreeCheckNetwork outputs");

        // Changing remaining checks on the same board must change the score
        // through check_biases (all raw stacks map to stack 0 here).
        let p_33 = Position::<ThreeCheck>::from_fen(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3+3 4 4",
            false,
        )
        .unwrap();
        let p_13 = Position::<ThreeCheck>::from_fen(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1+3 4 4",
            false,
        )
        .unwrap();
        let mut s_a = AccumulatorStack::new();
        let mut c_a = ThreeCheck::new_caches(&networks);
        let mut s_b = AccumulatorStack::new();
        let mut c_b = ThreeCheck::new_caches(&networks);
        let v_33 = ThreeCheck::evaluate(&networks, &p_33, &mut s_a, &mut c_a, UseNnue::Pure);
        let v_13 = ThreeCheck::evaluate(&networks, &p_13, &mut s_b, &mut c_b, UseNnue::Pure);
        assert_ne!(v_33, v_13, "check_biases should distinguish 3+3 from 1+3");

        // Bootstrap evaluation reflects check advantage symmetrically.
        let bootstrap = Networks {
            chess: EvalFile::with_network(ChessNetwork::zeroed_with_stacks(1)),
            atomic: EvalFile::default(),
            threecheck: EvalFile::default(),
            crazyhouse: EvalFile::default(),
            antichess: EvalFile::default(),
        };
        let p_ahead =
            Position::<ThreeCheck>::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 1+3 0 1", false).unwrap();
        let mut p_behind = p_ahead.clone();
        p_behind.flip().unwrap();
        let mut s1 = AccumulatorStack::new();
        let mut c1 = ThreeCheck::new_caches(&bootstrap);
        let mut s2 = AccumulatorStack::new();
        let mut c2 = ThreeCheck::new_caches(&bootstrap);
        let v_ahead = ThreeCheck::evaluate(&bootstrap, &p_ahead, &mut s1, &mut c1, UseNnue::Pure);
        let v_behind = ThreeCheck::evaluate(&bootstrap, &p_behind, &mut s2, &mut c2, UseNnue::Pure);
        assert!(v_ahead > 0, "1+3 should favor White: {v_ahead}");
        assert_eq!(v_ahead, v_behind);
    }
}
