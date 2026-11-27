//! Crazyhouse's networks in Stockfish's design, SFNNv20: Stockfish's
//! FullThreats, PP_3Wide and K32H3 feature transformer (reusing chess's
//! accumulators and caches verbatim, with drops supported incrementally),
//! plus 44 per-perspective additive pocket feature vectors (`2 × [8P, 4N,
//! 4B, 4R, 2Q]`), and 144 raw layer stacks conditioned on board piece count
//! (4 bands), queen presence on board or in hand (4 states), and pocket
//! activity (3 × 3 states).
//!
//! Its hash XORs Stockfish's feature transformer hash with `"ZhNN"`, so no
//! chess, Atomic or Three-check network reads as a Crazyhouse network.

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
use crate::position::{CrazyhouseState, HAND_TYPES, Position};
use crate::types::*;
use crate::variant::Crazyhouse;

/// Default filename for Crazyhouse's dedicated SFNNv20 network, which the
/// Makefile's `CRAZYHOUSE_NET` names too.
pub const EVAL_FILE_DEFAULT_NAME: &str = "crazyhouse-b39a7a9b9831.nnue";

/// What sets Crazyhouse's SFNNv20 networks apart from chess's: `"ZhNN"`.
pub const CRAZYHOUSE_HASH: u32 = 0x5A68_4E4E;

/// Maximum unary-encoded pocket count per piece type (`PAWN..=QUEEN`).
pub const HAND_CAPS: [usize; HAND_TYPES] = [8, 4, 4, 4, 2];

/// Starting feature index within a perspective's side for each piece type.
pub const HAND_OFFSETS: [usize; HAND_TYPES] = [0, 8, 12, 16, 20];

/// Pocket features per side: `8P + 4N + 4B + 4R + 2Q = 22`.
pub const HAND_FEATURES_PER_SIDE: usize = 22;

/// Total per-perspective pocket feature vectors (`ours`, then `theirs`).
pub const HAND_FEATURES: usize = 2 * HAND_FEATURES_PER_SIDE;

/// Hash of the per-perspective pocket weight table.
const HAND_WEIGHTS_HASH: u32 = CRAZYHOUSE_HASH ^ (HAND_FEATURES as u32 * L1 as u32);

/// Pocket activity states (`3 × 3`: `{0, 1..=2, 3+}` for ours × theirs).
pub const POCKET_STATES: usize = 9;

/// Raw layer stacks for Crazyhouse: 4 board piece-count bands × 4 queen
/// states (board or hand) × 9 pocket activity states.
pub const RAW_LAYER_STACKS: usize = 4 * 4 * POCKET_STATES;
pub const LAYER_STACKS: usize = RAW_LAYER_STACKS;

/// Evaluation by each stored layer stack of a Crazyhouse network, for `eval`.
pub struct CrazyhouseEvalTrace {
    pub positional: [Value; LAYER_STACKS],
    pub correct_bucket: usize,
    pub layer_stacks: usize,
}

/// Crazyhouse's dedicated SFNNv20 network.
pub struct CrazyhouseNetwork {
    pub(crate) feature_transformer: FeatureTransformer,
    /// Per-perspective additive weights on the L1 accumulator for unary pocket
    /// counts (`0..HAND_FEATURES`), permuted by `PACKUS_EPI16_ORDER` at load
    /// time to match the accumulators.
    hand_weights: [Aligned<[i16; L1]>; HAND_FEATURES],
    network: [NetworkArchitecture; LAYER_STACKS],
    layer_stack_fallback: [u16; RAW_LAYER_STACKS],
    layer_stack_index: [u16; RAW_LAYER_STACKS],
    stored_layer_stacks: usize,
}

#[inline(always)]
fn hand_band(total: u8) -> usize {
    match total {
        0 => 0,
        1..=2 => 1,
        _ => 2,
    }
}

/// Sums the active unary pocket feature rows for `perspective` into an L1
/// additive bias vector.
fn accumulate_hand_bias(
    hand_weights: &[Aligned<[i16; L1]>; HAND_FEATURES],
    extra: &CrazyhouseState,
    perspective: Color,
) -> Aligned<[i16; L1]> {
    let mut bias = Aligned([0i16; L1]);
    for (rel, color) in [perspective, !perspective].into_iter().enumerate() {
        let side_base = rel * HAND_FEATURES_PER_SIDE;
        for (pt_idx, &cap) in HAND_CAPS.iter().enumerate() {
            let cnt = (extra.hand[color.idx()][pt_idx] as usize).min(cap);
            let base = side_base + HAND_OFFSETS[pt_idx];
            for k in 0..cnt {
                let row = &hand_weights[base + k].0;
                for (dst, &src) in bias.0.iter_mut().zip(row.iter()) {
                    *dst = dst.wrapping_add(src);
                }
            }
        }
    }
    bias
}

impl CrazyhouseNetwork {
    /// Hash value of evaluation function structure.
    pub const HASH: u32 = FeatureTransformer::get_hash_value()
        ^ HAND_WEIGHTS_HASH
        ^ NetworkArchitecture::get_hash_value()
        ^ LAYER_STACK_TABLE_HASH;

    /// A network whose parameters are all zero.
    pub fn zeroed() -> LargePageBox<CrazyhouseNetwork> {
        // SAFETY: all parameters are integers.
        unsafe { LargePageBox::zeroed() }
    }

    /// A zeroed network storing `stored` layer stacks (1..=144), the others
    /// falling back to stack 0.
    #[cfg(test)]
    pub fn zeroed_with_stacks(stored: usize) -> LargePageBox<CrazyhouseNetwork> {
        let mut network = CrazyhouseNetwork::zeroed();
        for s in 0..RAW_LAYER_STACKS {
            let f = if s < stored { s } else { 0 };
            network.layer_stack_fallback[s] = f as u16;
            network.layer_stack_index[s] = f as u16;
        }
        network.stored_layer_stacks = stored;
        network
    }

    /// Reads a Crazyhouse SFNNv20 network file and its description.
    pub fn read(bytes: &[u8]) -> Option<(LargePageBox<CrazyhouseNetwork>, Vec<u8>)> {
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
        if !read_hash(&mut reader, HAND_WEIGHTS_HASH) {
            return None;
        }
        for row in &mut network.hand_weights {
            reader.read_i16s(&mut row.0);
            permute(&mut row.0, &PACKUS_EPI16_ORDER);
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

        write_u32(&mut out, HAND_WEIGHTS_HASH);
        for row in &self.hand_weights {
            let mut unpermuted = row.0;
            permute(&mut unpermuted, &INVERSE_PACKUS_EPI16_ORDER);
            write_i16s(&mut out, &unpermuted);
        }

        for stack in &self.network[..self.stored_layer_stacks] {
            write_u32(&mut out, NetworkArchitecture::get_hash_value());
            stack.write_parameters(&mut out);
        }
        out
    }

    #[inline(always)]
    fn transform<'a>(
        &self,
        pos: &Position<Crazyhouse>,
        accumulators: &mut AccumulatorStack,
        caches: &mut AccumulatorCaches,
        output: &'a mut MaybeUninit<Aligned<[u8; L1]>>,
        nnz_info: &mut NnzInfo,
    ) -> &'a [u8; L1] {
        let extra = &pos.state().extra;
        if !extra.has_any(WHITE) && !extra.has_any(BLACK) {
            return self
                .feature_transformer
                .transform(pos, accumulators, caches, output, nnz_info);
        }
        let stm = pos.side_to_move();
        let stm_bias = accumulate_hand_bias(&self.hand_weights, extra, stm);
        let ntm_bias = accumulate_hand_bias(&self.hand_weights, extra, !stm);
        self.feature_transformer.transform_with_bias(
            pos,
            accumulators,
            caches,
            [&stm_bias.0, &ntm_bias.0],
            output,
            nnz_info,
        )
    }

    /// Evaluates `pos` (not in check) from the side to move's point of view.
    pub fn evaluate(
        &self,
        pos: &Position<Crazyhouse>,
        accumulators: &mut AccumulatorStack,
        caches: &mut AccumulatorCaches,
    ) -> Value {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::default();
        let bucket = self.layer_stack(pos);

        let transformed_features = self.transform(
            pos,
            accumulators,
            caches,
            &mut transformed_features,
            &mut nnz_info,
        );
        let positional = self.network[bucket].propagate(transformed_features, &nnz_info);
        positional / OUTPUT_SCALE
    }

    /// Evaluates every stored layer stack on `pos`, for `eval`.
    pub fn trace_evaluate(
        &self,
        pos: &Position<Crazyhouse>,
        accumulators: &mut AccumulatorStack,
        caches: &mut AccumulatorCaches,
    ) -> CrazyhouseEvalTrace {
        let mut transformed_features = MaybeUninit::<Aligned<[u8; L1]>>::uninit();
        let mut nnz_info = NnzInfo::default();
        let correct_bucket = self.layer_stack(pos);

        let transformed_features = self.transform(
            pos,
            accumulators,
            caches,
            &mut transformed_features,
            &mut nnz_info,
        );

        let layer_stacks = self.stored_layer_stacks;
        CrazyhouseEvalTrace {
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

    /// Raw layer stack: 4 board piece-count bands × 4 queen states (board or
    /// hand) × 9 pocket activity states.
    #[inline(always)]
    pub fn raw_layer_stack(pos: &Position<Crazyhouse>) -> usize {
        let stm = pos.side_to_move();
        let extra = &pos.state().extra;
        let band = ((pos.count_both(ALL_PIECES) - 1) / 8).clamp(0, 3) as usize;
        let our_queen = pos.pieces_cpt(stm, QUEEN).any() || extra.count(stm, QUEEN) > 0;
        let their_queen = pos.pieces_cpt(!stm, QUEEN).any() || extra.count(!stm, QUEEN) > 0;
        let queens = 2 * usize::from(our_queen) + usize::from(their_queen);
        let pockets = hand_band(extra.total(stm)) * 3 + hand_band(extra.total(!stm));
        (band * 4 + queens) * POCKET_STATES + pockets
    }

    #[inline(always)]
    fn layer_stack(&self, pos: &Position<Crazyhouse>) -> usize {
        usize::from(self.layer_stack_index[Self::raw_layer_stack(pos)])
    }
}

impl NetworkFile for CrazyhouseNetwork {
    const DEFAULT_NAME: &'static str = EVAL_FILE_DEFAULT_NAME;

    fn read(bytes: &[u8]) -> Option<(LargePageBox<CrazyhouseNetwork>, Vec<u8>)> {
        CrazyhouseNetwork::read(bytes)
    }

    fn write(&self, description: &[u8]) -> Vec<u8> {
        CrazyhouseNetwork::write(self, description)
    }

    fn kind(&self) -> &'static str {
        "Crazyhouse FullThreats+PP_3Wide+K32H3+Hands SFNNv20 "
    }

    fn download_url() -> String {
        format!(
            "https://github.com/aiorla/varifish/releases/download/networks/{}",
            Self::DEFAULT_NAME
        )
    }

    fn size_and_dimensions(&self) -> String {
        let size = (size_of::<FeatureTransformer>()
            + size_of::<[Aligned<[i16; L1]>; HAND_FEATURES]>()
            + self.stored_layer_stacks * size_of::<NetworkArchitecture>())
            / (1024 * 1024);
        let stacks = self.stored_layer_stacks;
        format!("({size}MiB, ({INPUT_DIMENSIONS}, {L1}, {L2}, {L3}, 1) x {stacks})")
    }
}

/// Accumulator caches for Crazyhouse (identical layout to chess's).
#[repr(transparent)]
pub struct CrazyhouseCaches(pub AccumulatorCaches);

impl crate::nnue::Caches for CrazyhouseCaches {
    type Network = CrazyhouseNetwork;

    fn new(network: Option<&CrazyhouseNetwork>) -> LargePageBox<CrazyhouseCaches> {
        // SAFETY: `CrazyhouseCaches` is a transparent wrapper around `AccumulatorCaches`.
        let mut caches = unsafe { LargePageBox::<CrazyhouseCaches>::zeroed() };
        if let Some(network) = network {
            caches.clear(network);
        }
        caches
    }

    fn clear(&mut self, network: &CrazyhouseNetwork) {
        self.0
            .clear_with_biases(&network.feature_transformer.biases.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::Caches;
    use crate::nnue::network::Network as ChessNetwork;

    #[test]
    fn write_and_read_round_trip_and_reject_chess_network() {
        let mut net = CrazyhouseNetwork::zeroed_with_stacks(4);
        net.hand_weights[0].0[0] = 31;
        net.hand_weights[HAND_FEATURES - 1].0[L1 - 1] = -29;
        let bytes = net.write(b"crazyhouse test");

        let (read, desc) = CrazyhouseNetwork::read(&bytes).unwrap();
        assert_eq!(desc, b"crazyhouse test");
        assert_eq!(read.stored_layer_stacks, 4);
        assert_eq!(read.hand_weights[0].0, net.hand_weights[0].0);
        assert_eq!(
            read.hand_weights[HAND_FEATURES - 1].0,
            net.hand_weights[HAND_FEATURES - 1].0
        );
        assert_eq!(read.write(&desc), bytes);

        assert!(CrazyhouseNetwork::read(&bytes[..bytes.len() - 1]).is_none());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(CrazyhouseNetwork::read(&extra).is_none());

        let chess_bytes = ChessNetwork::zeroed_with_stacks(1).write(b"chess");
        assert!(CrazyhouseNetwork::read(&chess_bytes).is_none());
    }

    #[test]
    fn raw_layer_stack_covers_all_pocket_and_queen_states() {
        for (fen, expected_queens, expected_pockets) in [
            ("4k3/8/8/8/8/8/8/4K3[] w - - 0 1", 0, 0),
            ("4k3/8/8/8/8/8/8/4K3[Q] w - - 0 1", 2, 3),
            ("4k3/8/8/8/8/8/8/4K3[q] w - - 0 1", 1, 1),
            ("4k3/8/8/8/8/8/8/4K3[QNPrbn] w - - 0 1", 2, 8),
            ("3qk3/8/8/8/8/8/8/4K3[QPPpp] w - - 0 1", 3, 7),
        ] {
            let pos = Position::<Crazyhouse>::from_fen(fen, false).unwrap();
            let raw = CrazyhouseNetwork::raw_layer_stack(&pos);
            assert!(raw < RAW_LAYER_STACKS);
            assert_eq!(raw % POCKET_STATES, expected_pockets, "{fen}");
            assert_eq!((raw / POCKET_STATES) % 4, expected_queens, "{fen}");
        }
    }

    #[test]
    fn incremental_eval_matches_from_scratch_across_drops_and_captures() {
        use crate::misc::Prng;
        use crate::movegen::{ALL, MoveList};
        use crate::nnue::eval_file::{EvalFile, Networks};

        let mut net = CrazyhouseNetwork::zeroed_with_stacks(1);
        for (i, b) in net.feature_transformer.biases.0.iter_mut().enumerate() {
            *b = ((i as i16 * 13 + 5) % 25) - 12;
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
            *w = ((i * 5 + 3) % 5) as i8 - 2;
        }
        for (h, row) in net.hand_weights.iter_mut().enumerate() {
            for (i, v) in row.0.iter_mut().enumerate() {
                *v = (((h as i32 * 19 + i as i32 * 3) % 127) - 63) as i16;
            }
        }
        let mut stack_bytes = Vec::new();
        net.network[0].write_parameters(&mut stack_bytes);
        for (i, byte) in stack_bytes.iter_mut().enumerate() {
            *byte = (((i * 13 + 7) % 5) as i8 - 2) as u8;
        }
        let mut reader = Reader::new(&stack_bytes);
        assert!(net.network[0].read_parameters(&mut reader) && reader.at_end());

        let bytes = net.write(b"non-zero crazyhouse");
        let (loaded, _) = CrazyhouseNetwork::read(&bytes).unwrap();
        let mut caches = CrazyhouseCaches::new(Some(&loaded));
        let mut accumulators = AccumulatorStack::new();

        let networks = Networks {
            chess: EvalFile::with_network(ChessNetwork::zeroed_with_stacks(1)),
            atomic: EvalFile::default(),
            threecheck: EvalFile::default(),
            crazyhouse: EvalFile::with_network(loaded),
            antichess: EvalFile::default(),
        };
        let eval_net = networks.crazyhouse.network().unwrap();

        // Adding a piece to a hand must change the evaluation.
        let p_empty = Position::<Crazyhouse>::from_fen(
            "r1b1k2r/ppp2ppp/2n2n2/2b1p3/2B1P3/5N2/PPP2PPP/R1B1K2R[] w KQkq - 4 4",
            false,
        )
        .unwrap();
        let p_hand = Position::<Crazyhouse>::from_fen(
            "r1b1k2r/ppp2ppp/2n2n2/2b1p3/2B1P3/5N2/PPP2PPP/R1B1K2R[QNp] w KQkq - 4 4",
            false,
        )
        .unwrap();
        let mut a1 = AccumulatorStack::new();
        let mut c1 = CrazyhouseCaches::new(Some(eval_net));
        let mut a2 = AccumulatorStack::new();
        let mut c2 = CrazyhouseCaches::new(Some(eval_net));
        let v_empty = eval_net.evaluate(&p_empty, &mut a1, &mut c1.0);
        let v_hand = eval_net.evaluate(&p_hand, &mut a2, &mut c2.0);
        assert_ne!(v_empty, v_hand);

        // Play a random Crazyhouse game with pockets, drops, captures, and
        // promotions; incremental evaluation must match from-scratch at every ply.
        let mut pos = Position::<Crazyhouse>::from_fen(
            "r1b1k2r/ppp2ppp/2n1p3/4P3/1b1P4/2NB4/PP3PPP/R1B1K2R[QNPrbn] w KQkq - 0 1",
            false,
        )
        .unwrap();
        let mut rng = Prng::new(0x5A68_0001);
        let mut drop_plies = 0usize;
        for _ in 0..80 {
            if !pos.in_check() {
                let inc = eval_net.evaluate(&pos, &mut accumulators, &mut caches.0);
                let mut fresh_acc = AccumulatorStack::new();
                let mut fresh_cache = CrazyhouseCaches::new(Some(eval_net));
                let fresh = eval_net.evaluate(&pos, &mut fresh_acc, &mut fresh_cache.0);
                assert_eq!(inc, fresh, "incremental != fresh at {}", pos.fen());
            }
            let moves = MoveList::new::<ALL>(&pos);
            if moves.is_empty() {
                break;
            }
            let mv = moves.as_slice()[(rng.rand64() as usize) % moves.len()];
            if mv.is_drop() {
                drop_plies += 1;
            }
            let gives_check = pos.gives_check(mv);
            let dirties = accumulators.push(&pos);
            pos.do_move(mv, gives_check, dirties);
        }
        assert!(drop_plies > 0, "expected at least one drop in random play");
    }
}
