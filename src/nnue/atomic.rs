//! Atomic's networks: varifish's own, in Stockfish's design, SFNNv20:
//! FullThreats, PP_3Wide and K32H3 input features, 1024 accumulators a side
//! and SFNNv20's layer stacks of 1024, 32, 32 and 1 neurons, as chess's
//! (`sfnnv20`). varifish 0.2.0 is the last release that also reads
//! Atomic-Stockfish's Legacy Atomic V1 networks.

pub mod sfnnv20;

use crate::memory::LargePageBox;
use crate::nnue::eval_file::NetworkFile;
use crate::position::Position;
use crate::types::AtomicDirties;
use crate::variant::Atomic;

/// The network varifish's Atomic evaluates with by default, its own, in
/// Stockfish's design, which is not embedded; named `atomic-` and the start
/// of its SHA-256, as Stockfish names its networks
pub const EVAL_FILE_DEFAULT_NAME: &str = "atomic-b0d279b9e021.nnue";

/// An Atomic network, of the one format varifish reads.
pub enum AtomicNetwork {
    /// varifish's network in Stockfish's design, SFNNv20
    Sfnnv20(LargePageBox<sfnnv20::Network>),
}

impl AtomicNetwork {
    /// The network, in the box that EvalFile keeps networks in.
    pub fn boxed(self) -> LargePageBox<AtomicNetwork> {
        LargePageBox::new(self)
    }
}

impl NetworkFile for AtomicNetwork {
    const DEFAULT_NAME: &'static str = EVAL_FILE_DEFAULT_NAME;

    fn read(bytes: &[u8]) -> Option<(LargePageBox<AtomicNetwork>, Vec<u8>)> {
        let (network, description) = sfnnv20::Network::read(bytes)?;
        Some((AtomicNetwork::Sfnnv20(network).boxed(), description))
    }

    fn write(&self, description: &[u8]) -> Vec<u8> {
        match self {
            AtomicNetwork::Sfnnv20(network) => network.write(description),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            AtomicNetwork::Sfnnv20(_) => "Atomic FullThreats+PP_3Wide+K32H3 SFNNv20 ",
        }
    }

    fn download_url() -> String {
        format!(
            "https://github.com/aiorla/varifish/releases/download/networks/{}",
            Self::DEFAULT_NAME
        )
    }

    fn size_and_dimensions(&self) -> String {
        match self {
            AtomicNetwork::Sfnnv20(network) => network.size_and_dimensions(),
        }
    }
}

/// The accumulators along a search's line.
pub struct AtomicAccumulators {
    pub sfnnv20: sfnnv20::accumulator::AccumulatorStack,
}

impl crate::nnue::Accumulators<Atomic> for AtomicAccumulators {
    type Dirties = AtomicDirties;

    fn new() -> LargePageBox<Self> {
        // SAFETY: zero bytes make the stack, which reset() then empties.
        let mut accumulators = unsafe { LargePageBox::<AtomicAccumulators>::zeroed() };
        accumulators.sfnnv20.reset();
        accumulators
    }

    /// Empties the stack to the root.
    #[inline(always)]
    fn reset(&mut self, _network: Option<&AtomicNetwork>) {
        self.sfnnv20.reset();
    }

    #[inline(always)]
    fn push(&mut self, pos: &Position<Atomic>) -> &mut AtomicDirties {
        self.sfnnv20.push(pos)
    }

    #[inline(always)]
    fn pop(&mut self) {
        self.sfnnv20.pop();
    }
}

/// A search thread's caches of the accumulators.
pub struct AtomicCaches {
    pub sfnnv20: sfnnv20::accumulator::AccumulatorCaches,
}

impl crate::nnue::Caches for AtomicCaches {
    type Network = AtomicNetwork;

    fn new(network: Option<&AtomicNetwork>) -> LargePageBox<Self> {
        // SAFETY: zero bytes make entries, which a network then clears.
        let mut caches = unsafe { LargePageBox::<AtomicCaches>::zeroed() };
        if let Some(network) = network {
            caches.clear(network);
        }
        caches
    }

    /// Empties the caches, for `network`.
    fn clear(&mut self, network: &AtomicNetwork) {
        match network {
            AtomicNetwork::Sfnnv20(network) => self.sfnnv20.clear(network),
        }
    }
}
