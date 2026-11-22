//! Atomic's networks, in either of two formats, which the file's header
//! tells apart:
//!
//! - varifish's own networks, in Stockfish's design, SFNNv20: FullThreats,
//!   PP_3Wide and K32H3 input features, 1024 accumulators a side and
//!   SFNNv20's layer stacks of 1024, 32, 32 and 1 neurons, as chess's
//!   (`sfnnv20`);
//! - the format that Atomic-Stockfish calls Legacy Atomic V1, in which
//!   Fairy-Stockfish trained its Atomic networks and which run3b has:
//!   HalfKAv2Atomic input features, 512 accumulators and 8 PSQT values per
//!   perspective, then 8 layer stacks of 1024, 16, 32 and 1 neurons
//!   (`accumulator`, `features` and `network`), which the checks against
//!   Atomic-Stockfish evaluate with.
//!
//! The legacy format is derived, with changes for varifish in 2026, from
//! Atomic-Stockfish's `nnue/` (https://github.com/Belzedar94/Atomic-Stockfish),
//! a variant of Stockfish, a UCI chess playing engine derived from Glaurung
//! 2.1, Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under
//! the GNU General Public License, version 3 or later. It evaluates exactly as
//! Atomic-Stockfish does, with two changes for speed: the accumulators of a
//! perspective whose king moved start from the sums cached for its square
//! (Stockfish's Finny tables) instead of summing every feature anew, and the
//! first layer reads only the non-zero chunks of its input, as Stockfish's
//! does.
//!
//! A search keeps the accumulators and caches of both formats, and moves
//! update those of the network it searches with, which `reset` chooses.

pub mod accumulator;
pub mod features;
pub mod network;
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

/// An Atomic network, in either format.
pub enum AtomicNetwork {
    /// Legacy Atomic V1, as run3b
    Legacy(LargePageBox<network::Network>),
    /// varifish's network in Stockfish's design, SFNNv20
    Sfnnv20(LargePageBox<sfnnv20::Network>),
}

impl AtomicNetwork {
    /// The network, in the box that EvalFile keeps networks in.
    pub fn boxed(self) -> LargePageBox<AtomicNetwork> {
        LargePageBox::new(self)
    }

    /// The format of the network.
    pub fn kind(&self) -> NetworkKind {
        match self {
            AtomicNetwork::Legacy(_) => NetworkKind::Legacy,
            AtomicNetwork::Sfnnv20(_) => NetworkKind::Sfnnv20,
        }
    }
}

impl NetworkFile for AtomicNetwork {
    const DEFAULT_NAME: &'static str = EVAL_FILE_DEFAULT_NAME;

    fn read(bytes: &[u8]) -> Option<(LargePageBox<AtomicNetwork>, Vec<u8>)> {
        if let Some((network, description)) = sfnnv20::Network::read(bytes) {
            Some((AtomicNetwork::Sfnnv20(network).boxed(), description))
        } else {
            let (network, description) = network::Network::read(bytes)?;
            Some((AtomicNetwork::Legacy(network).boxed(), description))
        }
    }

    fn write(&self, description: &[u8]) -> Vec<u8> {
        match self {
            AtomicNetwork::Legacy(network) => network.write(description),
            AtomicNetwork::Sfnnv20(network) => network.write(description),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            AtomicNetwork::Legacy(_) => "Legacy Atomic V1 ",
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
            AtomicNetwork::Legacy(_) => network::Network::size_and_dimensions(),
            AtomicNetwork::Sfnnv20(network) => network.size_and_dimensions(),
        }
    }
}

/// The format of an Atomic network
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum NetworkKind {
    #[default]
    Legacy,
    Sfnnv20,
}

/// The accumulators of both formats along a search's line, of which the
/// moves update those of `kind`.
pub struct AtomicAccumulators {
    kind: NetworkKind,
    pub legacy: accumulator::AccumulatorStack,
    pub sfnnv20: sfnnv20::accumulator::AccumulatorStack,
}

impl AtomicAccumulators {
    /// The format whose accumulators the moves update
    #[inline(always)]
    pub fn kind(&self) -> NetworkKind {
        self.kind
    }
}

impl crate::nnue::Accumulators<Atomic> for AtomicAccumulators {
    type Dirties = AtomicDirties;

    fn new() -> LargePageBox<Self> {
        // SAFETY: zero bytes make the kind and the stacks, which reset()
        // then empties.
        let mut accumulators = unsafe { LargePageBox::<AtomicAccumulators>::zeroed() };
        accumulators.legacy.reset();
        accumulators.sfnnv20.reset();
        accumulators
    }

    /// Empties the stacks to the root, and has the moves update the
    /// accumulators of `network`'s format from now on.
    #[inline(always)]
    fn reset(&mut self, network: Option<&AtomicNetwork>) {
        if let Some(network) = network {
            self.kind = network.kind();
        }
        self.legacy.reset();
        self.sfnnv20.reset();
    }

    #[inline(always)]
    fn push(&mut self, pos: &Position<Atomic>) -> &mut AtomicDirties {
        match self.kind {
            NetworkKind::Legacy => self.legacy.push(),
            NetworkKind::Sfnnv20 => self.sfnnv20.push(pos),
        }
    }

    #[inline(always)]
    fn pop(&mut self) {
        match self.kind {
            NetworkKind::Legacy => self.legacy.pop(),
            NetworkKind::Sfnnv20 => self.sfnnv20.pop(),
        }
    }
}

/// A search thread's caches of the accumulators of both formats.
pub struct AtomicCaches {
    pub legacy: accumulator::AccumulatorCaches,
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

    /// Empties the caches of `network`'s format, for it.
    fn clear(&mut self, network: &AtomicNetwork) {
        match network {
            AtomicNetwork::Legacy(network) => self.legacy.clear(network),
            AtomicNetwork::Sfnnv20(network) => self.sfnnv20.clear(network),
        }
    }
}
