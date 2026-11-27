//! The efficiently updatable neural networks that evaluate positions: chess's,
//! as Stockfish's, and Atomic's, as Atomic-Stockfish's.

pub mod accumulator;
pub mod antichess;
pub(crate) mod architecture;
pub mod atomic;
pub(crate) mod common;
pub mod crazyhouse;
pub mod eval_file;
pub(crate) mod feature_transformer;
pub mod features;
pub(crate) mod layers;
pub(crate) mod misc;
pub mod network;
pub(crate) mod nnz_helper;
pub(crate) mod simd;
pub mod threecheck;

use crate::memory::LargePageBox;
use crate::position::Position;
use crate::variant::Variant;

/// The accumulators of the positions along a search's line of variant V,
/// which a move pushes, recording its changes in the dirties `push` returns,
/// and its undoing pops.
pub trait Accumulators<V: Variant>: Sized + Send + 'static {
    type Dirties;

    /// A stack holding only the root.
    fn new() -> LargePageBox<Self>;

    /// Empties the stack to the root, whose accumulations are computed anew,
    /// for `network`, the network the search evaluates with, if loaded.
    fn reset(&mut self, network: Option<&V::Network>);

    /// Adds the state of the position after a move from `pos`.
    fn push(&mut self, pos: &Position<V>) -> &mut Self::Dirties;

    fn pop(&mut self);
}

/// A search thread's caches of the accumulations of the boards last
/// refreshed with each king square.
pub trait Caches: Sized + Send + 'static {
    type Network;

    /// The caches of `network`, or caches to clear once one is loaded.
    fn new(network: Option<&Self::Network>) -> LargePageBox<Self>;

    /// Empties the caches, for `network`.
    fn clear(&mut self, network: &Self::Network);
}
