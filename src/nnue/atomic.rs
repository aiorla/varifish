//! The Atomic network format that Atomic-Stockfish calls Legacy Atomic V1,
//! in which Fairy-Stockfish trained its Atomic networks and which run3b
//! has: HalfKAv2Atomic input features, 512 accumulators and 8 PSQT values
//! per perspective, then 8 layer stacks of 1024, 16, 32 and 1 neurons.
//!
//! Derived, with changes for varifish in 2026, from Atomic-Stockfish's
//! `nnue/` (https://github.com/Belzedar94/Atomic-Stockfish), a variant of
//! Stockfish, a UCI chess playing engine derived from Glaurung 2.1,
//! Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS), under the
//! GNU General Public License, version 3 or later. It evaluates exactly as
//! Atomic-Stockfish does, with two changes for speed: the accumulators of a
//! perspective whose king moved start from the sums cached for its square
//! (Stockfish's Finny tables) instead of summing every feature anew, and the
//! first layer reads only the non-zero chunks of its input, as Stockfish's
//! does.

pub mod accumulator;
pub mod features;
pub mod network;
