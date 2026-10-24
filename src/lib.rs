//! varifish: a UCI engine for chess and its variants, translated from
//! Stockfish.

pub mod attacks;
pub mod benchmark;
pub mod bitboard;
pub mod engine;
pub mod evaluate;
pub mod history;
pub mod memory;
pub mod misc;
pub mod movegen;
pub mod movepick;
pub mod nnue;
pub mod options;
pub mod perft;
pub mod position;
pub mod score;
pub mod search;
pub mod syzygy;
pub mod thread;
pub mod timeman;
pub mod tt;
pub mod tune;
pub mod types;
pub mod uci;
pub mod variant;
