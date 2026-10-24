//! The network's types and constants, and the reading and writing of its
//! file.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `nnue/nnue_common.h`: Stockfish, a UCI chess playing engine derived from
//! Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! AUTHORS), under the GNU General Public License, version 3 or later.

/// Version of the evaluation file
pub const VERSION: u32 = 0x6A44_8AFA;

// Constants used in evaluation value calculation
pub const OUTPUT_SCALE: i32 = 16;
pub const WEIGHT_SCALE_BITS: u32 = 6;
pub const FT_MAX_VAL: i16 = 255;
pub const HIDDEN_ONE_VAL: i32 = 128;

pub const MAX_SIMD_WIDTH: usize = 32;

/// Round n up to be a multiple of base
pub const fn ceil_to_multiple(n: usize, base: usize) -> usize {
    n.div_ceil(base) * base
}

/// A value aligned to a cache line.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Aligned<T>(pub T);

/// A network file being read, as Stockfish reads its stream: reading past
/// the end fails, and a failed reader reads nothing more, returning zeros.
pub struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    failed: bool,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader {
            bytes,
            pos: 0,
            failed: false,
        }
    }

    /// Whether every read so far succeeded.
    pub fn good(&self) -> bool {
        !self.failed
    }

    /// Whether every byte was read.
    pub fn at_end(&self) -> bool {
        self.pos == self.bytes.len()
    }

    /// The next `n` bytes, or None after failing when fewer remain.
    pub fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.failed || self.bytes.len() - self.pos < n {
            self.failed = true;
            return None;
        }
        let bytes = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Some(bytes)
    }

    pub fn read_u32(&mut self) -> u32 {
        self.take(4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    /// Reads little-endian integers in bulk.
    pub fn read_i8s(&mut self, out: &mut [i8]) {
        if let Some(bytes) = self.take(out.len()) {
            for (o, &b) in out.iter_mut().zip(bytes) {
                *o = b as i8;
            }
        }
    }

    pub fn read_i16s(&mut self, out: &mut [i16]) {
        if let Some(bytes) = self.take(2 * out.len()) {
            for (o, &b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = i16::from_le_bytes(b);
            }
        }
    }

    pub fn read_i32s(&mut self, out: &mut [i32]) {
        if let Some(bytes) = self.take(4 * out.len()) {
            for (o, &b) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *o = i32::from_le_bytes(b);
            }
        }
    }
}

pub fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Writes little-endian integers in bulk.
pub fn write_i8s(out: &mut Vec<u8>, values: &[i8]) {
    out.extend(values.iter().map(|&v| v as u8));
}

pub fn write_i16s(out: &mut Vec<u8>, values: &[i16]) {
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

pub fn write_i32s(out: &mut Vec<u8>, values: &[i32]) {
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_reader_reads_zeros() {
        let mut reader = Reader::new(&[1, 0, 0, 0, 2]);
        assert_eq!(reader.read_u32(), 1);
        assert_eq!(reader.read_u32(), 0);
        assert!(!reader.good());
        let mut out = [7];
        reader.read_i8s(&mut out);
        assert_eq!(out, [7]);
    }
}
