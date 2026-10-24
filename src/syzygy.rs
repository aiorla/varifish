//! Syzygy endgame tablebases: the WDL tables that the search probes and the
//! DTZ tables that rank the moves at the root. Each variant reads its own
//! tables: chess the .rtbw and .rtbz files, and Atomic the .atbw and .atbz
//! files, in which the kings may stand next to each other and a capture may
//! explode a king.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `syzygy/tbprobe.h` and `.cpp`, with the Atomic changes of
//! Atomic-Stockfish's (https://github.com/Belzedar94/Atomic-Stockfish):
//! Stockfish, a UCI chess playing engine derived from Glaurung 2.1, Copyright
//! (C) 2004-2026 The Stockfish developers (see AUTHORS), under the GNU
//! General Public License, version 3 or later.
//!
//! Atomic-Stockfish took this code from an older Stockfish. varifish keeps
//! Stockfish's later checks of the files, which reject a corrupt table when
//! it is mapped rather than read past its end, and stops on a table whose
//! magic is wrong, as Stockfish does, where Atomic-Stockfish takes it for a
//! missing one.

use std::cmp::Ordering;
use std::fs::File;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::bitboard::{Bitboard, edge_distance};
use crate::movegen::{ALL, MoveList};
use crate::position::{Position, material_key_of};
use crate::search::RootMove;
use crate::types::*;
use crate::variant::{Atomic, Chess, Variant, VariantKind};

/// How a variant's tables are stored and indexed.
pub struct SyzygyFormat {
    /// The suffixes of the WDL and the DTZ files
    pub suffixes: [&'static str; 2],
    /// The first bytes of the WDL and the DTZ files
    pub magics: [[u8; 4]; 2],
    /// The most pieces of a table
    pub max_pieces: usize,
    /// Whether the kings may stand next to each other, which places the
    /// kings of a table without other unique pieces in 518 ways, rather
    /// than chess's 462.
    pub connected_kings: bool,
}

/// Chess's tables, of up to 7 pieces
pub const CHESS: SyzygyFormat = SyzygyFormat {
    suffixes: [".rtbw", ".rtbz"],
    magics: [[0x71, 0xE8, 0x23, 0x5D], [0xD7, 0x66, 0x0C, 0xA5]],
    max_pieces: 7,
    connected_kings: false,
};

/// Atomic's tables, of up to 6 pieces, which are all that were published:
/// a 7-piece Atomic format would place the kings differently.
pub const ATOMIC: SyzygyFormat = SyzygyFormat {
    suffixes: [".atbw", ".atbz"],
    magics: [[0x55, 0x8D, 0xA4, 0x49], [0x91, 0xA9, 0x5E, 0xEB]],
    max_pieces: 6,
    connected_kings: true,
};

/// The most pieces of the tables of any variant
const TBPIECES: usize = 7;
/// Max DTZ supported times 2, large enough to deal with the syzygy TB limit.
const MAX_DTZ: i32 = 1 << 18;

/// A WDL score, from the point of view of the side to move
pub type WdlScore = i32;
/// Loss
pub const WDL_LOSS: WdlScore = -2;
/// Loss, but draw under 50-move rule
pub const WDL_BLESSED_LOSS: WdlScore = -1;
/// Draw
pub const WDL_DRAW: WdlScore = 0;
/// Win, but draw under 50-move rule
pub const WDL_CURSED_WIN: WdlScore = 1;
/// Win
pub const WDL_WIN: WdlScore = 2;

/// Possible states after a probing operation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeState {
    /// Probe failed (missing file table)
    Fail = 0,
    /// Probe successful
    Ok = 1,
    /// DTZ should check the other side
    ChangeStm = -1,
    /// Best move zeroes DTZ (capture or pawn move)
    ZeroingBestMove = 2,
}

/// How the tablebases are probed in a search.
#[derive(Clone, Copy, Debug, Default)]
pub struct TbConfig {
    pub cardinality: i32,
    pub root_in_tb: bool,
    pub use_rule50: bool,
    pub probe_depth: Depth,
}

/// The options that steer probing: SyzygyProbeDepth, Syzygy50MoveRule and
/// SyzygyProbeLimit.
#[derive(Clone, Copy, Debug)]
pub struct SyzygyOptions {
    pub probe_depth: Depth,
    pub rule50: bool,
    pub probe_limit: i32,
}

impl Default for SyzygyOptions {
    fn default() -> Self {
        SyzygyOptions {
            probe_depth: 1,
            rule50: true,
            probe_limit: 7,
        }
    }
}

// Each table has a set of flags: all of them refer to DTZ tables, the last one
// to WDL tables
const STM: u8 = 1;
const MAPPED: u8 = 2;
const WIN_PLIES: u8 = 4;
const LOSS_PLIES: u8 = 8;
const WIDE: u8 = 16;
const SINGLE_VALUE: u8 = 128;

const WDL: usize = 0;
const DTZ: usize = 1;

const PIECE_TO_CHAR: &[u8; 16] = b" PNBRQK  pnbrqk ";

const WDL_TO_VALUE: [Value; 5] = [
    -VALUE_MATE + MAX_PLY + 1,
    VALUE_DRAW - 2,
    VALUE_DRAW,
    VALUE_DRAW + 2,
    VALUE_MATE - MAX_PLY - 1,
];

/// Huffman symbols are 12 bits long.
const SYM_COUNT: usize = 4096;

/// How far above the a1-h8 diagonal `s` stands, in ranks: negative below it.
const fn off_a1h8(s: usize) -> i32 {
    (s >> 3) as i32 - (s & 7) as i32
}

/// The tables that map squares and groups of pieces to indices.
struct Maps {
    pawns: [i32; SQUARE_NB],
    b1h1h7: [i32; SQUARE_NB],
    a1d1d4: [i32; SQUARE_NB],
    // [MapA1D1D4][SQUARE_NB]
    kk: [[i32; SQUARE_NB]; 10],
    // [k][n] k elements from a set of n elements
    binomial: [[i32; SQUARE_NB]; 6],
    // [leadPawnsCnt][SQUARE_NB]
    lead_pawn_idx: [[i32; SQUARE_NB]; 6],
    // [leadPawnsCnt][FILE_A..FILE_D]
    lead_pawns_size: [[i32; 4]; 6],
}

static MAPS: Maps = Maps::new();

impl Maps {
    const fn new() -> Maps {
        let mut m = Maps {
            pawns: [0; SQUARE_NB],
            b1h1h7: [0; SQUARE_NB],
            a1d1d4: [0; SQUARE_NB],
            kk: [[0; SQUARE_NB]; 10],
            binomial: [[0; SQUARE_NB]; 6],
            lead_pawn_idx: [[0; SQUARE_NB]; 6],
            lead_pawns_size: [[0; 4]; 6],
        };

        // MapB1H1H7[] encodes a square below a1-h8 diagonal to 0..27
        let mut code = 0;
        let mut s = 0;
        while s < SQUARE_NB {
            if off_a1h8(s) < 0 {
                m.b1h1h7[s] = code;
                code += 1;
            }
            s += 1;
        }

        // MapA1D1D4[] encodes a square in the a1-d1-d4 triangle to 0..9
        let mut diagonal = [0; 4];
        let mut diagonal_len = 0;
        code = 0;
        s = 0;
        while s <= Square::D4.0 as usize {
            if off_a1h8(s) < 0 && s & 7 <= FILE_D as usize {
                m.a1d1d4[s] = code;
                code += 1;
            } else if off_a1h8(s) == 0 && s & 7 <= FILE_D as usize {
                diagonal[diagonal_len] = s;
                diagonal_len += 1;
            }
            s += 1;
        }

        // Diagonal squares are encoded as last ones
        let mut i = 0;
        while i < diagonal_len {
            m.a1d1d4[diagonal[i]] = code;
            code += 1;
            i += 1;
        }

        // MapKK[] encodes all the 462 possible legal positions of two kings
        // where the first is in the a1-d1-d4 triangle. If the first king is
        // on the a1-d4 diagonal, the other one shall not be above the a1-h8
        // diagonal.
        let mut both_on_diagonal = [(0, 0); 64];
        let mut both_len = 0;
        code = 0;
        let mut idx = 0;
        while idx < 10 {
            let mut s1 = 0;
            while s1 <= Square::D4.0 as usize {
                // SQ_B1 is mapped to 0
                if m.a1d1d4[s1] == idx as i32 && (idx != 0 || s1 == Square::B1.0 as usize) {
                    let mut s2 = 0;
                    while s2 < SQUARE_NB {
                        let near =
                            (s1 & 7).abs_diff(s2 & 7) <= 1 && (s1 >> 3).abs_diff(s2 >> 3) <= 1;
                        if near {
                            // Illegal position
                        } else if off_a1h8(s1) == 0 && off_a1h8(s2) > 0 {
                            // First on diagonal, second above
                        } else if off_a1h8(s1) == 0 && off_a1h8(s2) == 0 {
                            both_on_diagonal[both_len] = (idx, s2);
                            both_len += 1;
                        } else {
                            m.kk[idx][s2] = code;
                            code += 1;
                        }
                        s2 += 1;
                    }
                }
                s1 += 1;
            }
            idx += 1;
        }

        // Legal positions with both kings on a diagonal are encoded as last
        // ones
        i = 0;
        while i < both_len {
            let (idx, s2) = both_on_diagonal[i];
            m.kk[idx][s2] = code;
            code += 1;
            i += 1;
        }

        // Binomial[] stores the Binomial Coefficients using Pascal rule.
        // There are Binomial[k][n] ways to choose k elements from a set of n
        // elements.
        m.binomial[0][0] = 1;

        let mut n = 1;
        while n < 64 {
            let mut k = 0;
            while k < 6 && k <= n {
                m.binomial[k][n] = (if k > 0 { m.binomial[k - 1][n - 1] } else { 0 })
                    + (if k < n { m.binomial[k][n - 1] } else { 0 });
                k += 1;
            }
            n += 1;
        }

        // MapPawns[s] encodes squares a2-h7 to 0..47. This is the number of
        // possible available squares when the leading one is in 's'.
        // Moreover the pawn with highest MapPawns[] is the leading pawn, the
        // one nearest the edge, and among pawns with the same file, the one
        // with the lowest rank.
        let mut available_squares = 47; // Available squares when lead pawn is in a2

        // Init the tables for the encoding of leading pawns group: with 7-men
        // TB we can have up to 5 leading pawns (KPPPPPK).
        let mut lead_pawns_cnt = 1;
        while lead_pawns_cnt <= 5 {
            let mut f = FILE_A as usize;
            while f <= FILE_D as usize {
                // Restart the index at every file because TB table is split
                // by file, so we can reuse the same index for different
                // files.
                let mut idx = 0;

                // Sum all possible combinations for a given file, starting
                // with the leading pawn on rank 2 and increasing the rank.
                let mut r = RANK_2 as usize;
                while r <= RANK_7 as usize {
                    let sq = r * 8 + f;

                    // Compute MapPawns[] at first pass. If sq is the leading
                    // pawn square, any other pawn cannot be below or more
                    // toward the edge of sq. There are 47 available squares
                    // when sq = a2 and reduced by 2 for any rank increase due
                    // to mirroring: sq == a3 -> no a2, h2, so MapPawns[a3] =
                    // 45
                    if lead_pawns_cnt == 1 {
                        m.pawns[sq] = available_squares;
                        available_squares -= 1;
                        m.pawns[sq ^ 7] = available_squares;
                        available_squares -= 1;
                    }
                    m.lead_pawn_idx[lead_pawns_cnt][sq] = idx;
                    idx += m.binomial[lead_pawns_cnt - 1][m.pawns[sq] as usize];
                    r += 1;
                }
                // After a file is traversed, store the cumulated per-file
                // index
                m.lead_pawns_size[lead_pawns_cnt][f] = idx;
                f += 1;
            }
            lead_pawns_cnt += 1;
        }

        m
    }
}

// DTZ tables don't store valid scores for moves that reset the rule50 counter
// like captures and pawn moves but we can easily recover the correct dtz of
// the previous move if we know the position's WDL score.
fn dtz_before_zeroing(wdl: WdlScore) -> i32 {
    match wdl {
        WDL_WIN => 1,
        WDL_CURSED_WIN => 101,
        WDL_BLESSED_LOSS => -101,
        WDL_LOSS => -1,
        _ => 0,
    }
}

/// Return the sign of a number (-1, 0, 1)
fn sign_of(val: i32) -> i32 {
    val.signum()
}

fn u16_le(bytes: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([bytes[i], bytes[i + 1]])
}

fn u32_le(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
}

fn u32_be(bytes: &[u8], i: usize) -> u32 {
    u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
}

fn u64_be(bytes: &[u8], i: usize) -> u64 {
    let mut b = [0; 8];
    b.copy_from_slice(&bytes[i..i + 8]);
    u64::from_be_bytes(b)
}

/// Whether `count` items of `stride` bytes from offset `p` end by `end`
fn fits(p: usize, count: u64, stride: u64, end: usize) -> bool {
    if p > end {
        return false;
    }

    let room = (end - p) as u64;
    count == 0 || stride <= room / count
}

/// Leaves the process, as Stockfish does on a file it cannot use.
fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// A table file mapped into memory, which it reads at first access only.
struct Mapping {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
    #[cfg(not(unix))]
    _buffer: Vec<Chunk>,
}

// SAFETY: the mapping is read-only and owned by its table.
unsafe impl Send for Mapping {}
// SAFETY: the mapping is only read.
unsafe impl Sync for Mapping {}

#[cfg(not(unix))]
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Chunk([u8; 64]);

#[cfg(unix)]
mod mmap {
    use std::ffi::{c_int, c_long, c_void};

    pub const PROT_READ: c_int = 1;
    pub const MAP_SHARED: c_int = 1;
    pub const MADV_RANDOM: c_int = 1;
    pub const MAP_FAILED: *mut c_void = !0 as *mut c_void;

    unsafe extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: c_long,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, length: usize) -> c_int;
        pub fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
    }
}

impl Mapping {
    /// Maps the file found at `path`, checking its size and its magic.
    /// None if it cannot be read; a corrupt file stops the engine.
    fn new(file: File, path: &Path, magic: [u8; 4]) -> Option<Mapping> {
        let size = file.metadata().ok()?.len();

        if size % 64 != 16 {
            fail(&format!("Corrupt tablebase file {}", path.display()));
        }

        let len = usize::try_from(size)
            .unwrap_or_else(|_| fail(&format!("Could not mmap() {}", path.display())));
        let mapping = Mapping::map(&file, len, path);

        if mapping.bytes()[..4] != magic {
            fail(&format!("Corrupted table in file {}", path.display()));
        }

        Some(mapping)
    }

    #[cfg(unix)]
    fn map(file: &File, len: usize, path: &Path) -> Mapping {
        use std::os::unix::io::AsRawFd;

        // SAFETY: a new read-only mapping of the whole file, whose size is
        // not zero. Like Stockfish, varifish expects table files not to
        // change while they are mapped.
        let ptr = unsafe {
            mmap::mmap(
                std::ptr::null_mut(),
                len,
                mmap::PROT_READ,
                mmap::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == mmap::MAP_FAILED {
            fail(&format!("Could not mmap() {}", path.display()));
        }

        // SAFETY: the range was just mapped. Failure only loses the hint.
        unsafe {
            mmap::madvise(ptr, len, mmap::MADV_RANDOM);
        }

        Mapping {
            ptr: std::ptr::NonNull::new(ptr.cast()).expect("mmap() maps no null page"),
            len,
        }
    }

    /// Elsewhere than on Unix, the file is read into memory aligned as a
    /// mapping would be.
    #[cfg(not(unix))]
    fn map(file: &File, len: usize, path: &Path) -> Mapping {
        use std::io::Read;

        let mut buffer = vec![Chunk([0; 64]); len.div_ceil(64)];
        // SAFETY: the chunks are plain bytes, at least len of them.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<u8>(), len) };
        let mut file = file;
        if file.read_exact(bytes).is_err() {
            fail(&format!("Could not mmap() {}", path.display()));
        }

        Mapping {
            ptr: std::ptr::NonNull::new(buffer.as_mut_ptr().cast()).expect("a vector is not null"),
            len,
            _buffer: buffer,
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping holds len readable bytes as long as it lives.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

#[cfg(unix)]
impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the mapping is no longer read, as its table goes.
        unsafe {
            mmap::munmap(self.ptr.as_ptr().cast(), self.len);
        }
    }
}

// TB tables are compressed with canonical Huffman code. The compressed data is
// divided into blocks of size d->sizeofBlock, and each block stores a variable
// number of symbols. Each symbol represents either a WDL or a (remapped) DTZ
// value, or a pair of other symbols (recursively). If you keep expanding the
// symbols in a block, you end up with up to 65536 WDL or DTZ values. Each
// symbol represents up to 256 values and will correspond after Huffman coding
// to at least 1 bit. So a block of 32 bytes corresponds to at most 32 x 8 x 256
// = 65536 values. This maximum is only reached for tables that consist mostly
// of draws or mostly of wins, but such tables are actually quite common. In
// principle, the blocks in WDL tables are 64 bytes long (and will be aligned
// on cache lines). But for mostly-draw or mostly-win tables this can leave many
// 64-byte blocks only half-filled, so in such cases blocks are 32 bytes long.
// The blocks of DTZ tables are up to 1024 bytes long. The generator picks the
// size that leads to the smallest table. The "book" of symbols and Huffman
// codes are the same for all blocks in the table. A non-symmetric pawnless TB
// file will have one table for wtm and one for btm, a TB file with pawns will
// have tables per file a,b,c,d also, in this case, one set for wtm and one for
// btm.

/// Low-level indexing information to access TB data. There are 8, 4, or 2
/// PairsData records for each table, according to the type of table and if
/// positions have pawns or not. It is populated at first access. Offsets
/// are into the table's file.
#[derive(Clone)]
struct PairsData {
    /// Table flags, see the flags above
    flags: u8,
    /// Maximum length in bits of the Huffman symbols
    max_sym_len: u8,
    /// Minimum length in bits of the Huffman symbols
    min_sym_len: u8,
    /// Number of blocks in the TB file
    blocks_num: u32,
    /// Block size in bytes
    sizeof_block: usize,
    /// About every span values there is a SparseIndex[] entry
    span: usize,
    /// lowestSym[l] is the symbol of length l with the lowest value
    lowest_sym: usize,
    /// The left and right symbols that expand each symbol: the 12 bits of
    /// each half of its 3 bytes. A symbol whose right half is 0xFFF has
    /// length 1, and its left half is the stored value. Symbols past the
    /// file's are such leaves of value 0, as Stockfish pads them.
    btree: Vec<[u16; 2]>,
    /// Number of stored positions (minus one) for each block: 1..65536
    block_length: usize,
    /// Size of blockLength[] table: padded so it's bigger than blocksNum
    block_length_size: u32,
    /// Partial indices into blockLength[], of 6 bytes each: a block number
    /// and an offset within it
    sparse_index: usize,
    /// Size of SparseIndex[] table
    sparse_index_size: usize,
    /// Start of Huffman compressed data
    data: usize,
    data_end: usize,
    /// base64[l - min_sym_len] is the 64bit-padded lowest symbol of length l
    base64: Vec<u64>,
    /// Number of values (-1) represented by a given Huffman symbol: 1..256
    symlen: Vec<u8>,
    /// Position pieces: the order of pieces defines the groups
    pieces: [Piece; TBPIECES],
    /// Start index used for the encoding of the group's pieces
    group_idx: [u64; TBPIECES + 1],
    /// Number of pieces in a given group: KRKN -> (3, 1)
    group_len: [usize; TBPIECES + 1],
    /// WDLWin, WDLLoss, WDLCursedWin, WDLBlessedLoss (used in DTZ)
    map_idx: [u16; 4],
}

impl Default for PairsData {
    fn default() -> Self {
        PairsData {
            flags: 0,
            max_sym_len: 0,
            min_sym_len: 0,
            blocks_num: 0,
            sizeof_block: 0,
            span: 0,
            lowest_sym: 0,
            btree: Vec::new(),
            block_length: 0,
            block_length_size: 0,
            sparse_index: 0,
            sparse_index_size: 0,
            data: 0,
            data_end: 0,
            base64: Vec::new(),
            symlen: Vec::new(),
            pieces: [NO_PIECE; TBPIECES],
            group_idx: [0; TBPIECES + 1],
            group_len: [0; TBPIECES + 1],
            map_idx: [0; 4],
        }
    }
}

impl PairsData {
    #[inline]
    fn left(&self, s: usize) -> usize {
        self.btree.get(s).map_or(0, |lr| lr[0] as usize)
    }

    #[inline]
    fn right(&self, s: usize) -> usize {
        self.btree.get(s).map_or(0xFFF, |lr| lr[1] as usize)
    }

    #[inline]
    fn symlen(&self, s: usize) -> i32 {
        self.symlen.get(s).map_or(0, |&l| i32::from(l))
    }
}

/// A table file, mapped with its indexing information.
struct MappedTable {
    mapping: Mapping,
    /// [wtm / btm][FILE_A..FILE_D or 0]
    items: [[PairsData; 4]; 2],
    /// The DTZ map, in DTZ tables
    map: usize,
}

impl MappedTable {
    fn get(&self, sides: usize, has_pawns: bool, stm: usize, f: usize) -> &PairsData {
        &self.items[stm % sides][if has_pawns { f } else { 0 }]
    }

    fn get_mut(&mut self, sides: usize, has_pawns: bool, stm: usize, f: usize) -> &mut PairsData {
        &mut self.items[stm % sides][if has_pawns { f } else { 0 }]
    }
}

/// The indexing information of a WDL file and its DTZ file, populated at
/// init time; their files are mapped at first access.
struct TbTable {
    key: Key,
    key2: Key,
    piece_count: usize,
    has_pawns: bool,
    has_unique_pieces: bool,
    /// [Lead color / other color]
    pawn_count: [u8; 2],
    /// The WDL and the DTZ file, None if missing
    files: [OnceLock<Option<MappedTable>>; 2],
}

impl TbTable {
    /// The table of `code`, such as "KRvK".
    fn new(code: &str) -> TbTable {
        // The strong side, before the 'v', is White for key and Black for
        // key2.
        let (strong, weak) = code.split_once('v').expect("a table's code has a 'v'");
        let mut counts = [[0i32; PIECE_TYPE_NB]; COLOR_NB];
        for (side, pieces) in [strong, weak].into_iter().enumerate() {
            for ch in pieces.bytes() {
                let pt = PIECE_TO_CHAR[1..7]
                    .iter()
                    .position(|&c| c == ch)
                    .expect("a table's code names pieces");
                counts[side][pt + 1] += 1;
            }
        }

        let piece_count_for = |strong_color: Color| {
            let mut piece_count = [0; PIECE_NB];
            for pt in PAWN.0..=KING.0 {
                let pt = PieceType(pt);
                piece_count[make_piece(strong_color, pt).idx()] = counts[0][pt.idx()];
                piece_count[make_piece(!strong_color, pt).idx()] = counts[1][pt.idx()];
            }
            piece_count
        };

        let count = |c: Color, pt: PieceType| counts[c.idx()][pt.idx()];

        let mut has_unique_pieces = false;
        for c in [WHITE, BLACK] {
            for pt in PAWN.0..KING.0 {
                if count(c, PieceType(pt)) == 1 {
                    has_unique_pieces = true;
                }
            }
        }

        // Set the leading color. In case both sides have pawns the leading
        // color is the side with fewer pawns because this leads to better
        // compression.
        let c = count(BLACK, PAWN) == 0
            || (count(WHITE, PAWN) != 0 && count(BLACK, PAWN) >= count(WHITE, PAWN));
        let (lead, other) = if c { (WHITE, BLACK) } else { (BLACK, WHITE) };

        TbTable {
            key: material_key_of(&piece_count_for(WHITE)),
            key2: material_key_of(&piece_count_for(BLACK)),
            piece_count: counts.iter().flatten().sum::<i32>() as usize,
            has_pawns: count(WHITE, PAWN) + count(BLACK, PAWN) > 0,
            has_unique_pieces,
            pawn_count: [count(lead, PAWN) as u8, count(other, PAWN) as u8],
            files: [OnceLock::new(), OnceLock::new()],
        }
    }
}

/// The tables of a variant: one for each WDL file found, with its DTZ file,
/// in a hash table by material key.
pub struct Tables<V: Variant> {
    paths: Vec<PathBuf>,
    hash_table: Vec<Option<(Key, u32)>>,
    tables: Vec<TbTable>,
    found_dtz_files: usize,
    found_wdl_files: usize,
    max_cardinality: i32,
    variant: PhantomData<V>,
}

/// 4K table, indexed by key's 12 lsb
const HASH_SIZE: usize = 1 << 12;
/// Number of elements allowed to map to the last bucket
const HASH_OVERFLOW: usize = 1;

impl<V: Variant> Default for Tables<V> {
    fn default() -> Self {
        Tables {
            paths: Vec::new(),
            hash_table: Vec::new(),
            tables: Vec::new(),
            found_dtz_files: 0,
            found_wdl_files: 0,
            max_cardinality: 0,
            variant: PhantomData,
        }
    }
}

/// The tables of each variant: those of the variant played, and none of the
/// others.
#[derive(Default)]
pub struct Tablebases {
    pub chess: Tables<Chess>,
    pub atomic: Tables<Atomic>,
}

impl Tablebases {
    /// Finds the tables of `variant` in `paths`, as SyzygyPath lists them,
    /// dropping all the tables found before, and tells what it found unless
    /// `paths` names no directory.
    pub fn init(&mut self, variant: VariantKind, paths: &str) -> Option<String> {
        *self = Tablebases::default();
        match variant {
            VariantKind::Chess => {
                self.chess = Tables::new(paths);
                self.chess.info()
            }
            VariantKind::Atomic => {
                self.atomic = Tables::new(paths);
                self.atomic.info()
            }
        }
    }
}

impl<V: Variant> Tables<V> {
    /// The tables found in `paths`, the directories that SyzygyPath lists,
    /// separated by ":" on Unix-based operating systems and by ";" on
    /// Windows, such as "C:\tb\wdl345;C:\tb\wdl6;D:\tb\dtz345;D:\tb\dtz6".
    /// Created at startup and after every change to SyzygyPath, which drops
    /// the tables mapped before.
    pub fn new(paths: &str) -> Tables<V> {
        let sep = if cfg!(windows) { ';' } else { ':' };
        let mut tables = Tables::<V> {
            paths: paths
                .split(sep)
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .collect(),
            ..Tables::default()
        };

        if tables.paths.is_empty() {
            return tables;
        }

        tables.hash_table = vec![None; HASH_SIZE + HASH_OVERFLOW];

        // Add entries in TB tables if the corresponding WDL file exists
        let range = |from: u8, to: u8| (from..=to).map(PieceType);
        for p1 in range(PAWN.0, QUEEN.0) {
            tables.add(&[KING, p1, KING]);

            for p2 in range(PAWN.0, p1.0) {
                tables.add(&[KING, p1, p2, KING]);
                tables.add(&[KING, p1, KING, p2]);

                for p3 in range(PAWN.0, QUEEN.0) {
                    tables.add(&[KING, p1, p2, KING, p3]);
                }

                for p3 in range(PAWN.0, p2.0) {
                    tables.add(&[KING, p1, p2, p3, KING]);

                    for p4 in range(PAWN.0, p3.0) {
                        tables.add(&[KING, p1, p2, p3, p4, KING]);

                        for p5 in range(PAWN.0, p4.0) {
                            tables.add(&[KING, p1, p2, p3, p4, p5, KING]);
                        }

                        for p5 in range(PAWN.0, QUEEN.0) {
                            tables.add(&[KING, p1, p2, p3, p4, KING, p5]);
                        }
                    }

                    for p4 in range(PAWN.0, QUEEN.0) {
                        tables.add(&[KING, p1, p2, p3, KING, p4]);

                        for p5 in range(PAWN.0, p4.0) {
                            tables.add(&[KING, p1, p2, p3, KING, p4, p5]);
                        }
                    }
                }

                for p3 in range(PAWN.0, p1.0) {
                    for p4 in range(PAWN.0, if p1 == p3 { p2.0 } else { p3.0 }) {
                        tables.add(&[KING, p1, p2, KING, p3, p4]);
                    }
                }
            }
        }

        tables
    }

    /// What tables were found, unless no directory was given
    pub fn info(&self) -> Option<String> {
        (!self.paths.is_empty()).then(|| {
            format!(
                "Found {} WDL and {} DTZ tablebase files (up to {}-man).",
                self.found_wdl_files, self.found_dtz_files, self.max_cardinality
            )
        })
    }

    /// The most pieces of the tables found
    pub fn max_cardinality(&self) -> i32 {
        self.max_cardinality
    }

    /// Opens a file among the directories of the tables.
    fn open(&self, name: &str) -> Option<(File, PathBuf)> {
        self.paths.iter().find_map(|path| {
            let path = path.join(name);
            File::open(&path).ok().map(|file| (file, path))
        })
    }

    /// If the corresponding file exists, a new table is added to the list
    /// and the hash table. Called at init time.
    fn add(&mut self, pieces: &[PieceType]) {
        if pieces.len() > V::SYZYGY.max_pieces {
            return;
        }

        let mut code: String = pieces
            .iter()
            .map(|pt| PIECE_TO_CHAR[pt.idx()] as char)
            .collect();
        let second_king = 1 + code[1..].find('K').expect("a table has two kings");
        code.insert(second_king, 'v'); // KRK -> KRvK

        let [wdl_suffix, dtz_suffix] = V::SYZYGY.suffixes;

        if self.open(&format!("{code}{dtz_suffix}")).is_some() {
            self.found_dtz_files += 1;
        }

        // Only WDL file is checked
        if self.open(&format!("{code}{wdl_suffix}")).is_none() {
            return;
        }

        self.found_wdl_files += 1;
        self.max_cardinality = self.max_cardinality.max(pieces.len() as i32);

        let table = TbTable::new(&code);
        let (key, key2) = (table.key, table.key2);
        let i = self.tables.len() as u32;
        self.tables.push(table);

        // Insert into the hash keys for both colors: KRvK with KR white and
        // black
        self.insert(key, i);
        self.insert(key2, i);
    }

    fn insert(&mut self, mut key: Key, i: u32) {
        let mut home_bucket = key as u32 as usize & (HASH_SIZE - 1);
        let mut entry = Some((key, i));

        // Ensure last element is empty to avoid overflow when looking up
        let mut bucket = home_bucket;
        while bucket < HASH_SIZE + HASH_OVERFLOW - 1 {
            let other = self.hash_table[bucket];
            let other_key = match other {
                Some((other_key, _)) if other_key != key => other_key,
                _ => {
                    self.hash_table[bucket] = entry;
                    return;
                }
            };

            // Robin Hood hashing: If we've probed for longer than this
            // element, insert here and search for a new spot for the other
            // element instead.
            let other_home_bucket = other_key as u32 as usize & (HASH_SIZE - 1);
            if other_home_bucket > home_bucket {
                self.hash_table[bucket] = entry;
                entry = other;
                key = other_key;
                home_bucket = other_home_bucket;
            }
            bucket += 1;
        }

        fail("TB hash table size too low!");
    }

    fn get(&self, key: Key) -> Option<&TbTable> {
        if self.hash_table.is_empty() {
            return None;
        }

        let mut bucket = key as u32 as usize & (HASH_SIZE - 1);
        loop {
            match self.hash_table[bucket] {
                Some((k, i)) if k == key => return Some(&self.tables[i as usize]),
                None => return None,
                Some(_) => bucket += 1,
            }
        }
    }

    /// The file of the table, mapped and initialized at first access, or
    /// None if missing. Thread safe: concurrent probes wait for the first.
    fn mapped<'a, const T: usize>(
        &self,
        e: &'a TbTable,
        pos: &Position<V>,
    ) -> Option<&'a MappedTable> {
        // Because TB is the only usage of materialKey, check it here in
        // debug mode
        debug_assert!(pos.material_key_is_ok());

        e.files[T]
            .get_or_init(|| {
                // Pieces strings in decreasing order for each color, like
                // ("KPP","KR")
                let (mut w, mut b) = (String::new(), String::new());
                for pt in (PAWN.0..=KING.0).rev() {
                    let ch = PIECE_TO_CHAR[pt as usize] as char;
                    for _ in 0..pos.count(WHITE, PieceType(pt)) {
                        w.push(ch);
                    }
                    for _ in 0..pos.count(BLACK, PieceType(pt)) {
                        b.push(ch);
                    }
                }

                let fname = if e.key == pos.material_key() {
                    format!("{w}v{b}")
                } else {
                    format!("{b}v{w}")
                } + V::SYZYGY.suffixes[T];

                let (file, path) = self.open(&fname)?;
                let mapping = Mapping::new(file, &path, V::SYZYGY.magics[T])?;
                let mut table = MappedTable {
                    mapping,
                    items: Default::default(),
                    map: 0,
                };

                if set::<T>(e, &mut table, V::SYZYGY.connected_kings).is_none() {
                    fail(&format!("Corrupted table in file {fname}"));
                }
                Some(table)
            })
            .as_ref()
    }

    fn probe_table<const T: usize>(
        &self,
        pos: &Position<V>,
        result: &mut ProbeState,
        wdl: WdlScore,
    ) -> i32 {
        // An Atomic capture can explode a king. Such terminal positions are
        // not represented in a table file and must be classified before
        // material lookup (including the otherwise hard-coded KvK draw).
        if V::ATOMIC && (!pos.has_king(WHITE) || !pos.has_king(BLACK)) {
            *result = ProbeState::Ok;
            let stm_wins = pos.has_king(pos.side_to_move());

            return if T == WDL {
                if stm_wins { WDL_WIN } else { WDL_LOSS }
            } else {
                // Stored DTZ magnitudes are unsigned here; probe_dtz()
                // applies the WDL sign exactly once.
                1
            };
        }

        // KvK
        if pos.count_both(ALL_PIECES) == 2 {
            return WDL_DRAW;
        }

        let Some(entry) = self.get(pos.material_key()) else {
            *result = ProbeState::Fail;
            return 0;
        };

        let Some(table) = self.mapped::<T>(entry, pos) else {
            *result = ProbeState::Fail;
            return 0;
        };

        do_probe_table::<V, T>(pos, entry, table, wdl, result)
    }

    // For a position where the side to move has a winning capture it is not
    // necessary to store a winning value so the generator treats such
    // positions as "don't care" and tries to assign to it a value that
    // improves the compression ratio. Similarly, if the side to move has a
    // drawing capture, then the position is at least drawn. If the position
    // is won, then the TB needs to store a win value. But if the position is
    // drawn, the TB may store a loss value if that is better for
    // compression. All of this means that during probing, the engine must
    // look at captures and probe their results and must probe the position
    // itself. The "best" result of these probes is the correct result for
    // the position. DTZ tables do not store values when a following move is
    // a zeroing winning move (winning capture or winning pawn move). Also,
    // DTZ store wrong values for positions where the best move is an ep-move
    // (even if losing). So in all these cases set the state to
    // ZEROING_BEST_MOVE.
    fn search<const CHECK_ZEROING_MOVES: bool>(
        &self,
        pos: &mut Position<V>,
        result: &mut ProbeState,
    ) -> WdlScore {
        let mut best_value = WDL_LOSS;
        let mut scratch = V::Dirties::default();

        let move_list = MoveList::new::<ALL>(pos);
        let total_count = move_list.len();
        let mut move_count = 0;

        for &m in &move_list {
            if !pos.capture(m) && (!CHECK_ZEROING_MOVES || pos.moved_piece(m).type_of() != PAWN) {
                continue;
            }

            move_count += 1;

            pos.make_move(m, &mut scratch);
            let value = -self.search::<false>(pos, result);
            pos.undo_move(m);

            if *result == ProbeState::Fail {
                return WDL_DRAW;
            }

            if value > best_value {
                best_value = value;

                if value >= WDL_WIN {
                    // Winning DTZ-zeroing move
                    *result = ProbeState::ZeroingBestMove;
                    return value;
                }
            }
        }

        // In case we have already searched all the legal moves we don't have
        // to probe the TB because the stored score could be wrong. For
        // instance TB tables do not contain information on position with ep
        // rights, so in this case the result of probe_wdl_table is wrong.
        // Also in case of only capture moves, for instance here
        // 4K3/4q3/6p1/2k5/6p1/8/8/8 w - - 0 7, we have to return with
        // ZEROING_BEST_MOVE set.
        let no_more_moves = move_count != 0 && move_count == total_count;

        let value = if no_more_moves {
            best_value
        } else {
            let value = self.probe_table::<WDL>(pos, result, WDL_DRAW);

            if *result == ProbeState::Fail {
                return WDL_DRAW;
            }
            value
        };

        // DTZ stores a "don't care" value if bestValue is a win
        if best_value >= value {
            *result = if best_value > WDL_DRAW || no_more_moves {
                ProbeState::ZeroingBestMove
            } else {
                ProbeState::Ok
            };
            return best_value;
        }

        *result = ProbeState::Ok;
        value
    }

    /// Probe the WDL table for a particular position. If *result != FAIL,
    /// the probe was successful. The return value is from the point of view
    /// of the side to move:
    ///
    /// -2 : loss
    /// -1 : loss, but draw under 50-move rule
    ///  0 : draw
    ///  1 : win, but draw under 50-move rule
    ///  2 : win
    pub fn probe_wdl(&self, pos: &mut Position<V>, result: &mut ProbeState) -> WdlScore {
        *result = ProbeState::Ok;
        self.search::<false>(pos, result)
    }

    /// Probe the DTZ table for a particular position. If *result != FAIL,
    /// the probe was successful. The return value is from the point of view
    /// of the side to move:
    ///
    /// ```text
    ///         n < -100 : loss, but draw under 50-move rule
    /// -100 <= n < -1   : loss in n ply (assuming 50-move counter == 0)
    ///        -1        : loss, the side to move is mated
    ///         0        : draw
    ///     1 < n <= 100 : win in n ply (assuming 50-move counter == 0)
    ///   100 < n        : win, but draw under 50-move rule
    /// ```
    ///
    /// The return value n can be off by 1: a return value -n can mean a loss
    /// in n+1 ply and a return value +n can mean a win in n+1 ply. This
    /// cannot happen for tables with positions exactly on the "edge" of the
    /// 50-move rule.
    ///
    /// This implies that if dtz > 0 is returned, the position is certainly a
    /// win if dtz + 50-move-counter <= 99. Care must be taken that the engine
    /// picks moves that preserve dtz + 50-move-counter <= 99.
    ///
    /// If n = 100 immediately after a capture or pawn move, then the position
    /// is also certainly a win, and during the whole phase until the next
    /// capture or pawn move, the inequality to be preserved is dtz +
    /// 50-move-counter <= 100.
    ///
    /// In short, if a move is available resulting in dtz + 50-move-counter
    /// <= 99, then do not accept moves leading to dtz + 50-move-counter ==
    /// 100.
    pub fn probe_dtz(&self, pos: &mut Position<V>, result: &mut ProbeState) -> i32 {
        *result = ProbeState::Ok;
        let wdl = self.search::<true>(pos, result);

        // DTZ tables don't store draws
        if *result == ProbeState::Fail || wdl == WDL_DRAW {
            return 0;
        }

        // DTZ stores a 'don't care value in this case, or even a plain wrong
        // one as in case the best move is a losing ep, so it cannot be
        // probed.
        if *result == ProbeState::ZeroingBestMove {
            return dtz_before_zeroing(wdl);
        }

        let dtz = self.probe_table::<DTZ>(pos, result, wdl);

        if *result == ProbeState::Fail {
            return 0;
        }

        if *result != ProbeState::ChangeStm {
            let cursed = wdl == WDL_BLESSED_LOSS || wdl == WDL_CURSED_WIN;
            return (dtz + 100 * i32::from(cursed)) * sign_of(wdl);
        }

        // DTZ stores results for the other side, so we need to do a 1-ply
        // search and find the winning move that minimizes DTZ.
        let mut scratch = V::Dirties::default();
        let mut min_dtz = 0xFFFF;

        for &m in &MoveList::new::<ALL>(pos) {
            let zeroing = pos.capture(m) || pos.moved_piece(m).type_of() == PAWN;

            pos.make_move(m, &mut scratch);

            // For zeroing moves we want the dtz of the move _before_ doing
            // it, otherwise we will get the dtz of the next move sequence.
            // Search the position after the move to get the score sign
            // (because even in a winning position we could make a losing
            // capture or go for a draw).
            let mut dtz = if zeroing {
                -dtz_before_zeroing(self.search::<false>(pos, result))
            } else {
                -self.probe_dtz(pos, result)
            };

            // If the move mates, force minDTZ to 1
            if dtz == 1 && pos.in_check() && MoveList::new::<ALL>(pos).is_empty() {
                min_dtz = 1;
            }

            // Convert result from 1-ply search. Zeroing moves are already
            // accounted by dtz_before_zeroing() that returns the DTZ of the
            // previous move.
            if !zeroing {
                dtz += sign_of(dtz);
            }

            // Skip the draws and if we are winning only pick positive dtz
            if dtz < min_dtz && sign_of(dtz) == sign_of(wdl) {
                min_dtz = dtz;
            }

            pos.undo_move(m);

            if *result == ProbeState::Fail {
                return 0;
            }
        }

        // When there are no legal moves, the position is mate: we return -1
        if min_dtz == 0xFFFF { -1 } else { min_dtz }
    }

    /// Use the DTZ tables to rank root moves.
    ///
    /// A return value false indicates that not all probes were successful.
    pub fn root_probe(
        &self,
        pos: &mut Position<V>,
        root_moves: &mut [RootMove],
        rule50: bool,
        rank_dtz: bool,
        time_abort: &dyn Fn() -> bool,
    ) -> bool {
        let mut result = ProbeState::Ok;
        let mut scratch = V::Dirties::default();

        // Obtain 50-move counter for the root position
        let cnt50 = pos.rule50_count();

        // Check whether a position was repeated since the last zeroing move.
        let rep = pos.has_repeated();

        let bound = if rule50 { MAX_DTZ / 2 - 100 } else { 1 };

        // Probe and rank each move
        for m in root_moves.iter_mut() {
            pos.make_move(m.pv[0], &mut scratch);

            // Calculate dtz for the current move counting from the root
            // position
            let mut dtz = if pos.rule50_count() == 0 {
                // In case of a zeroing move, dtz is one of -101/-1/0/1/101
                let wdl = -self.probe_wdl(pos, &mut result);
                dtz_before_zeroing(wdl)
            } else if (rule50 && pos.is_draw(1)) || pos.is_repetition(1) {
                // In case a root move leads to a draw by repetition or 50-move
                // rule, we set dtz to zero. Note: since we are only 1 ply from
                // the root, this must be a true 3-fold repetition inside the
                // game history.
                0
            } else {
                // Otherwise, take dtz for the new position and correct by 1
                // ply
                let dtz = -self.probe_dtz(pos, &mut result);
                dtz + sign_of(dtz)
            };

            // Make sure that a mating move is assigned a dtz value of 1
            if pos.in_check() && dtz == 2 && MoveList::new::<ALL>(pos).is_empty() {
                dtz = 1;
            }

            pos.undo_move(m.pv[0]);

            if time_abort() || result == ProbeState::Fail {
                return false;
            }

            // Better moves are ranked higher. Certain wins are ranked equally.
            // Losing moves are ranked equally unless a 50-move draw is in
            // sight.
            let rank_dtz_term = if rank_dtz { dtz } else { 0 };
            let r = match dtz.cmp(&0) {
                Ordering::Greater if dtz + cnt50 <= 99 && !rep => MAX_DTZ - rank_dtz_term,
                Ordering::Greater => MAX_DTZ / 2 - (dtz + cnt50),
                Ordering::Less if -dtz * 2 + cnt50 < 100 => -MAX_DTZ - rank_dtz_term,
                Ordering::Less => -MAX_DTZ / 2 + (-dtz + cnt50),
                Ordering::Equal => 0,
            };
            m.tb_rank = r;

            // Determine the score to be displayed for this move. Assign at
            // least 1 cp to cursed wins and let it grow to 49 cp as the
            // positions gets closer to a real win.
            m.tb_score = if r >= bound {
                VALUE_MATE - MAX_PLY - 1
            } else if r > 0 {
                (3.max(r - (MAX_DTZ / 2 - 200)) * PAWN_VALUE) / 200
            } else if r == 0 {
                VALUE_DRAW
            } else if r > -bound {
                ((-3).min(r + (MAX_DTZ / 2 - 200)) * PAWN_VALUE) / 200
            } else {
                -VALUE_MATE + MAX_PLY + 1
            };
        }

        true
    }

    /// Use the WDL tables to rank root moves. This is a fallback for the case
    /// that some or all DTZ tables are missing.
    ///
    /// A return value false indicates that not all probes were successful.
    pub fn root_probe_wdl(
        &self,
        pos: &mut Position<V>,
        root_moves: &mut [RootMove],
        rule50: bool,
    ) -> bool {
        const WDL_TO_RANK: [i32; 5] = [-MAX_DTZ, -MAX_DTZ + 101, 0, MAX_DTZ - 101, MAX_DTZ];

        let mut result = ProbeState::Ok;
        let mut scratch = V::Dirties::default();

        // Probe and rank each move
        for m in root_moves.iter_mut() {
            pos.make_move(m.pv[0], &mut scratch);

            let mut wdl = if pos.is_draw(1) {
                WDL_DRAW
            } else {
                -self.probe_wdl(pos, &mut result)
            };

            pos.undo_move(m.pv[0]);

            if result == ProbeState::Fail {
                return false;
            }

            m.tb_rank = WDL_TO_RANK[(wdl + 2) as usize];

            if !rule50 {
                wdl = wdl.signum() * WDL_WIN;
            }
            m.tb_score = WDL_TO_VALUE[(wdl + 2) as usize];
        }

        true
    }

    /// Ranks the root moves by the tables, best first, if the root position
    /// is in them, and says how the search is to probe them.
    pub fn rank_root_moves(
        &self,
        options: &SyzygyOptions,
        pos: &mut Position<V>,
        root_moves: &mut [RootMove],
        rank_dtz: bool,
        time_abort: &dyn Fn() -> bool,
    ) -> TbConfig {
        let mut config = TbConfig::default();

        if root_moves.is_empty() {
            return config;
        }

        config.root_in_tb = false;
        config.use_rule50 = options.rule50;
        config.probe_depth = options.probe_depth;
        config.cardinality = options.probe_limit;

        let mut dtz_available = true;

        // Tables with fewer pieces than SyzygyProbeLimit are searched with
        // probeDepth == DEPTH_ZERO
        if config.cardinality > self.max_cardinality {
            config.cardinality = self.max_cardinality;
            config.probe_depth = 0;
        }

        if config.cardinality >= pos.count_both(ALL_PIECES) && !pos.can_castle(ANY_CASTLING) {
            // Use DTZ to rank the moves if checkmate is the only zeroing move
            let rank_dtz = rank_dtz || pos.dtz_is_dtm();

            // Rank moves using DTZ tables, bail out if time_abort flags
            // zeitnot
            config.root_in_tb =
                self.root_probe(pos, root_moves, options.rule50, rank_dtz, time_abort);

            if !config.root_in_tb && !time_abort() {
                // DTZ tables are missing; try to rank moves using WDL tables
                dtz_available = false;
                config.root_in_tb = self.root_probe_wdl(pos, root_moves, options.rule50);
            }
        }

        if config.root_in_tb {
            // Sort moves according to TB rank
            root_moves.sort_by_key(|m| std::cmp::Reverse(m.tb_rank));

            // Probe during search only if DTZ is not available and we are
            // winning
            if dtz_available || root_moves[0].tb_score <= VALUE_DRAW {
                config.cardinality = 0;
            }
        } else {
            // Clean up if root_probe() and root_probe_wdl() have failed
            for m in root_moves.iter_mut() {
                m.tb_rank = 0;
            }
        }

        config
    }
}

/// The value stored at index `idx` of the table `d` in `bytes`.
fn decompress_pairs(d: &PairsData, bytes: &[u8], idx: u64) -> i32 {
    // Special case where all table positions store the same value
    if d.flags & SINGLE_VALUE != 0 {
        return i32::from(d.min_sym_len);
    }

    // First we need to locate the right block that stores the value at index
    // "idx". Because each block n stores blockLength[n] + 1 values, the index
    // i of the block that contains the value at position idx is:
    //
    //                    for (i = -1, sum = 0; sum <= idx; i++)
    //                        sum += blockLength[i + 1] + 1;
    //
    // This can be slow, so we use SparseIndex[] populated with a set of
    // SparseEntry that point to known indices into blockLength[]. Namely
    // SparseIndex[k] is a SparseEntry that stores the blockLength[] index and
    // the offset within that block of the value with index I(k), where:
    //
    //       I(k) = k * d->span + d->span / 2      (1)

    // First step is to get the 'k' of the I(k) nearest to our idx, using
    // definition (1)
    let span = d.span as u64;
    let k = (idx / span) as u32 as usize;

    // Then we read the corresponding SparseIndex[] entry
    let entry = d.sparse_index + 6 * k;
    let mut block = u32_le(bytes, entry);
    let mut offset = i32::from(u16_le(bytes, entry + 4));

    // Now compute the difference idx - I(k). From the definition of k, we
    // know that
    //
    //       idx = k * d->span + idx % d->span    (2)
    //
    // So from (1) and (2) we can compute idx - I(K):
    let diff = (idx % span).wrapping_sub(span / 2) as i32;

    // Sum the above to offset to find the offset corresponding to our idx
    offset = offset.wrapping_add(diff);

    if block >= d.block_length_size {
        block = d.block_length_size - 1;
    }

    let block_length = |block: u32| i32::from(u16_le(bytes, d.block_length + 2 * block as usize));

    // Move to the previous/next block, until we reach the correct block that
    // contains idx, that is when 0 <= offset <= d->blockLength[block]
    while offset < 0 && block > 0 {
        block -= 1;
        offset += block_length(block) + 1;
    }

    while offset > block_length(block) && block + 1 < d.block_length_size {
        offset -= block_length(block) + 1;
        block += 1;
    }

    // Finally, we find the start address of our block of canonical Huffman
    // symbols
    let mut ptr = (block as usize)
        .saturating_mul(d.sizeof_block)
        .saturating_add(d.data);

    // Read the first 64 bits in our block, this is a (truncated) sequence of
    // unknown number of symbols of unknown length but we know the first one
    // is at the beginning of this 64-bit sequence.
    let mut buf64 = if ptr.saturating_add(8) <= d.data_end {
        u64_be(bytes, ptr)
    } else {
        0
    };
    ptr = ptr.saturating_add(8);
    let mut buf64_size: i32 = 64;
    let mut sym;

    loop {
        // This is the symbol length - d->min_sym_len
        let mut len = 0;

        // Now get the symbol length. For any symbol s64 of length l
        // right-padded to 64 bits we know that d->base64[l-1] >= s64 >=
        // d->base64[l] so we can find the symbol length iterating through
        // base64[].
        while buf64 < d.base64[len] {
            len += 1;
        }

        // All the symbols of a given length are consecutive integers
        // (numerical sequence property), so we can compute the offset of our
        // symbol of length len, stored at the beginning of buf64.
        sym = ((buf64 - d.base64[len]) >> (64 - len - usize::from(d.min_sym_len))) as usize;

        // Now add the value of the lowest symbol of length len to get our
        // symbol, of 12 bits
        sym =
            sym.wrapping_add(usize::from(u16_le(bytes, d.lowest_sym + 2 * len))) & (SYM_COUNT - 1);

        // If our offset is within the number of values represented by symbol
        // sym, we are done.
        if offset < d.symlen(sym) + 1 {
            break;
        }

        // ...otherwise update the offset and continue to iterate
        offset -= d.symlen(sym) + 1;
        len += usize::from(d.min_sym_len); // Get the real length
        buf64 <<= len; // Consume the just processed symbol
        buf64_size -= len as i32;

        if buf64_size <= 32 {
            // Refill the buffer
            buf64_size += 32;
            if ptr.saturating_add(4) <= d.data_end {
                buf64 |= u64::from(u32_be(bytes, ptr))
                    .checked_shl((64 - buf64_size) as u32)
                    .unwrap_or(0);
            }
            ptr = ptr.saturating_add(4);
        }
    }

    // Now we have our symbol that expands into d->symlen[sym] + 1 symbols.
    // We binary-search for our value recursively expanding into the left and
    // right child symbols until we reach a leaf node where symlen[sym] + 1 ==
    // 1 that will store the value we need.
    while d.symlen(sym) != 0 {
        let left = d.left(sym);

        // If a symbol contains 36 sub-symbols (d->symlen[sym] + 1 = 36) and
        // expands in a pair (d->symlen[left] = 23, d->symlen[right] = 11),
        // then we know that, for instance, the tenth value (offset = 10) will
        // be on the left side because in Recursive Pairing child symbols are
        // adjacent.
        if offset < d.symlen(left) + 1 {
            sym = left;
        } else {
            offset -= d.symlen(left) + 1;
            sym = d.right(sym);
        }
    }

    d.left(sym) as i32
}

// DTZ scores are sorted by frequency of occurrence and then assigned the
// values 0, 1, 2, ... in order of decreasing frequency. This is done for each
// of the four WDLScore values. The mapping information necessary to
// reconstruct the original values are stored in the TB file and read during
// map[] init.
fn map_score<const T: usize>(table: &MappedTable, d: &PairsData, value: i32, wdl: WdlScore) -> i32 {
    if T == WDL {
        return value - 2;
    }

    const WDL_MAP: [usize; 5] = [1, 3, 0, 2, 0];

    let flags = d.flags;
    let mut value = value;

    if flags & MAPPED != 0 {
        let bytes = table.mapping.bytes();
        let i = usize::from(d.map_idx[WDL_MAP[(wdl + 2) as usize]]) + value as usize;
        value = if flags & WIDE != 0 {
            i32::from(u16_le(bytes, table.map + 2 * i))
        } else {
            i32::from(bytes[table.map + i])
        };
    }

    // DTZ tables store distance to zero in number of moves or plies. We want
    // to return plies, so we have to convert to plies when needed.
    if (wdl == WDL_WIN && flags & WIN_PLIES == 0)
        || (wdl == WDL_LOSS && flags & LOSS_PLIES == 0)
        || wdl == WDL_CURSED_WIN
        || wdl == WDL_BLESSED_LOSS
    {
        value *= 2;
    }

    value + 1
}

/// The index of the two kings that lead a table without other unique
/// pieces, mirrored into the a1-d1-d4 triangle and, when the first is on the
/// a1-h8 diagonal, the second not above it: one of 462 pairs in chess, and of
/// 518 where the kings may stand next to each other, as in Atomic, ordered
/// then as three unique pieces are.
fn kings_index(s0: usize, s1: usize, connected_kings: bool) -> i32 {
    if !connected_kings {
        return MAPS.kk[MAPS.a1d1d4[s0] as usize][s1];
    }

    let rank = |s: usize| (s >> 3) as i32;
    let off = |s: usize| off_a1h8(s) != 0;
    let adjust = i32::from(s1 > s0);

    if off(s0) {
        MAPS.a1d1d4[s0] * 63 + (s1 as i32 - adjust)
    } else if off(s1) {
        6 * 63 + rank(s0) * 28 + MAPS.b1h1h7[s1]
    } else {
        6 * 63 + 4 * 28 + rank(s0) * 7 + (rank(s1) - adjust)
    }
}

/// Computes a unique index out of a position and uses it to probe the TB
/// file. To encode k pieces of the same type and color, first sort the
/// pieces by square in ascending order s1 <= s2 <= ... <= sk then compute the
/// unique index as:
///
/// ```text
/// idx = Binomial[1][s1] + Binomial[2][s2] + ... + Binomial[k][sk]
/// ```
fn do_probe_table<V: Variant, const T: usize>(
    pos: &Position<V>,
    entry: &TbTable,
    table: &MappedTable,
    wdl: WdlScore,
    result: &mut ProbeState,
) -> i32 {
    let sides = if T == WDL { 2 } else { 1 };
    let mut squares = [Square(0); TBPIECES];
    let mut pieces = [NO_PIECE; TBPIECES];
    let mut size = 0;
    let mut lead_pawns_cnt = 0;
    let mut lead_pawns = Bitboard(0);
    let mut tb_file = FILE_A as usize;

    // A given TB entry like KRK has associated two material keys: KRvk and
    // Kvkr. If both sides have the same pieces keys are equal. In this case
    // TB tables only stores the 'white to move' case, so if the position to
    // lookup has black to move, we need to switch the color and flip the
    // squares before to lookup.
    let symmetric_black_to_move = entry.key == entry.key2 && pos.side_to_move() == BLACK;

    // TB files are calculated for white as the stronger side. For instance,
    // we have KRvK, not KvKR. A position where the stronger side is white
    // will have its material key == entry->key, otherwise we have to switch
    // the color and flip the squares before to lookup.
    let black_stronger = pos.material_key() != entry.key;

    let flip = symmetric_black_to_move || black_stronger;
    let flip_color = if flip { 8 } else { 0 };
    let flip_squares = if flip { 56 } else { 0 };
    let stm = usize::from(flip) ^ pos.side_to_move().idx();

    // For pawns, TB files store 4 separate tables according if leading pawn
    // is on file a, b, c or d after reordering. The leading pawn is the one
    // with maximum MapPawns[] value, that is the one most toward the edges
    // and with lowest rank.
    if entry.has_pawns {
        // In all the 4 tables, pawns are at the beginning of the piece
        // sequence and their color is the reference one. So we just pick the
        // first one.
        let pc = Piece(table.items[0][0].pieces[0].0 ^ flip_color);

        debug_assert!(pc.type_of() == PAWN);

        lead_pawns = pos.pieces_cpt(pc.color(), PAWN);
        for s in lead_pawns {
            squares[size] = Square(s.0 ^ flip_squares);
            size += 1;
        }

        lead_pawns_cnt = size;

        let lead = (1..lead_pawns_cnt).fold(0, |lead, i| {
            if MAPS.pawns[squares[lead].idx()] < MAPS.pawns[squares[i].idx()] {
                i
            } else {
                lead
            }
        });
        squares.swap(0, lead);

        tb_file = edge_distance(squares[0].file()) as usize;
    }

    // DTZ tables are one-sided, i.e. they store positions only for white to
    // move or only for black to move, so check for side to move to be stm,
    // early exit otherwise.
    if T == DTZ
        && usize::from(table.get(sides, entry.has_pawns, stm, tb_file).flags & STM) != stm
        && !(entry.key == entry.key2 && !entry.has_pawns)
    {
        *result = ProbeState::ChangeStm;
        return 0;
    }

    // Now we are ready to get all the position pieces (but the lead pawns)
    // and directly map them to the correct color and square.
    for s in pos.pieces() ^ lead_pawns {
        squares[size] = Square(s.0 ^ flip_squares);
        pieces[size] = Piece(pos.piece_on(s).0 ^ flip_color);
        size += 1;
    }

    debug_assert!(size >= 2);

    let d = table.get(sides, entry.has_pawns, stm, tb_file);

    // Then we reorder the pieces to have the same sequence as the one stored
    // in pieces[i]: the sequence that ensures the best compression.
    for i in lead_pawns_cnt..size - 1 {
        for j in i + 1..size {
            if d.pieces[i] == pieces[j] {
                pieces.swap(i, j);
                squares.swap(i, j);
                break;
            }
        }
    }

    // Now we map again the squares so that the square of the lead piece is
    // in the triangle A1-D1-D4.
    if squares[0].file() > FILE_D {
        for s in &mut squares[..size] {
            *s = s.flip_file();
        }
    }

    let mut idx: u64;

    if entry.has_pawns {
        // Encode leading pawns starting with the one with minimum MapPawns[]
        // and proceeding in ascending order.
        idx = MAPS.lead_pawn_idx[lead_pawns_cnt][squares[0].idx()] as u64;

        squares[1..lead_pawns_cnt].sort_by_key(|s| MAPS.pawns[s.idx()]);

        for (i, s) in squares.iter().enumerate().take(lead_pawns_cnt).skip(1) {
            idx += MAPS.binomial[i][MAPS.pawns[s.idx()] as usize] as u64;
        }
    } else {
        // In positions without pawns, we further flip the squares to ensure
        // leading piece is below RANK_5.
        if squares[0].rank() > RANK_4 {
            for s in &mut squares[..size] {
                *s = s.flip_rank();
            }
        }

        // Look for the first piece of the leading group not on the A1-D4
        // diagonal and ensure it is mapped below the diagonal.
        for i in 0..d.group_len[0] {
            if off_a1h8(squares[i].idx()) == 0 {
                continue;
            }

            if off_a1h8(squares[i].idx()) > 0 {
                // A1-H8 diagonal flip: SQ_A3 -> SQ_C1
                for s in &mut squares[i..size] {
                    *s = Square(((s.0 >> 3) | (s.0 << 3)) & 63);
                }
            }
            break;
        }

        let s0 = squares[0].idx();
        let s1 = squares[1].idx();
        let rank = |s: usize| (s >> 3) as i32;
        let off = |s: usize| off_a1h8(s) != 0;

        // Encode the leading group.
        //
        // Suppose we have KRvK. Let's say the pieces are on square numbers
        // wK, wR and bK (each 0...63). The simplest way to map this position
        // to an index is like this:
        //
        //   index = wK * 64 * 64 + wR * 64 + bK;
        //
        // But this way the TB is going to have 64*64*64 = 262144 positions,
        // with lots of positions being equivalent (because they are mirrors
        // of each other) and lots of positions being invalid (two pieces on
        // one square, adjacent kings, etc.). Usually the first step is to take
        // the wK and bK together. There are just 462 ways legal and
        // not-mirrored ways to place the wK and bK on the board. Once we have
        // placed the wK and bK, there are 62 squares left for the wR Mapping
        // its square from 0..63 to available squares 0..61 can be done like:
        //
        //   wR -= (wR > wK) + (wR > bK);
        //
        // In words: if wR "comes later" than wK, we deduct 1, and the same if
        // wR "comes later" than bK. In case of two same pieces like KRRvK we
        // want to place the two Rs "together". If we have 62 squares left, we
        // can place two Rs "together" in 62 * 61 / 2 ways (we divide by 2
        // because rooks can be swapped and still get the same position.)
        //
        // In case we have at least 3 unique pieces (including kings) we
        // encode them together.
        let leading = if entry.has_unique_pieces {
            let s2 = squares[2].idx();
            let adjust1 = i32::from(s1 > s0);
            let adjust2 = i32::from(s2 > s0) + i32::from(s2 > s1);
            let (s1i, s2i) = (s1 as i32, s2 as i32);

            if off(s0) {
                // First piece is below a1-h8 diagonal. MapA1D1D4[] maps the
                // b1-d1-d3 triangle to 0...5. There are 63 squares for second
                // piece and 62 (mapped to 0...61) for the third.
                (MAPS.a1d1d4[s0] * 63 + (s1i - adjust1)) * 62 + s2i - adjust2
            } else if off(s1) {
                // First piece is on a1-h8 diagonal, second below: map this
                // occurrence to 6 to differentiate from the above case,
                // rank_of() maps a1-d4 diagonal to 0...3 and finally
                // MapB1H1H7[] maps the b1-h1-h7 triangle to 0..27.
                (6 * 63 + rank(s0) * 28 + MAPS.b1h1h7[s1]) * 62 + s2i - adjust2
            } else if off(s2) {
                // First two pieces are on a1-h8 diagonal, third below
                6 * 63 * 62
                    + 4 * 28 * 62
                    + rank(s0) * 7 * 28
                    + (rank(s1) - adjust1) * 28
                    + MAPS.b1h1h7[s2]
            } else {
                // All 3 pieces on the diagonal a1-h8
                6 * 63 * 62
                    + 4 * 28 * 62
                    + 4 * 7 * 28
                    + rank(s0) * 7 * 6
                    + (rank(s1) - adjust1) * 6
                    + (rank(s2) - adjust2)
            }
        } else {
            // We don't have at least 3 unique pieces, like in KRRvKBB, just
            // map the kings.
            kings_index(s0, s1, V::SYZYGY.connected_kings)
        };

        idx = leading as u64;
    }

    idx = idx.wrapping_mul(d.group_idx[0]);
    let mut group_start = d.group_len[0];

    // Encode remaining pawns and then pieces according to square, in
    // ascending order
    let mut remaining_pawns = entry.has_pawns && entry.pawn_count[1] != 0;
    let mut next = 1;

    while d.group_len[next] != 0 {
        let group_len = d.group_len[next];
        squares[group_start..group_start + group_len].sort_unstable();
        let mut n: u64 = 0;

        // Map down a square if "comes later" than a square in the previous
        // groups (similar to what was done earlier for leading group pieces).
        for i in 0..group_len {
            let s = squares[group_start + i];
            let adjust = squares[..group_start]
                .iter()
                .filter(|&&prev| s > prev)
                .count();
            n += MAPS.binomial[i + 1][s.idx() - adjust - 8 * usize::from(remaining_pawns)] as u64;
        }

        remaining_pawns = false;
        idx = idx.wrapping_add(n.wrapping_mul(d.group_idx[next]));
        group_start += group_len;
        next += 1;
    }

    // Now that we have the index, decompress the pair and get the score
    map_score::<T>(
        table,
        d,
        decompress_pairs(d, table.mapping.bytes(), idx),
        wdl,
    )
}

// In Recursive Pairing each symbol represents a pair of children symbols. So
// read d->btree[] symbols data and expand each one in his left and right child
// symbol until reaching the leaves that represent the symbol value.
const WHITE_COLOUR: u8 = 0;
const GREY_COLOUR: u8 = 1;
const BLACK_COLOUR: u8 = 2;

fn set_symlen(d: &mut PairsData, s: usize, colour: &mut [u8], cyclic: &mut bool) -> u8 {
    colour[s] = GREY_COLOUR;
    let sr = d.right(s);

    if sr == 0xFFF {
        colour[s] = BLACK_COLOUR;
        return 0;
    }

    let sl = d.left(s);

    if colour[sl] == GREY_COLOUR || colour[sr] == GREY_COLOUR {
        *cyclic = true;
        colour[s] = BLACK_COLOUR;
        return 0;
    }

    for child in [sl, sr] {
        if colour[child] == WHITE_COLOUR {
            let len = set_symlen(d, child, colour, cyclic);
            if let Some(l) = d.symlen.get_mut(child) {
                *l = len;
            }
        }
    }

    colour[s] = BLACK_COLOUR;
    (d.symlen(sl) + d.symlen(sr) + 1) as u8
}

fn set_sizes(d: &mut PairsData, bytes: &[u8], mut data: usize, end: usize) -> Option<usize> {
    if !fits(data, 1, 1, end) {
        return None;
    }

    d.flags = bytes[data];
    data += 1;

    if d.flags & SINGLE_VALUE != 0 {
        d.blocks_num = 0;
        d.block_length_size = 0;
        d.span = 0;
        d.sparse_index_size = 0;

        if !fits(data, 1, 1, end) {
            return None;
        }

        // Here we store the single value
        d.min_sym_len = bytes[data];
        data += 1;
        return Some(data);
    }

    // groupLen[] is a zero-terminated list of group lengths, the last
    // groupIdx[] element stores the biggest index that is the tb size.
    let last = d.group_len[..TBPIECES]
        .iter()
        .position(|&l| l == 0)
        .unwrap_or(TBPIECES);
    let tb_size = d.group_idx[last];

    if !fits(data, 9, 1, end) {
        return None;
    }

    if bytes[data] >= 64 || bytes[data + 1] >= 64 {
        return None;
    }

    d.sizeof_block = 1 << bytes[data];
    d.span = 1 << bytes[data + 1];
    data += 2;
    // Round up
    d.sparse_index_size = (tb_size.wrapping_add(d.span as u64 - 1) / d.span as u64) as usize;
    let padding = bytes[data];
    data += 1;
    d.blocks_num = u32_le(bytes, data);
    data += 4;
    // Padded to ensure SparseIndex[] does not point out of range.
    d.block_length_size = d.blocks_num.wrapping_add(u32::from(padding));
    d.max_sym_len = bytes[data];
    d.min_sym_len = bytes[data + 1];
    data += 2;

    if d.min_sym_len == 0 || d.min_sym_len > d.max_sym_len || d.max_sym_len >= 64 {
        return None;
    }

    // Values are stored in blocks: a table without any is corrupt, and
    // Stockfish would read before its block lengths.
    if d.block_length_size == 0 {
        return None;
    }

    d.lowest_sym = data;
    let base64_size = usize::from(d.max_sym_len - d.min_sym_len) + 1;
    d.base64 = vec![0; base64_size];

    if !fits(data, base64_size as u64, 2, end) {
        return None;
    }

    // See https://en.wikipedia.org/wiki/Huffman_coding
    // The canonical code is ordered such that longer symbols (in terms of the
    // number of bits of their Huffman code) have a lower numeric value, so
    // that d->lowestSym[i] >= d->lowestSym[i+1] (when read as LittleEndian).
    // Starting from this we compute a base64[] table indexed by symbol length
    // and containing 64 bit values so that d->base64[i] >= d->base64[i+1].
    for i in (0..base64_size - 1).rev() {
        d.base64[i] = d.base64[i + 1]
            .wrapping_add(u64::from(u16_le(bytes, d.lowest_sym + 2 * i)))
            .wrapping_sub(u64::from(u16_le(bytes, d.lowest_sym + 2 * (i + 1))))
            / 2;

        if d.base64[i].wrapping_mul(2) < d.base64[i + 1] {
            return None;
        }
    }

    // Now left-shift by an amount so that d->base64[i] gets shifted 1 bit
    // more than d->base64[i+1] and given the above assert condition, we
    // ensure that d->base64[i] >= d->base64[i+1]. Moreover for any symbol s64
    // of length i and right-padded to 64 bits holds d->base64[i-1] >= s64 >=
    // d->base64[i].
    for (i, base) in d.base64.iter_mut().enumerate() {
        // Right-padding to 64 bits
        *base <<= 64 - i - usize::from(d.min_sym_len);
    }

    data += base64_size * 2;

    if !fits(data, 2, 1, end) {
        return None;
    }

    let symlen_size = usize::from(u16_le(bytes, data));
    data += 2;

    // Symbols are 12 bits long: more of them make a corrupt table, which
    // Stockfish would read past its buffers.
    if !fits(data, symlen_size as u64, 3, end) || symlen_size > SYM_COUNT {
        return None;
    }

    d.btree = bytes[data..data + 3 * symlen_size]
        .as_chunks::<3>()
        .0
        .iter()
        .map(|lr| {
            [
                (u16::from(lr[1] & 0xF) << 8) | u16::from(lr[0]),
                (u16::from(lr[2]) << 4) | u16::from(lr[1] >> 4),
            ]
        })
        .collect();

    // The compression scheme used is "Recursive Pairing", that replaces the
    // most frequent adjacent pair of symbols in the source message by a new
    // symbol, reevaluating the frequencies of all of the symbol pairs with
    // respect to the extended alphabet, and then repeating the process.
    // See https://web.archive.org/web/20201106232444/http://www.larsson.dogma.net/dcc99.pdf
    d.symlen = vec![0; symlen_size];
    let mut colour = vec![WHITE_COLOUR; SYM_COUNT];
    let mut cyclic = false;

    for sym in 0..symlen_size {
        if colour[sym] == WHITE_COLOUR {
            d.symlen[sym] = set_symlen(d, sym, &mut colour, &mut cyclic);
        }
    }

    if cyclic {
        return None;
    }

    Some(data + symlen_size * 3 + (symlen_size & 1))
}

fn set_dtz_map(
    e: &TbTable,
    table: &mut MappedTable,
    mut data: usize,
    max_file: usize,
    end: usize,
) -> Option<usize> {
    table.map = data;
    let bytes = table.mapping.bytes();

    for f in 0..=max_file {
        let d = &table.items[0][if e.has_pawns { f } else { 0 }];
        let flags = d.flags;
        let mut map_idx = d.map_idx;

        if flags & MAPPED != 0 {
            if flags & WIDE != 0 {
                // Word alignment, we may have a mixed table
                data += data & 1;
                for idx in &mut map_idx {
                    // Sequence like 3,x,x,x,1,x,0,2,x,x
                    if !fits(data, 2, 1, end) {
                        return None;
                    }

                    *idx = ((data - table.map) / 2 + 1) as u16;

                    let step = 2 * u64::from(u16_le(bytes, data)) + 2;
                    if !fits(data, step, 1, end) {
                        return None;
                    }

                    data += step as usize;
                }
            } else {
                for idx in &mut map_idx {
                    if !fits(data, 1, 1, end) {
                        return None;
                    }

                    *idx = (data - table.map + 1) as u16;

                    let step = u64::from(bytes[data]) + 1;
                    if !fits(data, step, 1, end) {
                        return None;
                    }

                    data += step as usize;
                }
            }
        }

        table.items[0][if e.has_pawns { f } else { 0 }].map_idx = map_idx;
    }

    // Word alignment
    Some(data + (data & 1))
}

// Group together pieces that will be encoded together. The general rule is
// that a group contains pieces of the same type and color. The exception is
// the leading group that, in case of positions without pawns, can be formed by
// 3 different pieces (default) or by the king pair when there is not a unique
// piece apart from the kings. When there are pawns, pawns are always first in
// pieces[].
//
// As example KRKN -> KRK + N, KNNK -> KK + NN, KPPKP -> P + PP + K + K
//
// The actual grouping depends on the TB generator and can be inferred from
// the sequence of pieces in piece[] array. Groups that no table has make a
// corrupt table.
fn set_groups(
    e: &TbTable,
    d: &mut PairsData,
    order: [usize; 2],
    f: usize,
    connected_kings: bool,
) -> Option<()> {
    let mut n = 0;
    let mut first_len: i32 = if e.has_pawns {
        0
    } else if e.has_unique_pieces {
        3
    } else {
        2
    };
    d.group_len[n] = 1;

    // Number of pieces per group is stored in groupLen[], for instance in
    // KRKN the encoder will default on '111', so groupLen[] will be (3, 1).
    for i in 1..e.piece_count {
        first_len -= 1;
        if first_len > 0 || d.pieces[i] == d.pieces[i - 1] {
            d.group_len[n] += 1;
        } else {
            n += 1;
            d.group_len[n] = 1;
        }
    }

    n += 1;
    d.group_len[n] = 0; // Zero-terminated

    if d.group_len[..n]
        .iter()
        .any(|&len| len >= MAPS.binomial.len())
    {
        return None;
    }

    // The sequence in pieces[] defines the groups, but not the order in which
    // they are encoded. If the pieces in a group g can be combined on the
    // board in N(g) different ways, then the position encoding will be of the
    // form:
    //
    //           g1 * N(g2) * N(g3) + g2 * N(g3) + g3
    //
    // This ensures unique encoding for the whole position. The order of the
    // groups is a per-table parameter and could not follow the canonical
    // leading pawns/pieces -> remaining pawns -> remaining pieces. In
    // particular the first group is at order[0] position and the remaining
    // pawns, when present, are at order[1] position.
    let pp = e.has_pawns && e.pawn_count[1] != 0; // Pawns on both sides
    let mut next = if pp { 2 } else { 1 };
    let mut free_squares =
        64usize.checked_sub(d.group_len[0] + if pp { d.group_len[1] } else { 0 })?;
    let mut idx: u64 = 1;

    let mut k = 0;
    while next < n || k == order[0] || k == order[1] {
        if k == order[0] {
            // Leading pawns or pieces
            d.group_idx[0] = idx;
            idx = idx.wrapping_mul(if e.has_pawns {
                *MAPS.lead_pawns_size.get(d.group_len[0])?.get(f)? as u64
            } else if e.has_unique_pieces {
                31332
            } else if connected_kings {
                518
            } else {
                462
            });
        } else if k == order[1] {
            // Remaining pawns
            d.group_idx[1] = idx;
            idx = idx.wrapping_mul(
                MAPS.binomial[d.group_len[1]][48usize.checked_sub(d.group_len[0])?] as u64,
            );
        } else {
            // Remaining pieces
            if next > TBPIECES {
                return None;
            }
            d.group_idx[next] = idx;
            idx = idx.wrapping_mul(MAPS.binomial[d.group_len[next]][free_squares] as u64);
            free_squares = free_squares.checked_sub(d.group_len[next])?;
            next += 1;
        }
        k += 1;
    }

    d.group_idx[n] = idx;
    Some(())
}

// Populate entry's PairsData records with data from the just memory-mapped
// file. Called at first access.
fn set<const T: usize>(e: &TbTable, table: &mut MappedTable, connected_kings: bool) -> Option<()> {
    const SPLIT: u8 = 1;
    const HAS_PAWNS: u8 = 2;

    // Skip the magic
    let mut data = 4;
    let end = table.mapping.len;
    let sides = if T == WDL && e.key != e.key2 { 2 } else { 1 };
    let max_file = if e.has_pawns {
        FILE_D as usize
    } else {
        FILE_A as usize
    };

    {
        let bytes = table.mapping.bytes();
        if !fits(data, 1, 1, end) {
            return None;
        }

        if e.has_pawns != (bytes[data] & HAS_PAWNS != 0)
            || (e.key != e.key2) != (bytes[data] & SPLIT != 0)
        {
            return None;
        }
    }

    data += 1; // First byte stores flags

    let pp = e.has_pawns && e.pawn_count[1] != 0; // Pawns on both sides

    debug_assert!(!pp || e.pawn_count[0] != 0);

    for f in 0..=max_file {
        for i in 0..sides {
            *table.get_mut(sides, e.has_pawns, i, f) = PairsData::default();
        }

        let bytes = table.mapping.bytes();
        if !fits(data, 1 + u64::from(pp) + e.piece_count as u64, 1, end) {
            return None;
        }

        let order = [
            [
                usize::from(bytes[data] & 0xF),
                if pp {
                    usize::from(bytes[data + 1] & 0xF)
                } else {
                    0xF
                },
            ],
            [
                usize::from(bytes[data] >> 4),
                if pp {
                    usize::from(bytes[data + 1] >> 4)
                } else {
                    0xF
                },
            ],
        ];
        data += 1 + usize::from(pp);

        let mut pieces = [[NO_PIECE; TBPIECES]; 2];
        for k in 0..e.piece_count {
            for (i, side_pieces) in pieces.iter_mut().enumerate().take(sides) {
                side_pieces[k] = Piece(if i != 0 {
                    bytes[data] >> 4
                } else {
                    bytes[data] & 0xF
                });
            }
            data += 1;
        }

        for (i, side_pieces) in pieces.iter().enumerate().take(sides) {
            let d = table.get_mut(sides, e.has_pawns, i, f);
            d.pieces = *side_pieces;
            set_groups(e, d, order[i], f, connected_kings)?;
        }
    }

    data += data & 1; // Word alignment

    for f in 0..=max_file {
        for i in 0..sides {
            let mut d = std::mem::take(table.get_mut(sides, e.has_pawns, i, f));
            let next = set_sizes(&mut d, table.mapping.bytes(), data, end);
            *table.get_mut(sides, e.has_pawns, i, f) = d;
            data = next?;
        }
    }

    if T == DTZ {
        data = set_dtz_map(e, table, data, max_file, end)?;
    }

    for f in 0..=max_file {
        for i in 0..sides {
            let d = table.get_mut(sides, e.has_pawns, i, f);
            d.sparse_index = data;

            if !fits(data, d.sparse_index_size as u64, 6, end) {
                return None;
            }

            data += d.sparse_index_size * 6;
        }
    }

    for f in 0..=max_file {
        for i in 0..sides {
            let d = table.get_mut(sides, e.has_pawns, i, f);
            d.block_length = data;

            if !fits(data, u64::from(d.block_length_size), 2, end) {
                return None;
            }

            data += d.block_length_size as usize * 2;
        }
    }

    for f in 0..=max_file {
        for i in 0..sides {
            data = (data + 0x3F) & !0x3F; // 64 byte alignment
            let d = table.get_mut(sides, e.has_pawns, i, f);
            d.data = data;

            if !fits(data, u64::from(d.blocks_num), d.sizeof_block as u64, end) {
                return None;
            }

            data += d.blocks_num as usize * d.sizeof_block;
            d.data_end = data;
        }
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Two kings' squares as a probe mirrors them: the first into the
    /// a1-d1-d4 triangle and, when it is on the a1-h8 diagonal, the second
    /// not above it.
    fn mirrored(mut s0: usize, mut s1: usize) -> (usize, usize) {
        if s0 & 7 > 3 {
            (s0, s1) = (s0 ^ 7, s1 ^ 7);
        }
        if s0 >> 3 > 3 {
            (s0, s1) = (s0 ^ 56, s1 ^ 56);
        }
        let flip = |s: usize| ((s >> 3) | (s << 3)) & 63;
        if off_a1h8(s0) > 0 {
            (flip(s0), flip(s1))
        } else if off_a1h8(s0) == 0 && off_a1h8(s1) > 0 {
            (s0, flip(s1))
        } else {
            (s0, s1)
        }
    }

    /// Every placement of the kings, next to each other only when they may
    /// be, takes one of the indices, which only its mirror images share.
    #[test]
    fn kings_take_each_index_once() {
        for (connected_kings, pairs) in [(false, 462), (true, 518)] {
            let mut placements = HashMap::new();
            for k0 in 0..SQUARE_NB {
                for k1 in 0..SQUARE_NB {
                    let near = (k0 & 7).abs_diff(k1 & 7) <= 1 && (k0 >> 3).abs_diff(k1 >> 3) <= 1;
                    if k0 == k1 || (near && !connected_kings) {
                        continue;
                    }
                    let (s0, s1) = mirrored(k0, k1);
                    let index = kings_index(s0, s1, connected_kings);
                    assert!((0..pairs).contains(&index), "{k0} {k1}: {index}");
                    assert_eq!(*placements.entry(index).or_insert((s0, s1)), (s0, s1));
                }
            }
            assert_eq!(placements.len(), pairs as usize);
        }
    }
}
