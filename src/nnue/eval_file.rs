//! The networks varifish evaluates with, one per variant, each kept as
//! Stockfish keeps its own: the network last loaded, the name it was loaded
//! by, and the description in its file. EvalFile names a network of either
//! variant, which its file's header tells apart, so that the options of
//! Stockfish and of Atomic-Stockfish both work.
//!
//! A translation into Rust, modified for varifish in 2026, of the network's
//! parts of Stockfish's `nnue/network.cpp` and `engine.cpp`: Stockfish, a UCI
//! chess playing engine derived from Glaurung 2.1, Copyright (C) 2004-2026
//! The Stockfish developers (see AUTHORS), under the GNU General Public
//! License, version 3 or later.

use std::fs;
use std::path::{Path, PathBuf};

use super::atomic::network::Network as AtomicNetwork;
use super::network::Network as ChessNetwork;
use crate::memory::LargePageBox;

/// A network as its file holds it.
pub trait NetworkFile: Send + Sync + Sized + 'static {
    /// The file loaded when EvalFile is empty or names it
    const DEFAULT_NAME: &'static str;
    /// What the evaluation is called when verified
    const KIND: &'static str;

    /// The network in the bytes of a network file, and the file's
    /// description, or None if the bytes are not a network of this kind.
    fn read(bytes: &[u8]) -> Option<(LargePageBox<Self>, Vec<u8>)>;

    /// The network's file, with the description given.
    fn write(&self, description: &[u8]) -> Vec<u8>;

    /// Where the default network can be downloaded
    fn download_url() -> String;

    /// The size in MiB and the dimensions, as verify reports them.
    fn size_and_dimensions() -> String;
}

/// A network, the name it was loaded by, and the description in its file.
pub struct EvalFile<N: NetworkFile> {
    network: Option<LargePageBox<N>>,
    // Selected net name, either via uci option or default
    current: Option<PathBuf>,
    // Net description extracted from the net file
    net_description: Vec<u8>,
}

impl<N: NetworkFile> Default for EvalFile<N> {
    fn default() -> Self {
        EvalFile {
            network: None,
            current: None,
            net_description: Vec::new(),
        }
    }
}

impl<N: NetworkFile> EvalFile<N> {
    /// The network, once one is loaded.
    #[inline(always)]
    pub fn network(&self) -> Option<&N> {
        self.network.as_deref()
    }

    /// The name EvalFile gives, the default network's if it is empty.
    fn requested(eval_file_path: &Path) -> &Path {
        if eval_file_path.as_os_str().is_empty() {
            Path::new(N::DEFAULT_NAME)
        } else {
            eval_file_path
        }
    }

    /// Loads the network `eval_file_path` names, or the default one if it
    /// is empty, unless it is the one loaded: the default network embedded
    /// in the binary, or else the file at that path, relative to the
    /// working directory or else to `root_directory`. The network loaded
    /// stays if none of these is a network.
    pub fn load(&mut self, root_directory: &Path, eval_file_path: &Path, embedded: &[u8]) {
        let eval_file_path = Self::requested(eval_file_path);

        if !self.is_current(eval_file_path) && eval_file_path == Path::new(N::DEFAULT_NAME) {
            self.load_bytes(eval_file_path, embedded);
        }

        for directory in [Path::new(""), root_directory] {
            if !self.is_current(eval_file_path) {
                // A file that cannot be read loads nothing.
                if let Ok(bytes) = fs::read(directory.join(eval_file_path)) {
                    self.load_bytes(eval_file_path, &bytes);
                }
            }
        }
    }

    /// A network of zeros, as if loaded from the default network's file.
    #[cfg(test)]
    pub fn with_network(network: LargePageBox<N>) -> EvalFile<N> {
        EvalFile {
            network: Some(network),
            current: Some(PathBuf::from(N::DEFAULT_NAME)),
            net_description: Vec::new(),
        }
    }

    fn is_current(&self, path: &Path) -> bool {
        self.current.as_deref() == Some(path)
    }

    /// Loads the network in `bytes`, if they are one, by the name `name`.
    fn load_bytes(&mut self, name: &Path, bytes: &[u8]) {
        if let Some((network, description)) = N::read(bytes) {
            self.network = Some(network);
            self.current = Some(name.to_path_buf());
            self.net_description = description;
        }
    }

    /// Writes the network to `filename`, or for the default network to its
    /// name, and returns what to tell the user.
    pub fn save(&self, filename: Option<&Path>) -> String {
        let (Some(network), Some(current)) = (&self.network, &self.current) else {
            return "Failed to export a net. No network file is currently loaded. \
                    Please load a network file first."
                .to_string();
        };

        let default_name = Path::new(N::DEFAULT_NAME);
        if filename.is_none() && current != default_name {
            return "Failed to export a net. A non-embedded net can only be \
                    saved if the filename is specified"
                .to_string();
        }

        let actual_filename = filename.unwrap_or(default_name);
        let saved = fs::write(actual_filename, network.write(&self.net_description)).is_ok();

        if saved {
            format!(
                "Network saved successfully to {}",
                actual_filename.display()
            )
        } else {
            "Failed to export a net".to_string()
        }
    }

    /// The lines explaining that the network `path` names is not loaded,
    /// after which the engine must exit.
    fn not_loaded(path: &Path) -> String {
        let msg1 = "Network evaluation parameters compatible with the engine must be available.";
        let msg2 = format!(
            "The network file {} was not loaded successfully.",
            path.display()
        );
        let msg3 = "The UCI option EvalFile might need to specify the full path, \
                    including the directory name, to the network file.";
        let url = N::download_url();
        let msg4 = format!("The default net can be downloaded from: {url}");
        // A download named otherwise, as Atomic-Stockfish's, must be renamed.
        let rename = if url.ends_with(&format!("/{}", N::DEFAULT_NAME)) {
            String::new()
        } else {
            format!(
                "ERROR: Save it as {} next to the engine.\n",
                N::DEFAULT_NAME
            )
        };
        let msg5 = "The engine will be terminated now.";

        format!(
            "ERROR: {msg1}\nERROR: {msg2}\nERROR: {msg3}\nERROR: {msg4}\n{rename}ERROR: {msg5}\n"
        )
    }

    /// The line describing the network loaded.
    fn describe(&self) -> String {
        let current = self.current.as_deref().unwrap_or(Path::new(""));
        format!(
            "NNUE evaluation using {}{} {}",
            N::KIND,
            current.display(),
            N::size_and_dimensions()
        )
    }
}

/// How Atomic is evaluated, as Atomic-Stockfish's "Use NNUE" chooses: by
/// the network as Fairy-Stockfish trained it, or by the network's output
/// alone, for generating training data. Chess always uses its network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UseNnue {
    #[default]
    True,
    Pure,
}

impl UseNnue {
    pub const ALL: [UseNnue; 2] = [UseNnue::True, UseNnue::Pure];

    /// The mode "Use NNUE" names, in any case.
    pub fn from_name(name: &str) -> Option<UseNnue> {
        UseNnue::ALL
            .into_iter()
            .find(|mode| mode.name().eq_ignore_ascii_case(name))
    }

    pub fn name(self) -> &'static str {
        match self {
            UseNnue::True => "true",
            UseNnue::Pure => "pure",
        }
    }
}

/// The network of each variant.
#[derive(Default)]
pub struct Networks {
    pub chess: EvalFile<ChessNetwork>,
    pub atomic: EvalFile<AtomicNetwork>,
}

impl Networks {
    /// Loads the network or networks EvalFile names, as EvalFile::load does,
    /// into the variant whose network the file holds: if EvalFile is empty,
    /// each variant's default network, and otherwise the file named, which
    /// is read once for both.
    pub fn load(&mut self, root_directory: &Path, eval_file_path: &Path, embedded: &[u8]) {
        if eval_file_path.as_os_str().is_empty() {
            self.chess.load(root_directory, eval_file_path, embedded);
            self.atomic.load(root_directory, eval_file_path, &[]);
            return;
        }

        let loaded = |networks: &Networks| {
            networks.chess.is_current(eval_file_path) || networks.atomic.is_current(eval_file_path)
        };

        if !loaded(self) && eval_file_path == Path::new(ChessNetwork::DEFAULT_NAME) {
            self.chess.load_bytes(eval_file_path, embedded);
        }

        for directory in [Path::new(""), root_directory] {
            if !loaded(self) {
                // A file that cannot be read loads nothing.
                if let Ok(bytes) = fs::read(directory.join(eval_file_path)) {
                    self.chess.load_bytes(eval_file_path, &bytes);
                    self.atomic.load_bytes(eval_file_path, &bytes);
                }
            }
        }
    }

    /// Checks that the network EvalFile names is loaded, and that the
    /// variant played, with its network `eval_file`, has one: Ok with a line
    /// describing the network, or Err with the lines explaining what is
    /// missing, after which the engine must exit. When EvalFile is empty or
    /// names the default network of a variant, it names the default network
    /// of the variant played.
    pub fn verify<N: NetworkFile>(
        &self,
        eval_file: &EvalFile<N>,
        eval_file_path: &Path,
    ) -> Result<String, String> {
        let defaults = [ChessNetwork::DEFAULT_NAME, AtomicNetwork::DEFAULT_NAME];
        let requested = if defaults
            .iter()
            .any(|&name| eval_file_path == Path::new(name))
        {
            Path::new(N::DEFAULT_NAME)
        } else {
            EvalFile::<N>::requested(eval_file_path)
        };
        if !self.chess.is_current(requested) && !self.atomic.is_current(requested) {
            return Err(EvalFile::<N>::not_loaded(requested));
        }
        if eval_file.network().is_none() {
            return Err(EvalFile::<N>::not_loaded(Path::new(N::DEFAULT_NAME)));
        }

        Ok(eval_file.describe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(target_os = "wasi", ignore = "WASI has no temporary directory")]
    fn files_load_into_the_variant_of_their_network() {
        let directory =
            std::env::temp_dir().join(format!("varifish-networks-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let chess_file = directory.join("chess.nnue");
        let atomic_file = directory.join("atomic.nnue");
        fs::write(&chess_file, ChessNetwork::zeroed().write(b"chess")).unwrap();
        fs::write(&atomic_file, AtomicNetwork::zeroed().write(b"atomic")).unwrap();

        let mut networks = Networks::default();
        networks.load(&directory, Path::new("atomic.nnue"), &[]);
        assert!(networks.chess.network().is_none());
        assert_eq!(
            networks.atomic.current.as_deref(),
            Some(Path::new("atomic.nnue"))
        );
        assert_eq!(networks.atomic.net_description, b"atomic");

        // The chess network loads beside it; neither file is the other's.
        networks.load(&directory, Path::new("chess.nnue"), &[]);
        assert_eq!(
            networks.chess.current.as_deref(),
            Some(Path::new("chess.nnue"))
        );
        assert_eq!(
            networks.atomic.current.as_deref(),
            Some(Path::new("atomic.nnue"))
        );

        let atomic = &networks.atomic;
        assert!(networks.verify(atomic, Path::new("chess.nnue")).is_ok());
        assert!(networks.verify(atomic, Path::new("atomic.nnue")).is_ok());
        let missing = networks
            .verify(atomic, Path::new("missing.nnue"))
            .unwrap_err();
        assert!(missing.contains("The network file missing.nnue was not loaded successfully."));

        // Without an Atomic network, Atomic cannot be evaluated, whatever
        // EvalFile names.
        let mut chess_only = Networks::default();
        chess_only.load(&directory, Path::new("chess.nnue"), &[]);
        let missing = chess_only
            .verify(&chess_only.atomic, Path::new("chess.nnue"))
            .unwrap_err();
        assert!(missing.contains(&format!(
            "The network file {} was not loaded successfully.",
            AtomicNetwork::DEFAULT_NAME
        )));
        assert!(
            chess_only
                .verify(&chess_only.chess, Path::new("chess.nnue"))
                .unwrap()
                .starts_with("NNUE evaluation using chess.nnue (")
        );

        fs::remove_dir_all(&directory).unwrap();
    }
}
