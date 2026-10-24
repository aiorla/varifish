//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `main.cpp` and of the network embedding in `nnue/network.cpp`: Stockfish,
//! a UCI chess playing engine derived from Glaurung 2.1, Copyright (C)
//! 2004-2026 The Stockfish developers (see AUTHORS), under the GNU General
//! Public License, version 3 or later.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use varifish::uci::{UciEngine, engine_info};

/// The default network, which build.rs finds in nets/, or nothing if it is
/// not there.
static EMBEDDED_NETWORK: &[u8] = include_bytes!(env!("VARIFISH_EMBEDDED_NETWORK"));

/// The directory of the binary that `argv0` names, where network files are
/// also looked for.
fn binary_directory(argv0: &str) -> PathBuf {
    match Path::new(argv0).parent() {
        Some(directory) if !directory.as_os_str().is_empty() => directory.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

fn main() {
    let _ = writeln!(io::stdout(), "{}", engine_info(false));

    let args: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let directory = args
        .first()
        .map_or_else(PathBuf::new, |argv0| binary_directory(argv0));
    let out = Arc::new(Mutex::new(io::stdout()));
    let err = Arc::new(Mutex::new(io::stderr()));
    let mut engine = UciEngine::new(out, err, directory, EMBEDDED_NETWORK);
    let code = engine.run(args.get(1..).unwrap_or_default(), io::stdin().lock());

    // A critical error exits at once, as Stockfish does, with any search
    // still running.
    std::process::exit(code);
}
