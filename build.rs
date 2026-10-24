//! Names the SIMD instructions the build targets with cfgs, after the defines
//! Stockfish's Makefile passes for each ARCH and, for WebAssembly, the ones
//! its compiler adds, and points the environment variable
//! `VARIFISH_EMBEDDED_NETWORK` at the network for the binary to embed: the
//! default network in nets/, where `make net` downloads it, or an empty file
//! when it is not there or the target is WebAssembly, whose pages fetch it
//! instead.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |feature: &str| features.split(',').any(|f| f == feature);
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // Every x86-64 build has SSE2, as Stockfish's define USE_SSE2, and the
    // AVX2 builds also have SSSE3 and SSE4.1, as theirs define USE_SSSE3 and
    // USE_SSE41.
    let use_sse2 = arch == "x86_64" && has("sse2");
    let use_ssse3 = use_sse2 && has("ssse3");
    let use_sse41 = use_ssse3 && has("sse4.1");
    let use_avx2 = use_sse41 && has("avx2");
    let use_avx512 = use_avx2 && has("avx512f") && has("avx512bw") && has("avx512vl");
    let use_vnni = use_avx512 && has("avx512vnni");
    // The instruction sets of Stockfish's x86-64-avx512icl, which Ice Lake
    // and Zen 4 have
    let use_avx512icl = use_vnni
        && [
            "avx512cd",
            "avx512dq",
            "avx512ifma",
            "avx512vbmi",
            "avx512vbmi2",
            "avx512vpopcntdq",
            "avx512bitalg",
            "vpclmulqdq",
            "gfni",
            "vaes",
        ]
        .iter()
        .all(|f| has(f));
    let use_neon = arch == "aarch64" && has("neon");
    let use_neon_dotprod = use_neon && has("dotprod");
    let use_simd128 = arch == "wasm32" && has("simd128");
    let use_relaxed_simd = use_simd128 && has("relaxed-simd");
    let cfgs = [
        ("use_sse2", use_sse2),
        ("use_ssse3", use_ssse3),
        ("use_sse41", use_sse41),
        ("use_avx2", use_avx2),
        ("use_avx512", use_avx512),
        ("use_vnni", use_vnni),
        ("use_avx512icl", use_avx512icl),
        ("use_neon", use_neon),
        ("use_neon_dotprod", use_neon_dotprod),
        ("use_simd128", use_simd128),
        ("use_relaxed_simd", use_relaxed_simd),
        // Derived as in Stockfish's simd.h, affine_transform.h and
        // affine_transform_sparse_input.h, where WebAssembly's SIMD takes the
        // paths of SSSE3's: `sparse_input` is their condition for reading
        // only the non-zero chunks of the first layer's input, which SSE2
        // alone reads too here, though not in Stockfish. Without a dot
        // product of bytes, SSSE3's or relaxed SIMD's, `widening_dot` widens
        // that layer's inputs and weights to 16 bits.
        ("vector", use_sse2 || use_neon || use_simd128),
        ("sparse_input", use_sse2 || use_neon || use_simd128),
        (
            "widening_dot",
            (use_sse2 && !use_ssse3) || (use_simd128 && !use_relaxed_simd),
        ),
        ("use_pair_activations", use_avx2),
        ("use_paired_widening_add", use_ssse3),
        (
            "enable_seq_opt",
            use_ssse3 || use_neon_dotprod || use_simd128,
        ),
    ];
    for (name, enabled) in cfgs {
        println!("cargo::rustc-check-cfg=cfg({name})");
        if enabled {
            println!("cargo::rustc-cfg={name}");
        }
    }

    // The default network is named in src/evaluate.rs, as Stockfish names it
    // in evaluate.h for its Makefile.
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source = fs::read_to_string(root.join("src/evaluate.rs")).unwrap();
    let name = source
        .lines()
        .find_map(|line| {
            line.strip_prefix("pub const EVAL_FILE_DEFAULT_NAME: &str = \"")?
                .strip_suffix("\";")
        })
        .expect("src/evaluate.rs names the default network");

    // Watching the directory rather than the file notices when it appears.
    let nets = root.join("nets");
    let network = nets.join(name);
    let embedded = if network.is_file() && arch != "wasm32" {
        network
    } else {
        let empty = PathBuf::from(env::var("OUT_DIR").unwrap()).join("no-network");
        fs::write(&empty, b"").unwrap();
        empty
    };
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/evaluate.rs");
    println!("cargo::rerun-if-changed={}", nets.display());
    println!(
        "cargo::rustc-env=VARIFISH_EMBEDDED_NETWORK={}",
        embedded.display()
    );
}
