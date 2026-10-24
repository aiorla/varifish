# varifish

varifish is a free UCI chess engine written in Rust, as a translation of
[Stockfish](https://github.com/official-stockfish/Stockfish). In chess it
plays exactly like Stockfish, at nearly the same speed.

It also plays [Atomic](https://lichess.org/variant/atomic), based on
[Atomic-Stockfish](https://github.com/Belzedar94/Atomic-Stockfish), whose
rules, network and tablebase support it uses. Its Atomic search began as a
port of Atomic-Stockfish's and has been improved since, and it now plays
stronger than Atomic-Stockfish. More variants will follow: varifish is
developed for [solvedchess.com](https://solvedchess.com), which works out who
wins in chess variants, Atomic first.

Like Stockfish, varifish has no graphical interface: use it from a chess GUI
that supports UCI engines, or from the command line.

## Download

Each [release](https://github.com/aiorla/varifish/releases) has binaries for
Linux, Windows and macOS:

- Linux and Windows on x86-64: choose the first build your processor
  supports, in this order: `x86-64-avx512icl`, `x86-64-vnni512`,
  `x86-64-avx512`, `x86-64-bmi2`, `x86-64-avx2`, `x86-64-sse41-popcnt`, and
  `x86-64`, which runs on any.
- 64-bit ARM on Linux: `armv8-dotprod` if the processor has dot product
  instructions, or else `armv8`.
- Macs with Apple silicon: `apple-silicon`.

On Linux, varifish needs glibc 2.34 or later: Ubuntu 22.04, Debian 12 or
newer.

varifish also runs in the browser, without installing, at
[solvedchess.com/varifish](https://solvedchess.com/varifish).

## Usage

varifish speaks UCI, with Stockfish's commands and options, and
`UCI_Variant` to choose `chess` or `atomic`. A command given as arguments
runs on its own, as in `./varifish bench`.

To play Atomic, download `atomic-run3b.nnue` from
[Atomic-Stockfish's v1.0 release](https://github.com/Belzedar94/Atomic-Stockfish/releases/tag/v1.0),
save it as `atomic_run3b_e202_l05.nnue` next to the binary (or set
`EvalFile` to its path), and set `UCI_Variant` to `atomic`.

`SyzygyPath` takes Syzygy tablebases: `.rtbw` and `.rtbz` files for chess,
and `.atbw` and `.atbz` files for Atomic, which
[Lichess offers](https://tablebase.lichess.ovh/tables/atomic/).

## Strength

On a Ryzen 7 7700, one thread each:

- In chess, varifish searches the same nodes as Stockfish, at 0.980 to 1.003
  of the speed of Stockfish built as its releases are, and at 1.21 in the
  plain `x86-64` build.
- In Atomic, it scored +15.6 Elo against Atomic-Stockfish at 8+0.08 (2,000
  game pairs) and +11.5 Elo at 40+0.4 (1,000 pairs).

## Building from source

With `make`, `curl` and [Rust](https://rustup.rs):

```sh
make build ARCH=x86-64-bmi2
./varifish bench
```

`ARCH` is `native` if not given, and `make help` lists the others. The build
downloads Stockfish's network and embeds it in the binary.
After `rustup target add wasm32-wasip1-threads`,
`make build ARCH=wasm32-relaxed-simd`, or `ARCH=wasm32`, builds
`varifish.wasm` for web pages, which [web/varifish.js](web/varifish.js) runs
on a cross-origin isolated page. `cargo test` runs the tests.

## License

varifish is free software under the GNU General Public License, version 3 or
later ([LICENSE](LICENSE)). As a translation of Stockfish, and for Atomic of
Atomic-Stockfish, its authors include theirs ([AUTHORS](AUTHORS)). The
network built into the binary is Stockfish's, trained on
[data from Leela Chess Zero](https://storage.lczero.org/files/training_data)
under the [Open Database License](https://opendatacommons.org/licenses/odbl/odbl-10.txt).
