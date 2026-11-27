<div align="center">

<h1>varifish</h1>

A free and strong UCI engine for chess and its variants, written in Rust.

[Download](https://github.com/aiorla/varifish/releases/latest)
·
[Try it in your browser](https://solvedchess.com/varifish)
·
[Report a bug](https://github.com/aiorla/varifish/issues)

[![Release](https://img.shields.io/github/v/release/aiorla/varifish?label=release)](https://github.com/aiorla/varifish/releases/latest)
[![License](https://img.shields.io/github/license/aiorla/varifish)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/aiorla/varifish/test.yml?label=tests)](https://github.com/aiorla/varifish/actions/workflows/test.yml)

</div>

## Overview

varifish is a translation of [Stockfish](https://github.com/official-stockfish/Stockfish)
into Rust. In chess it plays exactly like Stockfish, move for move, at nearly
the same speed.

It also plays [Atomic](https://lichess.org/variant/atomic),
[Three-check](https://lichess.org/variant/threeCheck),
[Crazyhouse](https://lichess.org/variant/crazyhouse) and
[Antichess](https://lichess.org/variant/antichess), each with a neural
network of its own, stronger than
[Atomic-Stockfish](https://github.com/Belzedar94/Atomic-Stockfish) in Atomic
and than [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish)
in the other three. More variants will follow: varifish is developed for
[solvedchess.com](https://solvedchess.com), which works out who wins in chess
variants.

varifish has no board of its own: use it from a chess GUI, or try it in your
browser at [solvedchess.com/varifish](https://solvedchess.com/varifish).

## Strength

One thread each, on a Ryzen 7 7700, from balanced openings that strong
players reach on Lichess:

| Variant | varifish against | Time control | Game pairs | Elo |
| --- | --- | --- | --- | --- |
| Atomic | Atomic-Stockfish | 40 s + 0.4 s | 1,000 | +38.7 [+33.6, +43.9] |
| Three-check | Fairy-Stockfish | 8 s + 0.08 s | 1,000 | +183.7 [+168.6, +199.5] |
| Crazyhouse | Fairy-Stockfish | 8 s + 0.08 s | 1,000 | +96.6 [+82.4, +111.0] |
| Antichess | Fairy-Stockfish | 8 s + 0.08 s | 1,000 | +53.6 [+42.6, +64.6] |

Atomic's games were played by varifish 0.2.0, the others by 0.3.0, against
Atomic-Stockfish's master at 63238876 and Fairy-Stockfish's at 52838835.

In chess, varifish plays the same moves as Stockfish, at 96% to 99% of its
speed.

## Getting started

### Download

Download the archive for your computer from the
[latest release](https://github.com/aiorla/varifish/releases/latest):

| Computer | Archive |
| --- | --- |
| Windows or Linux on Intel or AMD | The fastest your processor runs (see below) |
| Mac with Apple silicon | `apple-silicon` |
| Linux on 64-bit ARM | `armv8-dotprod`, or `armv8` on older processors |

> [!TIP]
> On Intel and AMD, the builds from fastest to most compatible are
> `x86-64-avx512icl`, `x86-64-vnni512`, `x86-64-avx512`, `x86-64-bmi2`,
> `x86-64-avx2`, `x86-64-sse41-popcnt` and `x86-64`, which runs on any.
> Most computers from the last ten years run `x86-64-avx2`.

On Linux, varifish needs Ubuntu 22.04, Debian 12 or newer (glibc 2.34).

### Playing chess

Add varifish to any GUI that takes UCI engines, such as
[En Croissant](https://encroissant.org),
[Cute Chess](https://github.com/cutechess/cutechess),
[Nibbler](https://github.com/rooklift/nibbler) or
[Arena](http://www.playwitharena.de). Its network is built in.

### Playing Atomic

1. Download `atomic-b0d279b9e021.nnue` from the
   [networks release](https://github.com/aiorla/varifish/releases/tag/networks)
   and save it next to the program.
2. In a GUI that plays variants, such as
   [Cute Chess](https://github.com/cutechess/cutechess), choose Atomic;
   elsewhere, set the option `UCI_Variant` to `atomic`.

### Playing Three-check

1. Download `3check-ae396b7e8123.nnue` from the
   [networks release](https://github.com/aiorla/varifish/releases/tag/networks)
   and save it next to the program.
2. Choose Three-check in a GUI that plays variants, or set the option
   `UCI_Variant` to `3check`.

Without its network, varifish plays Three-check far weaker, with chess's
network, and says so when it starts to search. Its FENs count the checks each side has still to
give after the en passant square, as Lichess's do (`3+3` at the start), or the
checks given at the end, as XBoard's do (`+0+0`).

### Playing Crazyhouse

1. Download `crazyhouse-b39a7a9b9831.nnue` from the
   [networks release](https://github.com/aiorla/varifish/releases/tag/networks)
   and save it next to the program.
2. Choose Crazyhouse in a GUI that plays variants, or set the option
   `UCI_Variant` to `crazyhouse`.

As in Three-check, varifish plays far weaker without its network. Its FENs accept pockets in brackets (`[QNp]`) or after an
extra slash (`/QNp`), mark promoted pieces with `~`, and write drops in UCI as
`P@e4`.

### Playing Antichess

1. Download `antichess-64def68f336c.nnue` from the
   [networks release](https://github.com/aiorla/varifish/releases/tag/networks)
   and save it next to the program.
2. Choose Antichess in a GUI that plays variants, or set the option
   `UCI_Variant` to `antichess`.

As in the other variants, varifish plays far weaker without its network. It
follows Lichess rules: captures are mandatory, there is no
check or castling, pawns may also promote to a king (`e7e8k`), and losing all
pieces or being stalemated wins.

### Options

varifish takes Stockfish's UCI commands and options. The main ones:

| Option | Default | What it does |
| --- | --- | --- |
| `Threads` | 1 | The cores to search with. |
| `Hash` | 16 | The memory for the search, in MB. |
| `MultiPV` | 1 | The number of best lines to show. |
| `UCI_Variant` | `chess` | `chess`, `atomic`, `3check`, `crazyhouse` or `antichess`. |
| `EvalFile` | the built-in network | Another network, such as a variant's in another folder. |
| `SyzygyPath` | empty | Syzygy tablebases: `.rtbw`/`.rtbz` for chess, `.atbw`/`.atbz` for Atomic, which [Lichess offers](https://tablebase.lichess.ovh/tables/atomic/); none for Three-check, Crazyhouse or Antichess yet. |

A command given on the command line runs on its own, as in `./varifish bench`.

## Building from source

With `make`, `curl` and [Rust](https://rustup.rs):

```sh
make build
./varifish bench
```

`make build` builds for this computer and downloads the chess network into
the program; `make help` lists the other builds, `ARCH=wasm32` for web pages
among them ([web/varifish.js](web/varifish.js) runs it). `cargo test` runs
the tests.

## Acknowledgements

- [Stockfish](https://github.com/official-stockfish/Stockfish), which
  varifish translates, and its developers.
- [Atomic-Stockfish](https://github.com/Belzedar94/Atomic-Stockfish) by
  Belzedar, whose Atomic rules, search and tablebase support varifish's
  Atomic began from, and whose network its own Atomic network grew from.
- [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish),
  from which parts of Atomic's and Antichess's search come, and whose games
  and scores, with networks by nishibuya for Three-check, by Belzedar for
  Crazyhouse and by Fabian Fichter for Antichess, trained varifish's networks
  for those variants.
- [Leela Chess Zero](https://storage.lczero.org/files/training_data), whose
  data, under the
  [Open Database License](https://opendatacommons.org/licenses/odbl/odbl-10.txt),
  trained Stockfish's chess network, which varifish builds in.

## License

varifish is free software under the
[GNU General Public License, version 3](LICENSE) or later. You may use it,
share it, sell it or build on it; whenever you distribute it, include the
license and the source code, or where to find it, and share your changes
under the same license. As a translation of Stockfish and Atomic-Stockfish,
its authors include theirs ([AUTHORS](AUTHORS)).

The variants' networks are varifish's own, under the same license. Atomic's
was trained on games that varifish played against itself, their positions
scored by its searches with a network fine-tuned from Atomic-Stockfish's;
Three-check's, Crazyhouse's and Antichess's on games that Fairy-Stockfish
played and scored, with the networks above, used with their authors'
permission.
