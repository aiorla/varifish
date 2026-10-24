# Builds varifish for the ARCH values of Stockfish's Makefile, so that both
# build alike from the same command:
#
#     make build ARCH=x86-64-bmi2
#
# The binary is copied to ./varifish, as Stockfish's is to ./stockfish, with
# the default network embedded, which `make net` first downloads into nets/.
# TARGET cross-compiles for a Rust target, as in
#
#     make build ARCH=armv8 TARGET=aarch64-unknown-linux-gnu
#
# The WebAssembly ARCH values build ./varifish.wasm for web pages, without
# the network, which pages fetch: with SIMD, and with relaxed SIMD's dot
# product for the browsers that have it. They need Rust's
# wasm32-wasip1-threads target, as `rustup target add` installs it.

ARCH ?= native
CARGO ?= cargo

# The x86-64 builds name their instruction sets on the generic x86-64, as
# Stockfish's builds name theirs without -march. The sets are those of
# x86-64-v2, of x86-64-v3 without FMA, as Stockfish builds avx2 and bmi2
# (where its compiler has FMA, it fuses multiply-adds, and varifish fuses
# the same ones), of x86-64-v4, and of Ice Lake.
X86_64_V2 = +cmpxchg16b,+popcnt,+sse3,+sse4.1,+sse4.2,+ssse3
X86_64_V3 = $(X86_64_V2),+avx,+avx2,+bmi1,+bmi2,+f16c,+lzcnt,+movbe,+xsave
X86_64_V4 = $(X86_64_V3),+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+fma
X86_64_ICL = $(X86_64_V4),+adx,+aes,+avx512bitalg,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vnni,+avx512vpopcntdq,+gfni,+pclmulqdq,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsavec,+xsaveopt,+xsaves
X86_64_GENERIC = -C target-cpu=x86-64 -C target-feature=

ifeq ($(ARCH),native)
	RUSTFLAGS_ARCH = -C target-cpu=native
else ifeq ($(ARCH),x86-64)
	RUSTFLAGS_ARCH = -C target-cpu=x86-64
else ifeq ($(ARCH),x86-64-sse41-popcnt)
	RUSTFLAGS_ARCH = $(X86_64_GENERIC)$(X86_64_V2)
else ifneq ($(filter $(ARCH),x86-64-avx2 x86-64-bmi2),)
	RUSTFLAGS_ARCH = $(X86_64_GENERIC)$(X86_64_V3)
else ifeq ($(ARCH),x86-64-avx512)
	RUSTFLAGS_ARCH = $(X86_64_GENERIC)$(X86_64_V4)
else ifeq ($(ARCH),x86-64-vnni512)
	RUSTFLAGS_ARCH = $(X86_64_GENERIC)$(X86_64_V4),+avx512vnni
else ifeq ($(ARCH),x86-64-avx512icl)
	RUSTFLAGS_ARCH = $(X86_64_GENERIC)$(X86_64_ICL)
else ifeq ($(ARCH),apple-silicon)
	RUSTFLAGS_ARCH = -C target-cpu=apple-m1
else ifeq ($(ARCH),armv8)
	RUSTFLAGS_ARCH = -C target-cpu=generic
else ifeq ($(ARCH),armv8-dotprod)
	RUSTFLAGS_ARCH = -C target-cpu=generic -C target-feature=+dotprod
else ifeq ($(ARCH),wasm32)
	RUSTFLAGS_ARCH = -C target-feature=+simd128 $(RUSTFLAGS_WASM)
	TARGET = wasm32-wasip1-threads
else ifeq ($(ARCH),wasm32-relaxed-simd)
	RUSTFLAGS_ARCH = -C target-feature=+simd128,+relaxed-simd $(RUSTFLAGS_WASM)
	TARGET = wasm32-wasip1-threads
else
	$(error Unsupported ARCH '$(ARCH)'; see 'make help')
endif

# A WebAssembly build may use all 4 GB of memory; the page that starts it
# gives it a memory of the size it chooses.
RUSTFLAGS_WASM = -C link-arg=--max-memory=4294967296

# Windows binaries, built there or cross-compiled, end in .exe, and carry
# their C runtime, which not every Windows has.
ifneq ($(findstring windows,$(TARGET)),)
	EXE_SUFFIX = .exe
else ifeq ($(TARGET)$(OS),Windows_NT)
	EXE_SUFFIX = .exe
endif
ifeq ($(EXE_SUFFIX),.exe)
	RUSTFLAGS_ARCH += -C target-feature=+crt-static
endif

ifeq ($(TARGET),wasm32-wasip1-threads)
	EXE = varifish.wasm
else
	EXE = varifish$(EXE_SUFFIX)
endif

# Each ARCH builds in its own directory, so switching does not rebuild all.
# The build names its Rust target, the machine's own without TARGET, so that
# cargo gives RUSTFLAGS to varifish alone: else build.rs gets them too, and
# it runs where it is built, which may lack the ARCH's instructions, as a
# runner without AVX-512 lacks those of x86-64-avx512.
TARGET_DIR = target/$(ARCH)
BUILD_TARGET = $(or $(TARGET),$(shell $(CARGO) -vV | sed -n 's/^host: //p'))
CARGO_BUILD = RUSTFLAGS="$(RUSTFLAGS_ARCH)" $(CARGO) build --release --locked --target-dir $(TARGET_DIR) --target $(BUILD_TARGET)
RELEASE_DIR = $(TARGET_DIR)/$(BUILD_TARGET)/release
ifdef FEATURES
	CARGO_BUILD += --features $(FEATURES)
endif

# The default network is named in src/evaluate.rs after the first 12 digits
# of its SHA-256, which the download is checked against.
NET := $(shell sed -n 's/^pub const EVAL_FILE_DEFAULT_NAME: &str = "\(nn-[0-9a-f]*\.nnue\)";$$/\1/p' src/evaluate.rs)
NET_DIGEST := $(patsubst nn-%.nnue,%,$(NET))
NET_URLS := https://tests.stockfishchess.org/api/nn/$(NET) \
	https://github.com/official-stockfish/networks/raw/master/$(NET) \
	https://media.githubusercontent.com/media/official-stockfish/networks/master/$(NET)

.PHONY: build bolt evalspeed tablebases net atomic-net clean help

# The WebAssembly builds embed no network, so they download none.
ifneq ($(TARGET),wasm32-wasip1-threads)
build: net
endif
build:
	$(CARGO_BUILD)
	cp $(RELEASE_DIR)/$(EXE) $(EXE)

# `make bolt` builds a Linux binary as `make build` does, then has BOLT lay
# its code out by a profile of both variants' benches: chess's bench, and
# Atomic's bench 16 1 17 on Atomic's network, which `make atomic-net`
# downloads into nets/. It needs llvm-bolt and merge-fdata, named as LLVM
# 20's bolt package names them. RUN_PREFIX runs the profiled binary under
# another program, as Intel's SDE runs one whose instructions the machine
# lacks, as in Stockfish's profile-build.
LLVM_BOLT ?= llvm-bolt-20
MERGE_FDATA ?= merge-fdata-20
BOLT_DIR = target/$(ARCH)-bolt
BOLT_FLAGS = -reorder-blocks=ext-tsp -reorder-functions=cdsort -split-functions \
	-split-all-cold -split-eh -dyno-stats -icf=all -use-gnu-stack

# BOLT reads the linker's relocations, which a build of its own keeps.
bolt: TARGET_DIR = target/$(ARCH)-relocs
bolt: RUSTFLAGS_ARCH += -C link-arg=-Wl,--emit-relocs
bolt: net atomic-net
	$(CARGO_BUILD)
	rm -rf $(BOLT_DIR)
	mkdir -p $(BOLT_DIR)
	$(LLVM_BOLT) $(RELEASE_DIR)/$(EXE) -instrument -o $(BOLT_DIR)/instrumented \
		--instrumentation-file=$(CURDIR)/$(BOLT_DIR)/profile.fdata \
		--instrumentation-file-append-pid > $(BOLT_DIR)/instrument.log
	printf 'bench\nquit\n' | $(RUN_PREFIX) $(BOLT_DIR)/instrumented > $(BOLT_DIR)/chess.txt 2>&1
	printf 'setoption name UCI_Variant value atomic\nsetoption name EvalFile value %s\nbench 16 1 17\nquit\n' \
		nets/$(ATOMIC_NET) | $(RUN_PREFIX) $(BOLT_DIR)/instrumented > $(BOLT_DIR)/atomic.txt 2>&1
	$(MERGE_FDATA) $(BOLT_DIR)/profile.fdata.*.fdata > $(BOLT_DIR)/merged.fdata 2> $(BOLT_DIR)/merge.log
	$(LLVM_BOLT) $(RELEASE_DIR)/$(EXE) -o $(EXE) \
		-data=$(BOLT_DIR)/merged.fdata $(BOLT_FLAGS) > $(BOLT_DIR)/optimize.log

# The examples: the evaluation timed along random walks, and the tablebases'
# probes
evalspeed: net
	$(CARGO_BUILD) --example evalspeed
	cp $(RELEASE_DIR)/examples/evalspeed$(EXE_SUFFIX) evalspeed$(EXE_SUFFIX)

tablebases:
	$(CARGO_BUILD) --example tablebases
	cp $(RELEASE_DIR)/examples/tablebases$(EXE_SUFFIX) tablebases$(EXE_SUFFIX)

# Atomic's network, which Atomic-Stockfish's v1.0 release has, checked
# against its SHA-256: Atomic's builds embed none, and `make bolt` profiles
# Atomic's bench on it.
ATOMIC_NET = atomic_run3b_e202_l05.nnue
ATOMIC_NET_SHA256 = 99dc67eabf26a64faeeca3a88b4c38597a840b8d4a874b9f2cf658c6f92a04a6
ATOMIC_NET_URL = https://github.com/Belzedar94/Atomic-Stockfish/releases/download/v1.0/atomic-run3b.nnue

atomic-net: nets/$(ATOMIC_NET)

nets/$(ATOMIC_NET):
	@echo "Downloading $(ATOMIC_NET_URL)"
	@if curl -fsSL -o $@.part "$(ATOMIC_NET_URL)" && \
		[ "$$( (sha256sum $@.part 2>/dev/null || shasum -a 256 $@.part) | cut -c1-64)" = "$(ATOMIC_NET_SHA256)" ]; then \
		mv $@.part $@; \
	else \
		rm -f $@.part; echo "Failed to download $(ATOMIC_NET)" >&2; exit 1; \
	fi

net: nets/$(NET)

nets/$(NET):
	@for url in $(NET_URLS); do \
		echo "Downloading $$url"; \
		if curl -fsSL -o $@.part "$$url" && \
			[ "$$( (sha256sum $@.part 2>/dev/null || shasum -a 256 $@.part) | cut -c1-12)" = "$(NET_DIGEST)" ]; then \
			mv $@.part $@; exit 0; \
		fi; \
	done; \
	rm -f $@.part; echo "Failed to download $(NET)" >&2; exit 1

clean:
	$(CARGO) clean
	rm -f varifish varifish.exe varifish.wasm evalspeed evalspeed.exe tablebases tablebases.exe

help:
	@echo "make net: download the default network into nets/"
	@echo "make build ARCH=<arch>, where <arch> is one of:"
	@echo "  native (the default), x86-64, x86-64-sse41-popcnt, x86-64-avx2,"
	@echo "  x86-64-bmi2, x86-64-avx512, x86-64-vnni512, x86-64-avx512icl,"
	@echo "  apple-silicon, armv8, armv8-dotprod, wasm32, wasm32-relaxed-simd"
	@echo "make bolt ARCH=<arch>: on Linux, the build laid out by BOLT from a profile"
	@echo "  of both variants' benches (needs LLVM's bolt; RUN_PREFIX runs the profile)"
	@echo "make atomic-net: download Atomic's network into nets/"
	@echo "make evalspeed ARCH=<arch>: the example that times the evaluation"
	@echo "make tablebases ARCH=<arch>: the example that prints the tablebases' probes"
	@echo "TARGET=<Rust target> cross-compiles, as TARGET=aarch64-unknown-linux-gnu"
	@echo "FEATURES=tune builds the search constants src/tune.rs lists as options"
