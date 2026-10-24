#!/bin/sh
# Builds varifish for web pages: its two WebAssembly builds, from `make build
# ARCH=wasm32` and `ARCH=wasm32-relaxed-simd`, with the scripts that run them
# (uci-worker.js, varifish.js and varifish-worker.js), in a folder named after
# their contents, so that browsers never mix two builds; and the files the
# engine opens: the Atomic network given and the 3-4 piece tables of wins and
# losses from the tablebases given, if any (1.2 MB). Into web/dist, or into
# the directory given. manifest.json names them all, and a page fetches it
# fresh; each file but the tables has a gzipped copy beside it, for servers
# that send it compressed (the network shrinks from 48 to 27 MB).
#
# With --share, a folder served beside the one built (another engine's build,
# say), a file whose bytes that folder already holds is named there rather
# than copied, so that a browser fetches it once for both pages; one that
# differs is copied. The build says which it shares.
# Usage: web/build.sh --network FILE [--tablebases DIR] [--share DIR] [OUT]
set -e
usage() {
  echo "usage: web/build.sh --network FILE [--tablebases DIR] [--share DIR] [OUT]" >&2
  exit 2
}
# Paths given are taken from where the script is called.
absolute() {
  case $1 in
    /*) echo "$1" ;;
    *) echo "$PWD/$1" ;;
  esac
}
network=
tablebases=
share=
out=
while [ $# -gt 0 ]; do
  case $1 in
    --network | --tablebases | --share)
      [ $# -gt 1 ] || usage
      path=$(absolute "$2")
      case $1 in
        --network) network=$path ;;
        --tablebases) tablebases=$path ;;
        --share) share=${path%/} ;;
      esac
      shift
      ;;
    -*) usage ;;
    *) out=$(absolute "$1") ;;
  esac
  shift
done
cd "$(dirname "$0")/.."
out=${out:-$PWD/web/dist}
out=${out%/}
[ -n "$network" ] || usage
if [ ! -f "$network" ]; then
  echo "no such network: $network" >&2
  exit 1
fi
if [ -n "$tablebases" ] && [ ! -d "$tablebases" ]; then
  echo "no such directory: $tablebases" >&2
  exit 1
fi
# A shared folder's files are named from the build's folder, as its siblings.
if [ -n "$share" ] && { [ ! -d "$share" ] || [ "$(dirname "$share")" != "$(dirname "$out")" ]; }; then
  echo "--share names a folder beside $out" >&2
  exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$@"; }
else
  echo "needs sha256sum or shasum" >&2
  exit 1
fi
digest() { sha256 "$1" | cut -c1-64; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
make build ARCH=wasm32
cp varifish.wasm "$work/varifish-wasm32.wasm"
make build ARCH=wasm32-relaxed-simd
cp varifish.wasm "$work/varifish-wasm32-relaxed-simd.wasm"
cp web/uci-worker.js web/varifish.js web/varifish-worker.js "$work/"
engine=engine-$(cat "$work"/* | sha256 | cut -c1-12)

mkdir -p "$out"
rm -rf "$out"/engine-* "$out"/network-*.nnue* "$out/tables" "$out/manifest.json"
mkdir "$out/$engine"
cp "$work"/* "$out/$engine/"
gzip -9 -k -n "$out/$engine"/*

# The shared folder's files by their SHA-256
if [ -n "$share" ]; then
  find "$share" -type f ! -name '*.gz' -exec sh -c 'for f; do echo "$f"; done' sh {} + |
    while read -r file; do echo "$(digest "$file") ${file#"$share"/}"; done > "$work/shared"
fi
# Prints the URL that the manifest names a file by, from the build's folder:
# the shared folder's file of the same bytes, if any, or else its copy at
# PATH in the build.
place() {
  if [ -n "$share" ]; then
    found=$(grep -m 1 "^$(digest "$1") " "$work/shared" | cut -d' ' -f2-)
    if [ -n "$found" ]; then
      echo "shares ../$(basename "$share")/$found" >&2
      echo "../$(basename "$share")/$found"
      return
    fi
  fi
  mkdir -p "$(dirname "$out/$2")"
  cp "$1" "$out/$2"
  echo "copies $2" >&2
  echo "$2"
}

# The network, under the name its file has, which EvalFile names
network_url=$(place "$network" "network-$(digest "$network" | cut -c1-12).nnue")
case $network_url in
  ../*) ;;
  *) gzip -9 -k -n "$out/$network_url" ;;
esac
network_bytes=$(wc -c < "$network" | tr -d ' ')

tables=
if [ -n "$tablebases" ]; then
  for table in "$tablebases"/*.atbw; do
    name=$(basename "$table")
    pieces=$(printf %s "${name%.*}" | tr -d v | wc -c)
    if [ -f "$table" ] && [ "$pieces" -le 4 ]; then
      url=$(place "$table" "tables/$name")
      tables="$tables${tables:+, }{\"name\": \"$name\", \"url\": \"$url\"}"
    fi
  done
  if [ -z "$tables" ]; then
    echo "no tables of 3 or 4 pieces (.atbw) in $tablebases" >&2
    exit 1
  fi
fi

cat > "$out/manifest.json" <<JSON
{"engine": "$engine/uci-worker.js", "wasm": "$engine/varifish-wasm32.wasm", "relaxed": "$engine/varifish-wasm32-relaxed-simd.wasm", "network": {"name": "$(basename "$network")", "url": "$network_url", "bytes": $network_bytes}, "tables": [$tables]}
JSON
ls -la "$out/$engine"
