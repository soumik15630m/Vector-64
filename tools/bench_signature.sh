#!/usr/bin/env bash
# Deterministic search signature (Stockfish-style bench check). The engine's
# node count for a fixed depth / threads / hash is a fingerprint of search
# behaviour: if it changes, the search tree changed. Intentional changes must
# update EXPECTED in the same commit; unintended ones are caught here.
#
#   bash tools/bench_signature.sh <path-to-ChessEngine>
#   BENCH_SIG_DEPTH=13 BENCH_SIG_NODES=13410056 bash tools/bench_signature.sh ...

set -euo pipefail

ENGINE="${1:?usage: bench_signature.sh <path-to-ChessEngine>}"
DEPTH="${BENCH_SIG_DEPTH:-13}"
EXPECTED="${BENCH_SIG_NODES:-13410056}"

got=$(printf "setoption name Threads value 1\nsetoption name Hash value 8\nbench %s\nquit\n" "$DEPTH" \
  | "$ENGINE" 2>/dev/null | grep -oE "nodes [0-9]+" | grep -oE "[0-9]+" | head -1)

if [ "${got:-}" = "$EXPECTED" ]; then
  echo "bench signature OK: $got nodes (depth $DEPTH, 1 thread, 8 MB hash)"
else
  echo "BENCH SIGNATURE MISMATCH"
  echo "  expected : $EXPECTED"
  echo "  got      : ${got:-<no output>}"
  echo "Search behaviour changed. If this was intentional, update the expected"
  echo "node count (BENCH_SIG_NODES / the CI value) in the same commit."
  exit 1
fi
