#!/usr/bin/env bash
# Deterministic search signature (Stockfish-style bench check). The engine's
# node count for a fixed depth / threads / hash is a fingerprint of search
# behaviour: if it changes, the search tree changed. The expected value is the
# single source of truth in tools/bench_signature.txt -- intentional changes
# regenerate it (tools/update_bench_signature.sh) in the same commit; unintended
# ones are caught here.
#
#   bash tools/bench_signature.sh <path-to-ChessEngine>
#   BENCH_SIG_DEPTH=13 BENCH_SIG_NODES=5253789 bash tools/bench_signature.sh ...  # override

set -euo pipefail

ENGINE="${1:?usage: bench_signature.sh <path-to-ChessEngine>}"
SIG_FILE="$(dirname "$0")/bench_signature.txt"

DEPTH="${BENCH_SIG_DEPTH:-$(grep -E '^DEPTH=' "$SIG_FILE" | cut -d= -f2)}"
EXPECTED="${BENCH_SIG_NODES:-$(grep -E '^NODES=' "$SIG_FILE" | cut -d= -f2)}"

got=$(printf "setoption name Threads value 1\nsetoption name Hash value 8\nbench %s\nquit\n" "$DEPTH" \
  | "$ENGINE" 2>/dev/null | grep -oE "nodes [0-9]+" | grep -oE "[0-9]+" | head -1)

if [ "${got:-}" = "$EXPECTED" ]; then
  echo "bench signature OK: $got nodes (depth $DEPTH, 1 thread, 8 MB hash)"
else
  echo "BENCH SIGNATURE MISMATCH"
  echo "  expected : $EXPECTED"
  echo "  got      : ${got:-<no output>}"
  echo "Search behaviour changed. If this was intentional, regenerate the"
  echo "signature in the same commit:"
  echo "  bash tools/update_bench_signature.sh $ENGINE"
  echo "(A missing count usually means the engine crashed -- e.g. a stack"
  echo " overflow in deep recursion -- rather than a mere count change.)"
  exit 1
fi
