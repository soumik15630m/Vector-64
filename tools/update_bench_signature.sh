#!/usr/bin/env bash
# Regenerate the committed search fingerprint after an INTENTIONAL search change.
# Run this, review the one-line diff in tools/bench_signature.txt, and commit it
# together with the search change so the two never drift.
#
#   bash tools/update_bench_signature.sh <path-to-ChessEngine>

set -euo pipefail

ENGINE="${1:?usage: update_bench_signature.sh <path-to-ChessEngine>}"
SIG_FILE="$(dirname "$0")/bench_signature.txt"
DEPTH="$(grep -E '^DEPTH=' "$SIG_FILE" | cut -d= -f2)"
OLD="$(grep -E '^NODES=' "$SIG_FILE" | cut -d= -f2)"

got=$(printf "setoption name Threads value 1\nsetoption name Hash value 8\nbench %s\nquit\n" "$DEPTH" \
  | "$ENGINE" 2>/dev/null | grep -oE "nodes [0-9]+" | grep -oE "[0-9]+" | head -1)

if [ -z "${got:-}" ]; then
  echo "no bench output from $ENGINE (crash? check the engine runs 'bench $DEPTH')"
  exit 1
fi

# Sanity: the count must be stable across runs, or it is not a valid signature.
again=$(printf "setoption name Threads value 1\nsetoption name Hash value 8\nbench %s\nquit\n" "$DEPTH" \
  | "$ENGINE" 2>/dev/null | grep -oE "nodes [0-9]+" | grep -oE "[0-9]+" | head -1)
if [ "$got" != "$again" ]; then
  echo "NON-DETERMINISTIC bench: $got vs $again -- refusing to record a signature."
  exit 1
fi

sed -i -E "s/^NODES=.*/NODES=$got/" "$SIG_FILE"
echo "bench signature updated: $OLD -> $got (depth $DEPTH, 1 thread, 8 MB hash)"
echo "review and commit tools/bench_signature.txt alongside the search change."
