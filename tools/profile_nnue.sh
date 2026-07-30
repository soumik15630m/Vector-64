#!/usr/bin/env bash
# NNUE hot-path profiler. Builds a *dedicated* -DENGINE_PROF binary (in
# build-prof/, leaving your normal build untouched), runs a fixed bench, and
# aggregates the per-position PROF lines into one cycle breakdown.
#
# The point of this on ARM: the x86 (AVX2/VNNI) baseline spends ~65% of the
# forward in L1, ~19% pairwise, ~10% L2, ~6% out, and ~90% of updates are
# incremental. If the NEON run is lopsided vs that (e.g. L1 or pairwise far
# higher), that's the kernel to attack next.
#
# Ticks are arch-specific (x86 TSC / arm64 cntvct_el0), so only the PERCENTAGES
# are comparable across machines -- never the raw tick counts.
#
#   bash tools/profile_nnue.sh          # depth 13 (matches the bench signature)
#   bash tools/profile_nnue.sh 15       # deeper = more samples, longer run
set -euo pipefail

DEPTH="${1:-13}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build-prof"

NET="$ROOT/nets/stk-vector-64.nnue"
if [ ! -f "$NET" ]; then
  echo "ERROR: embedded net missing: $NET"
  echo "It is git-tracked (~45 MB). Fetch it, then re-run:"
  echo "  git -C \"$ROOT\" pull"
  exit 1
fi

CFGLOG="$(mktemp)"
echo "[1/3] configuring $BUILD (-DENGINE_PROF, Release, native march)"
if ! cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_FLAGS="-DENGINE_PROF" >"$CFGLOG" 2>&1; then
  cat "$CFGLOG"; rm -f "$CFGLOG"; exit 1
fi
grep -iE 'NNUE binary:|^-- Compiler:' "$CFGLOG" || true
rm -f "$CFGLOG"

echo "[2/3] building ChessEngine-nnue"
if ! cmake --build "$BUILD" --target ChessEngine-nnue -j >/dev/null 2>&1; then
  echo "ERROR: ChessEngine-nnue did not build (see the 'NNUE binary:' line"
  echo "above). If it says 'skipped', the net is missing or the toolchain is"
  echo "unsupported; otherwise re-run without -j to see the compile error."
  exit 1
fi

# Binary location differs by generator (Unix vs MSVC multi-config).
EXE="$BUILD/bin/ChessEngine-nnue"
[ -x "$EXE" ] || EXE="$BUILD/bin/ChessEngine-nnue.exe"
[ -x "$EXE" ] || EXE="$BUILD/bin/Release/ChessEngine-nnue.exe"

ERR="$(mktemp)"
OUT="$(mktemp)"
trap 'rm -f "$ERR" "$OUT"' EXIT

echo "[3/3] bench $DEPTH (1 thread, 8 MB hash) -- aggregating PROF lines"
printf 'setoption name Threads value 1\nsetoption name Hash value 8\nbench %s\nquit\n' \
  "$DEPTH" | "$EXE" 2>"$ERR" >"$OUT" || true

NODES=$(grep -oE 'nodes [0-9]+' "$OUT" | grep -oE '[0-9]+' | tail -1)
echo
echo "==== STK-Vector-64 NNUE profile (depth $DEPTH, nodes ${NODES:-?}) ===="
echo "(percentages only -- ticks are arch-specific)"
awk '
  /^PROF / {
    for (i=1;i<=NF;i++) if ($i ~ /=/) { split($i,a,"="); v[a[1]]+=a[2] }
  }
  /^PROF-FWD/ {
    for (i=1;i<=NF;i++) if ($i ~ /=/) {
      split($i,a,"="); val=a[2]; gsub(/\(.*/,"",val); f[a[1]]+=val
    }
  }
  /^PROF-UPD/ {
    for (i=1;i<=NF;i++) if ($i ~ /=/) {
      split($i,a,"="); split(a[2],b,"/"); cyc=b[2]; gsub(/c\(.*/,"",cyc)
      u[a[1]]+=cyc
    }
  }
  function pct(x,t) { return t>0 ? 100*x/t : 0 }
  END {
    et = v["evalcyc"]+v["updcyc"]+v["smallcyc"]
    printf "eval loop:  eval=%.1f%%  update=%.1f%%  small=%.1f%%\n",
      pct(v["evalcyc"],et), pct(v["updcyc"],et), pct(v["smallcyc"],et)
    ft = f["pairwise"]+f["l1"]+f["l2"]+f["out"]
    printf "forward:    pairwise=%.1f%%  L1=%.1f%%  L2=%.1f%%  out=%.1f%%\n",
      pct(f["pairwise"],ft), pct(f["l1"],ft), pct(f["l2"],ft), pct(f["out"],ft)
    ut = u["incr"]+u["kingRefresh"]
    printf "update:     incremental=%.1f%%  kingRefresh=%.1f%%\n",
      pct(u["incr"],ut), pct(u["kingRefresh"],ut)
  }
' "$ERR"
