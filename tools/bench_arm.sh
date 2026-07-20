#!/usr/bin/env bash
# STK-Vector-64 — ARM (aarch64) native build benchmark.
# Run this ON an ARM host (Apple Silicon, Raspberry Pi 5, Ampere, Graviton, …).
# The engine is portable: PEXT falls back to magic bitboards and the NNUE AVX2
# path falls back to scalar automatically, so no source changes are needed —
# this script just selects the right ARM tuning flag and reports.
#
#   bash tools/bench_arm.sh
#
# It picks the first arch flag the toolchain accepts: -mcpu=native, then
# -march=native, then a portable build with no arch flag.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_bench.sh"

BUILD_DIR="build-arm"
detect_platform

ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64|armv8*|armv7*) ;;
  *) echo "${C_YEL}Warning: uname -m is '$ARCH' (not ARM). This script targets ARM;"
     echo "         on x86 use tools/bench.sh instead. Continuing anyway.${C_RESET}" ;;
esac

title "STK-Vector-64  —  ARM ($ARCH) build benchmark"
kv "CPU"   "$(cpu_name)  ($(cpu_threads) threads)"
kv "OS"    "$OSNAME"

# Select the best-accepted ARM tuning flag with graceful fallback.
step "Configuring & building (Release, ARM tuning, LTO)…"
ARCHFLAG=""
if try_cmake_build "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DENGINE_NATIVE=OFF -DCMAKE_CXX_FLAGS="-mcpu=native"; then
  ARCHFLAG="-mcpu=native"
elif try_cmake_build "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release; then
  ARCHFLAG="-march=native (cmake default)"
elif try_cmake_build "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DENGINE_NATIVE=OFF; then
  ARCHFLAG="portable (no arch flag)"
else
  echo "${C_RED}${C_BOLD}Build failed for all ARM flag choices. Check your toolchain.${C_RESET}"; exit 1
fi
kv "Build" "Release · -O3 $ARCHFLAG · LTO · $(isa_features "${ARCHFLAG%% *}")"
rule "─"

run_correctness "$BUILD_DIR"

EXE="$REPO_ROOT/$BUILD_DIR/bin/ChessEngine$EXE"
step "Measuring (best-of-$SEARCH_RUNS search, best-of-$PERFT_RUNS perft)…"
ST=$(measure_search_st "$EXE")
MT=$(measure_search_mt "$EXE")
read -r PST PMT < <(measure_perft "$EXE")

print_report "$ST" "$MT" "$PST" "$PMT" "$(cpu_threads)" \
  "ARM native · magic bitboards + scalar NNUE fallback · depth $BENCH_DEPTH bench"

NET=$(find_evalfile)
if [[ -n "$NET" ]]; then
  step "Measuring NNUE search (best-of-$SEARCH_RUNS, net: $(basename "$NET"))..."
  NST=$(measure_search_nnue_st "$EXE" "$NET")
  NMT=$(measure_search_nnue_mt "$EXE" "$NET")
  print_nnue_lines "$NST" "$NMT" "$NET" "$(cpu_threads)"
else
  printf "  NNUE bench skipped: no .nnue found (set EVALFILE=/path/to/net.nnue)\n"
fi
