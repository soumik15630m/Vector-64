#!/usr/bin/env bash
# STK-Vector-64 — default build benchmark.
# Builds the best native Release configuration for this machine (auto-detects
# toolchain; PEXT/AVX2 enabled where available, magic/scalar fallback elsewhere)
# and reports perft + search NPS, single- and multi-threaded.
#
#   bash tools/bench.sh
#   BENCH_DEPTH=14 SEARCH_RUNS=8 bash tools/bench.sh   # override sampling

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_bench.sh"

BUILD_DIR="build-bench"
detect_platform

title "STK-Vector-64  —  default build benchmark"
kv "CPU"   "$(cpu_name)  ($(cpu_threads) threads)"
kv "OS"    "$OSNAME"
kv "Build" "Release · -O3 -march=native · LTO · $(isa_features "-march=native")"
rule "─"

step "Configuring & building (Release, native, LTO)…"
cmake_build "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
run_correctness "$BUILD_DIR"

EXE="$REPO_ROOT/$BUILD_DIR/bin/ChessEngine$EXE"
step "Measuring (best-of-$SEARCH_RUNS search, best-of-$PERFT_RUNS perft)…"
ST=$(measure_search_st "$EXE")
MT=$(measure_search_mt "$EXE")
read -r PST PMT < <(measure_perft "$EXE")

print_report "$ST" "$MT" "$PST" "$PMT" "$(cpu_threads)" \
  "best-of-N · material+PSQT eval (no NNUE) · depth $BENCH_DEPTH bench"

NET=$(find_evalfile)
if [[ -n "$NET" ]]; then
  step "Measuring NNUE search (best-of-$SEARCH_RUNS, net: $(basename "$NET"))..."
  NST=$(measure_search_nnue_st "$EXE" "$NET")
  NMT=$(measure_search_nnue_mt "$EXE" "$NET")
  print_nnue_lines "$NST" "$NMT" "$NET" "$(cpu_threads)"
else
  printf "  NNUE bench skipped: no .nnue found (set EVALFILE=/path/to/net.nnue)\n"
fi
