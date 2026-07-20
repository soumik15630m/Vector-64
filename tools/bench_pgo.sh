#!/usr/bin/env bash
# STK-Vector-64 — profile-guided (PGO) build benchmark.
# Runs the full PGO cycle automatically: build instrumented -> exercise a
# search-representative learning set (both thread modes, all game phases, the
# time-managed abort path) -> rebuild optimized -> benchmark.
#
#   bash tools/bench_pgo.sh
#
# PGO tunes codegen for the search workload, so perft may regress slightly —
# that is expected and correct (games run search, not perft).

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_bench.sh"

BUILD_DIR="build-pgo"
detect_platform

if [[ -n "${GEN_ARGS+x}" ]] && printf '%s\n' "${GEN_ARGS[@]:-}" | grep -qi "visual studio\|msvc"; then
  echo "${C_RED}PGO here targets GCC/Clang (-fprofile-*). Use bench.sh on MSVC.${C_RESET}"; exit 1
fi

title "STK-Vector-64  —  PGO build benchmark"
kv "CPU"   "$(cpu_name)  ($(cpu_threads) threads)"
kv "OS"    "$OSNAME"
kv "Build" "Release · -O3 -march=native · LTO · PGO · $(isa_features "-march=native")"
rule "─"

step "1/3  Building instrumented binary (-fprofile-generate)…"
find "$REPO_ROOT/$BUILD_DIR" -name '*.gcda' -delete 2>/dev/null || true
cmake_build "$BUILD_DIR" -DENGINE_PGO=generate
GEN_EXE="$REPO_ROOT/$BUILD_DIR/bin/ChessEngine$EXE"

step "2/3  Running learning set (search, both thread modes, diverse positions)…"
run_pgo_learning_set "$GEN_EXE"
GCDA=$(find "$REPO_ROOT/$BUILD_DIR" -name '*.gcda' | wc -l | tr -d ' ')
echo "  ${C_GRN}✓${C_RESET} profile written ($GCDA translation units)"

step "3/3  Rebuilding optimized binary (-fprofile-use)…"
cmake_build "$BUILD_DIR" -DENGINE_PGO=use
run_correctness "$BUILD_DIR"

EXE="$REPO_ROOT/$BUILD_DIR/bin/ChessEngine$EXE"
step "Measuring (best-of-$SEARCH_RUNS search, best-of-$PERFT_RUNS perft)…"
ST=$(measure_search_st "$EXE")
MT=$(measure_search_mt "$EXE")
read -r PST PMT < <(measure_perft "$EXE")

print_report "$ST" "$MT" "$PST" "$PMT" "$(cpu_threads)" \
  "PGO (search-tuned) · perft may dip vs default — expected · depth $BENCH_DEPTH bench"

NET=$(find_evalfile)
if [[ -n "$NET" && "$(verify_evalfile "$EXE" "$NET")" == "1" ]]; then
  step "Measuring NNUE search (best-of-$SEARCH_RUNS, net: $(basename "$NET"))..."
  NST=$(measure_search_nnue_st "$EXE" "$NET")
  NMT=$(measure_search_nnue_mt "$EXE" "$NET")
  print_nnue_lines "$NST" "$NMT" "$NET" "$(cpu_threads)"
elif [[ -n "$NET" ]]; then
  printf "  NNUE bench FAILED: engine did not load %s\n" "$NET"
else
  printf "  NNUE bench skipped: no .nnue found (set EVALFILE=/path/to/net.nnue)\n"
fi
