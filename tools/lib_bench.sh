#!/usr/bin/env bash
# Shared helpers for the STK-Vector-64 benchmark scripts (bench.sh, bench_pgo.sh,
# bench_arm.sh). Handles platform/toolchain detection, building, the correctness
# gate, thermally-robust best-of-N measurement, and pretty terminal output.

set -euo pipefail

# ---- knobs (override via env) ----
BENCH_DEPTH="${BENCH_DEPTH:-13}"   # search bench depth
PERFT_RUNS="${PERFT_RUNS:-3}"      # perft samples (best taken)
SEARCH_RUNS="${SEARCH_RUNS:-6}"    # search samples (best taken)
RULE_W=60

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPD_ABS="$REPO_ROOT/test_data/standard.epd"

# ---- colors (auto-disabled when not a tty or NO_COLOR set) ----
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_CYAN=$'\033[36m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_RED=$'\033[31m'; C_MAG=$'\033[35m'
else
  C_RESET=; C_BOLD=; C_DIM=; C_CYAN=; C_GRN=; C_YEL=; C_RED=; C_MAG=
fi

detect_platform() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      EXE=".exe"; OSNAME="Windows"
      local mk; mk="$(command -v mingw32-make 2>/dev/null || true)"
      [[ -z "$mk" && -x /c/msys64/ucrt64/bin/mingw32-make.exe ]] && mk="/c/msys64/ucrt64/bin/mingw32-make.exe"
      if [[ -n "$mk" ]]; then GEN_ARGS=(-G "MinGW Makefiles" -DCMAKE_MAKE_PROGRAM="$mk"); else GEN_ARGS=(); fi ;;
    Darwin)
      EXE=""; OSNAME="macOS"
      if command -v ninja >/dev/null 2>&1; then GEN_ARGS=(-G Ninja); else GEN_ARGS=(); fi ;;
    *)
      EXE=""; OSNAME="Linux"
      if command -v ninja >/dev/null 2>&1; then GEN_ARGS=(-G Ninja); else GEN_ARGS=(); fi ;;
  esac
}

cpu_name() {
  if [[ -r /proc/cpuinfo ]] && grep -q "model name" /proc/cpuinfo 2>/dev/null; then
    grep -m1 "model name" /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'
  elif [[ -r /proc/cpuinfo ]] && grep -q "^Model" /proc/cpuinfo 2>/dev/null; then
    grep -m1 "^Model" /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'   # some ARM boards
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown"
  else echo "unknown"; fi
}
cpu_threads() { (command -v nproc >/dev/null 2>&1 && nproc) || echo "?"; }

# ISA features the compiler enables for a given arch flag (display only).
isa_features() {
  local flag="$1" defs out=""
  defs="$(echo | ${CXX:-c++} $flag -dM -E - 2>/dev/null || true)"
  grep -q "__BMI2__"    <<<"$defs" && out+="PEXT "
  grep -q "__AVX2__"    <<<"$defs" && out+="AVX2 "
  grep -q "__ARM_NEON"  <<<"$defs" && out+="NEON "
  [[ -z "$out" ]] && out="magic+scalar fallback"
  echo "${out% }"
}

# configure + build; aborts on failure. Args: build_dir, extra cmake args...
cmake_build() {
  local dir="$1"; shift
  ( cd "$REPO_ROOT" && cmake "${GEN_ARGS[@]}" "$@" -S . -B "$dir" >/dev/null )
  ( cd "$REPO_ROOT" && cmake --build "$dir" -j >/dev/null 2>&1 )
}
# non-fatal variant for flag probing. Returns nonzero on failure.
try_cmake_build() {
  local dir="$1"; shift
  ( cd "$REPO_ROOT" && cmake "${GEN_ARGS[@]}" "$@" -S . -B "$dir" >/dev/null 2>&1 ) || return 1
  ( cd "$REPO_ROOT" && cmake --build "$dir" -j >/dev/null 2>&1 ) || return 1
}

# Fail loudly if the build is not correct — never report perf on a broken build.
run_correctness() {
  local dir="$1"
  local exe="$REPO_ROOT/$dir/bin/ChessEngine$EXE"
  local tz="$REPO_ROOT/$dir/bin/test_zobrist$EXE"
  local ok=1
  step "Correctness gate"
  if [[ -x "$tz" ]]; then
    if "$tz" >/dev/null 2>&1; then echo "  ${C_GRN}✓${C_RESET} state / movegen / NNUE self-tests"
    else echo "  ${C_RED}✗ consistency test FAILED${C_RESET}"; ok=0; fi
  fi
  if "$exe" --perft "$EPD_ABS" 2>/dev/null | grep -q "0 Failed"; then
    echo "  ${C_GRN}✓${C_RESET} perft suite (6/6)"
  else echo "  ${C_RED}✗ perft FAILED${C_RESET}"; ok=0; fi
  [[ "$ok" -eq 1 ]] || { echo "${C_RED}${C_BOLD}Aborting: correctness gate failed.${C_RESET}"; exit 1; }
}

# best-of-N single-thread search, echoes nps
measure_search_st() {
  local exe="$1" best=0 n
  for _ in $(seq 1 "$SEARCH_RUNS"); do
    n=$(printf "setoption name Threads value 1\nbench %s\nquit\n" "$BENCH_DEPTH" | "$exe" 2>/dev/null | grep -oE "nps [0-9]+" | grep -oE "[0-9]+" || echo 0)
    [[ "${n:-0}" -gt "$best" ]] && best="$n"; sleep 0.3
  done; echo "$best"
}
# best-of-N multi-thread search (engine default = hardware threads), echoes nps
measure_search_mt() {
  local exe="$1" best=0 n
  for _ in $(seq 1 "$SEARCH_RUNS"); do
    n=$(printf "bench %s\nquit\n" "$BENCH_DEPTH" | "$exe" 2>/dev/null | grep -oE "nps [0-9]+" | grep -oE "[0-9]+" || echo 0)
    [[ "${n:-0}" -gt "$best" ]] && best="$n"; sleep 0.3
  done; echo "$best"
}
# Locate a primary net for the NNUE bench: $EVALFILE overrides, else the
# newest runs/*/stk_halfka_1024.nnue, else any .nnue beside the repo root.
find_evalfile() {
  if [[ -n "${EVALFILE:-}" && -f "${EVALFILE:-}" ]]; then echo "$EVALFILE"; return; fi
  local c
  c=$(ls -t "$REPO_ROOT"/runs/*/stk_halfka_1024.nnue 2>/dev/null | head -1 || true)
  [[ -n "$c" ]] && { echo "$c"; return; }
  c=$(ls -t "$REPO_ROOT"/*.nnue 2>/dev/null | head -1 || true)
  echo "${c:-}"
}

# best-of-N NNUE search (net loaded), single- and multi-thread, echoes nps
measure_search_nnue_st() {
  local exe="$1" net="$2" best=0 n
  for _ in $(seq 1 "$SEARCH_RUNS"); do
    n=$(printf "setoption name Threads value 1\nsetoption name EvalFile value %s\nbench %s\nquit\n" "$net" "$BENCH_DEPTH" | "$exe" 2>/dev/null | grep -oE "nps [0-9]+" | grep -oE "[0-9]+" || echo 0)
    [[ "${n:-0}" -gt "$best" ]] && best="$n"; sleep 0.3
  done; echo "$best"
}
measure_search_nnue_mt() {
  local exe="$1" net="$2" best=0 n
  for _ in $(seq 1 "$SEARCH_RUNS"); do
    n=$(printf "setoption name EvalFile value %s\nbench %s\nquit\n" "$net" "$BENCH_DEPTH" | "$exe" 2>/dev/null | grep -oE "nps [0-9]+" | grep -oE "[0-9]+" || echo 0)
    [[ "${n:-0}" -gt "$best" ]] && best="$n"; sleep 0.3
  done; echo "$best"
}

print_nnue_lines() {
  local st="$1" mt="$2" net="$3" thr="$4"
  awk -v st="$st" -v mt="$mt" -v thr="$thr" -v g="$C_GRN" -v r="$C_RESET" -v b="$C_BOLD" -v d="$C_DIM" \
    'BEGIN{
      printf "  %sNNUE  %s  ST      %s%8.2f%s Mnps\n", b, r, g, st/1e6, r;
      printf "  %sNNUE  %s  MT      %s%8.2f%s Mnps   %s(%s threads)%s\n", b, r, g, mt/1e6, r, d, thr, r;
    }'
  printf "  ${C_DIM}net: %s${C_RESET}\n" "$(basename "$net")"
}

# best-of-N perft, echoes "ST_MNPS MT_MNPS"
measure_perft() {
  local exe="$1" bs=0 bm=0 out s m
  for _ in $(seq 1 "$PERFT_RUNS"); do
    out=$("$exe" --perft "$EPD_ABS" 2>/dev/null || true)
    s=$(grep -oE "SINGLE CORE : [0-9.]+ seconds \([0-9]+ MNPS\)" <<<"$out" | grep -oE "\([0-9]+" | tr -d '(' || echo 0)
    m=$(grep -oE "MULTI CORE  : [0-9.]+ seconds \([0-9]+ MNPS\)"  <<<"$out" | grep -oE "\([0-9]+" | tr -d '(' || echo 0)
    [[ "${s:-0}" -gt "$bs" ]] && bs="$s"
    [[ "${m:-0}" -gt "$bm" ]] && bm="$m"
  done; echo "$bs $bm"
}

# PGO training workload: both thread modes + diverse phases + the deadline path.
run_pgo_learning_set() {
  local exe="$1"
  printf "setoption name Hash value 128\nbench %s\nquit\n" "$BENCH_DEPTH" | "$exe" >/dev/null 2>&1 || true
  printf "setoption name Threads value 1\nsetoption name Hash value 128\nbench %s\nquit\n" "$BENCH_DEPTH" | "$exe" >/dev/null 2>&1 || true
  local fens=(
    "startpos"
    "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    "fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    "fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    "fen r1bq1rk1/pp2bppp/2n2n2/2pp4/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8"
  )
  for f in "${fens[@]}"; do
    { echo "setoption name Hash value 128"; echo "position $f"; echo "go movetime 2500"; sleep 3; echo quit; } | "$exe" >/dev/null 2>&1 || true
  done
}

# ---- pretty output ----
# Build the rule by repeating the (possibly multibyte) glyph — tr is byte-wise
# and would corrupt box-drawing characters.
rule() {
  local ch="${1:-─}" out="" i
  for ((i = 0; i < RULE_W; i++)); do out+="$ch"; done
  printf "${C_CYAN}%s${C_RESET}\n" "$out"
}
step()  { printf "${C_DIM}» %s${C_RESET}\n" "$1"; }
title() { rule "═"; printf "  ${C_BOLD}%s${C_RESET}\n" "$1"; rule "═"; }
kv()    { printf "  ${C_DIM}%-7s${C_RESET} %s\n" "$1" "$2"; }

# print_report st_nps mt_nps perft_st perft_mt threads note
print_report() {
  local st="$1" mt="$2" pst="$3" pmt="$4" thr="$5" note="${6:-}"
  rule "─"
  awk -v st="$st" -v mt="$mt" -v thr="$thr" -v g="$C_GRN" -v r="$C_RESET" -v b="$C_BOLD" -v d="$C_DIM" '
    BEGIN{
      printf "  %sSearch%s  ST      %s%8.2f%s Mnps\n", b, r, g, st/1e6, r;
      printf "  %sSearch%s  MT      %s%8.2f%s Mnps   %s(%s threads)%s\n", b, r, g, mt/1e6, r, d, thr, r;
    }'
  awk -v pst="$pst" -v pmt="$pmt" -v y="$C_YEL" -v r="$C_RESET" -v b="$C_BOLD" '
    BEGIN{
      printf "  %sPerft %s  ST      %s%8d%s Mnps\n", b, r, y, pst, r;
      printf "  %sPerft %s  MT      %s%8d%s Mnps\n", b, r, y, pmt, r;
    }'
  rule "═"
  [[ -n "$note" ]] && printf "  ${C_DIM}%s${C_RESET}\n" "$note"
}
