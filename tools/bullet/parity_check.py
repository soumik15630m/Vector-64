#!/usr/bin/env python3
"""Phase-1 gate for the bullet port: verify the custom StkHalfKa input type
produces bit-identical features to the C++ engine (via halfka_features.py).

Build + run the Rust dumper (tools/bullet/stk_halfka.rs, built as a bullet
example), pipe its stdout to this script:

    cd <bullet clone>
    cargo run -r --example stk_halfka > parity_rust.txt
    python <repo>/tools/bullet/parity_check.py parity_rust.txt

Both must list the same FENs in the same order. Rust prints per FEN:
    FEN<i> STM <sorted idx...>
    FEN<i> NTM <sorted idx...>
and this maps STM/NTM -> white/black by side-to-move to compare against the
Python reference's `W`/`B` lines.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Must match the FEN list in tools/bullet/stk_halfka.rs.
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 6",
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b - - 6 6",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]
HALFKA = Path(__file__).resolve().parent.parent / "nnue" / "halfka_features.py"


def main() -> int:
    rust_path = sys.argv[1] if len(sys.argv) > 1 else "parity_rust.txt"

    out = subprocess.run([sys.executable, str(HALFKA)], input="\n".join(FENS) + "\n",
                         capture_output=True, text=True, check=True).stdout.splitlines()
    ref: list[tuple[set[int], set[int]]] = []
    w: set[int] = set()
    for line in out:
        line = line.strip()
        if line.startswith("W "):
            w = {int(x) for x in line[2:].split()}
        elif line.startswith("B "):
            ref.append((w, {int(x) for x in line[2:].split()}))

    rust: dict[tuple[str, str], set[int]] = {}
    for line in Path(rust_path).read_text().splitlines():
        p = line.split()
        if len(p) >= 2 and p[0].startswith("FEN"):
            rust[(p[0], p[1])] = {int(x) for x in p[2:]}

    ok = True
    total = 0
    for i, fen in enumerate(FENS):
        white = fen.split()[1] == "w"
        wref, bref = ref[i]
        stm, ntm = rust[(f"FEN{i}", "STM")], rust[(f"FEN{i}", "NTM")]
        exp_stm, exp_ntm = (wref, bref) if white else (bref, wref)
        good = stm == exp_stm and ntm == exp_ntm
        ok = ok and good
        total += len(stm) + len(ntm)
        print(f"FEN{i} ({'w' if white else 'b'}): {'OK' if good else 'MISMATCH'}")
        if not good:
            print(f"   STM delta = {stm ^ exp_stm}")
            print(f"   NTM delta = {ntm ^ exp_ntm}")

    print(f"\n==> FEATURE PARITY {'PASS' if ok else 'FAIL'} "
          f"({total} indices across {len(FENS)} positions)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
