#!/usr/bin/env python3
"""Build STK-HalfKA training shards from Stockfish-labelled positions.

Input: a .jsonl(.zst) file in the Lichess eval-database shape (each line has a
"fen" and an "evals" list of {depth, knodes, pvs:[{cp|mate}]}), or a simple
"<fen> | <cp>" text file (one per line) for quick tests.

Output: .npz shards with, per position:
  white_indices : int32 [N, MAX_FEATURES]  (-1 padded)  STK-HalfKA, white persp
  black_indices : int32 [N, MAX_FEATURES]  (-1 padded)  STK-HalfKA, black persp
  stm           : int8  [N]                 0=white, 1=black to move
  eval_cp       : float32 [N]               centipawns, white perspective
plus a manifest.json. Features come from tools/nnue/halfka_features.py, the same
encoder the C++ engine uses, so the data and the engine agree by construction.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import halfka_features as hk  # noqa: E402

MAX_FEATURES = 32  # >= 30 non-king pieces
MATE_CP = 8000.0


def iter_lines(path: Path):
    if path.suffix.lower() == ".zst":
        import zstandard as zstd

        with path.open("rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh) as reader:
                yield from io.TextIOWrapper(reader, encoding="utf-8")
    else:
        with path.open("r", encoding="utf-8") as fh:
            yield from fh


def extract_cp(rec: dict, min_depth: int) -> float | None:
    """Deepest cp eval, white perspective. Mate maps to +/-MATE_CP."""
    evals = rec.get("evals")
    if not isinstance(evals, list):
        cp = rec.get("cp")
        return float(cp) if cp is not None else None
    best_depth = -1
    best_cp: float | None = None
    for e in evals:
        if not isinstance(e, dict):
            continue
        depth = int(e.get("depth", 0))
        if depth < min_depth or depth <= best_depth:
            continue
        pvs = e.get("pvs")
        pv = pvs[0] if isinstance(pvs, list) and pvs else e
        if not isinstance(pv, dict):
            continue
        if pv.get("cp") is not None:
            best_cp, best_depth = float(pv["cp"]), depth
        elif pv.get("mate") is not None:
            best_cp = MATE_CP if float(pv["mate"]) > 0 else -MATE_CP
            best_depth = depth
    return best_cp


def parse_line(line: str, min_depth: int) -> tuple[str, float] | None:
    line = line.strip()
    if not line:
        return None
    if line[0] == "{":
        rec = json.loads(line)
        fen = rec.get("fen")
        if not isinstance(fen, str):
            return None
        cp = extract_cp(rec, min_depth)
        return (fen, cp) if cp is not None else None
    if "|" in line:
        fen_str, cp_str = line.rsplit("|", 1)
        return fen_str.strip(), float(cp_str)
    return None


def encode(fen: str) -> tuple[np.ndarray, np.ndarray, int]:
    pieces, stm = hk.parse_fen_pieces(fen)
    wf = hk.features_for(pieces, hk.WHITE)
    bf = hk.features_for(pieces, hk.BLACK)
    if len(wf) > MAX_FEATURES or len(bf) > MAX_FEATURES:
        raise ValueError("feature overflow")
    w = np.full(MAX_FEATURES, -1, dtype=np.int32)
    b = np.full(MAX_FEATURES, -1, dtype=np.int32)
    w[: len(wf)] = wf
    b[: len(bf)] = bf
    return w, b, stm


def main() -> int:
    p = argparse.ArgumentParser(description="Build STK-HalfKA training shards.")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--min-depth", type=int, default=0)
    p.add_argument("--max-abs-cp", type=float, default=6000.0)
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    stm_rows: list[int] = []
    cp_rows: list[float] = []
    shards: list[dict] = []
    kept = 0

    def flush() -> None:
        if not cp_rows:
            return
        idx = len(shards)
        name = f"stk_shard_{idx:05d}.npz"
        np.savez_compressed(
            out_dir / name,
            white_indices=np.stack(w_rows).astype(np.int32),
            black_indices=np.stack(b_rows).astype(np.int32),
            stm=np.asarray(stm_rows, dtype=np.int8),
            eval_cp=np.asarray(cp_rows, dtype=np.float32),
        )
        shards.append({"file": name, "samples": len(cp_rows)})
        w_rows.clear()
        b_rows.clear()
        stm_rows.clear()
        cp_rows.clear()

    for line in iter_lines(Path(args.input)):
        parsed = parse_line(line, args.min_depth)
        if parsed is None:
            continue
        fen, cp = parsed
        if abs(cp) > args.max_abs_cp:
            continue
        try:
            w, b, stm = encode(fen)
        except Exception:
            continue
        w_rows.append(w)
        b_rows.append(b)
        stm_rows.append(stm)
        cp_rows.append(cp)
        kept += 1
        if len(cp_rows) >= args.shard_size:
            flush()
        if args.max_samples and kept >= args.max_samples:
            break
    flush()

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {"total_samples": kept, "max_features": MAX_FEATURES, "shards": shards},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"kept {kept} samples in {len(shards)} shard(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
