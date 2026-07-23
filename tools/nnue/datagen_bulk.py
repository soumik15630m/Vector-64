#!/usr/bin/env python3
"""Bulk self-play datagen: build a large dataset over many crash-safe chunks.

Runs tools/nnue/datagen.py repeatedly (a fresh seed each chunk) into a shard
directory until --target-positions is reached. Resumable: every completed shard
is recorded in <out-dir>/bulk_state.json and skipped on restart, so a multi-day
run survives Ctrl-C / crashes / reboots. A chunk that died mid-write leaves a
partial shard that is simply regenerated (same seed -> same games) and
overwritten, so no positions are ever double-counted.

Point make_net.py --input at the shard directory to train on the whole set:

    python tools/nnue/datagen_bulk.py --engine build-bench/bin/ChessEngine.exe \
        --net runs/v2/stk_halfka_1024.nnue --out-dir runs/bulk/data \
        --target-positions 500000000 --nodes 5000 --concurrency 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


def run_chunk(cmd: list[str], log) -> None:
    log.write("\n$ " + " ".join(cmd) + "\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()  # else Tee-Object block-buffers the pipe (no live output)
        log.write(line)
        log.flush()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"datagen chunk failed ({proc.returncode})")


def main() -> int:
    p = argparse.ArgumentParser(description="Bulk resumable self-play datagen.")
    p.add_argument("--engine", required=True)
    p.add_argument("--net", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-positions", type=int, default=500_000_000)
    p.add_argument("--nodes", type=int, default=5000,
                   help="nodes/move (lower = more throughput; WDL carries the signal)")
    p.add_argument("--lam", type=float, default=0.5,
                   help="WDL blend weight (only used by --emit blend)")
    p.add_argument("--emit", choices=("blend", "raw"), default="raw",
                   help="shard line format: raw = bullet's '<fen> | <eval> | <wdl>' "
                        "(default); blend = '<fen> | <cp>' for the PyTorch trainer")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--chunk-games", type=int, default=40000,
                   help="games per chunk == crash-recovery granularity")
    p.add_argument("--log-interval", type=float, default=30.0,
                   help="seconds between datagen progress heartbeat lines")
    p.add_argument("--seed-base", type=int, default=100_000)
    p.add_argument("--python", action="store_true",
                   help="use the Python datagen.py driver instead of the engine's "
                        "native 'datagen' subcommand (native is ~30%% faster)")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "bulk_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"chunks": [], "positions": 0}
    log = open(out / "datagen_bulk.log", "a", encoding="utf-8")
    py = sys.executable

    t_start = time.time()
    done0 = state["positions"]
    print(f"[bulk] target {args.target_positions:,} positions @ {args.nodes} nodes; "
          f"have {state['positions']:,} in {len(state['chunks'])} shards", flush=True)

    while state["positions"] < args.target_positions:
        idx = len(state["chunks"])
        shard = out / f"shard_{idx:04d}.txt"
        seed = args.seed_base + idx
        if args.python:
            cmd = [py, "-u", str(HERE / "datagen.py"), "--engine", args.engine,
                   "--net", args.net, "--games", str(args.chunk_games),
                   "--nodes", str(args.nodes), "--lam", str(args.lam),
                   "--emit", args.emit, "--concurrency", str(args.concurrency),
                   "--log-interval", str(args.log_interval),
                   "--seed", str(seed), "--out", str(shard)]
        else:  # engine's native datagen subcommand (reuses search+NNUE, faster)
            cmd = [str(Path(args.engine).resolve()), "datagen", "--net", args.net,
                   "--games", str(args.chunk_games), "--nodes", str(args.nodes),
                   "--lam", str(args.lam), "--emit", args.emit,
                   "--threads", str(args.concurrency),
                   "--log-interval", str(args.log_interval),
                   "--seed", str(seed), "--out", str(shard)]
        run_chunk(cmd, log)
        got = count_lines(shard)
        state["chunks"].append({"file": shard.name, "seed": seed, "positions": got})
        state["positions"] += got
        state_path.write_text(json.dumps(state, indent=2))

        el = time.time() - t_start
        made = state["positions"] - done0
        rate = made / max(el, 1e-9)
        remain = max(0, args.target_positions - state["positions"])
        eta_h = remain / max(rate, 1e-9) / 3600
        print(f"[bulk] chunk {idx}: +{got:,} pos  total {state['positions']:,}/"
              f"{args.target_positions:,}  {rate:.0f} pos/s  eta {eta_h:.1f} h",
              flush=True)

    print(f"[bulk] DONE {state['positions']:,} positions in {len(state['chunks'])} "
          f"shards -> {out}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
