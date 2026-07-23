#!/usr/bin/env python3
"""One command: finished self-play shards -> final engine-ready .nnue.

Pipeline (each stage skips if its output already exists, so the whole thing is
resumable -- Ctrl-C and re-run):
  1. convert every <data-dir>/shard_*.txt -> <scratch>/<name>.bin   (bullet-utils)
  2. interleave all shard .bins            -> <scratch>/combined.bin (cross-shard mix)
  3. shuffle combined.bin                  -> <scratch>/train.bin    (full shuffle)
  4. train STK-HalfKA in bullet on the GPU for --epochs passes
  5. transfer the bullet checkpoint -> STKNet -> quantise/export + engine parity

    python tools/bullet/build_net.py --bullet D:/Soumik/Cpp/bullet

Then SPRT the result vs runs/v2 with tools/nnue/match.py -- that is the only
verdict that counts. Disk: needs the text shards (~45GB for 500M) plus ~16-48GB
scratch; intermediates are removed once train.bin exists (keep with --keep-scratch).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd[0]}")


def exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def main() -> int:
    p = argparse.ArgumentParser(description="Build the final .nnue from self-play shards.")
    p.add_argument("--bullet", required=True, help="bullet clone dir (has target/release + examples)")
    p.add_argument("--data-dir", default="runs/bulk/data", help="dir of shard_*.txt")
    p.add_argument("--scratch", default="runs/bulk/bin", help="dir for .bin intermediates")
    p.add_argument("--workdir", default="runs/bulk/train", help="output dir for model_float.pt + .nnue")
    p.add_argument("--engine", default="build-bench/bin/ChessEngine.exe")
    p.add_argument("--epochs", type=int, default=20,
                   help="passes over the dataset (20 x 500M = 10B visits; "
                        "500M unique tolerates it, overfitting risk only past ~30-40)")
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--shuffle-mem-mb", type=int, default=8192)
    p.add_argument("--keep-scratch", action="store_true", help="don't delete .bin intermediates")
    args = p.parse_args()

    bullet = Path(args.bullet).resolve()
    utils = bullet / "target/release/bullet-utils.exe"
    if not utils.exists():
        utils = bullet / "target/release/bullet-utils"
    if not utils.exists():
        raise SystemExit(f"bullet-utils not built at {utils} (cargo build -r --package bullet-utils)")

    data = Path(args.data_dir)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    # 1) convert shards -> per-shard .bin
    shards = sorted(data.glob("shard_*.txt"))
    if not shards:
        raise SystemExit(f"no shard_*.txt in {data}")
    bins = []
    for s in shards:
        b = scratch / (s.stem + ".bin")
        bins.append(b)
        if exists(b):
            print(f"[1/5] {b.name} exists, skip")
            continue
        run([utils, "convert", "--from", "text", "--input", s.resolve(), "--output", b.resolve()])

    # 2) interleave -> combined.bin
    combined = scratch / "combined.bin"
    if not exists(combined):
        run([utils, "interleave", *[b.resolve() for b in bins], "--output", combined.resolve()])

    # 3) shuffle -> train.bin
    trainbin = scratch / "train.bin"
    if not exists(trainbin):
        run([utils, "shuffle", "--input", combined.resolve(),
             "--mem-used-mb", str(args.shuffle_mem_mb), "--output", trainbin.resolve()])

    if not args.keep_scratch:
        for b in bins:
            b.unlink(missing_ok=True)
        combined.unlink(missing_ok=True)

    positions = trainbin.stat().st_size // 32
    bps = max(1, positions // args.batch)
    print(f"\n[4/5] training: {positions:,} positions, {args.epochs} epochs x {bps} batches/superbatch")

    # 4) train in the bullet clone (GPU)
    run(["cargo", "run", "-r", "--example", "stk_train", "--features", "cuda", "--",
         str(trainbin.resolve()), str(args.epochs), str(bps)], cwd=bullet)

    ckpt = bullet / f"checkpoints/stk-{args.epochs}" / "raw.bin"
    if not exists(ckpt):
        cands = sorted((bullet / "checkpoints").glob("stk-*/raw.bin"),
                       key=lambda q: int(q.parent.name.split("-")[1]))
        if not cands:
            raise SystemExit("no bullet checkpoint produced")
        ckpt = cands[-1]

    # 5) transfer -> STKNet -> export + engine parity
    print(f"\n[5/5] transfer {ckpt} -> .nnue")
    run([sys.executable, str(HERE / "transfer_to_stknet.py"), "--raw", str(ckpt),
         "--workdir", args.workdir, "--engine", args.engine])

    net = Path(args.workdir) / "stk_halfka_1024.nnue"
    print("\n==================  DONE  ==================")
    print(f"Final net: {net}")
    print("Next: SPRT it vs runs/v2 before trusting it, e.g.")
    print(f"  python tools/nnue/match.py --engine {args.engine} --base-engine {args.engine} \\")
    print(f"      --net {net} --base-net runs/v2/stk_halfka_1024.nnue \\")
    print("      --sprt 0 5 --games 12000 --nodes 10000 --concurrency 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
