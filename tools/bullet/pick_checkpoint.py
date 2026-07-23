#!/usr/bin/env python3
"""Pick the best bullet checkpoint by held-out validation loss.

bullet (this version) doesn't compute validation loss, so we do it here: for
each `checkpoints/stk-<N>/raw.bin`, transfer it into STKNet and score it on a
held-out shard with bullet's own objective
`(sigmoid(pred/scale) - target)^2`, where
`target = wdl_frac*result + (1-wdl_frac)*sigmoid(eval/scale)` (all stm-relative).

The loss-vs-N curve answers "how many epochs": still dropping at the last
checkpoint => train longer; bottomed then rose => overfit, use the min. With
--workdir the best checkpoint is transferred/exported to the final .nnue.

    python tools/bullet/pick_checkpoint.py --bullet D:/Soumik/Cpp/bullet \
        --val runs/bulk/data/shard_0093.txt --workdir runs/bulk/train \
        --engine build-bench/bin/ChessEngine.exe
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nnue"))
import halfka_features as hk  # noqa: E402

sys.path.insert(0, str(HERE))
import transfer_to_stknet as tr  # noqa: E402


def load_val(path: Path, k: int, seed: int, wdl_frac: float, scale: float):
    lines = path.read_text().splitlines()
    random.Random(seed).shuffle(lines)
    ws: list[np.ndarray] = []
    bs: list[np.ndarray] = []
    stms: list[int] = []
    bks: list[int] = []
    tgts: list[float] = []
    for ln in lines:
        if len(ws) >= k:
            break
        parts = ln.split("|")
        if len(parts) != 3:
            continue
        try:
            pieces, stm = hk.parse_fen_pieces(parts[0].strip())
            ev, wdl = float(parts[1]), float(parts[2])
        except (ValueError, KeyError, StopIteration):
            continue
        wf, bf = hk.features_for(pieces, hk.WHITE), hk.features_for(pieces, hk.BLACK)
        ev_stm = ev if stm == hk.WHITE else -ev
        wdl_stm = wdl if stm == hk.WHITE else 1.0 - wdl
        tgt = wdl_frac * wdl_stm + (1.0 - wdl_frac) / (1.0 + np.exp(-ev_stm / scale))
        w = np.full(40, -1, np.int64)
        w[:len(wf)] = wf
        b = np.full(40, -1, np.int64)
        b[:len(bf)] = bf
        ws.append(w)
        bs.append(b)
        stms.append(stm)
        bks.append(min(max((len(pieces) - 1) // 4, 0), 7))
        tgts.append(tgt)
    return (torch.tensor(np.array(ws)), torch.tensor(np.array(bs)),
            torch.tensor(stms), torch.tensor(bks),
            torch.tensor(np.array(tgts), dtype=torch.float32))


def main() -> int:
    p = argparse.ArgumentParser(description="Pick best bullet checkpoint by val loss.")
    p.add_argument("--bullet", required=True, help="bullet clone dir (has checkpoints/)")
    p.add_argument("--val", required=True, help="held-out shard_*.txt")
    p.add_argument("--workdir", default=None, help="if set, export the best checkpoint here")
    p.add_argument("--engine", default="build-bench/bin/ChessEngine.exe")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--eval-scale", type=float, default=400.0)
    p.add_argument("--wdl-frac", type=float, default=0.5)
    p.add_argument("--val-positions", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    mk_hidden = args.hidden
    ckpts = sorted((Path(args.bullet) / "checkpoints").glob("stk-*/raw.bin"),
                   key=lambda q: int(q.parent.name.split("-")[1]))
    if not ckpts:
        raise SystemExit(f"no stk-*/raw.bin under {args.bullet}/checkpoints")

    print(f"[val] loading {args.val_positions} positions from {args.val} ...")
    W, B, STM, BK, TGT = load_val(Path(args.val), args.val_positions, args.seed,
                                  args.wdl_frac, args.eval_scale)
    print(f"[val] scoring {len(ckpts)} checkpoints on {len(TGT)} positions\n")

    best_loss, best_ckpt = float("inf"), ckpts[-1]
    print(f"{'checkpoint':>12} {'val_loss':>12}")
    for c in ckpts:
        model = tr.build_stknet(tr.load_raw(c, mk_hidden), mk_hidden, args.eval_scale)
        with torch.no_grad():
            pred = torch.sigmoid(model(W, B, STM, BK) / args.eval_scale)
            loss = float(((pred - TGT) ** 2).mean())
        mark = ""
        if loss < best_loss:
            best_loss, best_ckpt, mark = loss, c, "  <- best"
        print(f"{c.parent.name:>12} {loss:12.6f}{mark}")

    print(f"\n[val] best: {best_ckpt.parent.name} (loss {best_loss:.6f})")
    if best_ckpt is ckpts[-1]:
        print("[val] NOTE: min is the last checkpoint -- val loss may still be "
              "dropping; consider more --epochs.")

    if args.workdir:
        print(f"[val] exporting {best_ckpt.parent.name} -> {args.workdir}")
        subprocess.run([sys.executable, str(HERE / "transfer_to_stknet.py"),
                        "--raw", str(best_ckpt), "--workdir", args.workdir,
                        "--engine", args.engine, "--hidden", str(args.hidden),
                        "--eval-scale", str(args.eval_scale)], check=True)
    print(f"best_checkpoint={best_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
