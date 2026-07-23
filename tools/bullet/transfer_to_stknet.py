#!/usr/bin/env python3
"""Transfer a bullet-trained STK-HalfKA net (raw.bin) into make_net's STKNet,
then reuse make_net's verified quantize/export + engine-parity to emit the
`.nnue`. This is phase 3 of the bullet port (tools/bullet/README.md).

bullet stores affine weights **column-major** with shape (out, in), so a tensor
reads back as `raw.reshape(in, out)` giving `M[in][out]`. bullet's raw graph
output equals cp / eval_scale, while STKNet outputs cp = dense*OUT_CP + psqt, so
the transfer folds the scale constants:

  ft.weight[f+1] = M0[f];  ft_bias = l0b + M0[DEAD]        (dead own-king row)
  l1w = (l1w^T reshaped [8,16,1024]) * 128/127             (pairwise PAIR_FACTOR)
  l2w = l2w^T reshaped [8,32,16];  out = l3w^T * eval_scale/OUT_CP
  psqt.weight[f+1] = MP[f] * eval_scale                    (psqt in cp)

Verify (optional): pass --bullet-evals <file> with lines `EVAL<i> <raw> <fen>`
(from the stk_eval bullet example); the script asserts STKNet_cp == raw*scale.

    python tools/bullet/transfer_to_stknet.py --raw <clone>/checkpoints/stk-N/raw.bin \
        --workdir runs/bulk/train --engine build-bench/bin/ChessEngine.exe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nnue"))
import halfka_features as hk  # noqa: E402
import make_net as mk  # noqa: E402

FT = 22528
NIN = FT + 1
DEAD = FT
NB, L1, L2 = 8, 16, 32


def load_raw(raw_path: Path, hidden: int) -> dict[str, np.ndarray]:
    raw = np.fromfile(raw_path, dtype=np.float32)
    meta = [("l0w", hidden, NIN), ("l0b", hidden, 1),
            ("l1w", NB * L1, hidden), ("l1b", NB * L1, 1),
            ("l2w", NB * L2, L1), ("l2b", NB * L2, 1),
            ("l3w", NB, L2), ("l3b", NB, 1),
            ("psqtw", NB, NIN), ("psqtb", NB, 1)]
    exp = sum(o * i for _, o, i in meta)
    if raw.size != exp:
        raise SystemExit(f"raw.bin has {raw.size} floats, expected {exp} "
                         f"(hidden={hidden}?)")
    out, off = {}, 0
    for nm, o, i in meta:
        out[nm] = raw[off:off + o * i]
        off += o * i
    return out


def build_stknet(R: dict[str, np.ndarray], hidden: int, eval_scale: float) -> mk.STKNet:
    pf = mk.PAIR_FACTOR  # QA/128 = 127/128
    out_cp = mk.OUT_CP  # 508
    # column-major (out,in) -> reshape (in,out) gives M[in][out]
    m0 = R["l0w"].reshape(NIN, hidden)                 # M0[feature] = hidden vec
    wl1 = R["l1w"].reshape(hidden, NB * L1).T          # (128, hidden) [out,in]
    wl2 = R["l2w"].reshape(L1, NB * L2).T              # (256, 16)
    wl3 = R["l3w"].reshape(L2, NB).T                   # (8, 32)
    mp = R["psqtw"].reshape(NIN, NB)                   # MP[feature] = psqt vec

    model = mk.STKNet()
    sd = model.state_dict()

    def setp(k: str, a: np.ndarray) -> None:
        sd[k].copy_(torch.tensor(np.ascontiguousarray(a), dtype=torch.float32))

    ftw = np.zeros((NIN, hidden), np.float32)
    ftw[1:] = m0[:FT]
    setp("ft.weight", ftw)
    setp("ft_bias", R["l0b"] + m0[DEAD])
    setp("l1w", wl1.reshape(NB, L1, hidden) * (1.0 / pf))
    setp("l1b", R["l1b"].reshape(NB, L1))
    setp("l2w", wl2.reshape(NB, L2, L1))
    setp("l2b", R["l2b"].reshape(NB, L2))
    setp("outw", wl3.reshape(NB, L2) * (eval_scale / out_cp))
    setp("outb", R["l3b"] * (eval_scale / out_cp))
    pw = np.zeros((NIN, NB), np.float32)
    pw[1:] = mp[:FT] * eval_scale
    setp("psqt.weight", pw)

    model.load_state_dict(sd)
    model.eval()
    return model


def stk_cp(model: mk.STKNet, fen: str) -> float:
    pieces, stm = hk.parse_fen_pieces(fen)
    wf, bf = hk.features_for(pieces, hk.WHITE), hk.features_for(pieces, hk.BLACK)
    w = torch.full((1, 40), -1, dtype=torch.long)
    b = torch.full((1, 40), -1, dtype=torch.long)
    w[0, :len(wf)] = torch.tensor(wf)
    b[0, :len(bf)] = torch.tensor(bf)
    bk = torch.tensor([min(max((len(pieces) - 1) // 4, 0), NB - 1)])
    with torch.no_grad():
        return float(model(w, b, torch.tensor([stm]), bk)[0])


def main() -> int:
    p = argparse.ArgumentParser(description="Transfer bullet raw.bin -> STKNet -> .nnue.")
    p.add_argument("--raw", required=True, help="bullet checkpoint raw.bin")
    p.add_argument("--workdir", required=True, help="output dir for model_float.pt + .nnue")
    p.add_argument("--engine", default="build-bench/bin/ChessEngine.exe")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--eval-scale", type=float, default=400.0)
    p.add_argument("--bullet-evals", default=None, help="stk_eval output for a parity check")
    p.add_argument("--tolerance-cp", type=int, default=1)
    args = p.parse_args()

    mk.HIDDEN = args.hidden
    mk.PAIR = args.hidden // 2

    R = load_raw(Path(args.raw), args.hidden)
    model = build_stknet(R, args.hidden, args.eval_scale)

    if args.bullet_evals:
        worst = 0.0
        for line in Path(args.bullet_evals).read_text().splitlines():
            t = line.split(maxsplit=2)
            if len(t) == 3 and t[0].startswith("EVAL"):
                raw_out, fen = float(t[1]), t[2].strip()
                worst = max(worst, abs(stk_cp(model, fen) - raw_out * args.eval_scale))
        print(f"[transfer] STKNet vs bullet oracle: worst {worst:.4f} cp")
        if worst > 0.5:
            raise SystemExit("[transfer] FAIL: STKNet != bullet (transfer bug)")

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    float_path = work / "model_float.pt"
    torch.save({"state_dict": model.state_dict()}, float_path)
    print(f"[transfer] wrote {float_path}")

    net_path = mk.stage_export(work, float_path)
    mk.stage_verify(argparse.Namespace(engine=args.engine, tolerance_cp=args.tolerance_cp),
                    float_path, net_path)
    print(f"[transfer] done -> {net_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
