#!/usr/bin/env python3
"""Train an STK-HalfKA NNUE (2048-wide) matching src/nnue/network.{h,cpp}.

The forward pass reproduces the engine's fixed-point math -- clip to [0,127],
the pairwise product of the accumulator halves, and the /128, /64, /16 shifts --
so the exported int weights evaluate (near) identically in the C++ engine. Loss
is in win-probability space: MSE(tanh(pred/400), tanh(cp/400)).

Data: .npz shards from build_stk_data.py (white_indices/black_indices/stm/eval_cp).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = 22528
HIDDEN = 2048
PAIR = HIDDEN // 2
L1_IN = 2 * PAIR
L1 = 16
L2 = 32
BUCKETS = 8
ACT_MAX = 127.0
PAIR_DIV = 128.0  # >> 7
DENSE_DIV = 64.0  # >> 6
OUT_DIV = 16.0  # >> 4
CP_SCALE = 400.0


def clip(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, ACT_MAX)


class STKNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Row 0 is a fixed zero padding row (feature indices are shifted +1).
        self.ft = nn.EmbeddingBag(FEATURES + 1, HIDDEN, mode="sum", padding_idx=0)
        self.ft_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.psqt = nn.EmbeddingBag(FEATURES + 1, BUCKETS, mode="sum", padding_idx=0)
        self.l1w = nn.Parameter(torch.empty(BUCKETS, L1, L1_IN).uniform_(-0.1, 0.1))
        self.l1b = nn.Parameter(torch.zeros(BUCKETS, L1))
        self.l2w = nn.Parameter(torch.empty(BUCKETS, L2, L1).uniform_(-0.2, 0.2))
        self.l2b = nn.Parameter(torch.zeros(BUCKETS, L2))
        self.outw = nn.Parameter(torch.empty(BUCKETS, L2).uniform_(-0.2, 0.2))
        self.outb = nn.Parameter(torch.zeros(BUCKETS))
        nn.init.uniform_(self.ft.weight, -0.02, 0.02)
        nn.init.uniform_(self.psqt.weight, -2.0, 2.0)
        with torch.no_grad():
            self.ft.weight[0].zero_()
            self.psqt.weight[0].zero_()

    def accumulate(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = idx + 1  # -1 padding -> row 0
        return self.ft(shifted) + self.ft_bias, self.psqt(shifted)

    def forward(
        self,
        white_idx: torch.Tensor,
        black_idx: torch.Tensor,
        stm: torch.Tensor,
        bucket: torch.Tensor,
    ) -> torch.Tensor:
        acc_w, psq_w = self.accumulate(white_idx)
        acc_b, psq_b = self.accumulate(black_idx)
        w_is_stm = (stm == 0).unsqueeze(1)
        us, them = torch.where(w_is_stm, acc_w, acc_b), torch.where(w_is_stm, acc_b, acc_w)
        psq_us = torch.where(stm.unsqueeze(1) == 0, psq_w, psq_b)
        psq_them = torch.where(stm.unsqueeze(1) == 0, psq_b, psq_w)

        def pairwise(a: torch.Tensor) -> torch.Tensor:
            c = clip(a)
            return c[:, :PAIR] * c[:, PAIR:] / PAIR_DIV

        x = torch.cat([pairwise(us), pairwise(them)], dim=1)  # [B, L1_IN]

        l1 = clip((torch.einsum("bi,koi->bko", x, self.l1w) + self.l1b) / DENSE_DIV)
        l2 = clip((torch.einsum("bko,klo->bkl", l1, self.l2w) + self.l2b) / DENSE_DIV)
        out_all = (torch.einsum("bkl,kl->bk", l2, self.outw) + self.outb) / OUT_DIV

        rows = torch.arange(bucket.shape[0], device=bucket.device)
        positional = out_all[rows, bucket]
        psqt_term = (psq_us[rows, bucket] - psq_them[rows, bucket]) / OUT_DIV
        return positional + psqt_term


def load_shards(data_dir: Path) -> TensorDataset:
    wi, bi, stm, cp = [], [], [], []
    files = sorted(glob.glob(str(data_dir / "*.npz")))
    if not files:
        raise SystemExit(f"no .npz shards in {data_dir}")
    for f in files:
        d = np.load(f)
        wi.append(d["white_indices"])
        bi.append(d["black_indices"])
        stm.append(d["stm"])
        cp.append(d["eval_cp"])
    w = torch.from_numpy(np.concatenate(wi)).long()
    b = torch.from_numpy(np.concatenate(bi)).long()
    s = torch.from_numpy(np.concatenate(stm)).long()
    c = torch.from_numpy(np.concatenate(cp)).float()
    piece_count = (w >= 0).sum(dim=1) + 1  # + own king
    bucket = torch.clamp((piece_count - 1) // 4, 0, BUCKETS - 1)
    return TensorDataset(w, b, s, c, bucket)


def main() -> int:
    p = argparse.ArgumentParser(description="Train STK-HalfKA NNUE (2048).")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    ds = load_shards(Path(args.data))
    loader: DataLoader[Any] = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = STKNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, args.epochs * len(loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    print(f"device={device} samples={len(ds)} batches/epoch={len(loader)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for w, b, s, cp, bucket in loader:
            w, b, s = w.to(device), b.to(device), s.to(device)
            cp, bucket = cp.to(device), bucket.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                pred = model(w, b, s, bucket)
                loss = torch.mean((torch.tanh(pred / CP_SCALE) - torch.tanh(cp / CP_SCALE)) ** 2)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss.detach()) * w.shape[0]
            seen += w.shape[0]
        print(f"epoch {epoch:3d}  loss {total / max(seen, 1):.6f}  lr {sched.get_last_lr()[0]:.2e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": "STK-HalfKA-2048"}, args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
