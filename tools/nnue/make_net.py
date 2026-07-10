#!/usr/bin/env python3
"""One-command STK-HalfKA NNUE pipeline: data prep -> train -> export -> verify.

    python tools/nnue/make_net.py --input lichess_db_eval.jsonl.zst --workdir runs/v1

Re-running the same command resumes wherever it stopped (prep line count,
training epoch/shard, export). Safe to Ctrl-C or lose power at any point.

Hyperparameters follow bullet's canonical recipe (examples/simple.rs): AdamW
(decay 0.01, weight clipping +/-127/64), lr 1e-3 with x0.1 drops at 45%/90% of
the run, batch 16384, sigmoid(eval/400) squared-error loss, and ~4B total
position-visits (e.g. 20 epochs over a 200M-position dataset). WDL blending is
omitted only because the Lichess eval DB carries no game results.

Training uses the standard NNUE quantization scheme (as in bullet/nnue-pytorch):
the model is trained in a normalized float domain (activations clipped to
[0, 1], dense weights clamped to the int8-representable +/-127/64), and the
fixed engine scales are applied only at export:

    ft weights/bias  x127 -> int16      dense weights x64  -> int8
    dense biases     x8128 -> int32     psqt          x16  -> int32

which reproduces the engine's integer pipeline (clip 127, >>7 pairwise, >>6
dense, >>4 output) so eval_cp = 508 * float_out + psqt_cp. The final step
loads the exported net into the engine and asserts Python == engine on a set
of test positions.

VRAM: the 1024-wide model needs < 1 GB; batches auto-halve on CUDA OOM.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_stk_data as prep  # noqa: E402
import halfka_features as hk  # noqa: E402

FEATURES = 22528
HIDDEN = 1024
PAIR = HIDDEN // 2
L1 = 16
L2 = 32
BUCKETS = 8

QA = 127.0  # activation quantization (clip ceiling)
QB = 64.0  # dense weight quantization
WMAX = QA / QB  # trainable dense-weight range, ~1.984
OUT_CP = QA * QB / 16.0  # 508: float output -> centipawns (engine >>4)
CP_SCALE = 400.0  # tanh() sharpness of the WDL loss
PAIR_FACTOR = QA / 128.0  # exact float image of the engine's (a*b)>>7

BENCH_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bq1rk1/pp2bppp/2n2n2/2pp4/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "8/8/1p1k4/p1p2p2/P1P2P2/1P1K4/8/8 w - - 0 1",
]


class STKNet(nn.Module):
    """Normalized-domain model of src/nnue/network.{h,cpp}."""

    def __init__(self) -> None:
        super().__init__()
        # Row 0 is a frozen zero row for -1 padding (indices are shifted +1).
        self.ft = nn.EmbeddingBag(FEATURES + 1, HIDDEN, mode="sum", padding_idx=0)
        self.ft_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.psqt = nn.EmbeddingBag(FEATURES + 1, BUCKETS, mode="sum", padding_idx=0)
        self.l1w = nn.Parameter(torch.empty(BUCKETS, L1, 2 * PAIR))
        self.l1b = nn.Parameter(torch.zeros(BUCKETS, L1))
        self.l2w = nn.Parameter(torch.empty(BUCKETS, L2, L1))
        self.l2b = nn.Parameter(torch.zeros(BUCKETS, L2))
        self.outw = nn.Parameter(torch.empty(BUCKETS, L2))
        self.outb = nn.Parameter(torch.zeros(BUCKETS))
        nn.init.uniform_(self.ft.weight, -0.05, 0.05)
        nn.init.uniform_(self.psqt.weight, -1.0, 1.0)
        for w in (self.l1w, self.l2w, self.outw):
            nn.init.uniform_(w, -0.5, 0.5)
        with torch.no_grad():
            self.ft.weight[0].zero_()
            self.psqt.weight[0].zero_()

    def clip_weights(self) -> None:
        with torch.no_grad():
            for w in (self.l1w, self.l2w, self.outw):
                w.clamp_(-WMAX, WMAX)

    def forward(
        self,
        white_idx: torch.Tensor,
        black_idx: torch.Tensor,
        stm: torch.Tensor,
        bucket: torch.Tensor,
    ) -> torch.Tensor:
        wi, bi = white_idx + 1, black_idx + 1
        acc_w = self.ft(wi) + self.ft_bias
        acc_b = self.ft(bi) + self.ft_bias
        psq_w, psq_b = self.psqt(wi), self.psqt(bi)

        white_stm = (stm == 0).unsqueeze(1)
        us = torch.where(white_stm, acc_w, acc_b)
        them = torch.where(white_stm, acc_b, acc_w)
        psq_us = torch.where(white_stm, psq_w, psq_b)
        psq_them = torch.where(white_stm, psq_b, psq_w)

        def pairwise(a: torch.Tensor) -> torch.Tensor:
            c = torch.clamp(a, 0.0, 1.0)
            return c[:, :PAIR] * c[:, PAIR:] * PAIR_FACTOR

        x = torch.cat([pairwise(us), pairwise(them)], dim=1)
        l1 = torch.clamp(torch.einsum("bi,koi->bko", x, self.l1w) + self.l1b, 0.0, 1.0)
        l2 = torch.clamp(torch.einsum("bko,klo->bkl", l1, self.l2w) + self.l2b, 0.0, 1.0)
        out = torch.einsum("bkl,kl->bk", l2, self.outw) + self.outb

        rows = torch.arange(bucket.shape[0], device=bucket.device)
        positional = out[rows, bucket] * OUT_CP
        psqt_cp = psq_us[rows, bucket] - psq_them[rows, bucket]
        return positional + psqt_cp


# ---------------------------------------------------------------- data prep


def stage_prep(args: argparse.Namespace, work: Path) -> Path:
    data_dir = work / "data"
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        total = json.loads(manifest.read_text())["total_samples"]
        print(f"[prep] complete ({total} samples), skipping")
        return data_dir
    if not args.input:
        raise SystemExit("[prep] no shards yet and --input not given")
    data_dir.mkdir(parents=True, exist_ok=True)

    progress_path = data_dir / "progress.json"
    lines_done = shards_done = kept = 0
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
        lines_done, shards_done, kept = state["lines"], state["shards"], state["kept"]
        print(f"[prep] resuming: {lines_done} lines / {kept} kept / {shards_done} shards")

    w_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    stm_rows: list[int] = []
    cp_rows: list[float] = []
    shards: list[dict[str, Any]] = [
        {"file": f"stk_shard_{i:05d}.npz", "samples": args.shard_size} for i in range(shards_done)
    ]

    def flush() -> None:
        nonlocal shards_done
        if not cp_rows:
            return
        name = f"stk_shard_{shards_done:05d}.npz"
        np.savez_compressed(
            data_dir / name,
            white_indices=np.stack(w_rows).astype(np.int32),
            black_indices=np.stack(b_rows).astype(np.int32),
            stm=np.asarray(stm_rows, dtype=np.int8),
            eval_cp=np.asarray(cp_rows, dtype=np.float32),
        )
        shards.append({"file": name, "samples": len(cp_rows)})
        shards_done += 1
        w_rows.clear()
        b_rows.clear()
        stm_rows.clear()
        cp_rows.clear()
        progress_path.write_text(json.dumps({"lines": line_no, "shards": shards_done, "kept": kept}))

    line_no = 0
    t0 = time.time()
    for line in prep.iter_lines(Path(args.input)):
        line_no += 1
        if line_no <= lines_done:
            continue  # fast-forward on resume
        parsed = prep.parse_line(line, args.min_depth)
        if parsed is None:
            continue
        fen, cp = parsed
        if abs(cp) > args.max_abs_cp:
            continue
        try:
            w, b, stm = prep.encode(fen)
        except Exception:
            continue
        w_rows.append(w)
        b_rows.append(b)
        stm_rows.append(stm)
        cp_rows.append(cp)
        kept += 1
        if kept % 500_000 == 0:
            rate = kept / max(time.time() - t0, 1e-9)
            print(f"[prep] kept {kept} ({rate:.0f}/s)")
        if len(cp_rows) >= args.shard_size:
            flush()
        if args.max_samples and kept >= args.max_samples:
            break
    flush()
    manifest.write_text(json.dumps({"total_samples": kept, "shards": shards}, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    print(f"[prep] done: {kept} samples, {shards_done} shards")
    return data_dir


# ------------------------------------------------------------------- train


def load_shard(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in ("white_indices", "black_indices", "stm", "eval_cp")}


def stage_train(args: argparse.Namespace, work: Path, data_dir: Path) -> Path:
    ckpt_path = work / "ckpt_latest.pt"
    done_path = work / "train_done.json"
    float_path = work / "model_float.pt"
    if done_path.exists():
        print("[train] complete, skipping")
        return float_path

    manifest = json.loads((data_dir / "manifest.json").read_text())
    shard_files = [data_dir / s["file"] for s in manifest["shards"]]
    if not shard_files:
        raise SystemExit("[train] no shards")

    device = torch.device(args.device)
    model = STKNet().to(device)
    # bullet's canonical recipe (examples/simple.rs): AdamW (decay 0.01, weight
    # clipping +/-1.98 == our +/-127/64), lr 1e-3 dropping x0.1 at 45% and 90%
    # of the run (their step 18 of 40 superbatches), batch 16384.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, round(0.45 * args.epochs)), gamma=0.1)

    epoch0, shard0, batch = 1, 0, args.batch_size
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        epoch0, shard0, batch = ck["epoch"], ck["next_shard"], ck["batch"]
        print(f"[train] resuming at epoch {epoch0}, shard {shard0}, batch {batch}")

    def save_ckpt(epoch: int, next_shard: int) -> None:
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "next_shard": next_shard,
                "batch": batch,
            },
            tmp,
        )
        os.replace(tmp, ckpt_path)

    print(f"[train] device={device} shards={len(shard_files)} epochs={args.epochs} batch={batch}")
    for epoch in range(epoch0, args.epochs + 1):
        order = np.random.default_rng(1000 + epoch).permutation(len(shard_files))
        model.train()
        loss_sum, seen, t0 = 0.0, 0, time.time()
        si = shard0
        while si < len(shard_files):
            # Load a group of --mix-shards shards and shuffle across their
            # union: the Lichess DB is not randomly ordered, so single-shard
            # batches are thematically correlated. mix=1 == original behavior.
            group_end = min(si + args.mix_shards, len(shard_files))
            loaded = [load_shard(shard_files[order[j]]) for j in range(si, group_end)]
            d = {k: np.concatenate([x[k] for x in loaded]) for k in loaded[0]}
            del loaded
            n = d["stm"].shape[0]
            perm = np.random.default_rng(epoch * 100_003 + si).permutation(n)
            pos = 0
            while pos < n:
                sel = perm[pos : pos + batch]
                try:
                    w = torch.from_numpy(d["white_indices"][sel]).long().to(device)
                    b = torch.from_numpy(d["black_indices"][sel]).long().to(device)
                    stm = torch.from_numpy(d["stm"][sel]).long().to(device)
                    cp = torch.from_numpy(d["eval_cp"][sel]).float().to(device)
                    pieces = (w >= 0).sum(dim=1) + 1
                    bucket = torch.clamp((pieces - 1) // 4, 0, BUCKETS - 1)
                    opt.zero_grad(set_to_none=True)
                    pred = model(w, b, stm, bucket)
                    # bullet loss form: sigmoid(eval/400) squared error.
                    loss = torch.mean((torch.sigmoid(pred / CP_SCALE) - torch.sigmoid(cp / CP_SCALE)) ** 2)
                    loss.backward()
                    opt.step()
                    model.clip_weights()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    batch = max(1024, batch // 2)
                    print(f"[train] CUDA OOM -> batch {batch}")
                    continue
                pos += len(sel)
                loss_sum += float(loss.detach()) * len(sel)
                seen += len(sel)
            save_ckpt(epoch, group_end)
            si = group_end
        sched.step()
        shard0 = 0
        save_ckpt(epoch + 1, 0)
        rate = seen / max(time.time() - t0, 1e-9)
        print(
            f"[train] epoch {epoch:3d}/{args.epochs}  loss {loss_sum / max(seen, 1):.6f}  "
            f"{rate:.0f} pos/s  lr {sched.get_last_lr()[0]:.2e}"
        )

    torch.save({"state_dict": model.state_dict()}, float_path)
    done_path.write_text(json.dumps({"epochs": args.epochs}) + "\n")
    print(f"[train] done -> {float_path}")
    return float_path


# ------------------------------------------------------------------ export


def quant(t: torch.Tensor, scale: float, lo: int, hi: int, dtype: str) -> np.ndarray:
    q = torch.round(t.detach().cpu().float() * scale).clamp(lo, hi)
    return q.numpy().astype(dtype)


def quantize_model(model: STKNet) -> dict[str, np.ndarray]:
    """The exact integer tensors written to the .nnue (padding row dropped)."""
    i32 = (-(2**31) + 1, 2**31 - 1)
    q: dict[str, np.ndarray] = {
        "ft_w": quant(model.ft.weight[1:], QA, -32767, 32767, "<i2"),
        "ft_b": quant(model.ft_bias, QA, -32767, 32767, "<i2"),
        "ft_p": quant(model.psqt.weight[1:], 16.0, *i32, "<i4"),
        "l1w": quant(model.l1w, QB, -127, 127, "i1"),
        "l1b": quant(model.l1b, QA * QB, *i32, "<i4"),
        "l2w": quant(model.l2w, QB, -127, 127, "i1"),
        "l2b": quant(model.l2b, QA * QB, *i32, "<i4"),
        "outw": quant(model.outw, QB, -127, 127, "i1"),
        "outb": quant(model.outb, QA * QB, *i32, "<i4"),
    }
    return q


def load_float_model(float_path: Path) -> STKNet:
    model = STKNet()
    model.load_state_dict(torch.load(float_path, map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    return model


def stage_export(work: Path, float_path: Path) -> Path:
    out = work / "stk_halfka_1024.nnue"
    q = quantize_model(load_float_model(float_path))
    with out.open("wb") as fh:
        fh.write(struct.pack("<9sIIII", b"STKHALFKA", 1, FEATURES, HIDDEN, BUCKETS))
        fh.write(q["ft_w"].tobytes())
        fh.write(q["ft_b"].tobytes())
        fh.write(q["ft_p"].tobytes())
        for k in range(BUCKETS):
            for name in ("l1w", "l1b", "l2w", "l2b", "outw", "outb"):
                fh.write(q[name][k].tobytes())

    # Worst-case accumulator estimate: ~32 active features of the largest
    # magnitude plus the bias must stay far inside int16.
    reach = 33 * int(np.abs(q["ft_w"]).max()) + int(np.abs(q["ft_b"]).max())
    print(f"[export] {out} ({out.stat().st_size / 1e6:.1f} MB), max |ft int16| {np.abs(q['ft_w']).max()}")
    if reach > 30000:
        print(f"[export] WARNING: accumulator may overflow int16 (worst-case reach {reach})")
    return out


# ------------------------------------------------------------------ verify


def python_eval(model: STKNet, fen: str) -> int:
    pieces, stm = hk.parse_fen_pieces(fen)
    wf = hk.features_for(pieces, hk.WHITE)
    bf = hk.features_for(pieces, hk.BLACK)
    pad = 40
    w = torch.full((1, pad), -1, dtype=torch.long)
    b = torch.full((1, pad), -1, dtype=torch.long)
    w[0, : len(wf)] = torch.tensor(wf)
    b[0, : len(bf)] = torch.tensor(bf)
    bucket = torch.tensor([min(max((len(pieces) - 1) // 4, 0), BUCKETS - 1)])
    with torch.no_grad():
        cp = model(w, b, torch.tensor([stm]), bucket)
    return int(round(float(cp[0])))


def python_eval_int(q: dict[str, np.ndarray], fen: str) -> int:
    """Mirror of the engine's integer forward over the exported tensors."""
    pieces, stm = hk.parse_fen_pieces(fen)
    feats = [hk.features_for(pieces, hk.WHITE), hk.features_for(pieces, hk.BLACK)]
    acc = [q["ft_b"].astype(np.int32) + q["ft_w"][f].astype(np.int32).sum(axis=0) for f in feats]
    psq = [q["ft_p"][f].astype(np.int64).sum(axis=0) for f in feats]
    bucket = min(max((len(pieces) - 1) // 4, 0), BUCKETS - 1)

    us, them = (0, 1) if stm == hk.WHITE else (1, 0)

    def pairwise(a: np.ndarray) -> np.ndarray:
        c = np.clip(a, 0, 127)
        return (c[:PAIR] * c[PAIR:]) >> 7

    x = np.concatenate([pairwise(acc[us]), pairwise(acc[them])])
    l1 = np.clip((q["l1b"][bucket] + q["l1w"][bucket].astype(np.int32) @ x) >> 6, 0, 127)
    l2 = np.clip((q["l2b"][bucket] + q["l2w"][bucket].astype(np.int32) @ l1) >> 6, 0, 127)
    raw = int(q["outb"][bucket]) + int(q["outw"][bucket].astype(np.int32) @ l2)
    positional = raw >> 4
    psqt_term = int(psq[us][bucket] - psq[them][bucket]) >> 4
    return positional + psqt_term


def find_engine(cli: str | None) -> Path | None:
    if cli:
        return Path(cli)
    for c in ("build-bench/bin/ChessEngine.exe", "build/bin/ChessEngine.exe", "build/bin/ChessEngine"):
        p = Path(c)
        if p.exists():
            return p
    return None


def stage_verify(args: argparse.Namespace, float_path: Path, net_path: Path) -> None:
    engine = find_engine(args.engine)
    if engine is None:
        print("[verify] WARNING: engine binary not found, skipping parity check")
        return
    model = load_float_model(float_path)
    q = quantize_model(model)

    lines = [f"setoption name EvalFile value {net_path}"]
    for fen in BENCH_FENS:
        lines += [f"position fen {fen}", "eval"]
    lines.append("quit")
    res = subprocess.run(
        [str(engine)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=120, check=False
    )
    scores = [int(tok.split("score:")[1].split("cp")[0]) for tok in res.stdout.splitlines() if "score:" in tok]
    if "EvalFile loaded" not in res.stdout or len(scores) != len(BENCH_FENS):
        raise SystemExit(f"[verify] FAIL: engine did not load/evaluate the net\n{res.stdout[-800:]}")

    # Engine must match the integer reference exactly (same math, same ints);
    # the float column is informational (quantization drift).
    worst = 0
    for fen, engine_cp in zip(BENCH_FENS, scores, strict=True):
        int_cp = python_eval_int(q, fen)
        float_cp = python_eval(model, fen)
        diff = abs(int_cp - engine_cp)
        worst = max(worst, diff)
        print(
            f"[verify] engine {engine_cp:6d}  py-int {int_cp:6d}  diff {diff:2d}   "
            f"(float {float_cp:6d})  {fen.split()[0][:20]}"
        )
    if worst > args.tolerance_cp:
        raise SystemExit(f"[verify] FAIL: engine != quantized reference (max diff {worst} cp)")
    print(f"[verify] PASS: engine == quantized reference (max diff {worst} cp)")


# -------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description="STK-HalfKA NNUE pipeline (prep/train/export/verify, resumable).")
    p.add_argument("--input", default=None, help=".jsonl(.zst) lichess evals or '<fen> | <cp>' text")
    p.add_argument("--workdir", required=True)
    p.add_argument("--epochs", type=int, default=20, help="20 x 200M ~= bullet's canonical 4B visits")
    p.add_argument("--mix-shards", type=int, default=1, help="shards shuffled together per group (8 recommended)")
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--max-samples", type=int, default=0, help="cap kept positions (0 = all)")
    p.add_argument("--min-depth", type=int, default=0)
    p.add_argument("--max-abs-cp", type=float, default=6000.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--engine", default=None, help="ChessEngine binary for the parity check")
    p.add_argument("--tolerance-cp", type=int, default=1)
    args = p.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    data_dir = stage_prep(args, work)
    float_path = stage_train(args, work, data_dir)
    net_path = stage_export(work, float_path)
    stage_verify(args, float_path, net_path)
    print(f"\nAll stages complete. Net: {net_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
