#!/usr/bin/env python3
"""
Train a Vector-64-compatible NNUE checkpoint in PyTorch.

The output checkpoint is intentionally shaped for tools/nnue/export_vector64.py:
  - feature_transform.weight  [81920, 512]
  - feature_transform.bias    [512]
  - dense1.weight/.bias       [32, 512] / [32]
  - dense2.weight/.bias       [32, 32]  / [32]
  - output.weight/.bias       [1, 32]   / [1]

Dataset format (.npz):
  Required:
    white_indices: int32/int64 [N, W] (HalfKP indices for white perspective, -1 padded)
    black_indices: int32/int64 [N, B] (HalfKP indices for black perspective, -1 padded)
    stm         : int8/int32   [N]     (0 = white to move, 1 = black to move)
    eval_cp     : float32      [N]     (centipawn eval from white perspective)

  Optional:
    white_values: float32      [N, W]   (feature coefficients; defaults to 1.0)
    black_values: float32      [N, B]   (feature coefficients; defaults to 1.0)
    result      : float32      [N]      (game result from white perspective, in [-1, 1])
    sample_weight: float32     [N]      (per-sample training weight)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


TOTAL_FEATURES = 81920
HIDDEN = 512
L1 = 32
L2 = 32

FEATURE_TO_DENSE_SCALE = 64.0
DENSE_TO_DENSE_SCALE = 64.0
OUTPUT_SCALE = 16.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Vector-64 NNUE model.")
    p.add_argument("--dataset", required=True, help="Path to dataset .npz")
    p.add_argument(
        "--dataset-manifest",
        default=None,
        help="Optional manifest/meta JSON path. If omitted, auto-resolves sidecar or parent manifest.json",
    )
    p.add_argument("--out-checkpoint", required=True, help="Path to output .pt checkpoint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--val-split", type=float, default=0.01, help="Validation split in [0, 0.5)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--result-blend", type=float, default=0.0, help="Blend factor for result target in [0,1]")
    p.add_argument("--result-scale", type=float, default=600.0, help="Result-to-centipawn scale")
    p.add_argument("--max-target-cp", type=float, default=2000.0, help="Clamp target cp to +/- this value")
    p.add_argument("--max-positions", type=int, default=0, help="Use first N positions (0 = all)")
    p.add_argument("--amp", action="store_true", help="Enable torch.cuda.amp autocast")
    p.add_argument("--log-every", type=int, default=100, help="Mini-batch logging interval")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_key(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        available = ", ".join(data.files)
        raise KeyError(f"Dataset missing '{key}'. Available keys: {available}")
    return data[key]


@dataclass
class LoadedDataset:
    white_idx: torch.Tensor
    black_idx: torch.Tensor
    stm: torch.Tensor
    target_cp: torch.Tensor
    white_vals: Optional[torch.Tensor]
    black_vals: Optional[torch.Tensor]
    sample_weight: Optional[torch.Tensor]
    raw_count: int


def load_dataset(
    path: Path,
    *,
    result_blend: float,
    result_scale: float,
    max_target_cp: float,
    max_positions: int,
) -> LoadedDataset:
    if not 0.0 <= result_blend <= 1.0:
        raise ValueError("--result-blend must be in [0, 1]")
    if max_target_cp <= 0:
        raise ValueError("--max-target-cp must be > 0")

    data = np.load(path)

    white_idx = require_key(data, "white_indices").astype(np.int64, copy=False)
    black_idx = require_key(data, "black_indices").astype(np.int64, copy=False)
    stm = require_key(data, "stm").astype(np.int64, copy=False).reshape(-1)
    eval_cp = require_key(data, "eval_cp").astype(np.float32, copy=False).reshape(-1)

    raw_count = int(eval_cp.shape[0])
    n = raw_count
    if white_idx.shape[0] != n or black_idx.shape[0] != n or stm.shape[0] != n:
        raise ValueError(
            "Dataset row mismatch: white_indices, black_indices, stm, eval_cp must share first dimension."
        )

    if max_positions > 0:
        n = min(n, max_positions)
        white_idx = white_idx[:n]
        black_idx = black_idx[:n]
        stm = stm[:n]
        eval_cp = eval_cp[:n]

    if np.any((stm != 0) & (stm != 1)):
        raise ValueError("'stm' must only contain 0 (white) or 1 (black).")

    # Validate index ranges with -1 padding convention.
    if np.any(white_idx < -1) or np.any(black_idx < -1):
        raise ValueError("Feature indices must be -1 (padding) or non-negative.")
    if np.any(white_idx >= TOTAL_FEATURES) or np.any(black_idx >= TOTAL_FEATURES):
        raise ValueError(f"Feature indices must be < {TOTAL_FEATURES}.")

    # Convert white-perspective centipawns to side-to-move perspective.
    stm_sign = np.where(stm == 0, 1.0, -1.0).astype(np.float32)
    target_cp = eval_cp * stm_sign

    if "result" in data and result_blend > 0.0:
        result = data["result"].astype(np.float32, copy=False).reshape(-1)
        if result.shape[0] < n:
            raise ValueError("'result' length is smaller than selected dataset size.")
        result = result[:n]
        result_stm = result * stm_sign
        blended = (1.0 - result_blend) * target_cp + result_blend * (result_stm * result_scale)
        target_cp = blended

    target_cp = np.clip(target_cp, -max_target_cp, max_target_cp)

    white_vals = None
    if "white_values" in data:
        white_vals_arr = data["white_values"].astype(np.float32, copy=False)
        if white_vals_arr.shape[0] < n or white_vals_arr.shape[1:] != white_idx.shape[1:]:
            raise ValueError("'white_values' shape must match 'white_indices' (or be a full superset on axis 0).")
        white_vals = torch.from_numpy(white_vals_arr[:n])

    black_vals = None
    if "black_values" in data:
        black_vals_arr = data["black_values"].astype(np.float32, copy=False)
        if black_vals_arr.shape[0] < n or black_vals_arr.shape[1:] != black_idx.shape[1:]:
            raise ValueError("'black_values' shape must match 'black_indices' (or be a full superset on axis 0).")
        black_vals = torch.from_numpy(black_vals_arr[:n])

    sample_weight = None
    if "sample_weight" in data:
        weight_arr = data["sample_weight"].astype(np.float32, copy=False).reshape(-1)
        if weight_arr.shape[0] < n:
            raise ValueError("'sample_weight' length is smaller than selected dataset size.")
        sample_weight = torch.from_numpy(weight_arr[:n])

    return LoadedDataset(
        white_idx=torch.from_numpy(white_idx[:n]),
        black_idx=torch.from_numpy(black_idx[:n]),
        stm=torch.from_numpy(stm[:n]),
        target_cp=torch.from_numpy(target_cp[:n]),
        white_vals=white_vals,
        black_vals=black_vals,
        sample_weight=sample_weight,
        raw_count=raw_count,
    )


def dataset_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_manifest_path(dataset_path: Path, explicit_manifest: Optional[str]) -> Path:
    if explicit_manifest:
        p = Path(explicit_manifest)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")
        return p

    sidecar = dataset_path.with_suffix(".meta.json")
    if sidecar.exists():
        return sidecar

    parent_manifest = dataset_path.parent / "manifest.json"
    if parent_manifest.exists():
        return parent_manifest

    raise FileNotFoundError(
        "No dataset manifest found. Provide --dataset-manifest, or ensure either "
        f"{sidecar.name} or manifest.json exists next to dataset shards."
    )


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return obj


def expected_samples_from_manifest(manifest: dict, dataset_path: Path) -> int:
    if "samples" in manifest:
        return int(manifest["samples"])

    shards = manifest.get("shards")
    if isinstance(shards, list):
        for shard in shards:
            if not isinstance(shard, dict):
                continue
            if str(shard.get("file", "")) == dataset_path.name:
                if "samples" not in shard:
                    raise ValueError(f"Shard entry has no 'samples' for {dataset_path.name}")
                return int(shard["samples"])

    # Fallback for single-dataset manifests.
    if "total_samples" in manifest:
        if isinstance(shards, list) and len(shards) > 1:
            raise ValueError(
                f"Dataset {dataset_path.name} not found in manifest shards list; cannot infer expected samples."
            )
        return int(manifest["total_samples"])

    raise ValueError("Manifest does not contain sample count fields ('samples' or 'total_samples').")


def cp_distribution(cp: np.ndarray) -> dict[str, Any]:
    if cp.ndim != 1:
        cp = cp.reshape(-1)
    if cp.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "hist_edges": [],
            "hist_counts": [],
        }

    hist_counts, hist_edges = np.histogram(cp, bins=20)
    return {
        "count": int(cp.size),
        "min": float(np.min(cp)),
        "max": float(np.max(cp)),
        "mean": float(np.mean(cp)),
        "std": float(np.std(cp)),
        "p01": float(np.percentile(cp, 1)),
        "p05": float(np.percentile(cp, 5)),
        "p50": float(np.percentile(cp, 50)),
        "p95": float(np.percentile(cp, 95)),
        "p99": float(np.percentile(cp, 99)),
        "hist_edges": [float(x) for x in hist_edges.tolist()],
        "hist_counts": [int(x) for x in hist_counts.tolist()],
    }


def assert_finite_tensor(name: str, t: torch.Tensor) -> None:
    arr = t.detach().cpu().numpy()
    if not np.isfinite(arr).all():
        raise ValueError(f"Detected NaN/Inf in tensor: {name}")


def run_dataset_preflight(
    *,
    dataset_path: Path,
    manifest_path: Path,
    manifest: dict,
    loaded: LoadedDataset,
) -> tuple[str, dict[str, Any]]:
    data_hash = dataset_sha256(dataset_path)
    expected_rows = expected_samples_from_manifest(manifest, dataset_path)
    if loaded.raw_count != expected_rows:
        raise ValueError(
            f"Sample count mismatch: dataset has {loaded.raw_count}, manifest expects {expected_rows} "
            f"(dataset={dataset_path.name}, manifest={manifest_path.name})."
        )

    manifest_hash = manifest.get("sha256")
    if manifest_hash is not None and str(manifest_hash).lower() != data_hash.lower():
        raise ValueError(
            f"Dataset hash mismatch: computed {data_hash}, manifest sha256={manifest_hash}"
        )

    assert_finite_tensor("target_cp", loaded.target_cp)
    assert_finite_tensor("white_idx", loaded.white_idx)
    assert_finite_tensor("black_idx", loaded.black_idx)
    assert_finite_tensor("stm", loaded.stm)
    if loaded.white_vals is not None:
        assert_finite_tensor("white_vals", loaded.white_vals)
    if loaded.black_vals is not None:
        assert_finite_tensor("black_vals", loaded.black_vals)
    if loaded.sample_weight is not None:
        assert_finite_tensor("sample_weight", loaded.sample_weight)

    cp = loaded.target_cp.detach().cpu().numpy().astype(np.float64, copy=False)
    cp_stats = cp_distribution(cp)
    if cp_stats["std"] <= 0.0:
        raise ValueError("Degenerate target distribution: stddev is zero.")

    print(f"Dataset SHA256: {data_hash}")
    print("Target CP distribution:")
    print(json.dumps(cp_stats, indent=2))

    return data_hash, cp_stats


class Vector64Dataset(Dataset):
    def __init__(self, loaded: LoadedDataset) -> None:
        self.loaded = loaded

    def __len__(self) -> int:
        return int(self.loaded.target_cp.shape[0])

    def __getitem__(self, idx: int):
        item = {
            "white_idx": self.loaded.white_idx[idx],
            "black_idx": self.loaded.black_idx[idx],
            "stm": self.loaded.stm[idx],
            "target_cp": self.loaded.target_cp[idx],
        }
        if self.loaded.white_vals is not None:
            item["white_vals"] = self.loaded.white_vals[idx]
        if self.loaded.black_vals is not None:
            item["black_vals"] = self.loaded.black_vals[idx]
        if self.loaded.sample_weight is not None:
            item["sample_weight"] = self.loaded.sample_weight[idx]
        return item


class FeatureTransform(nn.Module):
    def __init__(self, total_features: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(total_features, hidden))
        self.bias = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, indices: torch.Tensor, values: Optional[torch.Tensor]) -> torch.Tensor:
        # indices: [B, K], with -1 padding
        valid = indices >= 0
        safe = torch.clamp(indices, min=0)
        gathered = self.weight[safe]  # [B, K, H]

        if values is not None:
            gathered = gathered * values.unsqueeze(-1)

        gathered = gathered * valid.unsqueeze(-1)
        return gathered.sum(dim=1) + self.bias


class Vector64NNUE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_transform = FeatureTransform(TOTAL_FEATURES, HIDDEN)
        self.dense1 = nn.Linear(HIDDEN, L1)
        self.dense2 = nn.Linear(L1, L2)
        self.output = nn.Linear(L2, 1)

    def forward(
        self,
        white_idx: torch.Tensor,
        black_idx: torch.Tensor,
        stm: torch.Tensor,
        white_vals: Optional[torch.Tensor] = None,
        black_vals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        white_acc = self.feature_transform(white_idx, white_vals)
        black_acc = self.feature_transform(black_idx, black_vals)

        white_stm = (stm == 0).unsqueeze(1)
        us = torch.where(white_stm, white_acc, black_acc)
        them = torch.where(white_stm, black_acc, white_acc)

        x = F.relu((us - them) / FEATURE_TO_DENSE_SCALE)
        x = F.relu(self.dense1(x) / DENSE_TO_DENSE_SCALE)
        x = F.relu(self.dense2(x) / DENSE_TO_DENSE_SCALE)
        x = self.output(x).squeeze(1) / OUTPUT_SCALE
        return x


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    err = (pred - target) ** 2
    if weight is None:
        return err.mean()
    denom = torch.clamp(weight.sum(), min=1e-8)
    return (err * weight).sum() / denom


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True)
    return out


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    amp_enabled: bool,
    log_every: int,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    running_loss = 0.0
    running_abs = 0.0
    seen = 0
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            pred = model(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm"],
                batch.get("white_vals"),
                batch.get("black_vals"),
            )
            loss = weighted_mse(pred, batch["target_cp"], batch.get("sample_weight"))

        if train_mode:
            assert optimizer is not None
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bsz = int(batch["target_cp"].shape[0])
        seen += bsz
        running_loss += float(loss.detach().cpu().item()) * bsz
        running_abs += float(torch.mean(torch.abs(pred.detach() - batch["target_cp"])).cpu().item()) * bsz

        if log_every > 0 and step % log_every == 0:
            elapsed = time.time() - start
            sps = seen / max(elapsed, 1e-6)
            print(
                f"  step={step:5d}  loss={running_loss / seen:10.4f}  "
                f"mae_cp={running_abs / seen:9.3f}  samples/s={sps:8.1f}"
            )

    mean_loss = running_loss / max(seen, 1)
    mean_mae = running_abs / max(seen, 1)
    return mean_loss, mean_mae


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = choose_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(f"Device: {device} (amp={amp_enabled})")

    loaded = load_dataset(
        Path(args.dataset),
        result_blend=args.result_blend,
        result_scale=args.result_scale,
        max_target_cp=args.max_target_cp,
        max_positions=args.max_positions,
    )
    dataset = Vector64Dataset(loaded)

    total = len(dataset)
    if total < 2:
        raise RuntimeError("Dataset must have at least 2 positions.")

    if not 0.0 <= args.val_split < 0.5:
        raise ValueError("--val-split must be in [0, 0.5)")

    val_count = int(math.floor(total * args.val_split))
    val_count = min(max(val_count, 0), total - 1)
    train_count = total - val_count

    if val_count > 0:
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=generator)
    else:
        train_ds = dataset
        val_ds = None

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    model = Vector64NNUE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.min_lr,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    best_val = float("inf")
    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Samples: total={total}, train={train_count}, val={val_count}")
    print(f"Batches/epoch: {len(train_loader)}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_mae = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            log_every=args.log_every,
        )

        for _ in range(len(train_loader)):
            scheduler.step()
            global_step += 1

        msg = f"  train: loss={train_loss:.4f}, mae_cp={train_mae:.3f}, lr={scheduler.get_last_lr()[0]:.6g}"

        val_loss = None
        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_mae = run_epoch(
                    model,
                    val_loader,
                    device,
                    optimizer=None,
                    scaler=None,
                    amp_enabled=amp_enabled,
                    log_every=0,
                )
            msg += f" | val: loss={val_loss:.4f}, mae_cp={val_mae:.3f}"
        print(msg)

        should_save = val_loss is None or val_loss < best_val
        if should_save:
            if val_loss is not None:
                best_val = val_loss

            payload = {
                "state_dict": model.state_dict(),
                "meta": {
                    "project": "Vector-64",
                    "global_step": global_step,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
            }
            torch.save(payload, out_path)
            print(f"  saved: {out_path}")

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    summary = {
        "checkpoint": str(out_path),
        "total_samples": total,
        "train_samples": train_count,
        "val_samples": val_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "amp": amp_enabled,
        "result_blend": args.result_blend,
        "result_scale": args.result_scale,
        "max_target_cp": args.max_target_cp,
    }
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
