#!/usr/bin/env python3
"""
Export a PyTorch checkpoint to VECTOR64_NNUE binary format.

Binary layout (all little-endian):
  Header:
    magic[13] = "VECTOR64_NNUE"
    version   = uint32 (1)
    hidden    = uint32 (512)

  64-byte aligned sections:
    FeatureTransformWeights: int16[81920 * 512]
    HiddenLayer1Weights    : int8 [32 * 512]
    HiddenLayer2Weights    : int8 [32 * 32]
    OutputWeights          : int8 [32]
    Biases:
      FeatureTransformBias : int16[512]
      HiddenLayer1Bias     : int32[32]
      HiddenLayer2Bias     : int32[32]
      OutputBias           : int32[1]
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch


MAGIC = b"VECTOR64_NNUE"
VERSION = 1
ALIGNMENT = 64

TOTAL_FEATURES = 81920
HIDDEN = 512
L1 = 32
L2 = 32
I32_MAX = 2_147_483_647


@dataclass
class ExportStats:
    scales: Dict[str, float]
    sections: Dict[str, Dict[str, int]]
    file_size: int


def align64(offset: int) -> int:
    return (offset + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1)


def write_padding(fh) -> None:
    current = fh.tell()
    aligned = align64(current)
    if aligned > current:
        fh.write(b"\x00" * (aligned - current))


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")

    state = None
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    if state is None:
        if isinstance(obj, dict):
            state = obj
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        if isinstance(v, torch.nn.Parameter):
            v = v.detach()
        if torch.is_tensor(v):
            out[nk] = v.detach().cpu().contiguous()
    return out


def require_tensor(state: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state:
        available = ", ".join(sorted(state.keys())[:20])
        suffix = "..." if len(state) > 20 else ""
        raise KeyError(f"Missing tensor key '{key}'. Available (first 20): {available}{suffix}")
    return state[key]


def to_matrix(t: torch.Tensor, shape: Tuple[int, int], name: str) -> torch.Tensor:
    if t.ndim != 2:
        raise ValueError(f"{name}: expected 2D tensor, got shape {tuple(t.shape)}")
    if tuple(t.shape) == shape:
        return t.contiguous()
    if tuple(t.shape) == (shape[1], shape[0]):
        return t.transpose(0, 1).contiguous()
    raise ValueError(f"{name}: expected shape {shape} (or transpose), got {tuple(t.shape)}")


def to_vector(t: torch.Tensor, size: int, name: str) -> torch.Tensor:
    if t.numel() != size:
        raise ValueError(f"{name}: expected {size} values, got {t.numel()} (shape {tuple(t.shape)})")
    return t.reshape(size).contiguous()


def to_scalar(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.numel() != 1:
        raise ValueError(f"{name}: expected scalar tensor, got shape {tuple(t.shape)}")
    return t.reshape(()).contiguous()


def quantize_sym(
    t: torch.Tensor,
    *,
    max_q: int,
    dtype: torch.dtype,
    explicit_scale: float | None,
    name: str,
) -> Tuple[torch.Tensor, float]:
    if explicit_scale is not None and explicit_scale <= 0:
        raise ValueError(f"{name}: scale must be positive, got {explicit_scale}")

    if torch.is_floating_point(t):
        if explicit_scale is None:
            max_abs = float(torch.max(torch.abs(t)).item())
            scale = max_abs / max_q if max_abs > 0 else 1.0
        else:
            scale = explicit_scale
        q = torch.round(t / scale)
    else:
        scale = 1.0 if explicit_scale is None else explicit_scale
        if explicit_scale is not None and explicit_scale != 1.0:
            q = torch.round(t.to(torch.float32) / explicit_scale)
        else:
            q = t

    q = q.clamp(-max_q, max_q).to(dtype).contiguous()
    return q, float(scale)


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def write_tensor(fh, t: torch.Tensor) -> None:
    fh.write(t.cpu().contiguous().numpy().tobytes(order="C"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PyTorch checkpoint to VECTOR64_NNUE binary.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt/.pth checkpoint")
    p.add_argument("--output", required=True, help="Output .nnue path")

    p.add_argument("--ft-weight-key", default="feature_transform.weight")
    p.add_argument("--ft-bias-key", default="feature_transform.bias")
    p.add_argument("--l1-weight-key", default="dense1.weight")
    p.add_argument("--l1-bias-key", default="dense1.bias")
    p.add_argument("--l2-weight-key", default="dense2.weight")
    p.add_argument("--l2-bias-key", default="dense2.bias")
    p.add_argument("--out-weight-key", default="output.weight")
    p.add_argument("--out-bias-key", default="output.bias")

    p.add_argument("--ft-scale", type=float, default=None, help="FeatureTransform weight scale (float->int16)")
    p.add_argument("--l1-scale", type=float, default=None, help="Dense1 weight scale (float->int8)")
    p.add_argument("--l2-scale", type=float, default=None, help="Dense2 weight scale (float->int8)")
    p.add_argument("--out-scale", type=float, default=None, help="Output weight scale (float->int8)")

    p.add_argument(
        "--ft-bias-scale",
        type=float,
        default=None,
        help="FeatureTransform bias scale (defaults to resolved --ft-scale)",
    )
    p.add_argument("--l1-bias-scale", type=float, default=1.0, help="Dense1 bias scale (float->int32)")
    p.add_argument("--l2-bias-scale", type=float, default=1.0, help="Dense2 bias scale (float->int32)")
    p.add_argument("--out-bias-scale", type=float, default=1.0, help="Output bias scale (float->int32)")

    p.add_argument(
        "--meta",
        default=None,
        help="Optional metadata output path (defaults to <output>.meta.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    meta_path = Path(args.meta) if args.meta else out_path.with_suffix(out_path.suffix + ".meta.json")

    state = load_state_dict(ckpt_path)

    ft_w = to_matrix(require_tensor(state, args.ft_weight_key), (TOTAL_FEATURES, HIDDEN), "feature_transform.weight")
    ft_b = to_vector(require_tensor(state, args.ft_bias_key), HIDDEN, "feature_transform.bias")
    l1_w = to_matrix(require_tensor(state, args.l1_weight_key), (L1, HIDDEN), "dense1.weight")
    l1_b = to_vector(require_tensor(state, args.l1_bias_key), L1, "dense1.bias")
    l2_w = to_matrix(require_tensor(state, args.l2_weight_key), (L2, L1), "dense2.weight")
    l2_b = to_vector(require_tensor(state, args.l2_bias_key), L2, "dense2.bias")
    out_w = to_vector(require_tensor(state, args.out_weight_key), L2, "output.weight")
    out_b = to_scalar(require_tensor(state, args.out_bias_key), "output.bias")

    q_ft_w, s_ft_w = quantize_sym(ft_w, max_q=32767, dtype=torch.int16, explicit_scale=args.ft_scale, name="ft_w")
    ft_bias_scale = s_ft_w if args.ft_bias_scale is None else args.ft_bias_scale
    q_ft_b, s_ft_b = quantize_sym(ft_b, max_q=32767, dtype=torch.int16, explicit_scale=ft_bias_scale, name="ft_b")

    q_l1_w, s_l1_w = quantize_sym(l1_w, max_q=127, dtype=torch.int8, explicit_scale=args.l1_scale, name="l1_w")
    q_l1_b, s_l1_b = quantize_sym(l1_b, max_q=I32_MAX, dtype=torch.int32, explicit_scale=args.l1_bias_scale, name="l1_b")

    q_l2_w, s_l2_w = quantize_sym(l2_w, max_q=127, dtype=torch.int8, explicit_scale=args.l2_scale, name="l2_w")
    q_l2_b, s_l2_b = quantize_sym(l2_b, max_q=I32_MAX, dtype=torch.int32, explicit_scale=args.l2_bias_scale, name="l2_b")

    q_out_w, s_out_w = quantize_sym(out_w, max_q=127, dtype=torch.int8, explicit_scale=args.out_scale, name="out_w")
    q_out_b, s_out_b = quantize_sym(out_b, max_q=I32_MAX, dtype=torch.int32, explicit_scale=args.out_bias_scale, name="out_b")

    scales = {
        "feature_transform_weight": s_ft_w,
        "feature_transform_bias": s_ft_b,
        "dense1_weight": s_l1_w,
        "dense1_bias": s_l1_b,
        "dense2_weight": s_l2_w,
        "dense2_bias": s_l2_b,
        "output_weight": s_out_w,
        "output_bias": s_out_b,
    }

    sections: Dict[str, Dict[str, int]] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as fh:
        header = struct.pack("<13sII", MAGIC, VERSION, HIDDEN)
        fh.write(header)

        write_padding(fh)
        sections["FeatureTransformWeights"] = {"offset": fh.tell(), "size": tensor_nbytes(q_ft_w)}
        write_tensor(fh, q_ft_w)

        write_padding(fh)
        sections["HiddenLayer1Weights"] = {"offset": fh.tell(), "size": tensor_nbytes(q_l1_w)}
        write_tensor(fh, q_l1_w)

        write_padding(fh)
        sections["HiddenLayer2Weights"] = {"offset": fh.tell(), "size": tensor_nbytes(q_l2_w)}
        write_tensor(fh, q_l2_w)

        write_padding(fh)
        sections["OutputWeights"] = {"offset": fh.tell(), "size": tensor_nbytes(q_out_w)}
        write_tensor(fh, q_out_w)

        write_padding(fh)
        bias_size = tensor_nbytes(q_ft_b) + tensor_nbytes(q_l1_b) + tensor_nbytes(q_l2_b) + tensor_nbytes(q_out_b)
        sections["Biases"] = {"offset": fh.tell(), "size": bias_size}
        write_tensor(fh, q_ft_b)
        write_tensor(fh, q_l1_b)
        write_tensor(fh, q_l2_b)
        write_tensor(fh, q_out_b)

        final_size = fh.tell()

    stats = ExportStats(scales=scales, sections=sections, file_size=final_size)
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "output": str(out_path),
                "magic": MAGIC.decode("ascii"),
                "version": VERSION,
                "hidden_size": HIDDEN,
                "alignment": ALIGNMENT,
                "scales": stats.scales,
                "sections": stats.sections,
                "file_size": stats.file_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {out_path}")
    print(f"Size : {final_size} bytes")
    print(f"Meta : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

