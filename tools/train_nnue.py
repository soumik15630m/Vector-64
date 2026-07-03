#!/usr/bin/env python3
"""
Section A - Dataset: Lichess Evaluations

This section parses training data without depending on python-chess. It reads
ordinary JSONL, zstd-compressed JSONL, and EPD-like text, parses FEN strings
rank-8 downward, and builds the exact HalfKP feature indices used by the C++
engine. Seekable text files are indexed by byte offset so samples can be loaded
on demand without keeping the whole dataset in memory. Compressed zstd input is
streamed lazily because compressed byte offsets cannot be used as normal text
line offsets. Bad FENs, missing kings, missing centipawn scores, and malformed
records are skipped.

What can go wrong: zstandard may be missing for .zst files, a dataset may not
contain usable cp/mate values, or the FEN may not contain both kings.

Section B - Model Architecture

The NNUEModel mirrors the C++ inference path in float32: two HalfKP embedding
accumulators, side-to-move accumulator differencing, two dense ReLU layers, and
a centipawn output. The training loss compares tanh(out_cp / 400) to the target.

What can go wrong: feature indices outside [0, 81920) indicate a parser/formula
bug and are rejected before training.

Section C - Checkpoints & Auto-Resume

Training writes step checkpoints under checkpoints/, keeps the latest three,
saves best.pt when validation improves, auto-resumes the newest valid step
checkpoint, and writes emergency checkpoints for SIGINT, SIGTERM, and unhandled
exceptions.

What can go wrong: old checkpoints with missing keys are ignored for auto-resume
and rejected when explicitly requested.

Section D - Training Loop

The loop uses Adam, cosine learning-rate decay, optional CUDA AMP, periodic step
logs, and validation at the end of each epoch. The default settings are chosen
for the full pipeline but can be overridden from the CLI.

What can go wrong: this trainer intentionally refuses to start without CUDA when
run as a script, matching the pipeline contract.

Section E - Export to Engine Binary

Export writes the packed VECTOR64_NNUE binary exactly as nnue.cpp::load_file()
expects: 21-byte header, 64-byte aligned sections, zero padding, no internal
bias padding, and an exact 83,904,900-byte file. Quantized tensors are written
little-endian and a top-level .meta.json sidecar supplies positive float scales
for scaled engine inference.

What can go wrong: incompatible checkpoints, wrong tensor shapes, or accidental
padding changes fail with explicit assertions.

Section F - Verification Pass

Verification reloads the exported .nnue with a pure-Python implementation of
load_file() and infer_side_to_move_scaled(), then compares ten deterministic FEN
positions against the PyTorch model within +/-2 centipawns.

What can go wrong: missing sidecar keys, nonzero padding, clamped weights, or
shape mismatches will show up as verification failures.

Section G - CLI

The command-line interface supports training, auto-resume, export-only, and
post-export verification. The module is also importable by tools/smoke_test.py
and tools/run_pipeline.py.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import signal
import struct
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

WHITE = 0
BLACK = 1

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

NETWORK_MAGIC = b"VECTOR64_NNUE"
VERSION = 1
HALF_KP_TOTAL_FEATURES = 81920
HALF_KP_PIECE_BUCKETS = 10
HIDDEN_SIZE = 512
DENSE_L1_SIZE = 32
DENSE_L2_SIZE = 32
SECTION_ALIGNMENT = 64
FEATURE_TO_DENSE_SCALE = 64
DENSE_TO_DENSE_SCALE = 64
OUTPUT_SCALE = 16
EXPECTED_FILE_SIZE = 83_904_900

CHECKPOINT_REQUIRED_KEYS = {
    "step",
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "best_loss",
    "args",
}

META_KEYS = [
    "feature_transform_weight",
    "feature_transform_bias",
    "dense1_weight",
    "dense1_bias",
    "dense2_weight",
    "dense2_bias",
    "output_weight",
    "output_bias",
]

PIECE_TYPE_BY_CHAR = {
    "p": PAWN,
    "n": KNIGHT,
    "b": BISHOP,
    "r": ROOK,
    "q": QUEEN,
    "k": KING,
}

PIECE_BUCKETS = {
    (PAWN, WHITE): 0,
    (KNIGHT, WHITE): 1,
    (BISHOP, WHITE): 2,
    (ROOK, WHITE): 3,
    (QUEEN, WHITE): 4,
    (PAWN, BLACK): 5,
    (KNIGHT, BLACK): 6,
    (BISHOP, BLACK): 7,
    (ROOK, BLACK): 8,
    (QUEEN, BLACK): 9,
}


def align_to(offset: int, alignment: int = SECTION_ALIGNMENT) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def halfkp_feature_index(king_square: int, bucket: int, piece_square: int, perspective: int) -> int:
    side = 1 if perspective == BLACK else 0
    return (((king_square * HALF_KP_PIECE_BUCKETS + bucket) * 64 + piece_square) * 2) + side


def lround(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def clamp_array(values: torch.Tensor, low: int, high: int) -> torch.Tensor:
    return torch.clamp(torch.round(values), low, high)


@dataclass
class FenPosition:
    fen: str
    board: list[str | None]
    stm: int
    white_king: int
    black_king: int
    white_count: int
    black_count: int


def parse_fen(fen: str) -> FenPosition:
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError("FEN must contain at least board and side-to-move fields")

    board_part = parts[0]
    stm_part = parts[1]
    ranks = board_part.split("/")
    if len(ranks) != 8:
        raise ValueError("FEN board must contain exactly 8 ranks")

    board: list[str | None] = [None] * 64
    white_king = -1
    black_king = -1
    white_count = 0
    black_count = 0

    for fen_rank, rank_text in enumerate(ranks):
        rank = 7 - fen_rank
        file = 0
        for ch in rank_text:
            if ch.isdigit():
                file += int(ch)
                continue
            if ch.lower() not in PIECE_TYPE_BY_CHAR:
                raise ValueError(f"Invalid FEN piece character: {ch}")
            if file >= 8:
                raise ValueError("FEN rank has too many files")

            square = rank * 8 + file
            board[square] = ch
            if ch.isupper():
                white_count += 1
            else:
                black_count += 1
            if ch == "K":
                white_king = square
            elif ch == "k":
                black_king = square
            file += 1

        if file != 8:
            raise ValueError("FEN rank does not contain exactly 8 files")

    if stm_part == "w":
        stm = WHITE
    elif stm_part == "b":
        stm = BLACK
    else:
        raise ValueError("FEN side-to-move must be 'w' or 'b'")

    return FenPosition(
        fen=fen,
        board=board,
        stm=stm,
        white_king=white_king,
        black_king=black_king,
        white_count=white_count,
        black_count=black_count,
    )


def piece_color(piece: str) -> int:
    return WHITE if piece.isupper() else BLACK


def piece_type(piece: str) -> int:
    return PIECE_TYPE_BY_CHAR[piece.lower()]


def halfkp_features_from_position(pos: FenPosition) -> tuple[list[int], list[int]]:
    if pos.white_king < 0 or pos.black_king < 0:
        raise ValueError("Both kings must be present")

    white_indices: list[int] = []
    black_indices: list[int] = []

    for square, piece in enumerate(pos.board):
        if piece is None:
            continue
        pt = piece_type(piece)
        if pt == KING:
            continue
        color = piece_color(piece)
        bucket = PIECE_BUCKETS.get((pt, color))
        if bucket is None:
            continue
        white_idx = halfkp_feature_index(pos.white_king, bucket, square, WHITE)
        black_idx = halfkp_feature_index(pos.black_king, bucket, square, BLACK)
        if not (0 <= white_idx < HALF_KP_TOTAL_FEATURES and 0 <= black_idx < HALF_KP_TOTAL_FEATURES):
            raise ValueError("HalfKP feature index out of range")
        white_indices.append(white_idx)
        black_indices.append(black_idx)

    return white_indices, black_indices


def square_file(square: int) -> int:
    return square & 7


def square_rank(square: int) -> int:
    return square >> 3


def in_bounds(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


def is_square_attacked(pos: FenPosition, square: int, attacker: int) -> bool:
    target_file = square_file(square)
    target_rank = square_rank(square)

    knight_deltas = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    ]
    king_deltas = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ]
    bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for from_sq, piece in enumerate(pos.board):
        if piece is None or piece_color(piece) != attacker:
            continue
        pt = piece_type(piece)
        file = square_file(from_sq)
        rank = square_rank(from_sq)

        if pt == PAWN:
            direction = 1 if attacker == WHITE else -1
            for df in (-1, 1):
                if file + df == target_file and rank + direction == target_rank:
                    return True
        elif pt == KNIGHT:
            for df, dr in knight_deltas:
                if file + df == target_file and rank + dr == target_rank:
                    return True
        elif pt == KING:
            for df, dr in king_deltas:
                if file + df == target_file and rank + dr == target_rank:
                    return True
        elif pt in (BISHOP, ROOK, QUEEN):
            dirs: list[tuple[int, int]] = []
            if pt in (BISHOP, QUEEN):
                dirs.extend(bishop_dirs)
            if pt in (ROOK, QUEEN):
                dirs.extend(rook_dirs)
            for df, dr in dirs:
                f = file + df
                r = rank + dr
                while in_bounds(f, r):
                    sq = r * 8 + f
                    if sq == square:
                        return True
                    if pos.board[sq] is not None:
                        break
                    f += df
                    r += dr

    return False


def side_to_move_in_check(pos: FenPosition) -> bool:
    king_square = pos.white_king if pos.stm == WHITE else pos.black_king
    if king_square < 0:
        return True
    return is_square_attacked(pos, king_square, BLACK if pos.stm == WHITE else WHITE)


def target_from_cp(cp: float) -> float:
    return float(math.tanh(float(cp) / 400.0))


def target_from_mate(mate: float) -> float:
    return 1.0 if mate > 0 else -1.0


def best_lichess_eval(obj: dict[str, Any]) -> tuple[float | None, int | None, str]:
    best_cp: float | None = None
    best_depth: int | None = None
    saw_depth = False
    saw_mate_only = False

    for ev in obj.get("evals", []) or []:
        if not isinstance(ev, dict):
            continue
        depth = ev.get("depth")
        if depth is None:
            continue
        try:
            depth_i = int(depth)
        except (TypeError, ValueError):
            continue
        saw_depth = True
        for pv in ev.get("pvs", []) or []:
            if not isinstance(pv, dict):
                continue
            if pv.get("cp") is not None:
                try:
                    cp = float(pv["cp"])
                except (TypeError, ValueError):
                    continue
                if best_depth is None or depth_i > best_depth:
                    best_depth = depth_i
                    best_cp = cp
            elif pv.get("mate") is not None:
                saw_mate_only = True

    if best_cp is not None:
        return best_cp, best_depth, "ok"
    if not saw_depth:
        return None, None, "no_depth"
    if saw_mate_only:
        return None, best_depth, "mate_only"
    return None, best_depth, "no_cp"


def best_lichess_mate(obj: dict[str, Any]) -> float | None:
    best_mate: float | None = None
    best_depth: int | None = None
    for ev in obj.get("evals", []) or []:
        if not isinstance(ev, dict):
            continue
        depth_raw = ev.get("depth", -1)
        try:
            depth = int(depth_raw)
        except (TypeError, ValueError):
            depth = -1
        for pv in ev.get("pvs", []) or []:
            if not isinstance(pv, dict) or pv.get("mate") is None:
                continue
            try:
                mate = float(pv["mate"])
            except (TypeError, ValueError):
                continue
            if best_depth is None or depth > best_depth:
                best_depth = depth
                best_mate = mate
    return best_mate


EPD_SCORE_RE = re.compile(
    r"(?:^|[\s;])(?:ce|cp|eval|score)\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
EPD_MATE_RE = re.compile(r"(?:^|[\s;])(?:mate|bm)\s+(-?\d+)", re.IGNORECASE)


def parse_epd_line(line: str) -> tuple[str, float | None, float | None]:
    fields = line.strip().split()
    if len(fields) < 4:
        raise ValueError("EPD line must contain at least four FEN fields")
    fen = " ".join(fields[:4]) + " 0 1"
    score_match = EPD_SCORE_RE.search(line)
    mate_match = EPD_MATE_RE.search(line)
    cp = float(score_match.group(1)) if score_match else None
    mate = float(mate_match.group(1)) if mate_match else None
    return fen, cp, mate


def parse_training_line(line: str, source_kind: str) -> tuple[str, int, list[int], list[int], float] | None:
    try:
        if source_kind == "jsonl":
            obj = json.loads(line)
            fen = obj.get("fen")
            if not isinstance(fen, str):
                return None
            cp, _depth, reason = best_lichess_eval(obj)
            mate = None if cp is not None else best_lichess_mate(obj)
            if cp is None and mate is None:
                return None
        else:
            fen, cp, mate = parse_epd_line(line)
            if cp is None and mate is None:
                return None

        pos = parse_fen(fen)

        # Source evals are white-perspective; the network is trained
        # side-to-move relative, so flip targets for black to move.
        stm_sign = 1.0 if pos.stm == WHITE else -1.0
        if cp is not None:
            target = target_from_cp(stm_sign * float(cp))
        else:
            assert mate is not None  # guaranteed: cp and mate are not both None above
            target = target_from_mate(stm_sign * float(mate))

        white_indices, black_indices = halfkp_features_from_position(pos)
        return fen, pos.stm, white_indices, black_indices, target
    except Exception:
        return None


class IndexedTextEvalDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        source_kind: str,
        split: str,
        val_fraction: float,
        max_positions: int | None,
        offsets: list[int] | None = None,
    ) -> None:
        self.path = path
        self.source_kind = source_kind
        self.split = split
        self.val_fraction = val_fraction
        self.max_positions = max_positions
        all_offsets = offsets if offsets is not None else self._build_offsets()
        val_every = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
        if split == "val":
            self.offsets = [off for i, off in enumerate(all_offsets) if i % val_every == 0]
        else:
            self.offsets = [off for i, off in enumerate(all_offsets) if i % val_every != 0]
        self._fh: Any | None = None

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        with self.path.open("rb") as fh:
            while True:
                offset = fh.tell()
                raw = fh.readline()
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if parse_training_line(line, self.source_kind) is None:
                    continue
                offsets.append(offset)
                if self.max_positions and len(offsets) >= self.max_positions:
                    break
        if not offsets:
            raise RuntimeError(f"No usable training positions found in {self.path}")
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def _file(self) -> Any:
        if self._fh is None:
            self._fh = self.path.open("rb")
        return self._fh

    def __getitem__(self, index: int) -> dict[str, Any]:
        fh = self._file()
        fh.seek(self.offsets[index])
        line = fh.readline().decode("utf-8", errors="replace")
        parsed = parse_training_line(line, self.source_kind)
        if parsed is None:
            raise RuntimeError("Indexed line became unparsable")
        _fen, stm, white_indices, black_indices, target = parsed
        return {
            "white_indices": white_indices,
            "black_indices": black_indices,
            "stm": stm,
            "target": target,
        }


def build_seekable_offsets(
    path: Path,
    *,
    source_kind: str,
    max_positions: int | None,
) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as fh:
        while True:
            offset = fh.tell()
            raw = fh.readline()
            if not raw:
                break
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if parse_training_line(line, source_kind) is None:
                continue
            offsets.append(offset)
            if max_positions and len(offsets) >= max_positions:
                break
    if not offsets:
        raise RuntimeError(f"No usable training positions found in {path}")
    return offsets


class StreamingZstdEvalDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        *,
        split: str,
        val_fraction: float,
        max_positions: int | None,
        total_usable: int | None = None,
    ) -> None:
        self.path = path
        self.split = split
        self.val_fraction = val_fraction
        self.max_positions = max_positions
        self.total_usable = total_usable

    def __len__(self) -> int:
        if self.total_usable is None:
            return 0
        val_every = max(2, int(round(1.0 / max(self.val_fraction, 1e-6))))
        val_count = (self.total_usable + val_every - 1) // val_every
        return val_count if self.split == "val" else self.total_usable - val_count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install zstandard for .zst input: pip install zstandard") from exc

        val_every = max(2, int(round(1.0 / max(self.val_fraction, 1e-6))))
        usable = 0
        with self.path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    for line in text:
                        parsed = parse_training_line(line, "jsonl")
                        if parsed is None:
                            continue
                        split = "val" if usable % val_every == 0 else "train"
                        usable += 1
                        if self.max_positions and usable > self.max_positions:
                            break
                        if split != self.split:
                            continue
                        _fen, stm, white_indices, black_indices, target = parsed
                        yield {
                            "white_indices": white_indices,
                            "black_indices": black_indices,
                            "stm": stm,
                            "target": target,
                        }


def count_usable_zstd(path: Path, max_positions: int | None) -> int:
    try:
        import zstandard as zstd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install zstandard for .zst input: pip install zstandard") from exc

    count = 0
    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text:
                for line in text:
                    if parse_training_line(line, "jsonl") is not None:
                        count += 1
                        if max_positions and count >= max_positions:
                            break
    if count == 0:
        raise RuntimeError(f"No usable training positions found in {path}")
    return count


class NPZShardDataset(IterableDataset):
    def __init__(
        self,
        files: Sequence[Path],
        *,
        split: str,
        val_fraction: float,
        max_positions: int | None,
        total_positions: int | None = None,
    ) -> None:
        self.files = list(files)
        self.split = split
        self.val_fraction = val_fraction
        self.max_positions = max_positions
        self.total_positions = total_positions or self._count_rows()

    def _count_rows(self) -> int:
        total = 0
        for path in self.files:
            with np.load(path, allow_pickle=False) as data:
                key = "cp" if "cp" in data else "eval_cp"
                total += int(data[key].shape[0])
                if self.max_positions and total >= self.max_positions:
                    return self.max_positions
        return total

    def __len__(self) -> int:
        val_every = max(2, int(round(1.0 / max(self.val_fraction, 1e-6))))
        total = self.total_positions
        val_count = (total + val_every - 1) // val_every
        return val_count if self.split == "val" else total - val_count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        val_every = max(2, int(round(1.0 / max(self.val_fraction, 1e-6))))
        global_index = 0
        for path in self.files:
            with np.load(path, allow_pickle=False) as data:
                white = data["white_indices"]
                black = data["black_indices"]
                stm = data["stm"]
                cp_key = "cp" if "cp" in data else "eval_cp"
                cp = data[cp_key]
                rows = int(cp.shape[0])
                for i in range(rows):
                    if self.max_positions and global_index >= self.max_positions:
                        return
                    split = "val" if global_index % val_every == 0 else "train"
                    if split == self.split:
                        stm_i = int(stm[i])
                        # Shard cp values are white-perspective; train stm-relative.
                        cp_i = float(cp[i]) if stm_i == WHITE else -float(cp[i])
                        yield {
                            "white_indices": white[i].astype(np.int64, copy=False).tolist(),
                            "black_indices": black[i].astype(np.int64, copy=False).tolist(),
                            "stm": stm_i,
                            "target": target_from_cp(cp_i),
                        }
                    global_index += 1


def collate_samples(samples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not samples:
        raise RuntimeError("Empty batch")

    max_white = max(len([x for x in s["white_indices"] if int(x) >= 0]) for s in samples)
    max_black = max(len([x for x in s["black_indices"] if int(x) >= 0]) for s in samples)
    max_white = max(1, max_white)
    max_black = max(1, max_black)

    white = torch.full((len(samples), max_white), -1, dtype=torch.long)
    black = torch.full((len(samples), max_black), -1, dtype=torch.long)
    stm = torch.empty(len(samples), dtype=torch.long)
    target = torch.empty(len(samples), 1, dtype=torch.float32)

    for row, sample in enumerate(samples):
        w = [int(x) for x in sample["white_indices"] if int(x) >= 0]
        b = [int(x) for x in sample["black_indices"] if int(x) >= 0]
        if w:
            white[row, : len(w)] = torch.tensor(w, dtype=torch.long)
        if b:
            black[row, : len(b)] = torch.tensor(b, dtype=torch.long)
        stm[row] = int(sample["stm"])
        target[row, 0] = float(sample["target"])

    if torch.any(white >= HALF_KP_TOTAL_FEATURES) or torch.any(black >= HALF_KP_TOTAL_FEATURES):
        raise RuntimeError("Feature index out of range in batch")
    return {"white_indices": white, "black_indices": black, "stm": stm, "target": target}


class NNUEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_transform = nn.EmbeddingBag(
            HALF_KP_TOTAL_FEATURES,
            HIDDEN_SIZE,
            mode="sum",
            include_last_offset=True,
        )
        self.feature_bias = nn.Parameter(torch.zeros(HIDDEN_SIZE, dtype=torch.float32))
        self.dense1 = nn.Linear(HIDDEN_SIZE, DENSE_L1_SIZE)
        self.dense2 = nn.Linear(DENSE_L1_SIZE, DENSE_L2_SIZE)
        self.output = nn.Linear(DENSE_L2_SIZE, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_transform.weight, mean=0.0, std=0.015)
        nn.init.zeros_(self.feature_bias)
        nn.init.normal_(self.dense1.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.dense1.bias)
        nn.init.normal_(self.dense2.weight, mean=0.0, std=0.04)
        nn.init.zeros_(self.dense2.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.output.bias)

    def _accumulate(self, indices: torch.Tensor) -> torch.Tensor:
        mask = indices >= 0
        counts = mask.sum(dim=1, dtype=torch.long)
        offsets = torch.empty(indices.shape[0] + 1, dtype=torch.long, device=indices.device)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(counts, dim=0)
        flat = indices[mask]
        if flat.numel() == 0:
            accum = torch.zeros(indices.shape[0], HIDDEN_SIZE, dtype=self.feature_bias.dtype, device=indices.device)
        else:
            accum = self.feature_transform(flat, offsets)
        return accum + self.feature_bias

    def forward(self, white_indices: torch.Tensor, black_indices: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        white_acc = self._accumulate(white_indices.long())
        black_acc = self._accumulate(black_indices.long())

        stm_black = stm.long().view(-1, 1) == BLACK
        us = torch.where(stm_black, black_acc, white_acc)
        them = torch.where(stm_black, white_acc, black_acc)

        x0 = F.relu((us - them) / float(FEATURE_TO_DENSE_SCALE))
        x1 = F.relu(self.dense1(x0) / float(DENSE_TO_DENSE_SCALE))
        x2 = F.relu(self.dense2(x1) / float(DENSE_TO_DENSE_SCALE))
        out_cp = self.output(x2) / float(OUTPUT_SCALE)
        return out_cp


def discover_shard_files(path: Path) -> tuple[list[Path], int | None]:
    if path.is_dir():
        manifest = path / "manifest.json"
        total = None
        if manifest.exists():
            with manifest.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            total = int(meta.get("total_positions", 0)) or None
        return sorted(path.glob("*.npz")), total

    if path.suffix.lower() == ".json" and path.name == "manifest.json":
        with path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        base = path.parent
        files = sorted(base.glob("*.npz"))
        return files, int(meta.get("total_positions", 0)) or None

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = "cp" if "cp" in data else "eval_cp"
            total = int(data[key].shape[0])
        return [path], total

    return [], None


def build_datasets(
    data_path: Path,
    *,
    max_positions: int | None,
    val_fraction: float,
) -> tuple[Dataset[Any], Dataset[Any], int]:
    shard_files, shard_total = discover_shard_files(data_path)
    if shard_files:
        total = min(shard_total or 0, max_positions) if max_positions and shard_total else shard_total
        shard_train_ds = NPZShardDataset(
            shard_files,
            split="train",
            val_fraction=val_fraction,
            max_positions=max_positions,
            total_positions=total,
        )
        shard_val_ds = NPZShardDataset(
            shard_files,
            split="val",
            val_fraction=val_fraction,
            max_positions=max_positions,
            total_positions=shard_train_ds.total_positions,
        )
        return shard_train_ds, shard_val_ds, shard_train_ds.total_positions

    suffix = data_path.suffix.lower()
    source_kind = "jsonl" if suffix in {".jsonl", ".zst"} else "epd"
    if suffix == ".zst":
        print("Indexing compressed input by streaming usable records once...")
        total = count_usable_zstd(data_path, max_positions)
        return (
            StreamingZstdEvalDataset(
                data_path,
                split="train",
                val_fraction=val_fraction,
                max_positions=max_positions,
                total_usable=total,
            ),
            StreamingZstdEvalDataset(
                data_path,
                split="val",
                val_fraction=val_fraction,
                max_positions=max_positions,
                total_usable=total,
            ),
            total,
        )

    offsets = build_seekable_offsets(data_path, source_kind=source_kind, max_positions=max_positions)
    train_ds = IndexedTextEvalDataset(
        data_path,
        source_kind=source_kind,
        split="train",
        val_fraction=val_fraction,
        max_positions=max_positions,
        offsets=offsets,
    )
    val_ds = IndexedTextEvalDataset(
        data_path,
        source_kind=source_kind,
        split="val",
        val_fraction=val_fraction,
        max_positions=max_positions,
        offsets=offsets,
    )
    return train_ds, val_ds, len(train_ds) + len(val_ds)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"nnue_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def validate_checkpoint_obj(obj: Any, path: Path) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a checkpoint dictionary")
    missing = CHECKPOINT_REQUIRED_KEYS.difference(obj.keys())
    if missing:
        raise ValueError(f"{path} is missing checkpoint keys: {', '.join(sorted(missing))}")
    return obj


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("nnue_step_*.pt"), key=checkpoint_step)
    for path in reversed(candidates):
        try:
            obj = torch.load(path, map_location="cpu")
            validate_checkpoint_obj(obj, path)
            return path
        except Exception as exc:
            print(f"Skipping invalid checkpoint {path}: {exc}")
    return None


def save_checkpoint(
    path: Path,
    *,
    model: NNUEModel,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    step: int,
    epoch: int,
    best_loss: float,
    args: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else {},
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
            "best_loss": best_loss,
            "args": args,
        },
        path,
    )


def prune_step_checkpoints(checkpoint_dir: Path, keep: int = 3) -> None:
    checkpoints = sorted(checkpoint_dir.glob("nnue_step_*.pt"), key=checkpoint_step)
    for old in checkpoints[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


@dataclass
class TrainConfig:
    data: Path | None
    output: Path
    epochs: int = 10
    batch_size: int = 4096
    lr: float = 1e-3
    max_positions: int | None = None
    checkpoint_steps: int = 1000
    resume: Path | None = None
    export_only: Path | None = None
    verify: bool = False
    device: str = "cuda"
    checkpoint_dir: Path = Path("checkpoints")
    val_split: float = 0.05
    log_every: int = 100
    no_color: bool = False


@dataclass
class TrainResult:
    model: NNUEModel
    step: int
    epoch: int
    best_loss: float
    final_val_loss: float
    output_path: Path
    verification_passed: bool | None
    total_positions: int


class EmergencyCheckpoint:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.model: NNUEModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.args: dict[str, Any] = {}

    def update(
        self,
        *,
        model: NNUEModel,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        step: int,
        epoch: int,
        best_loss: float,
        args: dict[str, Any],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step = step
        self.epoch = epoch
        self.best_loss = best_loss
        self.args = args

    def save(self, prefix: str = "emergency") -> Path | None:
        if self.model is None:
            return None
        path = self.checkpoint_dir / f"{prefix}_step_{self.step}.pt"
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            epoch=self.epoch,
            best_loss=self.best_loss,
            args=self.args,
        )
        print(f"Emergency checkpoint saved: {path}")
        return path


def require_cuda_or_exit(device_arg: str) -> torch.device:
    lowered = device_arg.lower()
    if lowered == "cpu":
        print("ERROR: This pipeline requires a CUDA GPU. Use tools/smoke_test.py to diagnose.")
        raise SystemExit(1)
    if torch.version.cuda is None or not torch.cuda.is_available():
        print("ERROR: This pipeline requires a CUDA GPU. Use tools/smoke_test.py to diagnose.")
        raise SystemExit(1)
    if lowered in {"auto", "cuda"}:
        return torch.device("cuda:0")
    if re.fullmatch(r"cuda:\d+", lowered):
        index = int(lowered.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            print(f"ERROR: Requested {device_arg}, but only {torch.cuda.device_count()} CUDA GPU(s) are visible.")
            raise SystemExit(1)
        return torch.device(lowered)
    print("ERROR: --device must be cuda or a specific CUDA device like cuda:1.")
    raise SystemExit(1)


def make_loaders(
    data_path: Path,
    *,
    batch_size: int,
    max_positions: int | None,
    val_split: float,
) -> tuple[DataLoader, DataLoader, int]:
    train_ds, val_ds, total_positions = build_datasets(
        data_path,
        max_positions=max_positions,
        val_fraction=val_split,
    )
    train_loader: DataLoader[Any] = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_samples,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=collate_samples,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, total_positions


def estimate_batches(total_positions: int, batch_size: int, val_split: float, epochs: int) -> int:
    train_positions = max(1, int(total_positions * (1.0 - val_split)))
    return max(1, math.ceil(train_positions / batch_size) * max(1, epochs))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_one_epoch(
    *,
    model: NNUEModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    step: int,
    best_loss: float,
    cfg: TrainConfig,
    emergency: EmergencyCheckpoint,
    args_dict: dict[str, Any],
) -> tuple[int, float]:
    model.train()
    running_loss = 0.0
    batches = 0
    last_log_time = time.time()
    last_log_step = step
    previous_log_loss: float | None = None

    for batch in loader:
        step += 1
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out_cp = model(batch["white_indices"], batch["black_indices"], batch["stm"])
            pred = torch.tanh(out_cp / 400.0)
            loss = F.mse_loss(pred, batch["target"])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_value = float(loss.detach().cpu().item())
        running_loss += loss_value
        batches += 1

        emergency.update(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            best_loss=best_loss,
            args=args_dict,
        )

        if step % cfg.log_every == 0:
            now = time.time()
            elapsed = max(1e-6, now - last_log_time)
            positions = (step - last_log_step) * cfg.batch_size
            positions_per_sec = positions / elapsed
            lr = float(optimizer.param_groups[0]["lr"])
            trend = ""
            if previous_log_loss is not None:
                if loss_value <= previous_log_loss:
                    trend = " improved" if cfg.no_color else " \033[32mimproved\033[0m"
                else:
                    trend = " worse" if cfg.no_color else " \033[31mworse\033[0m"
            print(
                f"step={step} loss={loss_value:.6f}{trend} "
                f"lr={lr:.6g} positions/sec={positions_per_sec:,.0f}"
            )
            previous_log_loss = loss_value
            last_log_time = now
            last_log_step = step

        if step % cfg.checkpoint_steps == 0:
            path = cfg.checkpoint_dir / f"nnue_step_{step}.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=epoch,
                best_loss=best_loss,
                args=args_dict,
            )
            prune_step_checkpoints(cfg.checkpoint_dir)
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"checkpoint saved: {path} ({size_mb:.1f} MB)")

    return step, running_loss / max(1, batches)


@torch.no_grad()
def validate(model: NNUEModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out_cp = model(batch["white_indices"], batch["black_indices"], batch["stm"])
        pred = torch.tanh(out_cp / 400.0)
        loss = F.mse_loss(pred, batch["target"])
        total_loss += float(loss.detach().cpu().item())
        batches += 1
    return total_loss / max(1, batches)


def load_checkpoint_into(
    path: Path,
    *,
    model: NNUEModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | None = None,
) -> tuple[int, int, float]:
    if device is None:
        device = torch.device("cpu")
    obj = validate_checkpoint_obj(torch.load(path, map_location=device), path)
    model.load_state_dict(obj["model_state_dict"])
    if optimizer is not None and obj.get("optimizer_state_dict"):
        optimizer.load_state_dict(obj["optimizer_state_dict"])
    if scheduler is not None and obj.get("scheduler_state_dict"):
        scheduler.load_state_dict(obj["scheduler_state_dict"])
    step = int(obj["step"])
    epoch = int(obj["epoch"])
    best_loss = float(obj["best_loss"])
    return step, epoch, best_loss


def quantize_tensor(tensor: torch.Tensor, multiplier: float, low: int, high: int, dtype: np.dtype) -> np.ndarray:
    q = clamp_array(tensor.detach().cpu().to(torch.float32) * multiplier, low, high)
    return q.numpy().astype(dtype, copy=False)


def write_padding(fh: Any) -> None:
    current = fh.tell()
    aligned = align_to(current)
    if aligned > current:
        fh.write(b"\x00" * (aligned - current))


def export_model_to_nnue(model: NNUEModel, output_path: Path) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()

    ft_w = quantize_tensor(model_cpu.feature_transform.weight, FEATURE_TO_DENSE_SCALE, -32767, 32767, np.dtype("<i2"))
    ft_b = quantize_tensor(model_cpu.feature_bias, FEATURE_TO_DENSE_SCALE, -32767, 32767, np.dtype("<i2"))
    l1_w = quantize_tensor(model_cpu.dense1.weight, DENSE_TO_DENSE_SCALE, -127, 127, np.dtype("i1"))
    l1_b = quantize_tensor(model_cpu.dense1.bias, DENSE_TO_DENSE_SCALE, -2_147_483_647, 2_147_483_647, np.dtype("<i4"))
    l2_w = quantize_tensor(model_cpu.dense2.weight, DENSE_TO_DENSE_SCALE, -127, 127, np.dtype("i1"))
    l2_b = quantize_tensor(model_cpu.dense2.bias, DENSE_TO_DENSE_SCALE, -2_147_483_647, 2_147_483_647, np.dtype("<i4"))
    out_w = quantize_tensor(model_cpu.output.weight.reshape(-1), DENSE_TO_DENSE_SCALE, -127, 127, np.dtype("i1"))
    out_b = quantize_tensor(
        model_cpu.output.bias.reshape(-1), DENSE_TO_DENSE_SCALE, -2_147_483_647, 2_147_483_647, np.dtype("<i4")
    )

    if ft_w.shape != (HALF_KP_TOTAL_FEATURES, HIDDEN_SIZE):
        raise ValueError(f"featureTransformWeights shape mismatch: {ft_w.shape}")
    if ft_b.shape != (HIDDEN_SIZE,):
        raise ValueError(f"featureBias shape mismatch: {ft_b.shape}")
    if l1_w.shape != (DENSE_L1_SIZE, HIDDEN_SIZE):
        raise ValueError(f"l1Weights shape mismatch: {l1_w.shape}")
    if l2_w.shape != (DENSE_L2_SIZE, DENSE_L1_SIZE):
        raise ValueError(f"l2Weights shape mismatch: {l2_w.shape}")
    if out_w.shape != (DENSE_L2_SIZE,):
        raise ValueError(f"outWeights shape mismatch: {out_w.shape}")

    with output_path.open("wb") as fh:
        fh.write(struct.pack("<13sII", NETWORK_MAGIC, VERSION, HIDDEN_SIZE))

        write_padding(fh)
        if fh.tell() != 64:
            raise AssertionError("featureTransformWeights offset mismatch")
        fh.write(ft_w.tobytes(order="C"))

        write_padding(fh)
        if fh.tell() != 83_886_144:
            raise AssertionError("l1Weights offset mismatch")
        fh.write(l1_w.tobytes(order="C"))

        write_padding(fh)
        if fh.tell() != 83_902_528:
            raise AssertionError("l2Weights offset mismatch")
        fh.write(l2_w.tobytes(order="C"))

        write_padding(fh)
        if fh.tell() != 83_903_552:
            raise AssertionError("outWeights offset mismatch")
        fh.write(out_w.tobytes(order="C"))

        write_padding(fh)
        if fh.tell() != 83_903_616:
            raise AssertionError("bias block offset mismatch")
        fh.write(ft_b.tobytes(order="C"))
        fh.write(l1_b.tobytes(order="C"))
        fh.write(l2_b.tobytes(order="C"))
        fh.write(out_b.tobytes(order="C"))

        if fh.tell() != EXPECTED_FILE_SIZE:
            raise AssertionError(f"final file size mismatch: {fh.tell()} != {EXPECTED_FILE_SIZE}")

    scales = {
        "feature_transform_weight": 1.0 / float(FEATURE_TO_DENSE_SCALE),
        "feature_transform_bias": 1.0 / float(FEATURE_TO_DENSE_SCALE),
        "dense1_weight": 1.0 / float(DENSE_TO_DENSE_SCALE),
        "dense1_bias": 1.0 / float(DENSE_TO_DENSE_SCALE),
        "dense2_weight": 1.0 / float(DENSE_TO_DENSE_SCALE),
        "dense2_bias": 1.0 / float(DENSE_TO_DENSE_SCALE),
        "output_weight": 1.0 / float(DENSE_TO_DENSE_SCALE),
        "output_bias": 1.0 / float(DENSE_TO_DENSE_SCALE),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(scales, fh, indent=2, sort_keys=True)
        fh.write("\n")

    actual_size = output_path.stat().st_size
    if actual_size != EXPECTED_FILE_SIZE:
        raise AssertionError(f"Exported file size {actual_size} != {EXPECTED_FILE_SIZE}")
    return scales


@dataclass
class LoadedNNUE:
    ft_w: np.ndarray
    l1_w: np.ndarray
    l2_w: np.ndarray
    out_w: np.ndarray
    ft_b: np.ndarray
    l1_b: np.ndarray
    l2_b: np.ndarray
    out_b: np.ndarray
    scales: dict[str, float]


def assert_zero_padding(blob: bytes, begin: int, end: int) -> None:
    if begin > end:
        raise ValueError("Invalid padding range")
    if any(b != 0 for b in blob[begin:end]):
        raise ValueError(f"Nonzero padding bytes in range [{begin}, {end})")


def load_nnue_file(path: Path) -> LoadedNNUE:
    blob = path.read_bytes()
    if len(blob) != EXPECTED_FILE_SIZE:
        raise ValueError(f"{path} size {len(blob)} != {EXPECTED_FILE_SIZE}")
    magic, version, hidden = struct.unpack_from("<13sII", blob, 0)
    if magic != NETWORK_MAGIC:
        raise ValueError("Bad NNUE magic")
    if version != VERSION or hidden != HIDDEN_SIZE:
        raise ValueError("Bad NNUE version or hidden size")

    assert_zero_padding(blob, 21, 64)
    assert_zero_padding(blob, 83_903_584, 83_903_616)

    ft_w = np.frombuffer(blob, dtype="<i2", count=HALF_KP_TOTAL_FEATURES * HIDDEN_SIZE, offset=64)
    ft_w = ft_w.reshape(HALF_KP_TOTAL_FEATURES, HIDDEN_SIZE).copy()
    l1_w = np.frombuffer(blob, dtype="i1", count=DENSE_L1_SIZE * HIDDEN_SIZE, offset=83_886_144)
    l1_w = l1_w.reshape(DENSE_L1_SIZE, HIDDEN_SIZE).copy()
    l2_w = np.frombuffer(blob, dtype="i1", count=DENSE_L2_SIZE * DENSE_L1_SIZE, offset=83_902_528)
    l2_w = l2_w.reshape(DENSE_L2_SIZE, DENSE_L1_SIZE).copy()
    out_w = np.frombuffer(blob, dtype="i1", count=DENSE_L2_SIZE, offset=83_903_552).copy()
    ft_b = np.frombuffer(blob, dtype="<i2", count=HIDDEN_SIZE, offset=83_903_616).copy()
    l1_b = np.frombuffer(blob, dtype="<i4", count=DENSE_L1_SIZE, offset=83_904_640).copy()
    l2_b = np.frombuffer(blob, dtype="<i4", count=DENSE_L2_SIZE, offset=83_904_768).copy()
    out_b = np.frombuffer(blob, dtype="<i4", count=1, offset=83_904_896).copy()

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        cwd_candidate = Path(path.name + ".meta.json")
        meta_path = cwd_candidate if cwd_candidate.exists() else meta_path
    with meta_path.open("r", encoding="utf-8") as fh:
        scales_obj = json.load(fh)
    scales: dict[str, float] = {}
    for key in META_KEYS:
        value = float(scales_obj[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid sidecar scale for {key}: {value}")
        scales[key] = value

    return LoadedNNUE(
        ft_w=ft_w,
        l1_w=l1_w,
        l2_w=l2_w,
        out_w=out_w,
        ft_b=ft_b,
        l1_b=l1_b,
        l2_b=l2_b,
        out_b=out_b,
        scales=scales,
    )


def infer_scaled_python(net: LoadedNNUE, fen: str) -> int:
    pos = parse_fen(fen)
    white_features, black_features = halfkp_features_from_position(pos)

    white_acc = net.ft_b.astype(np.float32) * np.float32(net.scales["feature_transform_bias"])
    black_acc = white_acc.copy()
    if white_features:
        white_acc = white_acc + np.sum(
            net.ft_w[np.asarray(white_features, dtype=np.int64)].astype(np.float32)
            * np.float32(net.scales["feature_transform_weight"]),
            axis=0,
        )
    if black_features:
        black_acc = black_acc + np.sum(
            net.ft_w[np.asarray(black_features, dtype=np.int64)].astype(np.float32)
            * np.float32(net.scales["feature_transform_weight"]),
            axis=0,
        )

    us = white_acc if pos.stm == WHITE else black_acc
    them = black_acc if pos.stm == WHITE else white_acc

    x0 = np.maximum(np.float32(0.0), (us - them) / np.float32(FEATURE_TO_DENSE_SCALE))
    x1_sum = net.l1_b.astype(np.float32) * np.float32(net.scales["dense1_bias"])
    x1_sum = x1_sum + (net.l1_w.astype(np.float32) * np.float32(net.scales["dense1_weight"])) @ x0
    x1 = np.maximum(np.float32(0.0), x1_sum / np.float32(DENSE_TO_DENSE_SCALE))

    x2_sum = net.l2_b.astype(np.float32) * np.float32(net.scales["dense2_bias"])
    x2_sum = x2_sum + (net.l2_w.astype(np.float32) * np.float32(net.scales["dense2_weight"])) @ x1
    x2 = np.maximum(np.float32(0.0), x2_sum / np.float32(DENSE_TO_DENSE_SCALE))

    out = float(net.out_b[0]) * net.scales["output_bias"]
    out += float(np.dot(net.out_w.astype(np.float32) * np.float32(net.scales["output_weight"]), x2))
    return lround(out / float(OUTPUT_SCALE))


VERIFY_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n1p3/8/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
    "r2qkbnr/ppp2ppp/2npb3/4p3/4P3/2NPBN2/PPPQ1PPP/R3KB1R w KQkq - 4 6",
    "rn1qk2r/pp3ppp/2pbpn2/3p4/3P4/2NBPN2/PPP2PPP/R2QKB1R w KQkq - 0 7",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 7",
    "2rq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PPQ2PPP/2RR2K1 b - - 3 13",
    "r3r1k1/ppq2ppp/2p2n2/3p4/3P4/2N1PN2/PPQ2PPP/3RR1K1 w - - 0 18",
    "8/5pk1/3p2p1/2pPp3/2P1P3/3K1P2/6P1/8 w - - 0 40",
    "4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 b - - 0 30",
]


@torch.no_grad()
def verify_export(model: NNUEModel, output_path: Path, *, verbose: bool = True) -> bool:
    net = load_nnue_file(output_path)
    device = next(model.parameters()).device
    model.eval()
    all_ok = True

    for fen in VERIFY_FENS:
        pos = parse_fen(fen)
        white_features, black_features = halfkp_features_from_position(pos)
        batch = collate_samples(
            [
                {
                    "white_indices": white_features,
                    "black_indices": black_features,
                    "stm": pos.stm,
                    "target": 0.0,
                }
            ]
        )
        batch = move_batch(batch, device)
        torch_cp = float(model(batch["white_indices"], batch["black_indices"], batch["stm"]).cpu().item())
        py_cp = infer_scaled_python(net, fen)
        diff = abs(torch_cp - py_cp)
        ok = diff <= 2.0
        all_ok = all_ok and ok
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"{status}: torch={torch_cp:.2f} python_scaled={py_cp} diff={diff:.2f} fen={fen[:48]}")

    if verbose:
        print("PASS" if all_ok else "FAIL")
    return all_ok


def train_model(cfg: TrainConfig) -> TrainResult:
    if cfg.export_only is None and cfg.data is None:
        raise ValueError("--data is required unless --export-only is used")

    device = torch.device(cfg.device)
    args_dict = {
        "data": str(cfg.data) if cfg.data else None,
        "output": str(cfg.output),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "max_positions": cfg.max_positions,
        "checkpoint_steps": cfg.checkpoint_steps,
        "device": cfg.device,
    }

    model = NNUEModel().to(device)
    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any | None = None
    step = 0
    start_epoch = 0
    best_loss = float("inf")
    final_val_loss = float("inf")
    total_positions = 0
    verification_passed: bool | None = None
    emergency = EmergencyCheckpoint(cfg.checkpoint_dir)

    def handle_signal(signum: int, _frame: Any) -> None:
        emergency.save("emergency")
        raise SystemExit(128 + signum)

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if cfg.export_only is not None:
            load_checkpoint_into(cfg.export_only, model=model, device=device)
            model.to("cpu")
            export_model_to_nnue(model, cfg.output)
            if cfg.verify:
                verification_passed = verify_export(model, cfg.output)
            return TrainResult(
                model=model,
                step=0,
                epoch=0,
                best_loss=float("nan"),
                final_val_loss=float("nan"),
                output_path=cfg.output,
                verification_passed=verification_passed,
                total_positions=0,
            )

        assert cfg.data is not None
        train_loader, val_loader, total_positions = make_loaders(
            cfg.data,
            batch_size=cfg.batch_size,
            max_positions=cfg.max_positions,
            val_split=cfg.val_split,
        )
        total_steps = estimate_batches(total_positions, cfg.batch_size, cfg.val_split, cfg.epochs)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        resume_path = cfg.resume or find_latest_checkpoint(cfg.checkpoint_dir)
        if resume_path is not None:
            step, start_epoch, best_loss = load_checkpoint_into(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            print(f"Resumed checkpoint: {resume_path} at step {step}")

        emergency.update(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=start_epoch,
            best_loss=best_loss,
            args=args_dict,
        )

        print("epoch | train_loss | val_loss | best_val | lr | elapsed")
        wall_start = time.time()
        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.time()
            step, train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch + 1,
                step=step,
                best_loss=best_loss,
                cfg=cfg,
                emergency=emergency,
                args_dict=args_dict,
            )
            final_val_loss = validate(model, val_loader, device)
            if final_val_loss < best_loss:
                best_loss = final_val_loss
                save_checkpoint(
                    cfg.checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    epoch=epoch + 1,
                    best_loss=best_loss,
                    args=args_dict,
                )
                print(f"best checkpoint saved: {cfg.checkpoint_dir / 'best.pt'}")

            lr = float(optimizer.param_groups[0]["lr"])
            elapsed = time.time() - epoch_start
            print(
                f"{epoch + 1:5d} | {train_loss:10.6f} | {final_val_loss:8.6f} | "
                f"{best_loss:8.6f} | {lr:.6g} | {elapsed:7.1f}s"
            )

        best_path = cfg.checkpoint_dir / "best.pt"
        if best_path.exists():
            load_checkpoint_into(best_path, model=model, device=device)
            print(f"Loaded best checkpoint for export: {best_path}")

        model.to("cpu")
        export_model_to_nnue(model, cfg.output)
        if cfg.verify:
            verification_passed = verify_export(model, cfg.output)
        print(f"Training wall time: {time.time() - wall_start:.1f}s")
        return TrainResult(
            model=model,
            step=step,
            epoch=cfg.epochs,
            best_loss=best_loss,
            final_val_loss=final_val_loss,
            output_path=cfg.output,
            verification_passed=verification_passed,
            total_positions=total_positions,
        )
    except Exception:
        emergency.save("emergency")
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export VECTOR64_NNUE.")
    parser.add_argument("--data", type=Path, help="Lichess eval dataset (.jsonl, .jsonl.zst, .bin/.epd)")
    parser.add_argument("--output", type=Path, default=Path("network.nnue"), help="Output .nnue file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-positions", type=int, default=0, help="Cap dataset size")
    parser.add_argument("--checkpoint-steps", type=int, default=1000)
    parser.add_argument("--resume", type=Path, default=None, help="Explicit checkpoint")
    parser.add_argument("--export-only", type=Path, default=None, help="Skip training, export checkpoint at PATH")
    parser.add_argument("--verify", action="store_true", help="Run verification pass after export")
    parser.add_argument("--device", default="cuda", help="cuda or cuda:N. cpu is rejected by pipeline policy.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Reproducibility: seed every RNG involved in training.
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    device = require_cuda_or_exit(args.device)
    cfg = TrainConfig(
        data=args.data,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_positions=args.max_positions or None,
        checkpoint_steps=args.checkpoint_steps,
        resume=args.resume,
        export_only=args.export_only,
        verify=args.verify,
        device=str(device),
    )
    result = train_model(cfg)
    status = "PASS" if result.verification_passed else "SKIPPED" if result.verification_passed is None else "FAIL"
    print(f"Exported: {result.output_path}")
    print(f"Final val loss: {result.final_val_loss}")
    print(f"Verification: {status}")
    return 0 if result.verification_passed is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
