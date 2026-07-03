#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import train_nnue as nnue


EXTRACTED_PATH = Path("data") / "extracted_positions.jsonl"
SHARDS_DIR = Path("data") / "shards"
DEFAULT_OUTPUT = Path("artifacts") / "pipeline_output.nnue"
DEFAULT_HF_DATASET = "Lichess/chess-position-evaluations"
ROLLING_DEDUP_CLEAR = 10_000_000
MAX_FEATURES_PER_SIDE = 30


class PipelineError(RuntimeError):
    pass


class Colors:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def code(self, value: str) -> str:
        return value if self.enabled else ""

    @property
    def bold(self) -> str:
        return self.code("\033[1m")

    @property
    def green(self) -> str:
        return self.code("\033[32m")

    @property
    def red(self) -> str:
        return self.code("\033[31m")

    @property
    def yellow(self) -> str:
        return self.code("\033[33m")

    @property
    def cyan(self) -> str:
        return self.code("\033[36m")

    @property
    def reset(self) -> str:
        return self.code("\033[0m")


def stage_header(title: str, colors: Colors) -> None:
    line = "═" * max(8, 54 - len(title))
    print(f"\n{colors.cyan}══ {title} {line}{colors.reset}")


def boxed(lines: List[str], colors: Colors) -> None:
    width = max(len(line) for line in lines)
    top = "╔" + "═" * (width + 2) + "╗"
    bottom = "╚" + "═" * (width + 2) + "╝"
    print(colors.cyan + top + colors.reset)
    for line in lines:
        print(colors.cyan + "║ " + colors.reset + line.ljust(width) + colors.cyan + " ║" + colors.reset)
    print(colors.cyan + bottom + colors.reset)


def is_hf_source(source: Optional[str]) -> bool:
    if source is None:
        return True
    lowered = source.strip().lower()
    return lowered in {"hf", "huggingface", "huggingface://default"} or lowered.startswith("hf://")


def hf_dataset_id(source: Optional[str]) -> str:
    if source is None:
        return DEFAULT_HF_DATASET
    if source.lower().startswith("hf://"):
        dataset_id = source[5:].strip()
        return dataset_id or DEFAULT_HF_DATASET
    return DEFAULT_HF_DATASET


def source_label(source: Optional[str]) -> str:
    if is_hf_source(source):
        return f"hf://{hf_dataset_id(source)}"
    return str(source)


def open_jsonl_lines(path: Path) -> Iterator[str]:
    if path.suffix.lower() == ".zst":
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as exc:
            raise PipelineError("zstandard is required for .zst input. Fix: pip install zstandard") from exc
        with path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    yield from text
    else:
        with path.open("r", encoding="utf-8") as fh:
            yield from fh


def iter_source_rows(source: Optional[str]) -> Iterator[Dict[str, Any]]:
    if is_hf_source(source):
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise PipelineError("datasets is required for Hugging Face input. Fix: pip install datasets") from exc

        dataset_id = hf_dataset_id(source)
        dataset = load_dataset(dataset_id, split="train", streaming=True)
        for row in dataset:
            if isinstance(row, dict):
                yield row
        return

    assert source is not None
    path = Path(source)
    for line in open_jsonl_lines(path):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            yield {"__error__": "json_error"}
            continue
        if isinstance(obj, dict):
            yield obj


def row_eval_cp_depth(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[int], str]:
    if "evals" in row:
        return nnue.best_lichess_eval(row)

    depth_raw = row.get("depth")
    if depth_raw is None:
        return None, None, "no_depth"
    try:
        depth = int(depth_raw)
    except (TypeError, ValueError):
        return None, None, "bad_depth"

    cp_raw = row.get("cp")
    if cp_raw is not None:
        try:
            return float(cp_raw), depth, "ok"
        except (TypeError, ValueError):
            return None, depth, "bad_cp"

    if row.get("mate") is not None:
        return None, depth, "mate_only"
    return None, depth, "no_cp"


def fen_digest(fen: str) -> bytes:
    return hashlib.blake2b(fen.encode("utf-8"), digest_size=12).digest()


def extract_positions(
    *,
    data_source: Optional[str],
    output_path: Path,
    max_positions: int,
    colors: Colors,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    seen: set[bytes] = set()
    scanned = 0
    accepted = 0
    last_stats = time.time()
    started = time.time()
    current_fen = ""

    with output_path.open("w", encoding="utf-8") as out:
        with tqdm(
            total=max_positions,
            desc="extract accepted",
            unit="pos",
            dynamic_ncols=True,
        ) as pbar:
            for obj in iter_source_rows(data_source):
                scanned += 1
                if obj.get("__error__") == "json_error":
                    stats["json_error"] += 1
                    continue
                if scanned % ROLLING_DEDUP_CLEAR == 0:
                    seen.clear()

                fen = obj.get("fen")
                if not isinstance(fen, str):
                    stats["missing_fen"] += 1
                    continue
                current_fen = fen

                cp, depth, reason = row_eval_cp_depth(obj)
                if cp is None:
                    stats[reason] += 1
                    continue
                if depth is None:
                    stats["no_depth"] += 1
                    continue
                if depth < 10:
                    stats["depth_lt_10"] += 1
                    continue
                if abs(cp) > 2000:
                    stats["abs_cp_gt_2000"] += 1
                    continue

                key = fen_digest(fen)
                if key in seen:
                    stats["duplicate_fen"] += 1
                    continue
                seen.add(key)

                try:
                    pos = nnue.parse_fen(fen)
                except ValueError:
                    stats["bad_fen"] += 1
                    continue
                if pos.white_king < 0 or pos.black_king < 0:
                    stats["missing_king"] += 1
                    continue
                if pos.white_count + pos.black_count < 5:
                    stats["lt_5_pieces"] += 1
                    continue
                if pos.white_count + pos.black_count - 2 > MAX_FEATURES_PER_SIDE:
                    stats["too_many_pieces"] += 1
                    continue
                if nnue.side_to_move_in_check(pos):
                    stats["side_to_move_in_check"] += 1
                    continue

                out.write(json.dumps({"fen": fen, "cp": float(cp), "stm": int(pos.stm)}, separators=(",", ":")) + "\n")
                accepted += 1
                pbar.update(1)

                now = time.time()
                if now - last_stats >= 5.0:
                    rejected = scanned - accepted
                    rate = (accepted / scanned * 100.0) if scanned else 0.0
                    pbar.set_postfix(scanned=f"{scanned:,}", rejected=f"{rejected:,}", accept=f"{rate:.3f}%")
                    tqdm.write(
                        "stats: "
                        f"scanned={scanned:,} accepted={accepted:,} rejected={rejected:,} "
                        f"acceptance={rate:.3f}% current_fen={current_fen[:40]}"
                    )
                    last_stats = now

                if accepted >= max_positions:
                    break

    elapsed = time.time() - started
    if accepted == 0:
        raise PipelineError("Extraction produced zero positions. Check dataset path and filters.")

    print(
        f"{colors.green}extraction complete{colors.reset}: "
        f"accepted={accepted:,} scanned={scanned:,} elapsed={elapsed:.1f}s"
    )
    if accepted < max_positions:
        print(
            f"{colors.yellow}warning{colors.reset}: dataset ended before target; "
            f"accepted {accepted:,} of {max_positions:,}"
        )
    print("rejection summary: " + ", ".join(f"{k}={v:,}" for k, v in stats.most_common()))
    return {"accepted": accepted, "scanned": scanned, "rejected": scanned - accepted, "seconds": elapsed}


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as fh:
        for _ in fh:
            count += 1
    return count


def flush_shard(
    *,
    shard_index: int,
    rows: List[Tuple[List[int], List[int], int, float]],
    out_dir: Path,
) -> Tuple[Path, int]:
    n = len(rows)
    white = np.full((n, MAX_FEATURES_PER_SIDE), -1, dtype=np.int32)
    black = np.full((n, MAX_FEATURES_PER_SIDE), -1, dtype=np.int32)
    stm = np.empty(n, dtype=np.int8)
    cp = np.empty(n, dtype=np.float32)

    for i, (w, b, side, score) in enumerate(rows):
        if len(w) > MAX_FEATURES_PER_SIDE or len(b) > MAX_FEATURES_PER_SIDE:
            tqdm.write(
                f"warning: skipping position {i} in shard {shard_index} "
                f"— {max(len(w), len(b))} non-king features exceeds limit {MAX_FEATURES_PER_SIDE}"
            )
            continue
        white[i, : len(w)] = np.asarray(w, dtype=np.int32)
        black[i, : len(b)] = np.asarray(b, dtype=np.int32)
        stm[i] = side
        cp[i] = score

    out_path = out_dir / f"shard_{shard_index:04d}.npz"
    np.savez_compressed(out_path, white_indices=white, black_indices=black, stm=stm, cp=cp)
    return out_path, out_path.stat().st_size


def build_shards(
    *,
    extracted_path: Path,
    out_dir: Path,
    shard_size: int,
    colors: Colors,
) -> Dict[str, Any]:
    if not extracted_path.exists():
        raise PipelineError(f"Missing extracted positions file: {extracted_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_positions = count_lines(extracted_path)
    if total_positions == 0:
        raise PipelineError(f"{extracted_path} is empty")

    # Resume: detect already-written shards and skip their positions
    existing_shards = sorted(out_dir.glob("shard_*.npz"))
    start_shard_index = len(existing_shards)
    skip_positions = start_shard_index * shard_size
    written_bytes = sum(p.stat().st_size for p in existing_shards)

    if start_shard_index > 0:
        print(
            f"{colors.yellow}resuming batching{colors.reset}: "
            f"{start_shard_index} shards already exist, "
            f"skipping first {skip_positions:,} positions"
        )
        if skip_positions >= total_positions:
            print(f"{colors.green}batching already complete{colors.reset}: all {total_positions:,} positions sharded")
            return {"positions": total_positions, "shards": start_shard_index, "seconds": 0.0, "bytes": written_bytes}

    remaining_positions = total_positions - skip_positions
    rows: List[Tuple[List[int], List[int], int, float]] = []
    shard_index = start_shard_index
    started = time.time()

    with extracted_path.open("r", encoding="utf-8") as fh:
        # Skip lines already covered by existing shards
        for _ in range(skip_positions):
            if not fh.readline():
                break

        with tqdm(total=remaining_positions, desc="batch HalfKP", unit="pos", dynamic_ncols=True) as pbar:
            for line in fh:
                try:
                    obj = json.loads(line)
                    fen = str(obj["fen"])
                    score = float(obj["cp"])
                    side = int(obj["stm"])
                    pos = nnue.parse_fen(fen)
                    white_features, black_features = nnue.halfkp_features_from_position(pos)
                except Exception as exc:
                    raise PipelineError(f"Failed to batch extracted line {pbar.n + skip_positions + 1}: {exc}") from exc

                rows.append((white_features, black_features, side, score))
                pbar.update(1)

                if len(rows) >= shard_size:
                    path, size = flush_shard(shard_index=shard_index, rows=rows, out_dir=out_dir)
                    written_bytes += size
                    shard_index += 1
                    rows.clear()
                    elapsed = max(1e-6, time.time() - started)
                    pbar.set_postfix(shards=shard_index, mbps=f"{written_bytes / (1024 * 1024) / elapsed:.1f}")
                    tqdm.write(f"wrote {path} ({size / (1024 * 1024):.1f} MB)")

            if rows:
                path, size = flush_shard(shard_index=shard_index, rows=rows, out_dir=out_dir)
                written_bytes += size
                shard_index += 1
                tqdm.write(f"wrote {path} ({size / (1024 * 1024):.1f} MB)")

    manifest = {
        "total_positions": total_positions,
        "num_shards": shard_index,
        "shard_size": shard_size,
        "feature_count": nnue.HALF_KP_TOTAL_FEATURES,
        "max_features_per_side": MAX_FEATURES_PER_SIDE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elapsed = time.time() - started
    print(
        f"{colors.green}batching complete{colors.reset}: "
        f"shards={shard_index:,} positions={total_positions:,} elapsed={elapsed:.1f}s"
    )
    return {"positions": total_positions, "shards": shard_index, "seconds": elapsed, "bytes": written_bytes}


def ensure_file(path: Path, message: str) -> None:
    if not path.exists():
        raise PipelineError(f"{message}: {path}")


def ensure_shards(path: Path) -> None:
    manifest = path / "manifest.json"
    if not manifest.exists():
        raise PipelineError(f"Missing shard manifest for --skip-batching: {manifest}")
    if not list(path.glob("*.npz")):
        raise PipelineError(f"No .npz shards found in {path}")


def latest_checkpoint_for_export() -> Path:
    best = Path("checkpoints") / "best.pt"
    if best.exists():
        return best
    latest = nnue.find_latest_checkpoint(Path("checkpoints"))
    if latest is None:
        raise PipelineError("No checkpoint found for --skip-training. Expected checkpoints/best.pt or nnue_step_*.pt.")
    return latest


def run_training_or_export(
    *,
    output: Path,
    device: str,
    skip_training: bool,
    colors: Colors,
    no_color: bool,
) -> nnue.TrainResult:
    if skip_training:
        checkpoint = latest_checkpoint_for_export()
        print(f"exporting latest checkpoint: {checkpoint}")
        cfg = nnue.TrainConfig(
            data=None,
            output=output,
            export_only=checkpoint,
            verify=True,
            device=device,
            checkpoint_steps=5000,
            log_every=500,
            no_color=no_color,
        )
        return nnue.train_model(cfg)

    ensure_shards(SHARDS_DIR)
    cfg = nnue.TrainConfig(
        data=SHARDS_DIR,
        output=output,
        epochs=6,
        batch_size=4096,
        lr=1e-3,
        checkpoint_steps=5000,
        verify=True,
        device=device,
        checkpoint_dir=Path("checkpoints"),
        val_split=0.05,
        log_every=500,
        no_color=no_color,
    )
    return nnue.train_model(cfg)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full VECTOR64_NNUE training pipeline.")
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Raw Lichess eval dataset (.jsonl.zst/.jsonl), or hf://DATASET_ID. "
            f"Default: hf://{DEFAULT_HF_DATASET}"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Final .nnue output")
    parser.add_argument("--max-positions", type=int, default=120_000_000, help="Target extracted positions")
    parser.add_argument("--shard-size", type=int, default=500_000, help="Positions per shard")
    parser.add_argument("--skip-extraction", action="store_true", help="Resume from data/extracted_positions.jsonl")
    parser.add_argument("--skip-batching", action="store_true", help="Resume from data/shards/")
    parser.add_argument("--skip-training", action="store_true", help="Skip to export using latest checkpoint")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    parser.add_argument("--device", default="cuda", help="cuda or cuda:N. cpu is rejected by pipeline policy.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    colors = Colors(enabled=not args.no_color and "NO_COLOR" not in os.environ)
    no_color = args.no_color or "NO_COLOR" in os.environ

    device = nnue.require_cuda_or_exit(args.device)
    if args.data is not None and not is_hf_source(args.data) and not Path(args.data).exists() and not args.skip_extraction:
        print(f"{colors.red}ERROR{colors.reset}: dataset path does not exist: {args.data}")
        return 1

    started = time.time()
    boxed(
        [
            "VECTOR64_NNUE PIPELINE",
            f"date/time: {datetime.now().isoformat(timespec='seconds')}",
            f"device: {device}",
            f"dataset: {source_label(args.data)}",
            f"target positions: {args.max_positions:,}",
        ],
        colors,
    )

    extraction_stats: Dict[str, Any] = {}
    batching_stats: Dict[str, Any] = {}
    result: Optional[nnue.TrainResult] = None

    try:
        stage_header("STAGE 1 - EXTRACTION", colors)
        auto_skip_extraction = EXTRACTED_PATH.exists() and not args.skip_extraction
        if auto_skip_extraction:
            n_lines = count_lines(EXTRACTED_PATH)
            print(
                f"{colors.green}auto-skip{colors.reset}: found {EXTRACTED_PATH} "
                f"with {n_lines:,} positions — delete it to re-extract"
            )
        elif args.skip_extraction:
            ensure_file(EXTRACTED_PATH, "Missing extracted positions file for --skip-extraction")
            print(f"resuming from {EXTRACTED_PATH}")
        else:
            extraction_stats = extract_positions(
                data_source=args.data,
                output_path=EXTRACTED_PATH,
                max_positions=args.max_positions,
                colors=colors,
            )

        stage_header("STAGE 2 - BATCHING", colors)
        manifest_path = SHARDS_DIR / "manifest.json"
        auto_skip_batching = manifest_path.exists() and bool(list(SHARDS_DIR.glob("*.npz"))) and not args.skip_batching
        if auto_skip_batching:
            existing_npz = list(SHARDS_DIR.glob("*.npz"))
            print(
                f"{colors.green}auto-skip{colors.reset}: found {len(existing_npz)} shards in {SHARDS_DIR} "
                f"with manifest — delete shards dir to re-batch"
            )
        elif args.skip_batching:
            ensure_shards(SHARDS_DIR)
            print(f"resuming from {SHARDS_DIR}")
        else:
            batching_stats = build_shards(
                extracted_path=EXTRACTED_PATH,
                out_dir=SHARDS_DIR,
                shard_size=args.shard_size,
                colors=colors,
            )

        stage_header("STAGE 3 - TRAINING", colors)
        result = run_training_or_export(
            output=args.output,
            device=str(device),
            skip_training=args.skip_training,
            colors=colors,
            no_color=no_color,
        )

        stage_header("STAGE 4 - EXPORT & VERIFY", colors)
        verification = "PASS" if result.verification_passed else "FAIL"
        print(f"export path: {result.output_path}")
        print(f"verification: {verification}")
        if not result.verification_passed:
            raise PipelineError("Verification failed after export.")

    except PipelineError as exc:
        print(f"{colors.red}ERROR{colors.reset}: {exc}")
        return 1

    total_wall = time.time() - started
    trained_positions = result.total_positions if result is not None else 0
    final_val = result.final_val_loss if result is not None else float("nan")
    verification = "PASS" if result and result.verification_passed else "FAIL"
    boxed(
        [
            "PIPELINE COMPLETE",
            f"total wall time: {total_wall:.1f}s",
            f"positions trained on: {trained_positions:,}",
            f"final val loss: {final_val:.6f}" if math.isfinite(final_val) else "final val loss: n/a",
            f"export path: {args.output}",
            f"verification: {verification}",
        ],
        colors,
    )
    print(
        f"PIPELINE COMPLETE final_val_loss={final_val:.6f} "
        f"export={args.output} verification={verification}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())