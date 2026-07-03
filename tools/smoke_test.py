#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import shutil
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import train_nnue as nnue


DEFAULT_HF_DATASET = "Lichess/chess-position-evaluations"


class SmokeFailure(RuntimeError):
    pass


class Colors:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def code(self, value: str) -> str:
        return value if self.enabled else ""

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


def ok(colors: Colors, message: str) -> None:
    print(f"{colors.green}PASS{colors.reset}: {message}")


def warn(colors: Colors, message: str) -> None:
    print(f"{colors.yellow}WARN{colors.reset}: {message}")


def fail(message: str) -> None:
    raise SmokeFailure(message)


def check_python_and_deps(colors: Colors) -> None:
    print(f"{colors.cyan}1 - Python & dependency check{colors.reset}")
    version = sys.version_info
    print(f"python: {version.major}.{version.minor}.{version.micro}")
    if version < (3, 9):
        fail("Python >= 3.9 is required. Fix: install Python 3.9 or newer and rerun this script.")

    packages = [
        ("torch", "torch", "pip install torch --index-url https://download.pytorch.org/whl/cu121"),
        ("numpy", "numpy", "pip install numpy"),
        ("tqdm", "tqdm", "pip install tqdm"),
        ("zstandard", "zstandard", "pip install zstandard"),
        ("datasets", "datasets", "pip install datasets"),
    ]
    missing: List[str] = []
    for display, module_name, install in packages:
        try:
            module = importlib.import_module(module_name)
            version_text = getattr(module, "__version__", "unknown")
            print(f"{display}: {version_text}")
        except ImportError:
            print(f"{colors.red}MISSING{colors.reset}: {display}")
            print(f"Fix: {install}")
            missing.append(display)

    if missing:
        fail("Missing Python dependencies. Install the packages above and rerun.")
    ok(colors, "Python and imports are usable")


def check_constants(colors: Colors) -> None:
    print(f"{colors.cyan}2 - Constants self-consistency check{colors.reset}")
    formulas = [
        (
            "HALF_KP_TOTAL_FEATURES == 64 * HALF_KP_PIECE_BUCKETS * 64 * 2",
            64 * nnue.HALF_KP_PIECE_BUCKETS * 64 * 2,
            81_920,
        ),
        (
            "featureTransformWeight bytes == HALF_KP_TOTAL_FEATURES * HIDDEN_SIZE * 2",
            nnue.HALF_KP_TOTAL_FEATURES * nnue.HIDDEN_SIZE * 2,
            83_886_080,
        ),
        (
            "total expected file size",
            nnue.EXPECTED_FILE_SIZE,
            83_904_900,
        ),
        (
            "bias block bytes == HIDDEN_SIZE * 2 + DENSE_L1_SIZE * 4 + DENSE_L2_SIZE * 4 + 4",
            nnue.HIDDEN_SIZE * 2 + nnue.DENSE_L1_SIZE * 4 + nnue.DENSE_L2_SIZE * 4 + 4,
            1_284,
        ),
    ]
    for label, computed, expected in formulas:
        print(f"{label}: {computed}")
        if computed != expected:
            fail(f"{label} computed {computed}, expected {expected}. Fix the hardcoded constants.")
    ok(colors, "Constants match the architecture contract")


def check_feature_index(colors: Colors) -> None:
    print(f"{colors.cyan}3 - Feature index sanity check{colors.reset}")
    vectors = [
        (0, 0, 0, nnue.WHITE, 0),
        (0, 0, 0, nnue.BLACK, 1),
        (1, 0, 0, nnue.WHITE, 1280),
        (0, 1, 0, nnue.WHITE, 128),
        (60, 9, 63, nnue.BLACK, 78079),
    ]
    all_ok = True
    for king_sq, bucket, piece_sq, perspective, expected in vectors:
        actual = nnue.halfkp_feature_index(king_sq, bucket, piece_sq, perspective)
        passed = actual == expected
        all_ok = all_ok and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status}: halfkp_feature_index({king_sq}, {bucket}, {piece_sq}, "
            f"{perspective}) = {actual}, expected {expected}"
        )
    if not all_ok:
        fail("Feature index formula mismatch. Fix halfkp_feature_index before training.")
    ok(colors, "Feature index vectors match")


def check_fens(colors: Colors) -> None:
    print(f"{colors.cyan}4 - FEN parser check{colors.reset}")
    cases = [
        {
            "name": "starting position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "white_count": 16,
            "black_count": 16,
            "white_king": 4,
            "black_king": 60,
            "features": 30,
        },
        {
            "name": "open game middlegame",
            "fen": "r1bqkbnr/pppp1ppp/2n1p3/8/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
            "white_count": 16,
            "black_count": 16,
            "white_king": 4,
            "black_king": 60,
            "features": 30,
        },
        {
            "name": "castled middlegame",
            "fen": "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 7",
            "white_count": 16,
            "black_count": 16,
            "white_king": 6,
            "black_king": 62,
            "features": 30,
        },
    ]
    all_ok = True
    for case in cases:
        pos = nnue.parse_fen(case["fen"])
        white_features, black_features = nnue.halfkp_features_from_position(pos)
        passed = (
            pos.white_count == case["white_count"]
            and pos.black_count == case["black_count"]
            and pos.white_king == case["white_king"]
            and pos.black_king == case["black_king"]
            and len(white_features) == case["features"]
            and len(black_features) == case["features"]
            and all(0 <= idx < nnue.HALF_KP_TOTAL_FEATURES for idx in white_features + black_features)
        )
        all_ok = all_ok and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status}: {case['name']} pieces W/B={pos.white_count}/{pos.black_count}, "
            f"kings={pos.white_king}/{pos.black_king}, features={len(white_features)}/{len(black_features)}"
        )
    if not all_ok:
        fail("FEN parser or feature generation failed. Fix FEN square mapping before training.")
    ok(colors, "FEN parser and HalfKP feature generation are consistent")


def make_synthetic_batch() -> Dict[str, Any]:
    fens = nnue.VERIFY_FENS[:4]
    samples = []
    for fen in fens:
        pos = nnue.parse_fen(fen)
        white_features, black_features = nnue.halfkp_features_from_position(pos)
        samples.append(
            {
                "white_indices": white_features,
                "black_indices": black_features,
                "stm": pos.stm,
                "target": 0.0,
            }
        )
    return nnue.collate_samples(samples)


def check_model_forward(colors: Colors) -> nnue.NNUEModel:
    print(f"{colors.cyan}5 - Model forward pass check{colors.reset}")
    import torch

    torch.manual_seed(1337)
    model = nnue.NNUEModel()
    batch = make_synthetic_batch()
    with torch.no_grad():
        output = model(batch["white_indices"], batch["black_indices"], batch["stm"])
    shape_ok = tuple(output.shape) == (4, 1)
    finite_ok = bool(torch.isfinite(output).all().item())
    range_ok = bool(((output >= -3000.0) & (output <= 3000.0)).all().item())
    print(f"output shape: {tuple(output.shape)}")
    print(f"output min/max cp: {float(output.min()):.4f}/{float(output.max()):.4f}")
    if not (shape_ok and finite_ok and range_ok):
        fail("Random NNUEModel forward pass failed. Fix model initialization or forward shapes.")
    ok(colors, "Random model forward pass is finite and plausible")
    return model


def check_export_round_trip(colors: Colors, model: nnue.NNUEModel) -> None:
    print(f"{colors.cyan}6 - Export round-trip check{colors.reset}")
    output = Path("smoke_test_output.nnue")
    meta = output.with_suffix(output.suffix + ".meta.json")
    try:
        nnue.export_model_to_nnue(model, output)
        if not output.exists():
            fail("Export did not create smoke_test_output.nnue")
        size = output.stat().st_size
        print(f"file size: {size}")
        if size != nnue.EXPECTED_FILE_SIZE:
            fail(f"Exported file size {size}, expected {nnue.EXPECTED_FILE_SIZE}")

        blob = output.read_bytes()
        magic, version, hidden = struct.unpack_from("<13sII", blob, 0)
        if magic != nnue.NETWORK_MAGIC:
            fail(f"Bad magic bytes: {magic!r}")
        if version != nnue.VERSION or hidden != nnue.HIDDEN_SIZE:
            fail(f"Bad header fields: version={version}, hiddenSize={hidden}")

        padding_checks = [
            ("header padding [21,64)", 21, 64),
            ("l1 boundary zero-length alignment", 83_886_144, 83_886_144),
            ("output padding [83903584,83903616)", 83_903_584, 83_903_616),
        ]
        for name, begin, end in padding_checks:
            if any(b != 0 for b in blob[begin:end]):
                fail(f"Nonzero bytes in {name}")
            print(f"padding check: {name} OK")

        if not meta.exists():
            fail("Export did not create .meta.json sidecar")
        meta_obj = json.loads(meta.read_text(encoding="utf-8"))
        missing = [key for key in nnue.META_KEYS if key not in meta_obj]
        if missing:
            fail(f"Sidecar missing top-level keys: {', '.join(missing)}")
        bad = [key for key in nnue.META_KEYS if not (isinstance(meta_obj[key], (int, float)) and meta_obj[key] > 0)]
        if bad:
            fail(f"Sidecar keys must be positive floats: {', '.join(bad)}")
        ok(colors, "Exported binary and sidecar round-trip checks passed")
    finally:
        for path in (output, meta):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def open_data_lines(path: Path) -> Iterator[str]:
    if path.suffix.lower() == ".zst":
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as exc:
            raise SmokeFailure("Fix: pip install zstandard") from exc
        with path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    yield from text
    else:
        with path.open("r", encoding="utf-8") as fh:
            yield from fh


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


def iter_hf_rows() -> Iterator[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SmokeFailure("Fix: pip install datasets") from exc

    dataset = load_dataset(DEFAULT_HF_DATASET, split="train", streaming=True)
    for row in dataset:
        if isinstance(row, dict):
            yield row


def iter_local_json_rows(path: Path) -> Iterator[Dict[str, Any]]:
    for line in open_data_lines(path):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            yield {"__error__": "json_error"}
            continue
        if isinstance(obj, dict):
            yield obj


def check_data_file(colors: Colors, data_path: Optional[Path]) -> None:
    print(f"{colors.cyan}7 - Data file check{colors.reset}")
    if data_path is None:
        print(f"No --data provided; validating default Hugging Face stream: {DEFAULT_HF_DATASET}")
        row_iter = iter_hf_rows()
        source_hint = f"hf://{DEFAULT_HF_DATASET}"
    else:
        if not data_path.exists():
            fail(f"Data file does not exist: {data_path}. Fix: pass the correct --data PATH.")
        row_iter = iter_local_json_rows(data_path)
        source_hint = str(data_path)

    valid_json_with_fen = 0
    usable_cp = 0
    skipped: Counter[str] = Counter()
    total = 0

    try:
        for total, obj in enumerate(row_iter, start=1):
            if total > 500:
                break
            if obj.get("__error__") == "json_error":
                skipped["json_error"] += 1
                continue
            if not isinstance(obj.get("fen"), str):
                skipped["missing_fen"] += 1
                continue
            valid_json_with_fen += 1
            cp, depth, reason = row_eval_cp_depth(obj)
            if cp is not None:
                usable_cp += 1
            else:
                skipped[reason] += 1
    except OSError as exc:
        fail(f"Could not read {source_hint}: {exc}. Fix: check file permissions and path.")
    except Exception as exc:
        fail(f"Could not sample {source_hint}: {exc}. Fix: check internet access, install datasets, or pass --data PATH.")

    print(f"sampled lines: {min(total, 500)}")
    print(f"valid rows with fen: {valid_json_with_fen}")
    print(f"usable evals[].pvs[].cp values: {usable_cp}")
    if skipped:
        print("skipped reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
    if valid_json_with_fen == 0:
        fail("No sampled line had valid JSON with a fen field. Fix: pass a Lichess eval JSONL(.zst) file.")
    if usable_cp == 0:
        fail("No sampled line had a usable evals[].pvs[].cp value. Fix: verify the dataset format.")
    ok(colors, "Dataset sample is readable and contains usable FEN/cp records")


def check_cuda(colors: Colors) -> None:
    print(f"{colors.cyan}8 - CUDA availability check{colors.reset}")
    import torch

    if not torch.cuda.is_available():
        print("FATAL: CUDA is required. No CUDA-capable GPU was detected.")
        if torch.version.cuda is None:
            print(
                "FATAL: PyTorch was installed without CUDA support. Reinstall with: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
        raise SmokeFailure("CUDA requirement failed.")

    if torch.version.cuda is None:
        print(
            "FATAL: PyTorch was installed without CUDA support. Reinstall with: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        raise SmokeFailure("PyTorch CUDA build requirement failed.")

    print(f"torch CUDA build: {torch.version.cuda}")
    count = torch.cuda.device_count()
    print(f"CUDA GPUs: {count}")
    total_vram = 0
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        total_vram += props.total_memory
        capability = f"{props.major}.{props.minor}"
        print(f"gpu {i}: {torch.cuda.get_device_name(i)} | {total_gb:.2f} GB | compute {capability}")

    device = torch.device("cuda:0")
    a = torch.randn((512, 512), device=device, dtype=torch.float32)
    b = a @ a
    if not torch.isfinite(b).all().item():
        fail("CUDA matmul produced non-finite values. Fix: check GPU/driver health.")
    torch.cuda.synchronize(device)
    total_gb = total_vram / (1024 ** 3)
    print(f"total VRAM: {total_gb:.2f} GB")
    if total_gb < 6.0:
        warn(colors, "Total VRAM is below 6 GB; batch size 4096 may be uncomfortable.")
    ok(colors, "CUDA is available and usable")


def check_disk_space(colors: Colors) -> None:
    print(f"{colors.cyan}9 - Disk space check{colors.reset}")
    extracted_gb = 7.0
    shards_gb = 2.4
    checkpoints_gb = 1.7
    total_estimate = extracted_gb + shards_gb + checkpoints_gb
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free / (1024 ** 3)
    print(f"extracted positions estimate: ~{extracted_gb:.1f} GB")
    print(f"shards estimate: ~{shards_gb:.1f} GB")
    print(f"checkpoints estimate: ~{checkpoints_gb:.1f} GB")
    print(f"estimated subtotal: ~{total_estimate:.1f} GB")
    print(f"available in project directory: {free_gb:.1f} GB")
    if free_gb < 15.0:
        warn(colors, "Less than 15 GB free; the full pipeline may run out of disk space.")
    ok(colors, "Disk space check completed")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-flight checks for the VECTOR64_NNUE pipeline.")
    parser.add_argument("--data", type=Path, default=None, help="Optional Lichess dataset path to validate")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    colors = Colors(enabled=not args.no_color and "NO_COLOR" not in os.environ)
    try:
        check_python_and_deps(colors)
        check_constants(colors)
        check_feature_index(colors)
        check_fens(colors)
        model = check_model_forward(colors)
        check_export_round_trip(colors, model)
        check_data_file(colors, args.data)
        check_cuda(colors)
        check_disk_space(colors)
    except SmokeFailure as exc:
        print(f"{colors.red}FAILED{colors.reset}: {exc}")
        return 1

    print("ALL CHECKS PASSED — safe to run pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
