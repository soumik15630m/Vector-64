#!/usr/bin/env python3
"""
Convert Lichess eval JSONL(.zst) into HalfKP NPZ shards for Vector-64 training.

Output shard schema:
  - white_indices: int32 [N, max_features]
  - black_indices: int32 [N, max_features]
  - stm: int8 [N] where 0=white-to-move, 1=black-to-move
  - eval_cp: float32 [N] from white perspective

This script can filter:
  - mate scores
  - low-depth / low-knodes evals
  - extreme centipawn scores
  - duplicate positions (none/shard/global)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


PIECE_BUCKET = {
    1: 0,  # pawn
    2: 1,  # knight
    3: 2,  # bishop
    4: 3,  # rook
    5: 4,  # queen
}

PIECE_TYPE = {
    "p": 1,
    "n": 2,
    "b": 3,
    "r": 4,
    "q": 5,
    "k": 6,
}


@dataclass
class Stats:
    lines_total: int = 0
    lines_json_error: int = 0
    lines_missing_fen: int = 0
    lines_invalid_fen: int = 0
    lines_no_eval: int = 0
    lines_mate_filtered: int = 0
    lines_depth_filtered: int = 0
    lines_knodes_filtered: int = 0
    lines_extreme_filtered: int = 0
    lines_duplicate_filtered: int = 0
    lines_feature_overflow: int = 0
    samples_kept: int = 0
    shards_written: int = 0


@dataclass
class ShardInfo:
    file: str
    meta: str
    samples: int


@dataclass
class ParsedFen:
    stm: int  # 0=white, 1=black
    white_king_sq: int
    black_king_sq: int
    # tuples: (square, piece_type 1..6, color 0=white/1=black)
    pieces: List[Tuple[int, int, int]]


@dataclass
class CpSummary:
    min: Optional[float] = None
    max: Optional[float] = None
    mean: float = 0.0
    std: float = 0.0
    bins: List[float] = None  # type: ignore[assignment]
    counts: List[int] = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HalfKP NPZ shards from eval JSONL.")
    p.add_argument("--input", required=True, help="Input .jsonl or .jsonl.zst file path")
    p.add_argument("--out-dir", required=True, help="Output directory for shard .npz files")
    p.add_argument("--shard-size", type=int, default=500_000, help="Samples per shard (default: 500000)")
    p.add_argument("--max-features", type=int, default=64, help="Padded feature slots per sample (default: 64)")
    p.add_argument("--prefix", default="halfkp_shard", help="Shard filename prefix")
    p.add_argument("--min-depth", type=int, default=0, help="Minimum eval depth filter (default: 0)")
    p.add_argument("--min-knodes", type=int, default=0, help="Minimum eval knodes filter (default: 0)")
    p.add_argument("--max-abs-cp", type=float, default=2000.0, help="Drop samples with |cp| above this (default: 2000)")
    p.add_argument(
        "--cp-source",
        choices=("white", "stm"),
        default="white",
        help="Perspective of input cp in JSON (default: white).",
    )
    p.add_argument(
        "--dedup-scope",
        choices=("none", "shard", "global"),
        default="global",
        help="Dedup scope (default: global)",
    )
    p.add_argument(
        "--dedup-db",
        default=None,
        help="SQLite file for global dedup. Default: <out-dir>/dedup_seen.sqlite",
    )
    p.add_argument("--max-samples", type=int, default=0, help="Stop after writing this many samples (0 = unlimited)")
    p.add_argument("--log-every", type=int, default=100_000, help="Progress log interval in input lines")
    p.add_argument("--uncompressed", action="store_true", help="Write with np.savez instead of np.savez_compressed")
    return p.parse_args()


def _to_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _canonical_fen_key(fen: str) -> str:
    # Dedup key intentionally ignores castling/ep/clocks and keeps only:
    # piece placement + side to move.
    # This avoids leaking duplicates through move-history-only FEN fields.
    parts = fen.strip().split()
    if len(parts) >= 2:
        return " ".join(parts[:2])
    return fen.strip()


class Deduper:
    def __init__(self, scope: str, db_path: Optional[Path] = None):
        self.scope = scope
        self._shard_seen: set[bytes] = set()
        self._conn: Optional[sqlite3.Connection] = None
        self._txn_count = 0
        self._txn_flush_every = 20_000

        if scope == "global":
            assert db_path is not None
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path))
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=OFF")
            cur.execute("PRAGMA temp_store=MEMORY")
            cur.execute("CREATE TABLE IF NOT EXISTS seen (h BLOB PRIMARY KEY)")
            self._conn.commit()
            cur.execute("BEGIN")

    def begin_shard(self) -> None:
        if self.scope == "shard":
            self._shard_seen.clear()

    def is_duplicate(self, key: str) -> bool:
        if self.scope == "none":
            return False

        h = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()

        if self.scope == "shard":
            if h in self._shard_seen:
                return True
            self._shard_seen.add(h)
            return False

        assert self._conn is not None
        try:
            self._conn.execute("INSERT INTO seen(h) VALUES (?)", (h,))
            self._txn_count += 1
            if self._txn_count >= self._txn_flush_every:
                self._conn.commit()
                self._conn.execute("BEGIN")
                self._txn_count = 0
            return False
        except sqlite3.IntegrityError:
            return True

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None


def _iter_lines(path: Path) -> Iterator[str]:
    if path.suffix.lower() == ".zst":
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Input is .zst but python-zstandard is not installed. Install with: pip install zstandard"
            ) from exc

        with path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    for line in text:
                        yield line
        return

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            yield line


def _extract_cp(record: dict, min_depth: int, min_knodes: int) -> Tuple[Optional[float], str, int, int]:
    # Returns: (cp_or_none, reason, depth, knodes)
    # reason in {"ok","mate","depth","knodes","no_eval"}
    had_mate = False
    had_depth = False
    had_knodes = False
    candidates: List[Tuple[int, int, float]] = []

    evals = record.get("evals")
    has_evals_array = isinstance(evals, list)
    if has_evals_array:
        for e in evals:
            if not isinstance(e, dict):
                continue
            depth = _to_int(e.get("depth"), 0)
            if depth < min_depth:
                had_depth = True
                continue

            knodes = _to_int(e.get("knodes"), 0)
            if knodes < min_knodes:
                had_knodes = True
                continue

            score_candidates: List[dict] = []
            pvs = e.get("pvs")
            if isinstance(pvs, list):
                for pv in pvs:
                    if isinstance(pv, dict):
                        score_candidates.append(pv)

            # Fallback to direct eval-entry score fields.
            if not score_candidates:
                score_candidates.append(e)

            best_cp_for_eval: Optional[float] = None
            for s in score_candidates:
                mate = s.get("mate")
                if mate is not None:
                    had_mate = True
                    continue
                cp = _to_float(s.get("cp"))
                if cp is None:
                    continue
                best_cp_for_eval = cp
                break

            if best_cp_for_eval is None:
                continue
            candidates.append((depth, knodes, best_cp_for_eval))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        depth, knodes, cp = candidates[0]
        return cp, "ok", depth, knodes

    # Fallback for flattened schema only when eval-array is absent.
    # If eval-array exists, do not override its quality filters with top-level fields.
    if has_evals_array:
        if had_mate:
            return None, "mate", 0, 0
        if had_depth:
            return None, "depth", 0, 0
        if had_knodes:
            return None, "knodes", 0, 0
        return None, "no_eval", 0, 0

    depth = _to_int(record.get("depth"), 0)
    knodes = _to_int(record.get("knodes"), 0)
    if depth >= min_depth and knodes >= min_knodes:
        mate = record.get("mate")
        cp = _to_float(record.get("cp"))
        if mate is not None:
            return None, "mate", depth, knodes
        if cp is not None:
            return cp, "ok", depth, knodes

    if had_mate:
        return None, "mate", 0, 0
    if had_depth:
        return None, "depth", 0, 0
    if had_knodes:
        return None, "knodes", 0, 0
    return None, "no_eval", 0, 0


def _parse_fen_minimal(fen: str) -> ParsedFen:
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError("invalid FEN fields")

    board_field = parts[0]
    stm_field = parts[1]
    if stm_field == "w":
        stm = 0
    elif stm_field == "b":
        stm = 1
    else:
        raise ValueError("invalid side-to-move field")

    rank = 7
    file = 0
    pieces: List[Tuple[int, int, int]] = []

    for ch in board_field:
        if ch == "/":
            if file != 8:
                raise ValueError("invalid FEN rank width")
            rank -= 1
            file = 0
            continue

        if ch.isdigit():
            step = int(ch)
            if step <= 0 or step > 8:
                raise ValueError("invalid FEN empty run")
            file += step
            if file > 8:
                raise ValueError("invalid FEN overflow")
            continue

        lower = ch.lower()
        piece_type = PIECE_TYPE.get(lower)
        if piece_type is None:
            raise ValueError("invalid FEN piece")
        if not (0 <= rank <= 7 and 0 <= file <= 7):
            raise ValueError("invalid FEN square")

        sq = rank * 8 + file
        color = 0 if ch.isupper() else 1
        pieces.append((sq, piece_type, color))
        file += 1
        if file > 8:
            raise ValueError("invalid FEN overflow")

    if rank != 0 or file != 8:
        raise ValueError("invalid FEN terminal rank")

    white_kings = [sq for (sq, pt, c) in pieces if pt == 6 and c == 0]
    black_kings = [sq for (sq, pt, c) in pieces if pt == 6 and c == 1]
    if len(white_kings) != 1 or len(black_kings) != 1:
        raise ValueError("invalid king count")

    return ParsedFen(
        stm=stm,
        white_king_sq=white_kings[0],
        black_king_sq=black_kings[0],
        pieces=pieces,
    )


def _encode_halfkp(fen: str, max_features: int):
    parsed = _parse_fen_minimal(fen)
    return _encode_halfkp_from_parsed(parsed, max_features)


def _encode_halfkp_from_parsed(parsed: ParsedFen, max_features: int):

    white_idx = np.full((max_features,), -1, dtype=np.int32)
    black_idx = np.full((max_features,), -1, dtype=np.int32)

    k = 0
    for sq, piece_type, color in sorted(parsed.pieces, key=lambda x: x[0]):
        if piece_type == 6:
            continue
        bucket_base = PIECE_BUCKET.get(piece_type)
        if bucket_base is None:
            continue
        if k >= max_features:
            raise OverflowError("feature overflow")

        bucket = bucket_base + (5 if color == 1 else 0)
        white_idx[k] = (((parsed.white_king_sq * 10 + bucket) * 64 + sq) * 2) + 0
        black_idx[k] = (((parsed.black_king_sq * 10 + bucket) * 64 + sq) * 2) + 1
        k += 1

    return white_idx, black_idx, parsed.stm


def _phase_bucket_from_parsed(parsed: ParsedFen) -> str:
    # Simple non-pawn-material phase estimate over both sides:
    # knight/bishop=1, rook=2, queen=4; max=24 at initial position.
    units = 0
    for _, piece_type, _ in parsed.pieces:
        if piece_type == 2 or piece_type == 3:
            units += 1
        elif piece_type == 4:
            units += 2
        elif piece_type == 5:
            units += 4

    if units >= 18:
        return "opening"
    if units >= 9:
        return "middlegame"
    return "endgame"


def _finalize_cp_summary(
    cp_min: Optional[float],
    cp_max: Optional[float],
    cp_sum: float,
    cp_sum_sq: float,
    n: int,
    edges: List[float],
    counts: List[int],
) -> CpSummary:
    if n <= 0:
        return CpSummary(
            min=None,
            max=None,
            mean=0.0,
            std=0.0,
            bins=edges,
            counts=counts,
        )

    mean = cp_sum / float(n)
    var = max(0.0, (cp_sum_sq / float(n)) - (mean * mean))
    return CpSummary(
        min=cp_min,
        max=cp_max,
        mean=mean,
        std=var ** 0.5,
        bins=edges,
        counts=counts,
    )


def _write_shard(
    out_dir: Path,
    prefix: str,
    shard_index: int,
    white_rows: List[np.ndarray],
    black_rows: List[np.ndarray],
    stm_rows: List[int],
    cp_rows: List[float],
    compressed: bool,
) -> ShardInfo:
    shard_name = f"{prefix}_{shard_index:05d}.npz"
    shard_path = out_dir / shard_name
    meta_path = out_dir / f"{prefix}_{shard_index:05d}.meta.json"

    white = np.stack(white_rows, axis=0).astype(np.int32, copy=False)
    black = np.stack(black_rows, axis=0).astype(np.int32, copy=False)
    stm = np.asarray(stm_rows, dtype=np.int8)
    cp = np.asarray(cp_rows, dtype=np.float32)

    if compressed:
        np.savez_compressed(
            shard_path,
            white_indices=white,
            black_indices=black,
            stm=stm,
            eval_cp=cp,
        )
    else:
        np.savez(
            shard_path,
            white_indices=white,
            black_indices=black,
            stm=stm,
            eval_cp=cp,
        )

    shard_meta = {
        "file": shard_name,
        "samples": int(cp.shape[0]),
        "shape_white": list(white.shape),
        "shape_black": list(black.shape),
        "shape_stm": list(stm.shape),
        "shape_eval_cp": list(cp.shape),
    }
    meta_path.write_text(json.dumps(shard_meta, indent=2) + "\n", encoding="utf-8")

    return ShardInfo(file=shard_name, meta=meta_path.name, samples=int(cp.shape[0]))


def _print_progress(stats: Stats, shards: int) -> None:
    print(
        "[progress] "
        f"lines={stats.lines_total:,} "
        f"kept={stats.samples_kept:,} "
        f"shards={shards} "
        f"dup={stats.lines_duplicate_filtered:,} "
        f"mate={stats.lines_mate_filtered:,} "
        f"extreme={stats.lines_extreme_filtered:,} "
        f"invalid_fen={stats.lines_invalid_fen:,}",
        file=sys.stderr,
    )


def main() -> int:
    args = parse_args()

    if args.shard_size <= 0:
        raise ValueError("--shard-size must be > 0")
    if args.max_features < 30:
        raise ValueError("--max-features must be >= 30")
    if args.max_abs_cp <= 0:
        raise ValueError("--max-abs-cp must be > 0")
    if args.min_depth < 0:
        raise ValueError("--min-depth must be >= 0")
    if args.min_knodes < 0:
        raise ValueError("--min-knodes must be >= 0")

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dedup_db = Path(args.dedup_db) if args.dedup_db else (out_dir / "dedup_seen.sqlite")
    deduper = Deduper(args.dedup_scope, dedup_db if args.dedup_scope == "global" else None)
    deduper.begin_shard()

    stats = Stats()
    shards: List[ShardInfo] = []

    phase_hist: Dict[str, int] = {"opening": 0, "middlegame": 0, "endgame": 0}
    depth_hist: Dict[str, int] = {}

    cp_bins = 20
    cp_edges = np.linspace(-float(args.max_abs_cp), float(args.max_abs_cp), num=cp_bins + 1, dtype=np.float64).tolist()
    cp_counts = [0 for _ in range(cp_bins)]
    cp_min: Optional[float] = None
    cp_max: Optional[float] = None
    cp_sum = 0.0
    cp_sum_sq = 0.0

    white_rows: List[np.ndarray] = []
    black_rows: List[np.ndarray] = []
    stm_rows: List[int] = []
    cp_rows: List[float] = []

    def flush_shard() -> None:
        if not cp_rows:
            return
        shard_index = len(shards)
        info = _write_shard(
            out_dir=out_dir,
            prefix=args.prefix,
            shard_index=shard_index,
            white_rows=white_rows,
            black_rows=black_rows,
            stm_rows=stm_rows,
            cp_rows=cp_rows,
            compressed=(not args.uncompressed),
        )
        shards.append(info)
        stats.shards_written = len(shards)

        white_rows.clear()
        black_rows.clear()
        stm_rows.clear()
        cp_rows.clear()
        deduper.begin_shard()

    try:
        for line in _iter_lines(in_path):
            stats.lines_total += 1
            if args.log_every > 0 and stats.lines_total % args.log_every == 0:
                _print_progress(stats, len(shards))

            stripped = line.strip()
            if not stripped:
                continue

            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                stats.lines_json_error += 1
                continue

            fen = rec.get("fen")
            if not isinstance(fen, str) or not fen.strip():
                stats.lines_missing_fen += 1
                continue

            cp, reason, sel_depth, _ = _extract_cp(rec, args.min_depth, args.min_knodes)
            if cp is None:
                if reason == "mate":
                    stats.lines_mate_filtered += 1
                elif reason == "depth":
                    stats.lines_depth_filtered += 1
                elif reason == "knodes":
                    stats.lines_knodes_filtered += 1
                else:
                    stats.lines_no_eval += 1
                continue

            try:
                parsed = _parse_fen_minimal(fen)
            except Exception:
                stats.lines_invalid_fen += 1
                continue

            key = _canonical_fen_key(fen)
            if deduper.is_duplicate(key):
                stats.lines_duplicate_filtered += 1
                continue

            try:
                white_idx, black_idx, stm = _encode_halfkp_from_parsed(parsed, args.max_features)
            except OverflowError:
                stats.lines_feature_overflow += 1
                continue

            cp_white = float(cp)
            if args.cp_source == "stm":
                cp_white = cp_white if stm == 0 else -cp_white

            if abs(cp_white) > args.max_abs_cp:
                stats.lines_extreme_filtered += 1
                continue

            white_rows.append(white_idx)
            black_rows.append(black_idx)
            stm_rows.append(int(stm))
            cp_rows.append(cp_white)
            stats.samples_kept += 1

            depth_key = str(sel_depth)
            depth_hist[depth_key] = depth_hist.get(depth_key, 0) + 1

            phase = _phase_bucket_from_parsed(parsed)
            phase_hist[phase] = phase_hist.get(phase, 0) + 1

            cp_min = cp_white if cp_min is None else min(cp_min, cp_white)
            cp_max = cp_white if cp_max is None else max(cp_max, cp_white)
            cp_sum += cp_white
            cp_sum_sq += cp_white * cp_white

            # Map cp into fixed histogram bins over [-max_abs_cp, max_abs_cp].
            span = float(args.max_abs_cp) * 2.0
            idx = int(((cp_white + float(args.max_abs_cp)) / span) * cp_bins)
            if idx < 0:
                idx = 0
            elif idx >= cp_bins:
                idx = cp_bins - 1
            cp_counts[idx] += 1

            if len(cp_rows) >= args.shard_size:
                flush_shard()

            if args.max_samples > 0 and stats.samples_kept >= args.max_samples:
                break

        flush_shard()
    finally:
        deduper.close()

    cp_summary = _finalize_cp_summary(
        cp_min=cp_min,
        cp_max=cp_max,
        cp_sum=cp_sum,
        cp_sum_sq=cp_sum_sq,
        n=stats.samples_kept,
        edges=cp_edges,
        counts=cp_counts,
    )

    manifest = {
        "input": str(in_path),
        "out_dir": str(out_dir),
        "total_samples": stats.samples_kept,
        "config": {
            "shard_size": args.shard_size,
            "max_features": args.max_features,
            "min_depth": args.min_depth,
            "min_knodes": args.min_knodes,
            "max_abs_cp": args.max_abs_cp,
            "cp_source": args.cp_source,
            "target_scaling": "none_raw_centipawn",
            "dedup_key": "piece_placement+side_to_move",
            "dedup_scope": args.dedup_scope,
            "dedup_db": str(dedup_db) if args.dedup_scope == "global" else None,
            "max_samples": args.max_samples,
            "compressed": (not args.uncompressed),
        },
        "stats": asdict(stats),
        "cp_summary": asdict(cp_summary),
        "depth_hist": dict(sorted(depth_hist.items(), key=lambda kv: int(kv[0]))),
        "phase_hist": phase_hist,
        "shards": [asdict(s) for s in shards],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(shards)} shard(s) to: {out_dir}")
    print(f"Manifest: {manifest_path}")
    _print_progress(stats, len(shards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
