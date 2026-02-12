#!/usr/bin/env python3
"""
Validate VECTOR64_NNUE file header, section offsets, and section sizes.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Dict, Tuple


MAGIC = b"VECTOR64_NNUE"
VERSION = 1
ALIGNMENT = 64

TOTAL_FEATURES = 81920
HIDDEN = 512
L1 = 32
L2 = 32

HEADER_SIZE = struct.calcsize("<13sII")
SECTION_SIZES = {
    "FeatureTransformWeights": TOTAL_FEATURES * HIDDEN * 2,  # int16
    "HiddenLayer1Weights": L1 * HIDDEN,  # int8
    "HiddenLayer2Weights": L2 * L1,  # int8
    "OutputWeights": L2,  # int8
    "Biases": (HIDDEN * 2) + (L1 * 4) + (L2 * 4) + 4,  # int16 + int32 + int32 + int32
}


def align64(offset: int) -> int:
    return (offset + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1)


def expected_layout() -> Tuple[Dict[str, Dict[str, int]], int]:
    sections: Dict[str, Dict[str, int]] = {}
    pos = HEADER_SIZE

    for name in (
        "FeatureTransformWeights",
        "HiddenLayer1Weights",
        "HiddenLayer2Weights",
        "OutputWeights",
        "Biases",
    ):
        pos = align64(pos)
        size = SECTION_SIZES[name]
        sections[name] = {"offset": pos, "size": size}
        pos += size

    return sections, pos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check VECTOR64_NNUE binary file.")
    p.add_argument("path", help="Path to .nnue file")
    p.add_argument(
        "--allow-trailing",
        action="store_true",
        help="Allow extra bytes after the final expected section",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"[FAIL] File does not exist: {path}")
        return 1

    size = path.stat().st_size
    sections, expected_size = expected_layout()

    with path.open("rb") as fh:
        header_raw = fh.read(HEADER_SIZE)
        if len(header_raw) != HEADER_SIZE:
            print(f"[FAIL] Header too short: expected {HEADER_SIZE}, got {len(header_raw)}")
            return 1

        magic, version, hidden = struct.unpack("<13sII", header_raw)

        ok = True
        if magic != MAGIC:
            print(f"[FAIL] Magic mismatch: got {magic!r}, expected {MAGIC!r}")
            ok = False
        if version != VERSION:
            print(f"[FAIL] Version mismatch: got {version}, expected {VERSION}")
            ok = False
        if hidden != HIDDEN:
            print(f"[FAIL] Hidden-size mismatch: got {hidden}, expected {HIDDEN}")
            ok = False

        for name, spec in sections.items():
            offset = spec["offset"]
            section_size = spec["size"]
            if offset + section_size > size:
                print(
                    f"[FAIL] Section '{name}' truncated: "
                    f"needs [{offset}, {offset + section_size}), file size is {size}"
                )
                ok = False

    print(f"File: {path}")
    print(f"Size: {size} bytes")
    print(f"Expected minimum size: {expected_size} bytes")
    print("Sections:")
    for name, spec in sections.items():
        print(f"  - {name}: offset={spec['offset']} size={spec['size']}")

    if size != expected_size:
        if size > expected_size and args.allow_trailing:
            print(f"[WARN] Trailing bytes present: {size - expected_size} bytes")
        else:
            print(f"[FAIL] File size mismatch: got {size}, expected {expected_size}")
            ok = False

    if ok:
        print("[PASS] VECTOR64_NNUE file layout is valid.")
        return 0
    print("[FAIL] VECTOR64_NNUE file layout is invalid.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

