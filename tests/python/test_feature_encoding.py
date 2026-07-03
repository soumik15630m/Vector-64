"""Golden-value contract test for the HalfKP feature encoding.

The feature-index formula is shared by three places that must never disagree:
the C++ engine (nnue.cpp::halfkp_feature_index), the exporter, and the trainer
(tools/train_nnue.py). A mismatch there is exactly the class of bug behind the
earlier NNUE perspective regression, so the contract is locked with golden
values here. Runs without torch; also cross-checks the real trainer function
when torch happens to be installed.
"""
from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys

WHITE, BLACK = 0, 1
PIECE_BUCKETS = 10
SQUARES = 64
TOTAL_FEATURES = 81920


def feature_index(king_sq: int, bucket: int, piece_sq: int, perspective: int) -> int:
    """Canonical HalfKP index — must match the C++ engine and the exporter."""
    side = 1 if perspective == BLACK else 0
    return (((king_sq * PIECE_BUCKETS + bucket) * SQUARES + piece_sq) * 2) + side


def test_known_values() -> None:
    assert feature_index(0, 0, 0, WHITE) == 0
    assert feature_index(0, 0, 0, BLACK) == 1
    assert feature_index(63, 9, 63, BLACK) == TOTAL_FEATURES - 1


def test_perspective_is_the_low_bit() -> None:
    for k, b, s in [(0, 0, 0), (12, 3, 40), (63, 9, 63)]:
        assert feature_index(k, b, s, WHITE) + 1 == feature_index(k, b, s, BLACK)


def test_all_indices_unique_and_in_range() -> None:
    seen = bytearray(TOTAL_FEATURES)
    count = 0
    for k in range(SQUARES):
        for b in range(PIECE_BUCKETS):
            for s in range(SQUARES):
                for p in (WHITE, BLACK):
                    idx = feature_index(k, b, s, p)
                    assert 0 <= idx < TOTAL_FEATURES
                    assert seen[idx] == 0, "feature index collision"
                    seen[idx] = 1
                    count += 1
    assert count == TOTAL_FEATURES
    assert all(seen)


def test_matches_trainer_when_torch_available() -> None:
    """If torch is installed, the trainer's function must match the contract."""
    if importlib.util.find_spec("torch") is None:
        import pytest

        pytest.skip("torch not installed; contract verified against reference only")

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from tools.train_nnue import halfkp_feature_index

    for k, b, s, p in itertools.product((0, 7, 63), (0, 4, 9), (0, 33, 63), (WHITE, BLACK)):
        assert halfkp_feature_index(k, b, s, p) == feature_index(k, b, s, p)
