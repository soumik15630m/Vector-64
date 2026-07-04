#!/usr/bin/env python3
"""STK-HalfKA feature encoder (reference implementation).

Mirrors src/nnue/halfka.h exactly. Used both as the training-data encoder and
as the parity oracle for the C++ inference path. See the header for the scheme.

Squares: a1=0, file 0=a, rank 0=rank 1 (sq = rank*8 + file), matching Core.
Colours: WHITE=0, BLACK=1. Piece types: PAWN=1..KING=6.
"""

from __future__ import annotations

import sys

KING_BUCKETS = 32
PIECE_KINDS = 11
SQUARES = 64
FEATURES = KING_BUCKETS * PIECE_KINDS * SQUARES  # 22528

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

_PIECE_TYPE = {"p": PAWN, "n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN, "k": KING}


def flip_rank(sq: int) -> int:
    return sq ^ 56


def flip_file(sq: int) -> int:
    return sq ^ 7


def orient_sq(sq: int, side: int, mirror: bool) -> int:
    s = sq if side == WHITE else flip_rank(sq)
    return flip_file(s) if mirror else s


def make_orient(side: int, king_sq: int) -> tuple[int, bool, int]:
    ok = king_sq if side == WHITE else flip_rank(king_sq)
    mirror = (ok & 7) >= 4
    tk = flip_file(ok) if mirror else ok
    bucket = (tk >> 3) * 4 + (tk & 7)
    return side, mirror, bucket


def piece_kind(side: int, piece_color: int, pt: int) -> int:
    if piece_color == side:
        if pt == KING:
            return -1
        return pt - 1
    return 4 + pt


def feature_index(orient: tuple[int, bool, int], piece_color: int, pt: int, sq: int) -> int:
    side, mirror, bucket = orient
    kind = piece_kind(side, piece_color, pt)
    if kind < 0:
        return -1
    ts = orient_sq(sq, side, mirror)
    return (bucket * PIECE_KINDS + kind) * SQUARES + ts


def parse_fen_pieces(fen: str) -> tuple[list[tuple[int, int, int]], int]:
    """Return ([(sq, color, type)], side_to_move)."""
    parts = fen.strip().split()
    board_field, stm_field = parts[0], parts[1]
    pieces: list[tuple[int, int, int]] = []
    rank, file = 7, 0
    for ch in board_field:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            pt = _PIECE_TYPE[ch.lower()]
            color = WHITE if ch.isupper() else BLACK
            pieces.append((rank * 8 + file, color, pt))
            file += 1
    stm = WHITE if stm_field == "w" else BLACK
    return pieces, stm


def features_for(pieces: list[tuple[int, int, int]], perspective: int) -> list[int]:
    king_sq = next(sq for sq, c, t in pieces if t == KING and c == perspective)
    orient = make_orient(perspective, king_sq)
    out = []
    for sq, color, pt in pieces:
        f = feature_index(orient, color, pt, sq)
        if f >= 0:
            out.append(f)
    return sorted(out)


def _main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        pieces, _stm = parse_fen_pieces(line)
        for tag, persp in (("W", WHITE), ("B", BLACK)):
            feats = features_for(pieces, persp)
            assert all(0 <= f < FEATURES for f in feats), "feature out of range"
            print(tag + "".join(f" {f}" for f in feats))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
