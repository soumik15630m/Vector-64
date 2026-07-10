#!/usr/bin/env python3
"""Fixed-node engine-vs-engine match: measure a net against the classical eval.

    python tools/nnue/match.py --engine build-bench/bin/ChessEngine.exe \
        --net runs/v1/stk_halfka_1024.nnue --games 100 --nodes 10000

Plays color-swapped pairs from a built-in set of diverse openings, adjudicates
with python-chess (mate, stalemate, repetition, 50-move, insufficient
material), and reports W/D/L plus an Elo difference with a 95% interval.
Engines run single-threaded at fixed nodes, so results are hardware- and
load-independent (safe to run while training occupies the GPU).
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass

import chess

OPENINGS = [
    # Short, balanced lines giving structural variety (mainline theory).
    "e2e4 e7e5 g1f3 b8c6 f1b5",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c4d5 e6d5",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4",
    "d2d4 g8f6 c2c4 e7e6 g1f3 d7d5 b1c3 c7c6",
    "g1f3 d7d5 c2c4 e7e6 d2d4 g8f6 b1c3 c7c5",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3",
    "c2c4 c7c5 g1f3 g8f6 d2d4 c5d4 f3d4 e7e6",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4 f8g7",
    "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6 g1f3 f8e7",
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6 f2f4 g8f6",
    "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5 c4b5 a7a6",
    "e2e4 e7e5 f1c4 g8f6 d2d3 c7c6 g1f3 d7d5",
    "d2d4 e7e6 c2c4 f8b4 b1d2 g8f6 g1f3 b7b6",
    "g1f3 g8f6 c2c4 b7b6 g2g3 c8b7 f1g2 e7e6",
    "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5 d2d4 c5d4 c3d4",
    "d2d4 d7d5 g1f3 g8f6 c2c4 e7e6 g2g3 f8e7",
    "e2e4 e7e5 b1c3 g8f6 g2g3 d7d5 e4d5 f6d5",
]


class Engine:
    def __init__(self, binary: str, net: str | None):
        self.proc = subprocess.Popen(
            [os.path.abspath(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._send("uci")
        self._wait("uciok")
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 64")
        if net:
            self._send(f"setoption name EvalFile value {net}")
        self._send("isready")
        self._wait("readyok")

    def _send(self, s: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _wait(self, token: str) -> str:
        assert self.proc.stdout is not None
        out: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died:\n" + "".join(out[-10:]))
            out.append(line)
            if line.startswith(token):
                return line.strip()

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._wait("readyok")

    def best_move(self, moves: list[str], nodes: int) -> str:
        pos = "position startpos" + (" moves " + " ".join(moves) if moves else "")
        self._send(pos)
        self._send(f"go nodes {nodes}")
        line = self._wait("bestmove")
        return line.split()[1]

    def quit(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


@dataclass
class Tally:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    def elo(self) -> tuple[float, float]:
        """Elo difference and 95% half-interval from the score fraction."""
        n = max(self.n, 1)
        p = (self.wins + 0.5 * self.draws) / n
        p = min(max(p, 1e-6), 1 - 1e-6)
        elo = -400.0 * math.log10(1.0 / p - 1.0)
        var = (self.wins * (1 - p) ** 2 + self.draws * (0.5 - p) ** 2 + self.losses * p**2) / n
        se = math.sqrt(var / n)
        lo, hi = p - 1.96 * se, p + 1.96 * se
        half = 0.0
        if 0 < lo and hi < 1:
            e_lo = -400.0 * math.log10(1.0 / lo - 1.0)
            e_hi = -400.0 * math.log10(1.0 / hi - 1.0)
            half = (e_hi - e_lo) / 2.0
        return elo, half


def play_game(white: Engine, black: Engine, opening: str, nodes: int, max_plies: int) -> str:
    """Returns '1-0', '0-1' or '1/2-1/2' (from White's perspective)."""
    board = chess.Board()
    moves = opening.split()
    for m in moves:
        board.push_uci(m)
    white.new_game()
    black.new_game()
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return outcome.result()
        if len(moves) >= max_plies:
            return "1/2-1/2"
        eng = white if board.turn == chess.WHITE else black
        mv = eng.best_move(moves, nodes)
        try:
            board.push_uci(mv)
        except ValueError:
            # Illegal move from an engine loses the game outright.
            return "0-1" if board.turn == chess.WHITE else "1-0"
        moves.append(mv)


def main() -> int:
    p = argparse.ArgumentParser(description="Fixed-node match: net vs classical.")
    p.add_argument("--engine", required=True)
    p.add_argument("--net", required=True, help="EvalFile for the 'new' side")
    p.add_argument("--base-net", default=None, help="EvalFile for the base side (default: classical)")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--nodes", type=int, default=10000)
    p.add_argument("--max-plies", type=int, default=400)
    args = p.parse_args()

    new_eng = Engine(args.engine, args.net)
    base_eng = Engine(args.engine, args.base_net)
    tally = Tally()
    try:
        for g in range(args.games):
            opening = OPENINGS[(g // 2) % len(OPENINGS)]
            new_is_white = g % 2 == 0
            white, black = (new_eng, base_eng) if new_is_white else (base_eng, new_eng)
            result = play_game(white, black, opening, args.nodes, args.max_plies)
            if result == "1/2-1/2":
                tally.draws += 1
            elif (result == "1-0") == new_is_white:
                tally.wins += 1
            else:
                tally.losses += 1
            elo, half = tally.elo()
            print(
                f"game {g + 1:3d}/{args.games}  {result:7s}  "
                f"+{tally.wins} ={tally.draws} -{tally.losses}   elo {elo:+.0f} +/- {half:.0f}",
                flush=True,
            )
    finally:
        new_eng.quit()
        base_eng.quit()

    elo, half = tally.elo()
    print(f"\nfinal: +{tally.wins} ={tally.draws} -{tally.losses}  ->  elo {elo:+.1f} +/- {half:.1f} (95%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
