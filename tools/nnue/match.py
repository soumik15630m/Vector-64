#!/usr/bin/env python3
"""Fixed-node engine-vs-engine testing: fixed matches and SPRT.

    # fixed-length match (Elo estimate):
    python tools/nnue/match.py --engine <exe> --net <big.nnue> --games 200

    # SPRT (the gate for search/eval changes):
    python tools/nnue/match.py --engine <exe> --net <big.nnue> \
        --base-engine <base-exe> --sprt 0 5 --concurrency 6

Both sides play color-swapped pairs on generated openings (seeded random
walks, material-balanced, unlimited variety -- essential because fixed-node
games are deterministic, so every (opening, color) yields exactly one
outcome and repeats add no information). Games run on parallel workers;
fixed nodes make results load-independent. Adjudication via python-chess.

SPRT uses the trinomial GSPRT approximation with H0: elo <= elo0 and
H1: elo >= elo1, alpha = beta = 0.05; the run stops when the LLR crosses
+/- ln((1-beta)/alpha) ~= 2.944.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import random
import subprocess
import sys
import threading
from dataclasses import dataclass

import chess

LLR_BOUND = math.log((1 - 0.05) / 0.05)  # ~2.944 for alpha = beta = 0.05


def generate_openings(count: int, seed: int, plies: int = 8) -> list[str]:
    """Seeded random legal walks, filtered to quiet, material-balanced ends."""
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < count:
        board = chess.Board()
        moves: list[str] = []
        ok = True
        for _ in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                ok = False
                break
            mv = legal[rng.randrange(len(legal))]
            moves.append(mv.uci())
            board.push(mv)
        if not ok or board.is_game_over() or board.is_check():
            continue
        imbalance = 0
        for pt, val in values.items():
            imbalance += val * (len(board.pieces(pt, chess.WHITE)) -
                                len(board.pieces(pt, chess.BLACK)))
        if abs(imbalance) > 150:
            continue
        out.append(" ".join(moves))
    return out


class Engine:
    def __init__(self, binary: str, net: str | None, small: str | None = None,
                 options: str | None = None):
        self.proc = subprocess.Popen(
            [os.path.abspath(binary)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._send("uci")
        self._wait("uciok")
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 64")
        if net:
            self._send(f"setoption name EvalFile value {net}")
        if small:
            self._send(f"setoption name EvalFileSmall value {small}")
        for opt in (options or "").split(";"):
            if "=" in opt:
                name, value = opt.split("=", 1)
                self._send(f"setoption name {name.strip()} value {value.strip()}")
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
        return self._wait("bestmove").split()[1]

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

    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(self.n, 1)

    def elo(self) -> tuple[float, float]:
        """Elo difference and 95% half-interval from the score fraction."""
        n = max(self.n, 1)
        p = min(max(self.score(), 1e-6), 1 - 1e-6)
        elo = -400.0 * math.log10(1.0 / p - 1.0)
        var = (self.wins * (1 - p) ** 2 + self.draws * (0.5 - p) ** 2 +
               self.losses * p**2) / n
        se = math.sqrt(var / n)
        lo, hi = p - 1.96 * se, p + 1.96 * se
        half = 0.0
        if 0 < lo and hi < 1:
            e_lo = -400.0 * math.log10(1.0 / lo - 1.0)
            e_hi = -400.0 * math.log10(1.0 / hi - 1.0)
            half = (e_hi - e_lo) / 2.0
        return elo, half

    def llr(self, elo0: float, elo1: float) -> float:
        """Trinomial GSPRT approximation of the log-likelihood ratio."""
        n = self.n
        if n == 0 or self.wins == n or self.losses == n or self.draws == n:
            return 0.0
        s = self.score()
        m2 = (self.wins + 0.25 * self.draws) / n
        var = m2 - s * s
        if var <= 0:
            return 0.0
        s0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
        s1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
        return (s1 - s0) * (2 * s - s0 - s1) / (2 * var / n)


def play_game(white: Engine, black: Engine, opening: str, nodes: int,
              max_plies: int) -> str:
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
            return "0-1" if board.turn == chess.WHITE else "1-0"
        moves.append(mv)


def main() -> int:
    p = argparse.ArgumentParser(description="Fixed-node match / SPRT runner.")
    p.add_argument("--engine", required=True, help="binary for the 'new' side")
    p.add_argument("--base-engine", default=None,
                   help="binary for the base side (default: same as --engine)")
    p.add_argument("--net", default=None, help="EvalFile for the 'new' side")
    p.add_argument("--net-small", default=None,
                   help="EvalFileSmall for the 'new' side (dual-net)")
    p.add_argument("--base-net", default=None,
                   help="EvalFile for the base side (default: classical)")
    p.add_argument("--base-net-small", default=None)
    p.add_argument("--uci-new", default=None,
                   help='extra options for the new side, "Name=Val;Name=Val"')
    p.add_argument("--uci-base", default=None,
                   help="extra options for the base side")
    p.add_argument("--games", type=int, default=200,
                   help="game cap (SPRT may stop earlier)")
    p.add_argument("--nodes", type=int, default=8000)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--seed", type=int, default=2024,
                   help="opening-book seed (fixed seed = reproducible run)")
    p.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"),
                   default=None, help="run as SPRT with H0 elo<=ELO0, H1 elo>=ELO1")
    args = p.parse_args()

    pair_count = (args.games + 1) // 2
    openings = generate_openings(pair_count, args.seed)
    jobs: queue.Queue[str | None] = queue.Queue()
    for op in openings:
        jobs.put(op)
    for _ in range(args.concurrency):
        jobs.put(None)

    tally = Tally()
    lock = threading.Lock()
    stop = threading.Event()

    def report(final: bool = False) -> None:
        elo, half = tally.elo()
        line = (f"games {tally.n:4d}  +{tally.wins} ={tally.draws} "
                f"-{tally.losses}  elo {elo:+7.1f} +/- {half:5.1f}")
        if args.sprt:
            line += (f"  LLR {tally.llr(*args.sprt):+6.2f} "
                     f"[{-LLR_BOUND:.2f}, {LLR_BOUND:.2f}]")
        print(("final: " if final else "") + line, flush=True)

    def worker() -> None:
        new_eng = Engine(args.engine, args.net, args.net_small, args.uci_new)
        base_eng = Engine(args.base_engine or args.engine, args.base_net,
                          args.base_net_small, args.uci_base)
        try:
            while not stop.is_set():
                op = jobs.get()
                if op is None:
                    break
                for new_is_white in (True, False):
                    if stop.is_set():
                        break
                    white, black = ((new_eng, base_eng) if new_is_white
                                    else (base_eng, new_eng))
                    result = play_game(white, black, op, args.nodes,
                                       args.max_plies)
                    with lock:
                        if result == "1/2-1/2":
                            tally.draws += 1
                        elif (result == "1-0") == new_is_white:
                            tally.wins += 1
                        else:
                            tally.losses += 1
                        if tally.n % 10 == 0:
                            report()
                        if args.sprt and tally.n >= 20 and not stop.is_set():
                            llr = tally.llr(*args.sprt)
                            if llr >= LLR_BOUND:
                                print("SPRT: H1 accepted (change is a gain)",
                                      flush=True)
                                stop.set()
                            elif llr <= -LLR_BOUND:
                                print("SPRT: H0 accepted (no gain / a loss)",
                                      flush=True)
                                stop.set()
        finally:
            new_eng.quit()
            base_eng.quit()

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    report(final=True)
    if args.sprt:
        llr = tally.llr(*args.sprt)
        if llr >= LLR_BOUND:
            return 0
        if llr <= -LLR_BOUND:
            return 1
        print("SPRT: inconclusive at game cap", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
