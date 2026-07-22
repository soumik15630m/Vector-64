#!/usr/bin/env python3
"""Self-play data generation for the RL loop.

The C++ engine plays itself at fixed nodes from seeded, material-balanced
openings. Each quiet position (not in check, past the opening book, with a
completed search score) is labelled with a WDL-blended target and written as
``<fen> | <cp>`` lines -- exactly what tools/nnue/build_stk_data.py consumes, so
the rest of the training pipeline is unchanged.

The blend lives in win-probability space (the same space make_net.py's
sigmoid(cp/400) loss uses):

    p = (1 - lambda) * sigmoid(eval/400) + lambda * wdl
    target_cp = 400 * logit(p)

where ``eval`` is the search score (white perspective) and ``wdl`` in {1,0.5,0}
is the game result from white's perspective. lambda=0 is pure self-distillation,
lambda=1 is pure outcome; 0.5 is a balanced default.

    python tools/nnue/datagen.py --engine <exe> --net <big.nnue> \
        --games 20000 --nodes 6000 --concurrency 10 --out data/gen0.txt
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import subprocess
import threading
import time

import chess

CP_SCALE = 400.0  # must match make_net.py's CP_SCALE
MATE_CP = 8000


def generate_openings(count: int, seed: int, plies: int = 8) -> list[str]:
    """Seeded random legal walks, filtered to quiet, material-balanced ends
    (identical policy to match.py so datagen and SPRT share opening variety)."""
    import random

    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
              chess.ROOK: 500, chess.QUEEN: 900}
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
    def __init__(self, binary: str, net: str, options: str | None = None):
        self.proc = subprocess.Popen(
            [os.path.abspath(binary)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._send("uci")
        self._wait("uciok")
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 64")
        self._send(f"setoption name EvalFile value {net}")
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
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith(token):
                return line.strip()

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._wait("readyok")

    def search(self, moves: list[str], nodes: int) -> tuple[str, int | None]:
        """Return (bestmove_uci, score_cp) where score is side-to-move relative
        from the last info line (None if the engine reported no score)."""
        assert self.proc.stdout is not None
        pos = "position startpos" + (" moves " + " ".join(moves) if moves else "")
        self._send(pos)
        self._send(f"go nodes {nodes}")
        score: int | None = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith("info ") and " score " in line:
                t = line.split()
                try:
                    i = t.index("score")
                    if t[i + 1] == "cp":
                        score = int(t[i + 2])
                    elif t[i + 1] == "mate":
                        score = MATE_CP if int(t[i + 2]) > 0 else -MATE_CP
                except (ValueError, IndexError):
                    pass
            elif line.startswith("bestmove"):
                return line.split()[1], score

    def quit(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def blend_cp(eval_white: int, wdl: float, lam: float) -> int:
    """Blend a white-perspective eval with the game result in win-prob space."""
    p_eval = 1.0 / (1.0 + math.exp(-max(-4000, min(4000, eval_white)) / CP_SCALE))
    p = (1.0 - lam) * p_eval + lam * wdl
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return int(round(CP_SCALE * math.log(p / (1.0 - p))))


def play_and_record(engine: Engine, opening: str, nodes: int, max_plies: int,
                    skip_plies: int) -> tuple[list[tuple[str, int]], float]:
    """Play one self-play game; return (list of (fen, eval_white), wdl_white)."""
    board = chess.Board()
    moves = opening.split()
    for m in moves:
        board.push_uci(m)
    engine.new_game()
    rec: list[tuple[str, int]] = []
    wdl = 0.5
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            r = outcome.result()
            wdl = 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5
            break
        if len(moves) >= max_plies:
            wdl = 0.5  # adjudicate the (already balanced) cut game as a draw
            break
        bm, score = engine.search(moves, nodes)
        if (len(moves) >= skip_plies and score is not None
                and not board.is_check()):
            eval_white = score if board.turn == chess.WHITE else -score
            rec.append((board.fen(), eval_white))
        try:
            board.push_uci(bm)
        except ValueError:
            wdl = 0.0 if board.turn == chess.WHITE else 1.0
            break
        moves.append(bm)
    return rec, wdl


def main() -> int:
    p = argparse.ArgumentParser(description="Self-play RL data generator.")
    p.add_argument("--engine", required=True)
    p.add_argument("--net", required=True)
    p.add_argument("--uci", default=None, help="extra options 'Name=Val;...'")
    p.add_argument("--games", type=int, default=20000)
    p.add_argument("--nodes", type=int, default=6000)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--skip-plies", type=int, default=12,
                   help="drop the first N plies (opening noise)")
    p.add_argument("--lam", type=float, default=0.5, help="WDL blend weight")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    openings = generate_openings(args.games, args.seed)
    work: queue.Queue[str] = queue.Queue()
    for o in openings:
        work.put(o)

    lock = threading.Lock()
    counters = {"games": 0, "positions": 0, "w": 0, "d": 0, "l": 0}
    fh = open(args.out, "w", encoding="utf-8")
    t0 = time.time()
    print(f"[datagen] {args.games} games @ {args.nodes} nodes, {args.concurrency} "
          f"workers, lambda={args.lam}, net={os.path.basename(args.net)}", flush=True)

    def worker() -> None:
        eng = Engine(args.engine, args.net, args.uci)
        buf: list[str] = []
        try:
            while True:
                try:
                    opening = work.get_nowait()
                except queue.Empty:
                    break
                rec, wdl = play_and_record(eng, opening, args.nodes,
                                           args.max_plies, args.skip_plies)
                for fen, ev in rec:
                    buf.append(f"{fen} | {blend_cp(ev, wdl, args.lam)}\n")
                if len(buf) >= 2000:
                    with lock:
                        fh.writelines(buf)
                        counters["positions"] += len(buf)
                    buf.clear()
                with lock:
                    counters["games"] += 1
                    counters["w" if wdl == 1.0 else "l" if wdl == 0.0 else "d"] += 1
                    g = counters["games"]
                    if g % 500 == 0:
                        el = time.time() - t0
                        rate = g / max(el, 1e-9)
                        eta = (args.games - g) / max(rate, 1e-9) / 60.0
                        print(f"  {g}/{args.games} games  "
                              f"{counters['positions'] + len(buf)} pos  "
                              f"{rate:.1f} g/s  eta {eta:.1f} min", flush=True)
        finally:
            if buf:
                with lock:
                    fh.writelines(buf)
                    counters["positions"] += len(buf)
            eng.quit()

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fh.close()
    g = max(counters["games"], 1)
    el = time.time() - t0
    print(f"DATAGEN DONE  {counters['games']} games  {counters['positions']} "
          f"positions  ({counters['positions']/g:.1f} pos/game)  {el/60:.1f} min "
          f"({g/max(el,1e-9):.1f} g/s)")
    print(f"  result (white pov): {100*counters['w']/g:.0f}% W  "
          f"{100*counters['d']/g:.0f}% D  {100*counters['l']/g:.0f}% L  ->  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
