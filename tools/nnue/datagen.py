#!/usr/bin/env python3
"""DEPRECATED -- superseded by the engine's native data generator.

Use one of these instead; both write byte-identical rows because they share the
helpers in src/datagen/selfplay.h:

    ChessEngine datagen --net <net> --out <file> --games N --nodes N --threads N
    ChessEngine-viz             # datagen mode: live games, crash-safe resume

Driving the engine over UCI from Python is slower, and this path has drifted
from the native one (it predates the native generator, the warm-history default
and the current label handling). Kept so older commands in notes still run; not
maintained, and not for new datasets.

Original documentation follows.

Self-play data generation for the RL loop.

The C++ engine plays itself at fixed nodes from seeded, material-balanced
openings. Each quiet position (not in check, past the opening book, with a
completed search score) is recorded. Two output formats (``--emit``):

  blend : ``<fen> | <cp>``            WDL baked in via --lam; make_net.py input
  raw   : ``<fen> | <eval> | <wdl>``  white-relative, unblended; bullet's native
                                      text ingest (bullet does its own blend)

``blend`` feeds the PyTorch trainer (build_stk_data.py), so that pipeline is
unchanged; ``raw`` feeds bullet, which prefers to blend eval and game result at
train time. build_stk_data.py also reads the raw 3-field form (blending it) so
either dataset can train the PyTorch net.

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
from collections.abc import Hashable

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
    seen: dict[Hashable, int] = {}  # transposition-key counts for cheap threefold
    while True:
        # Cheap terminal detection. board.outcome(claim_draw=True) re-scanned the
        # whole move stack for threefold/fifty-move CLAIMS every ply (~11% of
        # datagen CPU); track repetitions incrementally instead. Draw thresholds
        # (threefold, 50-move, insufficient material) match the claim_draw
        # outcome; games end at most ~1 ply later (actual vs claimable rep).
        key = board._transposition_key()
        seen[key] = seen.get(key, 0) + 1
        if (seen[key] >= 3 or board.halfmove_clock >= 100
                or len(moves) >= max_plies or board.is_insufficient_material()):
            wdl = 0.5  # draw (rep / 50-move / material / balanced cut)
            break
        if not any(board.legal_moves):  # checkmate or stalemate
            wdl = (0.0 if board.turn == chess.WHITE else 1.0) \
                if board.is_check() else 0.5
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
    p.add_argument("--lam", type=float, default=0.5,
                   help="WDL blend weight (only used by --emit blend)")
    p.add_argument("--emit", choices=("blend", "raw"), default="blend",
                   help="blend: '<fen> | <cp>' with WDL baked in via --lam "
                        "(make_net input); raw: '<fen> | <eval> | <wdl>' "
                        "white-relative, unblended (bullet's native ingest format)")
    p.add_argument("--log-interval", type=float, default=30.0,
                   help="seconds between progress heartbeat lines")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    openings = generate_openings(args.games, args.seed)
    work: queue.Queue[str] = queue.Queue()
    for o in openings:
        work.put(o)

    lock = threading.Lock()
    counters = {"games": 0, "positions": 0, "w": 0, "d": 0, "l": 0}
    prog = {"last": 0.0}
    fh = open(args.out, "w", encoding="utf-8")
    t0 = time.time()
    prog["last"] = t0
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
                    if args.emit == "raw":
                        buf.append(f"{fen} | {ev} | {wdl:.1f}\n")
                    else:
                        buf.append(f"{fen} | {blend_cp(ev, wdl, args.lam)}\n")
                if len(buf) >= 2000:
                    with lock:
                        fh.writelines(buf)
                    buf.clear()
                with lock:
                    counters["games"] += 1
                    counters["positions"] += len(rec)
                    counters["w" if wdl == 1.0 else "l" if wdl == 0.0 else "d"] += 1
                    now = time.time()
                    if now - prog["last"] >= args.log_interval:
                        prog["last"] = now
                        g = counters["games"]
                        el = max(now - t0, 1e-9)
                        gps = g / el
                        pos = counters["positions"]
                        eta = (args.games - g) / max(gps, 1e-9) / 60.0
                        print(f"  {g}/{args.games} ({100.0 * g / args.games:.0f}%)  "
                              f"{pos:,} pos  {gps:.1f} g/s  {pos / el:.0f} pos/s  "
                              f"W/D/L {counters['w']}/{counters['d']}/{counters['l']}"
                              f"  eta {eta:.1f}m", flush=True)
        finally:
            if buf:
                with lock:
                    fh.writelines(buf)
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
