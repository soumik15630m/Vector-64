#!/usr/bin/env python3
"""RL self-play loop for the STK-HalfKA net.

One generation:
  1. datagen.py   : the current best net plays itself; positions get a
                    WDL-blended label ("<fen> | <cp>").
  2. make_net.py  : fine-tune (warm-start --init from the best net's
                    model_float.pt) on that data -> a new .nnue + model_float.pt.
  3. match.py     : SPRT the new net vs the best net at fixed nodes.
  4. gate         : accept the new net iff SPRT accepts H1 (a real gain);
                    otherwise keep the best and generate fresh data next round.

State lives in <root>/state.json, so the loop is resumable (Ctrl-C safe): it
never regresses because every generation is SPRT-gated against the current best.

    python tools/nnue/rl_loop.py --engine build-bench/bin/ChessEngine.exe \
        --init-net runs/v2/stk_halfka_1024.nnue \
        --init-float runs/v2/model_float.pt --root runs/rl --generations 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str], log, allow_fail: bool = False) -> str:
    log.write("\n$ " + " ".join(cmd) + "\n")
    log.flush()
    out_lines: list[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        out_lines.append(line)
    proc.wait()
    # match.py exits non-zero when SPRT rejects/is inconclusive -- that is a
    # verdict, not a crash, so the caller opts into tolerating it.
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return "".join(out_lines)


def main() -> int:
    p = argparse.ArgumentParser(description="RL self-play loop.")
    p.add_argument("--engine", required=True)
    p.add_argument("--init-net", required=True, help="starting best .nnue")
    p.add_argument("--init-float", required=True, help="starting best model_float.pt")
    p.add_argument("--root", default="runs/rl")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--games", type=int, default=20000, help="self-play games/gen")
    p.add_argument("--dg-nodes", type=int, default=6000, help="nodes/move in datagen")
    p.add_argument("--lam", type=float, default=0.5, help="WDL blend weight")
    p.add_argument("--epochs", type=int, default=3, help="fine-tune epochs/gen")
    p.add_argument("--sprt-nodes", type=int, default=8000)
    p.add_argument("--sprt-games", type=int, default=4000)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--device", default=None, help="passed to make_net.py --device")
    args = p.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"gen": 0, "best_net": str(Path(args.init_net).resolve()),
                 "best_float": str(Path(args.init_float).resolve()),
                 "history": []}
    log = open(root / "rl.log", "a", encoding="utf-8")

    py = sys.executable
    while state["gen"] < args.generations:
        gen = state["gen"]
        gdir = root / f"gen{gen:03d}"
        gdir.mkdir(exist_ok=True)
        data = gdir / "data.txt"
        n_acc = sum(1 for h in state["history"] if h["accepted"])
        print(f"\n{'='*66}\n  GENERATION {gen}   best={Path(state['best_net']).name}"
              f"   accepted so far: {n_acc}/{len(state['history'])}\n{'='*66}")

        # 1) self-play data from the current best net
        t = time.time()
        if not data.exists() or data.stat().st_size == 0:
            run([py, str(HERE / "datagen.py"), "--engine", args.engine,
                 "--net", state["best_net"], "--games", str(args.games),
                 "--nodes", str(args.dg_nodes), "--lam", str(args.lam),
                 "--concurrency", str(args.concurrency), "--seed", str(1000 + gen),
                 "--out", str(data)], log)
        t_dg = (time.time() - t) / 60

        # 2) fine-tune (warm-start from the best net)
        t = time.time()
        new_net = gdir / "stk_halfka_1024.nnue"
        if not new_net.exists():
            cmd = [py, str(HERE / "make_net.py"), "--input", str(data),
                   "--workdir", str(gdir), "--init", state["best_float"],
                   "--epochs", str(args.epochs), "--engine", args.engine]
            if args.device:
                cmd += ["--device", args.device]
            run(cmd, log)
        t_tr = (time.time() - t) / 60

        # 3) SPRT the new net vs the current best (different seed than datagen)
        t = time.time()
        out = run([py, str(HERE / "match.py"), "--engine", args.engine,
                   "--base-engine", args.engine, "--net", str(new_net),
                   "--base-net", state["best_net"], "--sprt", "0", "5",
                   "--games", str(args.sprt_games), "--nodes", str(args.sprt_nodes),
                   "--concurrency", str(args.concurrency),
                   "--seed", str(90000 + gen)], log, allow_fail=True)
        t_sprt = (time.time() - t) / 60

        accepted = "H1 accepted" in out
        elo = "?"
        for ln in reversed(out.splitlines()):
            if "elo" in ln and "+/-" in ln:
                try:
                    elo = ln.split("elo")[1].split("+/-")[0].strip()
                except IndexError:
                    pass
                break
        entry = {"gen": gen, "accepted": accepted, "elo": elo,
                 "net": str(new_net.resolve()),
                 "min": {"datagen": round(t_dg, 1), "train": round(t_tr, 1),
                         "sprt": round(t_sprt, 1)}}
        if accepted:
            state["best_net"] = str(new_net.resolve())
            state["best_float"] = str((gdir / "model_float.pt").resolve())
        state["history"].append(entry)
        state["gen"] = gen + 1
        state_path.write_text(json.dumps(state, indent=2))

        verdict = "ACCEPTED -> new best" if accepted else "rejected (best unchanged)"
        print(f"\n  gen {gen} {verdict}   elo {elo}   "
              f"[datagen {t_dg:.1f}m  train {t_tr:.1f}m  sprt {t_sprt:.1f}m]")

    print(f"\nRL loop complete. Accepted {sum(1 for h in state['history'] if h['accepted'])}"
          f"/{len(state['history'])} generations. Best net: {state['best_net']}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
