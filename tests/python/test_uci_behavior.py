"""Black-box UCI behaviour tests — drive the built engine over stdin/stdout and
assert protocol and search behaviour. Each test has a single, specific goal.

The engine binary is located via the ENGINE_BIN environment variable, falling
back to a search of the build directories. When no binary is found (e.g. the
lint-only CI job) every test skips rather than failing.
"""
from __future__ import annotations

import os
import pathlib
import queue
import subprocess
import threading
import time

import pytest


def _find_engine() -> str | None:
    env = os.environ.get("ENGINE_BIN")
    if env and pathlib.Path(env).exists():
        return env
    root = pathlib.Path(__file__).resolve().parents[2]
    for pattern in (
        "build*/bin/ChessEngine",
        "build*/bin/ChessEngine.exe",
        "build*/bin/Release/ChessEngine.exe",
    ):
        hits = sorted(root.glob(pattern))
        if hits:
            return str(hits[0])
    return None


ENGINE = _find_engine()


class Uci:
    def __init__(self, path: str) -> None:
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\r\n"))

    def send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def collect_until(self, predicate, timeout: float = 15.0):
        deadline = time.time() + timeout
        seen: list[str] = []
        while time.time() < deadline:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                break
            seen.append(line)
            if predicate(line):
                return line, seen
        return None, seen

    def bestmove(self, timeout: float = 15.0) -> tuple[str, list[str]]:
        line, seen = self.collect_until(lambda ln: ln.startswith("bestmove"), timeout)
        assert line is not None, f"no bestmove; saw: {seen[-5:]}"
        return line.split()[1], seen

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


@pytest.fixture()
def uci():
    if not ENGINE:
        pytest.skip("engine binary not found; set ENGINE_BIN")
    engine = Uci(ENGINE)
    yield engine
    engine.close()


def test_uci_handshake(uci) -> None:
    uci.send("uci")
    line, _ = uci.collect_until(lambda ln: ln == "uciok", timeout=5)
    assert line == "uciok"


def test_isready(uci) -> None:
    uci.send("isready")
    line, _ = uci.collect_until(lambda ln: ln == "readyok", timeout=5)
    assert line == "readyok"


def test_startpos_returns_a_move(uci) -> None:
    uci.send("position startpos")
    uci.send("go depth 6")
    move, _ = uci.bestmove()
    assert len(move) in (4, 5) and move != "0000"


def test_mate_in_one(uci) -> None:
    uci.send("position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    uci.send("go depth 6")
    move, seen = uci.bestmove()
    assert move == "a1a8"
    assert any("score mate 1" in ln for ln in seen)


def test_stalemate_returns_null(uci) -> None:
    uci.send("position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    uci.send("go depth 4")
    move, _ = uci.bestmove()
    assert move == "0000"


def test_checkmate_returns_null(uci) -> None:
    uci.send("position fen R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
    uci.send("go depth 4")
    move, _ = uci.bestmove()
    assert move == "0000"


def test_illegal_fen_rejected(uci) -> None:
    uci.send("position fen 8/R6k/1R6/8/8/8/8/4K3 w - - 0 1")
    uci.send("isready")
    line, seen = uci.collect_until(lambda ln: ln == "readyok", timeout=5)
    assert any("illegal" in ln.lower() for ln in seen)


def test_promotion_move(uci) -> None:
    uci.send("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    uci.send("go depth 8")
    move, _ = uci.bestmove()
    assert move == "a7a8q"


def test_searchmoves_restricts_root(uci) -> None:
    uci.send("position startpos")
    uci.send("go depth 6 searchmoves e2e4")
    move, _ = uci.bestmove()
    assert move == "e2e4"


def test_nodes_limit_is_respected(uci) -> None:
    # Pin to one thread: under Lazy SMP the node budget is per-thread, so the
    # aggregate would legitimately be ~threads x budget.
    uci.send("setoption name Threads value 1")
    uci.send("position startpos")
    uci.send("go nodes 50000")
    move, seen = uci.bestmove()
    assert move != "0000"
    node_counts = [int(ln.split("nodes")[1].split()[0]) for ln in seen if " nodes " in ln]
    # Stops within one 2048-node batch of the budget.
    assert node_counts and max(node_counts) < 60000


def _bench_signature() -> tuple[int, int]:
    """The (depth, nodes) fingerprint from the single source of truth."""
    path = pathlib.Path(__file__).resolve().parents[2] / "tools" / "bench_signature.txt"
    vals: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            vals[key.strip()] = value.strip()
    return int(vals["DEPTH"]), int(vals["NODES"])


def test_bench_signature_is_deterministic(uci) -> None:
    depth, expected = _bench_signature()
    uci.send("setoption name Threads value 1")
    uci.send("setoption name Hash value 8")
    uci.send(f"bench {depth}")
    line, _ = uci.collect_until(lambda ln: f"bench depth {depth}" in ln, timeout=30)
    assert line is not None
    nodes = int(line.split("nodes")[1].split()[0])
    assert nodes == expected


def test_ucinewgame_then_search(uci) -> None:
    uci.send("ucinewgame")
    uci.send("isready")
    line, _ = uci.collect_until(lambda ln: ln == "readyok", timeout=5)
    assert line == "readyok"
    uci.send("position startpos moves e2e4 e7e5")
    uci.send("go depth 5")
    move, _ = uci.bestmove()
    assert len(move) in (4, 5) and move != "0000"
