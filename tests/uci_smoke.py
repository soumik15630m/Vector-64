#!/usr/bin/env python3
import os
import queue
import subprocess
import sys
import threading
import time


class UciSession:
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"engine not found: {engine_path}")
        self.proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader.start()

    def _reader_loop(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\r\n"))

    def send(self, cmd: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_until(self, predicate, timeout_s: float):
        deadline = time.time() + timeout_s
        seen = []
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            try:
                line = self.lines.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            seen.append(line)
            if predicate(line):
                return line, seen
        raise TimeoutError("timeout waiting for expected UCI output\n" + "\n".join(seen))

    def close(self):
        try:
            if self.proc.poll() is None:
                self.send("quit")
                self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()


def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def run_test(engine_path: str):
    s = UciSession(engine_path)
    try:
        s.send("uci")
        _, uci_lines = s.read_until(lambda ln: ln == "uciok", timeout_s=3.0)
        assert_true(any(ln.startswith("id name ") for ln in uci_lines), "missing id name")
        assert_true(any("option name Threads" in ln for ln in uci_lines), "missing Threads option")

        s.send("isready")
        s.read_until(lambda ln: ln == "readyok", timeout_s=2.0)

        s.send("position startpos")
        s.send("go movetime 200")
        best_line, _ = s.read_until(lambda ln: ln.startswith("bestmove "), timeout_s=4.0)
        best = best_line.split()[1]
        assert_true(len(best) >= 4, f"bad bestmove token: {best}")

        s.send("position startpos")
        s.send("go infinite")
        time.sleep(0.2)
        s.send("stop")
        s.read_until(lambda ln: ln.startswith("bestmove "), timeout_s=4.0)
    finally:
        s.close()


def main():
    if len(sys.argv) < 2:
        print("usage: uci_smoke.py <engine_path>")
        return 2
    try:
        run_test(sys.argv[1])
        print("uci smoke test: PASS")
        return 0
    except Exception as e:
        print(f"uci smoke test: FAIL: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
