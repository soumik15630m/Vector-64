#!/usr/bin/env bash
# End-to-end smoke test for the NNUE visualizer.
#
# The important assertion is the last one: the eval the visualizer streams must
# equal what the engine's own `eval` command reports for the same position. That
# is what makes "the UI shows the real engine" a checked property rather than an
# intention. The rest covers the wire format, control surface and the loopback /
# no-filesystem security posture.
#
#   bash tools/viz_smoke.sh <path-to-ChessEngine-viz> [<path-to-ChessEngine-nnue>]
set -euo pipefail

VIZ="${1:?usage: viz_smoke.sh <ChessEngine-viz> [ChessEngine-nnue]}"
ENGINE="${2:-}"
PORT="${VIZ_SMOKE_PORT:-7911}"
BASE="http://127.0.0.1:${PORT}"
TMP="$(mktemp -d)"
PID=""

cleanup() {
  [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

echo "[viz-smoke] starting $VIZ on :$PORT"
"$VIZ" --port "$PORT" --headless --nodes 4000 --delay 0 >"$TMP/server.log" 2>&1 &
PID=$!

# Wait for the port to answer rather than sleeping a fixed amount.
for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 "$BASE/api/health" >/dev/null 2>&1; then break; fi
  sleep 0.5
done
curl -fsS --max-time 5 "$BASE/api/health" | grep -q '"ok":true'
echo "[viz-smoke] health ok"

# --- wire format ----------------------------------------------------------
curl -fsS --max-time 15 "$BASE/api/state?since=0" -o "$TMP/state.bin"
python3 - "$TMP/state.bin" <<'PY'
import json, struct, sys
b = open(sys.argv[1], 'rb').read()
n = struct.unpack('<I', b[:4])[0]
h = json.loads(b[4:4+n])
f = h["frame"]
size = {'i16': 2, 'u8': 1, 'i32': 4}
total = sum(x["len"] * size[x["type"]] for x in f["buffers"])
assert total == len(b) - 4 - n, f"payload {total} != {len(b)-4-n}"
a = h["arch"]
assert a["hidden"] == 1024 and a["l1"] == 16 and a["l2"] == 32, a
assert a["features"] == 22528 and a["kingBuckets"] == 32, a
if h["nnueActive"]:
    assert f["eval"] == f["psqt"] + f["positional"], "eval != psqt + positional"
    names = {x["name"] for x in f["buffers"]}
    for want in ("accUs", "accThem", "l1in", "l1out", "l2out", "outContrib",
                 "l2Contrib", "l1Top", "whiteFeatures", "blackFeatures"):
        assert want in names, f"missing buffer {want}"
print(f"[viz-smoke] wire ok (header {n} B, payload {total} B, seq {h['seq']})")
open(sys.argv[1] + '.fen', 'w').write(f["fen"])
open(sys.argv[1] + '.eval', 'w').write(str(f["eval"]))
open(sys.argv[1] + '.nnue', 'w').write('1' if h["nnueActive"] else '0')
PY

# --- control surface ------------------------------------------------------
curl -fsS --max-time 5 -X POST -d '{"cmd":"pause","value":true}' "$BASE/api/control" | grep -q '"ok":true'
curl -fsS --max-time 5 -X POST -d '{"cmd":"nodes","value":5000}' "$BASE/api/control" | grep -q '"ok":true'
code=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' -X POST -d '{"cmd":"bogus"}' "$BASE/api/control")
[ "$code" = "400" ] || { echo "unknown command should be 400, got $code"; exit 1; }
code=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' -X POST -d '{"cmd":"move","value":"a1a8"}' "$BASE/api/control")
[ "$code" = "400" ] || { echo "illegal move should be 400, got $code"; exit 1; }
code=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' -X POST -d 'not json' "$BASE/api/control")
[ "$code" = "400" ] || { echo "malformed body should be 400, got $code"; exit 1; }
echo "[viz-smoke] control ok (valid accepted, invalid rejected)"

# --- only embedded assets are served --------------------------------------
code=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "$BASE/../../etc/passwd")
[ "$code" = "404" ] || { echo "path traversal should 404, got $code"; exit 1; }
code=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "$BASE/etc/passwd")
[ "$code" = "404" ] || { echo "arbitrary path should 404, got $code"; exit 1; }
echo "[viz-smoke] serves embedded assets only"

# --- net inspector --------------------------------------------------------
curl -fsS --max-time 60 "$BASE/api/net" -o "$TMP/net.json"
python3 - "$TMP/net.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert len(d["kingBucketMap"]) == 64, "king bucket map must cover 64 squares"
assert d["arch"]["kingBuckets"] == 32
if d["loaded"]:
    assert len(d["buckets"]) == 8, "expected 8 PSQT buckets"
    ft = d["ftWeights"]
    assert sum(ft["bins"]) == ft["count"] == 22528 * 1024, "histogram must cover every FT weight"
print("[viz-smoke] net inspector ok")
PY

# --- the truth check ------------------------------------------------------
if [ -n "$ENGINE" ] && [ "$(cat "$TMP/state.bin.nnue")" = "1" ]; then
  fen="$(cat "$TMP/state.bin.fen")"
  want="$(cat "$TMP/state.bin.eval")"
  got=$(printf 'position fen %s\neval\nquit\n' "$fen" \
        | "$ENGINE" 2>/dev/null | grep -oE '\-?[0-9]+ cp' | grep -oE '\-?[0-9]+' | head -1)
  if [ "$got" != "$want" ]; then
    echo "MISMATCH: visualizer says $want cp, engine says ${got:-<none>} cp"
    echo "  fen: $fen"
    exit 1
  fi
  echo "[viz-smoke] eval matches the engine exactly ($want cp) -- real engine, not a demo"
else
  echo "[viz-smoke] skipping eval cross-check (no engine given or no net loaded)"
fi

echo "[viz-smoke] PASS"
