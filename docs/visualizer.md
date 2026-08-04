# Vector Scope — the NNUE live visualizer

Vector Scope renders what STK-Vector-64's neural network is actually computing
while the engine plays. It ships as part of the engine, as its own binary.

**It shows the real engine.** The full H=1024 net, the real search, exact
values. There is no reduced network, no demo mode and no illustrative
approximation: the eval it displays is checked in CI to equal what the engine's
own `eval` command reports for the same position, and the per-layer
contributions it draws are the exact integers that layer summed.

```bash
./ChessEngine-viz                 # opens http://127.0.0.1:7777
./ChessEngine-viz --port 8080 --nodes 50000 --threads 4
./ChessEngine-viz --headless      # serve without opening a browser
```

## Modes

| Mode | What it does |
|---|---|
| **self-play** | The engine plays itself continuously from random balanced openings (the same opening generator the data pipeline uses). |
| **analysis** | Paste a FEN; the engine thinks about that position and the network view follows it. |
| **play** | You play one side, the engine the other. Legal moves come from the engine, so the board can only offer moves it accepts. |
| **net inspector** | The net itself, independent of any game: the feature-transformer weight histogram, per-bucket dense weight statistics, and the king-bucket map. |

## Reading the neuron field

Left to right, the field is the forward pass:

```
active features → accumulator (2 × 1024 int16) → pairwise clipped ReLU (1024)
                → L1 (16) → L2 (32) → eval
```

- **Cell colour** is a real activation. Cool is negative, warm is positive; the
  pairwise and dense layers are clipped to `[0, 127]` so they use an intensity
  ramp instead.
- **Edges are attribution, not topology.** Every line is drawn from the
  engine's own `weight × activation`, and its width and opacity scale with
  `|weight × activation|`. A thick bright edge is a path that actually moved
  the evaluation. The L1 layer has 1024 inputs per neuron, far too many to
  draw, so only each neuron's strongest inputs are shown.
- **Accumulator contrast** is scaled by mean magnitude rather than the maximum,
  because a single outlier would otherwise wash the whole block out. The
  mapping stays monotonic — only the contrast is chosen, never the meaning.
- **While the engine is thinking**, the field shows the *leaf of the principal
  variation*: the position the engine is actually weighing at that depth, not
  the root.

## Architecture

Four layers. The first has no I/O and the last has no engine knowledge.

```
src/viz/probe.{h,cpp}     telemetry extraction (pure data, no I/O)
src/viz/session.{h,cpp}   drives the engine in one of the modes, publishes snapshots
src/viz/server.{h,cpp}    HTTP transport (cpp-httplib)
ui/                       React + TypeScript + PixiJS
```

**`chess_core` gains no dependencies.** cpp-httplib and nlohmann/json are
attached only to the visualizer targets; CI checks that `ChessEngine` and
`ChessEngine-nnue` link neither them nor `chess_viz`. The engine you run in a
tournament is exactly as lean as it was before this existed.

**The probe never runs in the search hot path.** `Viz::capture()` rebuilds its
own accumulator and runs its own forward pass, out of band. The bench
signatures (`5253789` classical, `2926142` NNUE) are unchanged and CI enforces
that.

### Wire format

One framed binary message per state update:

```
[uint32 LE headerLen][headerLen bytes of UTF-8 JSON][raw little-endian buffers]
```

The JSON header carries the game state, search telemetry, the architecture
constants (so the UI never hard-codes the network shape) and a table naming
each raw buffer, its element type and length, in payload order. Sending the
bulk arrays as binary keeps a frame around 9 KB instead of ~25 KB and hands the
client typed arrays with no parsing.

`GET /api/state?since=N` is a **long poll**: it blocks until something newer
than `N` exists. The client requests the next frame only after rendering the
last, so it applies natural backpressure and can never fall behind a queue.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | the embedded single-file UI |
| `GET /api/state?since=N` | long-poll; framed binary state |
| `GET /api/net` | net inspector data (JSON) |
| `POST /api/control` | `{"cmd":"pause"\|"step"\|"newgame"\|"mode"\|"nodes"\|"delay"\|"position"\|"move"\|"enginecolor", ...}` |
| `GET /api/health` | liveness |

### Security

The visualizer is **unauthenticated and exposes engine control**, so it binds
`127.0.0.1` and refuses to bind anything else. It serves only assets embedded
in the binary — there is no filesystem path to traverse — and caps request size
and timeouts. Treat it as a local developer tool.

## Building

The UI is built to a single self-contained `ui/dist/index.html` and **committed**,
then embedded into the binary with the same `.incbin` / RCDATA mechanism used
for the net. A C++ build therefore never needs Node.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ChessEngine-viz -j
```

Only when changing the UI:

```bash
cd ui && npm install && npm run build     # regenerates ui/dist/index.html
```

CMake declares the bundle and the net as `OBJECT_DEPENDS` of the embed objects,
so changing either really does rebuild the binary. (Without that, `npm run
build` silently appears to do nothing.)

Disable the visualizer entirely with `-DENGINE_VIZ=OFF` (also skips the
dependency fetch, useful offline).

## Browser build (WebAssembly)

The same engine and the same 46 MB net, compiled to WebAssembly and running in
the tab — not a cut-down demo. The net is downloaded once and kept in the Cache
API afterwards.

```bash
source /path/to/emsdk/emsdk_env.sh
bash tools/build_wasm.sh          # -> ui/dist-wasm/
python -m http.server -d ui/dist-wasm 8080
```

`ui/public/coi-serviceworker.js` (MIT) installs a service worker that supplies
the `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers a
static host like GitHub Pages cannot send. That unlocks `SharedArrayBuffer`,
which is what Emscripten pthreads need, so the browser build gets **real
multi-threaded search**. Where the shim cannot register, it falls back to a
single thread and the UI reports the true thread count rather than hiding it.

The UI picks its transport at runtime: if a local `ChessEngine-viz` answers
`/api/health` it uses that, otherwise it loads the WebAssembly engine. Both
implement the same `EngineSource` interface, and the WASM build reuses the same
C++ encoder, so the frames are byte-identical either way.

> **Note on SIMD.** The WebAssembly build uses the portable scalar NNUE kernels
> (compiled with `-msimd128`, so the compiler may still vectorise them). No
> hand-written `wasm_simd128` kernels have been added: the project's rule is
> that any SIMD path must first be proven bit-exact against the scalar
> reference by `test_zobrist`, and that gate needs to run under node in CI
> before such a path can be trusted.

## Tests

```bash
ctest --test-dir build -R viz            # telemetry + session driver
bash tools/viz_smoke.sh build/bin/ChessEngine-viz build/bin/ChessEngine-nnue
```

`viz.telemetry_exact` proves the captured eval equals `NNUE::Network::evaluate`
and that each layer's reported attribution sums back to that layer's output.
`viz.session_selfplay` replays the reported move list and requires it to
reproduce the reported FEN exactly. `viz_smoke.sh` covers the wire format, the
control surface, the loopback/no-filesystem posture, and cross-checks the
streamed eval against the engine's own `eval`.
