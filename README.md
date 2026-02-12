# STK-Vector-64

Status: Ongoing

Production-grade C++20 chess engine focused on fast legal move generation, UCI search, and a Vector-64 NNUE-ready architecture path.

## Highlights

- Bitboard-based board representation
- Magic-bitboard sliding attacks
- Bitwise legal-move validator with occupancy simulation
- Zobrist hashing for position identity and repetition checks
- EPD-driven perft verification harness
- UCI loop with iterative deepening negamax alpha-beta search

## Architecture Targets

- Current implementation notes: `docs/architecture.md`
- Production target profile: `docs/vector64-spec.md`

## Build

### Windows (MSVC)

```powershell
cmake -S . -B build
cmake --build build --config Release
```

### Linux (GCC/Clang)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

From the `build` directory:

```bash
./bin/ChessEngine
```

On Windows:

```powershell
.\bin\Release\ChessEngine.exe
```

The executable starts in UCI mode by default.

Run the perft EPD suite explicitly with:

```bash
./bin/ChessEngine --perft
```

On Windows:

```powershell
.\bin\Release\ChessEngine.exe --perft
```

You can also provide a custom EPD file path:

```bash
./bin/ChessEngine --perft <path-to-standard.epd>
```

Perft mode returns non-zero on mismatch/failure.

Run NNUE incremental consistency diagnostics with:

```bash
./bin/ChessEngine --nnue-consistency [games] [max-plies] [seed]
```

## UCI NNUE Loading

The UCI loop exposes:

```text
setoption name EvalFile value <path-to-network.nnue>
```

`EvalFile` expects the `VECTOR64_NNUE` binary layout described in `docs/vector64-spec.md`.

## NNUE Export Tools

Scripts are provided in `tools/nnue/`:

- Validate a `.nnue` file layout:
```powershell
python .\tools\nnue\check_vector64.py .\test_data\vector64_dummy.nnue
```

- Export from a PyTorch checkpoint to `VECTOR64_NNUE`:
```powershell
python .\tools\nnue\export_vector64.py `
  --checkpoint .\path\to\checkpoint.pt `
  --output .\test_data\vector64_model.nnue
```

Default key mapping expected in the checkpoint:
- `feature_transform.weight`, `feature_transform.bias`
- `dense1.weight`, `dense1.bias`
- `dense2.weight`, `dense2.bias`
- `output.weight`, `output.bias`

## NNUE Training (PyTorch)

Training is done offline in PyTorch, then exported to `VECTOR64_NNUE`.

- Full guide and dataset schema: `docs/nnue-training.md`
- Lichess eval JSONL -> HalfKP NPZ converter: `tools/nnue/build_halfkp_npz.py`
- Training script (baseline/recommended): `tools/nnue/train_vector64.py`
- Pure RL self-play script (experimental): `tools/nnue/rl_train_vector64.py`

Example workflow:

```powershell
# 1) Train float checkpoint from HalfKP dataset (.npz)
python .\tools\nnue\train_vector64.py `
  --dataset .\path\to\train_halfkp.npz `
  --out-checkpoint .\artifacts\vector64_ckpt.pt `
  --epochs 3 `
  --batch-size 1024 `
  --lr 1e-3 `
  --result-blend 0.2

# 2) Export to engine binary format
python .\tools\nnue\export_vector64.py `
  --checkpoint .\artifacts\vector64_ckpt.pt `
  --output .\artifacts\vector64.nnue

# 3) Validate binary layout
python .\tools\nnue\check_vector64.py .\artifacts\vector64.nnue
```

Then load in UCI:

```text
setoption name EvalFile value <path-to-network.nnue>
```
