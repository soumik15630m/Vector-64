# Vector-64 NNUE Training Guide

This repository trains NNUE offline in PyTorch and then exports to the engine's `VECTOR64_NNUE` format.

Recommended production path in this repository:
- Baseline: supervised training from labeled HalfKP datasets (`train_vector64.py`)
- Experimental: pure RL self-play (`rl_train_vector64.py`), typically requiring very large game counts before matching supervised quality

## 1. What You Train

`tools/nnue/train_vector64.py` trains this topology:

- Input features: `81920` (HalfKP)
- Feature transform: `81920 -> 512`
- Dense stack: `512 -> 32 -> 32 -> 1`
- Output unit: centipawns from side-to-move perspective

Checkpoint keys are intentionally compatible with `tools/nnue/export_vector64.py`:

- `feature_transform.weight`, `feature_transform.bias`
- `dense1.weight`, `dense1.bias`
- `dense2.weight`, `dense2.bias`
- `output.weight`, `output.bias`

## 2. Dataset Format (`.npz`)

Required arrays:

- `white_indices`: `int32/int64 [N, W]`
- `black_indices`: `int32/int64 [N, B]`
- `stm`: `int8/int32 [N]` where `0=white`, `1=black`
- `eval_cp`: `float32 [N]`, eval in centipawns from white perspective

Conventions:

- `white_indices` and `black_indices` are active HalfKP feature indices.
- Use `-1` as padding for unused slots in a row.

Optional arrays:

- `white_values`: `float32 [N, W]` (defaults to 1.0 if omitted)
- `black_values`: `float32 [N, B]` (defaults to 1.0 if omitted)
- `result`: `float32 [N]` in `[-1, 1]` from white perspective
- `sample_weight`: `float32 [N]`

`result` can be blended with eval target using:

- `--result-blend` (0 to 1)
- `--result-scale` (maps result into centipawns, default `600`)

### Build From Lichess Eval JSONL

Use `tools/nnue/build_halfkp_npz.py` to convert Lichess eval dumps into shardable HalfKP `.npz` files.

Example:

```powershell
python .\tools\nnue\build_halfkp_npz.py `
  --input .\path\to\lichess_db_eval.jsonl.zst `
  --out-dir .\dataset\halfkp_shards `
  --shard-size 500000 `
  --max-features 64 `
  --min-depth 20 `
  --max-abs-cp 2000 `
  --dedup-scope global
```

Notes:

- Requires `zstandard` only when reading `.zst` directly (`pip install zstandard`).
- `--cp-source` controls whether input centipawns are already white-perspective (`white`) or side-to-move (`stm`).
- `--dedup-scope global` persists position dedup state in SQLite at `<out-dir>/dedup_seen.sqlite`.
- Global dedup key is `piece placement + side to move` (ignores castling/ep/clocks) to prevent history-only duplicates.
- `manifest.json` includes CP summary (min/max/mean/std + histogram), depth distribution, phase distribution, shard list, and active filtering config.

## 3. Train

Example command:

```powershell
python .\tools\nnue\train_vector64.py `
  --dataset .\path\to\train_halfkp.npz `
  --out-checkpoint .\artifacts\vector64_ckpt.pt `
  --epochs 3 `
  --batch-size 1024 `
  --lr 1e-3 `
  --weight-decay 1e-2 `
  --min-lr 1e-5 `
  --result-blend 0.2 `
  --result-scale 600 `
  --val-split 0.01 `
  --device auto `
  --amp
```

Notes:

- `--amp` only applies on CUDA.
- For quick smoke runs, add `--max-positions 100000`.
- Target clamp defaults to `--max-target-cp 2000`.

## 4. Export + Validate

```powershell
python .\tools\nnue\export_vector64.py `
  --checkpoint .\artifacts\vector64_ckpt.pt `
  --output .\artifacts\vector64.nnue

python .\tools\nnue\check_vector64.py .\artifacts\vector64.nnue
```

## 5. Load in Engine

Use UCI:

```text
setoption name EvalFile value <path-to-network.nnue>
```

## 6. Pure RL Self-Play Loop (Experimental)

For pure RL (no prebuilt eval labels), use:

```powershell
python .\tools\nnue\rl_train_vector64.py `
  --out-checkpoint .\artifacts\vector64_rl.pt `
  --games 2000 `
  --batch-size 1024 `
  --updates-per-game 2 `
  --replay-capacity 250000 `
  --warmup-samples 8192 `
  --epsilon-start 0.25 `
  --epsilon-end 0.05 `
  --epsilon-decay-games 1500 `
  --result-scale 600 `
  --amp
```

Notes:

- Requires `python-chess` (`pip install python-chess`).
- This is Monte Carlo self-play value learning with replay buffer updates.
- Checkpoints are exporter-compatible, so you can run the same export/validate/load steps in sections 4 and 5.
