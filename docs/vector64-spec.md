# Vector-64 Production Architecture

Status: Target Architecture

This document defines forward production targets. For current implementation behavior, see `docs/architecture.md`.

## 1. Engine Type

- Production chess engine
- Search: Negamax Alpha-Beta
- Iterative deepening: enabled

## 2. Search Subsystem

### 2.1 Transposition Table

- Enabled
- Replacement strategy: depth-preferred
- Entry layout:
  - `key`: `uint64`
  - `depth`: `int16`
  - `score`: `int16`
  - `best_move`: `uint32`
  - `flag`: `EXACT | LOWER | UPPER`

### 2.2 Move Ordering

- TT move
- Captures (MVV-LVA)
- Killer moves
- History heuristic

## 3. NNUE Design

### 3.1 Feature System

- Type: HalfKP
- Encoding: king-relative piece-square
- Total feature space: `81920`
- Structure:
  - king square: `64`
  - piece types: `10`
  - square: `64`
  - sides: `2`
- Typical sparse activation estimate: `~30`

### 3.2 Network Topology

- Input: sparse HalfKP
- Layer stack:
  1. FeatureTransform: `512` (`int16`)
  2. Dense: `32`, `ReLU` (`int8`)
  3. Dense: `32`, `ReLU` (`int8`)
  4. Output: `1`, linear (`int32`)

### 3.3 Incremental Accumulator

- Hidden size: `512`
- Storage:
  - white: `int16[512]`
  - black: `int16[512]`
- Alignment: `64-byte`
- Update rules:
  - Piece move: subtract old features, add new features
  - Capture: subtract captured features
  - Promotion: remove pawn feature, add promoted feature
  - King move: full accumulator rebuild

### 3.4 Evaluation Policy

- Perspective: side to move
- Output unit: centipawns
- Blending:
  - PSQT: enabled
  - NNUE weight: `0.8`
  - PSQT weight: `0.2`

## 4. Training Stack

- Framework: PyTorch
- Baseline training path: supervised HalfKP training via `tools/nnue/train_vector64.py`
- Pure RL self-play path: `tools/nnue/rl_train_vector64.py` (experimental; expected to need very large self-play scale before parity)
- Training guide: `docs/nnue-training.md`
- Target hardware: NVIDIA RTX 3050 (6GB VRAM)
- Precision: FP32 for training, quantized for inference
- Optimizer: AdamW
- Learning rate: `1e-3`
- Scheduler: cosine decay
- Batch size: `1024`
- Loss:
  - Primary: MSE
  - Optional blend: eval + game result
- Dataset:
  - Minimum positions: `10,000,000`
  - Sources: self-play + strong-engine labeled positions

## 5. Quantization Contract

- Input weights: `int16`
- Hidden weights: `int8`
- Output weights: `int8`
- Accumulator type: `int32`
- Requirements:
  - Deterministic scaling
  - Symmetric clamping
  - Validation against float model

## 6. Binary Network Format

### 6.1 Header

- Magic: `VECTOR64_NNUE`
- Version: `1`
- Hidden size: `512`

### 6.2 Sections

- FeatureTransformWeights
- HiddenLayer1Weights
- HiddenLayer2Weights
- OutputWeights
- Biases

### 6.3 Packing

- Section alignment: `64-byte`

## 7. Performance Targets

- Single-core eval time: `300 ns`
- Target eval throughput: `4,000,000 evals/s`
- NPS drop vs classical eval: max `25%`

## 8. Multi-Network Support

- Enabled
- Network switch strategies:
  - Material phase
  - Move count
  - Game stage heuristic

## 9. Scalability

- Hidden-size variants: `512`, `768`
- Additional layers: supported
- Distributed training: supported
- SIMD targets: `AVX2`, `AVX512`
