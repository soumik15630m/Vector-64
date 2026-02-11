# STK-Vector-64

Status: Ongoing

Production-grade C++20 chess engine core focused on high-throughput perft validation and fast legal move generation.

## Highlights

- Bitboard-based board representation
- Magic-bitboard sliding attacks
- Bitwise legal-move validator with occupancy simulation
- Zobrist hashing for position identity and repetition checks
- EPD-driven perft verification harness

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

The executable runs the EPD test suite in `test_data/standard.epd` and returns non-zero on failure.

