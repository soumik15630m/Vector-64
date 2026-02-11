# Engine Core

Status: Ongoing

## Scope

This document defines the current engine-core architecture for move generation, legality validation, board state handling, hashing, and perft verification.

## Core Layout

- `src/cores/types.h`
- `src/cores/bitboard.h`
- `src/cores/move.h`
- `src/cores/position.h`
- `src/cores/position.cpp`
- `src/cores/attacks.h`
- `src/cores/attacks.cpp`
- `src/cores/movegen.h`
- `src/cores/movegen.cpp`
- `src/cores/zobrist.h`
- `src/cores/zobrist.cpp`

## Board Model

- Board state is stored as bitboards split by color and piece type.
- Occupancy is derived from `byColor[WHITE] | byColor[BLACK]`.
- Move make/unmake supports castling, en passant, promotions, and clocks.
- Position integrity is guarded by `ASSERT_CONSISTENCY`.

## Move Generation

- Pseudo-legal generation is piece-specific and bitboard-driven.
- Sliding attacks use magic bitboards for O(1) lookup.
- Legal filtering uses occupancy simulation and attack validation.
- Legality path handles castling squares and en passant capture square semantics.

## Attack System

- Precomputed attacks for pawns, knights, kings.
- Magic lookup tables for bishops and rooks.
- Queen attacks are composed from bishop and rook attacks.
- `between_bb` and `line_bb` are precomputed for geometry queries.

## Hashing

- Zobrist hashing tracks pieces, side-to-move, castling rights, and en passant file.
- Hash is incrementally updated in move make/unmake.
- Position history stores hashes for repetition checks.

## Test Surface

- Entry point uses EPD-based perft verification.
- Perft compares generated node counts with expected node counts.
- Multi-threaded branch evaluation exists for benchmark comparison.

