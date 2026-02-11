# STK-Vector-64 Architecture Specification

Status: Ongoing

## 1. Purpose

This document defines the architecture of the STK-Vector-64 engine core as currently implemented. It specifies module boundaries, data layout, move-generation flow, legality validation, hashing strategy, and test execution behavior.

## 2. Engine Boundaries

The repository currently implements an engine core and verification harness, not a full UCI search stack.

In scope:
- Board representation and state transitions
- Pseudo-legal and legal move generation
- Attack generation (precomputed + magic-bitboard)
- Incremental Zobrist hashing
- Perft validation from EPD input

Out of scope:
- Search (alpha-beta, qsearch, TT replacement policy)
- Evaluation model
- UCI command protocol
- Opening book / endgame tablebase integration

## 3. Source Topology

Core modules:
- `src/cores/types.h`: primitive enums and square/rank/file helpers
- `src/cores/bitboard.h`: bitboard constants and bit operations
- `src/cores/move.h`: 16-bit move packing and move list container
- `src/cores/position.h`, `src/cores/position.cpp`: position state and make/unmake
- `src/cores/attacks.h`, `src/cores/attacks.cpp`: attack tables + magic lookups
- `src/cores/movegen.h`, `src/cores/movegen.cpp`: pseudo/legal move generation
- `src/cores/zobrist.h`, `src/cores/zobrist.cpp`: zobrist key initialization
- `src/cores/invariants.h`: debug-time consistency guard

Support modules:
- `src/utils/debug.h`, `src/utils/debug.cpp`: bitboard visualization helper
- `tests/perfts.h`, `tests/perfts.cpp`: perft and EPD suite runner
- `main.cpp`: executable entry point for EPD suite execution

## 4. Fundamental Types

### 4.1 Bitboard

`Bitboard` is `uint64_t` with little-endian square indexing:
- `SQ_A1 = 0`
- `SQ_H8 = 63`

This mapping aligns directional deltas to simple arithmetic:
- North `+8`
- South `-8`
- East `+1`
- West `-1`

### 4.2 Color and Piece Domain

`Color`: `WHITE`, `BLACK`

`PieceType` includes:
- `PAWN`, `KNIGHT`, `BISHOP`, `ROOK`, `QUEEN`, `KING`
- `NO_PIECE_TYPE` sentinel

### 4.3 Move Encoding

`Move` packs into `uint16_t`:
- bits `[0..5]`: from square
- bits `[6..11]`: to square
- bits `[12..15]`: flag nibble

Flag space encodes:
- quiet, double push, castling, en passant
- captures
- promotions (quiet/capture variants)

## 5. Position State Model

`Position` stores board state as orthogonal bitboards:
- `byColor[2]`
- `byType[PIECE_TYPE_NB]`

Derived occupancy:
- `occupancy() = byColor[WHITE] | byColor[BLACK]`

Game-state fields:
- `sideToMove`
- `castlingRights` bitmask
- `epSquare`
- `halfmoveClock`
- `fullmoveNumber`
- `zobristHash`
- repetition history ring/array (`history`, `gamePly`)

### 5.1 Castling Rights Decay

`CastlingSpoilers[64]` masks rights when king/rook moves or target rook is captured. Rights are updated incrementally in `make_move` and restored via undo state in `unmake_move`.

### 5.2 FEN IO

`setFromFEN` parses:
- board occupancy
- active color
- castling rights
- en passant target
- half/full move counters

`toFEN` reserializes full state from current position.

## 6. Attack Subsystem

## 6.1 Non-sliding attacks

Precomputed arrays:
- `PawnAttacks[color][sq]`
- `KnightAttacks[sq]`
- `KingAttacks[sq]`

Computed once in `Attacks::init()`.

## 6.2 Sliding attacks via magics

Magic tables are generated at startup:
- `RookMagics[64]`
- `BishopMagics[64]`
- shared `AttackTable`

For each square:
1. Build occupancy mask of relevant blocker squares.
2. Enumerate all blocker combinations.
3. Compute ground-truth ray attacks.
4. Search sparse magic constant producing collision-safe mapping.

Runtime lookup:
- mask occupancy
- multiply by magic
- shift
- index prebuilt attack table

## 6.3 Geometry tables

Precomputed:
- `Between[s1][s2]`
- `Line[s1][s2]`

These support line/ray relations and are available for legality and future search heuristics.

## 7. Move Generation Pipeline

## 7.1 Pseudo-legal generation

`generate_pseudo_legal_moves` dispatches by side:
- pawns (pushes, captures, promotions, en passant)
- knights
- bishops
- rooks
- queens
- king
- castling candidates

Generation is bitboard-driven and avoids branch-heavy piece loops where possible.

## 7.2 Legal filtering

`generate_legal_moves` performs:
1. pseudo-legal generation into `MoveList`
2. context precompute (`LegalityContext`)
3. in-place compaction of legal moves only

This avoids:
- repeated board make/unmake for legality checks
- extra move-list allocations/copies

## 8. Bitwise Legality Validator

Legal validation uses occupancy simulation, not full board mutation.

For a candidate move:
1. Remove moving piece from source square in simulated occupancy.
2. If capture, remove captured square (special case en passant capture square).
3. Add moving piece at destination.
4. For castling, additionally simulate rook displacement and validate start/intermediate king squares.
5. Compute king square after move (destination for king moves, original otherwise).
6. Query `is_square_attacked_by_enemy` against simulated occupancy.

Move is legal iff king is not attacked in simulated result.

This path is O(1) for attack queries due to precomputed tables + magic lookups.

## 9. Move Make/Unmake

`make_move` responsibilities:
- capture removal (incl. en passant)
- piece displacement
- promotion replacement
- castling rook shift
- castling rights decay
- en passant setup/clear
- half/full move clock updates
- side-to-move flip
- incremental zobrist update
- history append

`unmake_move` restores full prior state from `UndoInfo` plus reverse piece transforms.

Undo struct includes:
- prior castling rights
- prior en passant square
- prior halfmove clock
- captured piece type
- zobrist delta

## 10. Zobrist Hashing

Random keys initialized once for:
- `psq[color][piece][square]`
- `enpassant[file]`
- `castling[0..15]`
- `side`

Hash is maintained incrementally to avoid full recompute.

Repetition detection scans same-side history window bounded by halfmove clock.

## 11. Invariants and Defensive Checks

`ASSERT_CONSISTENCY(pos)` is active in debug builds and validates:
- color bitboards are disjoint
- exactly one king per side
- en passant rank validity relative to side to move

These checks protect move/unmove integrity during development and perft debugging.

## 12. Test and Verification Surface

`main.cpp` invokes EPD suite runner:
- loads `test_data/standard.epd`
- executes perft to configured max depth
- returns nonzero on mismatch/failure

`tests/perfts.cpp` provides:
- recursive perft
- optional multithreaded branch fan-out for benchmark comparison
- pass/fail reporting against EPD expected node counts

## 13. CI Behavior

Workflow: `.github/workflows/ci.yml`

Matrix:
- `ubuntu-latest`
- `windows-latest`

Per platform:
- configure with CMake
- build engine target
- run executable so perft suite acts as correctness gate

## 14. Performance Characteristics

Current design focuses on high throughput in movegen/perft workloads:
- bitboard arithmetic throughout critical path
- O(1) sliding attack lookups with magics
- legal filter by occupancy simulation
- in-place legal list compaction
- depth-1 perft bulk counting

Observed benchmark outputs in this repository have reported >100 MNPS single-thread in release configuration on target hardware.

## 15. Known Architectural Constraints

- Magic initialization is runtime-generated, increasing startup cost.
- Current executable path is test-first (EPD/perft), not protocol-driven engine mode.
- Multithread perft currently spawns per-branch async tasks; this is benchmark-oriented and not a search scheduler.

## 16. Extension Points

Natural extension points without breaking core structure:
- add search module over existing legal move API
- add transposition table keyed by existing Zobrist hash
- add UCI front-end as separate interface layer
- add dedicated benchmark command path distinct from CI correctness path

