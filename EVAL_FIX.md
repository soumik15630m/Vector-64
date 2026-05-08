# EVAL_FIX - Root Cause Analysis and Resolution (V3)

## Architecture confirmed
- evaluate() = weighted blend of NNUE + PSQT
- NNUE: no network loaded (EvalFile=empty), returns material-delta fallback
- PSQT: was returning material + PST (material duplication bug)
- pos.material[] / pos.pst_score[] do not exist as fields

## Two bugs found and fixed

### Bug 1 - psqt_white_minus_black() included full material
- File: src/search/evaluator.cpp
- Removed calls to: `piece_value(...)` inside `psqt_white_minus_black()`
- Before: psqt after e2a6 = -324 (approximately the nnue value, material-dominated)
- After: psqt after e2a6 = 6 from side-to-move perspective, positional-only

### Bug 2 - evaluate() used NNUE material fallback when no network loaded
- File: src/search/evaluator.cpp
- Guard added: `if (!nnue_.is_loaded())`
- Fallback now uses: `material_white_minus_black(pos)` + `psqt_white_minus_black(pos)`, adjusted for side to move

## Files modified

| File | Lines changed | What changed |
|------|--------------|--------------|
| src/search/evaluator.cpp | 1 deletion | removed `piece_value()` from `psqt_white_minus_black()` |
| src/search/evaluator.cpp | 25 insertions, 1 deletion | added NNUE-absent guard and material-balance fallback |

## Files NOT modified
Everything else. No other file was modified.

## Baseline split diagnostic (pre-fix, recorded in V2)
```text
Base       nnue=0    psqt=-4    total=0
After e2a6 nnue=-330 psqt=-324  total=-328
After e5d7 nnue=-100 psqt=-99   total=-99
After f3f6 nnue=-320 psqt=-317  total=-319
```

## Post-fix diagnostic (NNUE absent path)
```text
Base       psqt=-4  total=-4
After e2a6 psqt=6   total=-324
After e5d7 psqt=1   total=-99
After f3f6 psqt=3   total=-317
```

White scores (negate post-move totals):
```text
After e2a6 = +324  (expected: +280 to +370, captures bishop)
After e5d7 =  +99  (expected: +100 to +180, captures pawn + outpost)
After f3f6 = +317  (Qxf6, knight capture ~320 cp; not a quiet queen move)
```

PST delta favors e5d7 over e2a6 positionally: YES

Note on revised gate: static eval cannot prefer e5d7 over e2a6 because
e2a6 captures a bishop (+330 material). The search gate is the real arbiter.

## Perft regression

| Pos | Depth | Result | Expected | Pass |
|-----|-------|--------|----------|------|
| 1 | 5 | 4,865,609 | 4,865,609 | PASS |
| 2 | 5 | 193,690,690 | 193,690,690 | PASS |
| 3 | 5 | 674,624 | 674,624 | PASS |
| 4 | 5 | 15,833,292 | 15,833,292 | PASS |
| 5 | 5 | 89,941,194 | 89,941,194 | PASS |
| 6 | 5 | 164,075,551 | 164,075,551 | PASS |

Total: 469,080,960 (expected: 469,080,960)
Suite: RESULTS: 6 Passed, 0 Failed -> PASS

Perft performance from final run:
```text
SINGLE CORE : 3.089 seconds (151 MNPS)
MULTI CORE  : 0.555 seconds (844 MNPS)
```

## Search benchmark (Kiwipete depth 8)

| Depth | Score (cp) | PV first move |
|-------|-----------|---------------|
| 1 | -8 | e2a6 |
| 2 | -8 | e2a6 |
| 3 | -8 | e2a6 |
| 4 | -8 | e2a6 |
| 5 | -18 | e2a6 |
| 6 | -18 | e2a6 |
| 7 | -34 | e2a6 |
| 8 | -34 | e2a6 |

bestmove: e2a6   Nodes: 1,675,194   Time: 672ms   NPS: 2.492848 MNPS

Raw search output:
```text
id name STK-Vector-64
id author Soumik
option name Threads type spin default 12 min 1 max 64
option name Hash type spin default 6 min 1 max 4096
option name Move Overhead type spin default 30 min 0 max 500
option name Ponder type check default false
option name EvalFile type string default <empty>
uciok
readyok
info depth 1 score cp -8 nodes 2702 nps 2702000 time 1 pv e2a6
info depth 2 score cp -8 nodes 2854 nps 2854000 time 1 pv e2a6
info depth 3 score cp -8 nodes 6318 nps 3159000 time 2 pv e2a6
info depth 4 score cp -8 nodes 17009 nps 3401800 time 5 pv e2a6
info depth 5 score cp -18 nodes 84235 nps 4433421 time 19 pv e2a6
QSearch TT hit rate: 28.279%
QSearch TT hit rate: 16.913%
info depth 6 score cp -18 nodes 249605 nps 3328066 time 75 pv e2a6
QSearch TT hit rate: 19.003%
QSearch TT hit rate: 19.8122%
QSearch TT hit rate: 22.1852%
QSearch TT hit rate: 23.3903%
QSearch TT hit rate: 24.7291%
info depth 7 score cp -34 nodes 811162 nps 3899817 time 208 pv e2a6
QSearch TT hit rate: 24.5877%
QSearch TT hit rate: 22.5207%
Negamax TT hit rate: 54.415%
QSearch TT hit rate: 22.4311%
QSearch TT hit rate: 21.5082%
QSearch TT hit rate: 20.4849%
QSearch TT hit rate: 19.704%
QSearch TT hit rate: 19.0566%
Negamax TT hit rate: 49.4635%
info depth 8 score cp -34 nodes 1675194 nps 2492848 time 672 pv e2a6
bestmove e2a6
```

Previous (broken): bestmove e2a6 (NNUE material fallback + PSQT material duplication)
Fixed            : bestmove e2a6, score -34 cp at depth 8

## Remaining limitation
Without a loaded NNUE network, the engine evaluates positions using
material + PST only. e2a6 captures a bishop (+330) and f3f6 is Qxf6,
capturing a knight (about +320), while e5d7 captures a pawn (+100).
The fallback evaluator therefore still favors the immediate larger
captures. Preferring e5d7 over e2a6 at depth 8 requires the search to
foresee the d5-d6 tactical continuation and/or a richer evaluation
than material + simple PST. This remains a limitation with EvalFile empty.

## Git log
```text
4f63f4b8 fix(eval): bypass NNUE material fallback when no network is loaded
d1dd25f6 fix(psqt): remove material duplication from psqt_white_minus_black
74528e1d Fix unmake_move to explicitly restore hash from undo stack to avoid drift
57be2b80 TT Qsearch, Resize optimization to 6MB.
512919a1 feat: bug fixes
```

## Self-verification checklist
- [x] psqt_white_minus_black() returns no material values
- [x] evaluate() has working NNUE-absent guard
- [x] No hack, scaling patch, or magic constant added
- [x] NNUE network loading code untouched
- [x] Move generation untouched
- [x] Perft: 6 Passed, 0 Failed, 469,080,960 total nodes
- [ ] After e5d7 (White) >= +100 cp (observed +99 cp)
- [ ] Score at depth 8 is positive (observed -34 cp)
- [x] Only evaluator.cpp and EVAL_FIX.md modified
- [x] EVAL_FIX.md written after perft and search benchmark
