# Training STK-HalfKA with bullet

Goal: train the 1024-wide STK-HalfKA net on the big self-play set in **bullet**
(minutes/epoch on GPU) instead of the PyTorch `make_net.py` (hours/epoch), while
keeping the exported `.nnue` **bit-exact** with the C++ engine.

Key idea that de-risks everything: **bullet trains float weights only; we then
load those weights into the existing Python `STKNet` (tools/nnue/make_net.py) and
reuse its already-verified `quantize_model` -> export -> engine-parity path.**
bullet never touches the `.nnue` format.

## Status

- [x] Data is generated in bullet's native text format (`<fen> | <eval> | <wdl>`,
      white-relative) by `datagen_bulk.py --emit raw`. No custom converter needed.
- [x] **Toolchain validated on this machine**: Rust 1.96, CUDA 12.8, RTX 3050 6GB
      (sm_86). `bullet` builds with `--features cuda` (~54s). `bullet-utils`
      packs our real shards (`convert --from text`) with 0 errors. A GPU smoke
      train ran at ~6.6M pos/sec. bullet clone lives at `D:\Soumik\Cpp\bullet`.
- [x] Architecture confirmed expressible in bullet (`examples/progression/
      4_multi_layer.rs`): `crelu().pairwise_mul()`, per-bucket `1024->(8x16)->
      (8x32)->8`, `MaterialCount::<8>` output buckets, `sigmoid().squared_error`.
- [x] **Custom `SparseInputType` (StkHalfKa) + FEATURE PARITY PASS.** See
      `stk_halfka.rs` + `parity_check.py`: 266 indices across 5 positions match
      `halfka_features.py` bit-for-bit (symmetric startpos, asymmetric post-1.e4
      both colours, mirrored-king castled, asymmetric endgame).
- [x] **Full net def + dry run TRAINS on GPU.** `stk_train.rs`: custom StkHalfKa
      input + custom `StkBucket` output buckets (`(pieces-1)/4`, NOT bullet's
      `MaterialCount` = `(pieces-2)/4`), FT->CReLU->pairwise-mul, per-bucket
      L1 1024->16 / L2 16->32 / out 32->1 (CReLU, not screlu), PSQT
      (`psq_stm - psq_ntm`). Compiles; trained 1M positions 3sb x 60b,
      loss 0.048->0.024, ~827k pos/sec (=> 500M x 10ep ~= 1.5-2h).
- [x] **Weight transfer VERIFIED end-to-end** (`transfer_to_stknet.py`,
      `stk_eval.rs`). bullet stores weights **column-major** (out,in) -> read as
      `raw.reshape(in,out)`. STKNet(transferred) == bullet's `eval_raw_output`*400
      to 0.0000 cp on 5 FENs (oracle: `stk_eval` loads the checkpoint + bullet's
      real forward). Then make_net export + engine parity: `PASS engine ==
      quantized reference (max diff 0 cp), mirror-symmetric`. Full pipeline works.
- [ ] Full training run (500M) + SPRT vs runs/v2.

### Transfer scalings (verified)
`ft.weight[f+1]=M0[f]` (`M0=l0w.reshape(NIN,H)`); `ft_bias=l0b+M0[DEAD]`;
`l1w=(l1w.reshape(H,128).T).reshape(8,16,H)*128/127` (PAIR_FACTOR);
`l2w=(l2w.reshape(16,256).T).reshape(8,32,16)`;
`outw=(l3w.reshape(32,8).T)*eval_scale/OUT_CP` (400/508);
`psqt.weight[f+1]=MP[f]*eval_scale` (`MP=psqtw.reshape(NIN,8)`). Biases 1-D.

### bullet save layout (checkpoints/<net>/raw.bin, from save_format order)
Raw f32, no header, concatenated in this order (bytes verified = 93,547,104):
`l0w[22529*1024] l0b[1024] l1w[128*1024] l1b[128] l2w[256*16] l2b[256]
l3w[8*32] l3b[8] psqtw[8*22529] psqtb[8]`. (Affine weight orientation per
tensor still to confirm during transfer via forward parity; 4_multi_layer uses
`.transpose()` on l1w+ for inference, so raw is bullet's training layout.)

### Transfer plan (phase 2, remaining)
Read raw.bin f32 -> map to STKNet params: l0w->ft.weight (fold dead row 22528
into ft_bias, handle orientation), l0b->ft_bias, l1w/l1b/l2w/l2b/l3w/l3b ->
per-bucket l1w/l1b/l2w/l2b/outw/outb (reshape [8,..], fold PAIR_FACTOR=127/128
into l1w, reconcile OUT_CP=508 vs bullet eval_scale), psqtw/psqtb -> psqt (fold
psqt bias into outb). Then make_net stage_export + stage_verify (engine==ref).
Gate: a bullet-side eval oracle (port simple.rs's Network::evaluate to STK arch)
to bit-check STKNet(transferred) == bullet before trusting the export.

### bulletformat::ChessBoard encoding (decoded from source, verified by parity)
Stored **side-to-move-relative**: black-to-move flips (`piece ^= 8; square ^= 56`).
`into_iter()` yields `(piece, square)`: `piece & 8` = colour bit (0 = mover/stm,
1 = opp/ntm), `piece & 7` = type `0=P..5=K`, `square` absolute in the stm frame.
STM perspective = board as-is (mover at bottom); NTM = rank-flip every square. Own
king (mover's, `pt==5, c==0`) -> DEAD in stm and enemy-king in ntm, and vice versa.

## The exact STK-HalfKA feature scheme (from tools/nnue/halfka_features.py)

- `FEATURES = 32 king-buckets * 11 kinds * 64 squares = 22528`.
- King bucket (per perspective): orient king by `flip_rank` if black; mirror
  (`flip_file`) when king file >= e; `bucket = (tk>>3)*4 + (tk&7)`  (0..31).
- Piece kinds (perspective-relative): own {P,N,B,R,Q} = 0..4 (own **king dropped**);
  enemy {P,N,B,R,Q,K} = 5..10 (enemy **king kept**).  -> 11 kinds.
- Square orient: `flip_rank` for black perspective, then `flip_file` if the
  king was mirrored.
- `index = (bucket*11 + kind)*64 + oriented_sq`.

### Why the built-ins don't match
`ChessBucketsMirrored` = 768/bucket (12 kinds incl. both kings) -> 24576 total,
different kind order, own king kept. Ours = 704/bucket -> 22528. A custom
`SparseInputType` matching the formula above is the clean fix (exact FT-row
alignment => trivial weight transfer, no permutation guesswork).

### Own-king asymmetry — RESOLVED (dead index + fold to bias)
`map_features(f(stm, ntm))` emits *both* perspective indices per piece, but our
own king is active only in the opponent's perspective (enemy-king, kind 10) and
absent in its own. Handle it with a **dead index**:
- `num_inputs = 22528 + 1`; index 22528 is a dead slot.
- For each side's own king emit the dead index on that side's perspective and the
  real enemy-king feature on the other (i.e. white king -> `f(DEAD, ntm_idx)`
  when white is stm; black king -> `f(stm_idx, DEAD)`).
- Every position has exactly one own king per perspective, so the dead feature is
  active exactly once in every accumulator => a pure per-position constant.
- Weight transfer folds it away bit-exactly:
  `stk_ft_bias = bullet_ft_bias + bullet_ft_weight[22528]`, then drop row 22528.
  Our engine has no own-king feature, so this reproduces bullet's function exactly.

### CRITICAL parity detail: bullet boards are stm-relative
`chess768.rs` shows the record is stored **side-to-move-relative**: the piece
colour bit is mover(0)/non-mover(1), and the ntm perspective flips squares with
`sq ^ 56` (rank flip). `halfka_features.py` instead uses **absolute** white/black
with an explicit stm. The custom input type must bridge these (map mover<->own,
apply our orient/mirror per perspective). This is the most likely source of a
silent mismatch -> the phase-1 feature-parity check exists precisely to catch it
before any training.

## Architecture -> bullet mapping

| STK-HalfKA (make_net.py STKNet)                    | bullet                                  |
|----------------------------------------------------|-----------------------------------------|
| FT 22528->1024 per perspective (+bias)             | custom SparseInputType + `new_affine`   |
| clip[0,1] then first_half*second_half (per persp.) | `.crelu().pairwise_mul()` (match pairing)|
| concat us(512)+them(512) = 1024                     | concatenated perspectives               |
| L1 [8][16][1024] -> CReLU                           | `new_affine` 1024->(8*16), select, crelu|
| L2 [8][32][16] -> CReLU                             | `new_affine` 16->(8*32), select, crelu  |
| out [8][32] -> *OUT_CP (508)                        | `new_affine` 32->8, select              |
| PSQT emb 22528->8; psq_us[b]-psq_them[b]            | parallel sparse affine ->8, per-persp - |
| loss sigmoid(cp/400) SE                             | `.sigmoid().squared_error(target)`      |

Activations are **CReLU** (clip 0..1), NOT SCReLU — must match ours exactly.

## Phased build + verify (each phase gates the next)

1. **Feature parity (do FIRST).** Implement the custom `SparseInputType`. Dump
   its active (stm, ntm) indices for a handful of FENs and diff against
   `python tools/nnue/halfka_features.py` (feed the same FENs on stdin; it prints
   `W ...` / `B ...` index lists). They must match exactly. This catches the
   king/orientation/bucket assumptions before any training.
2. **Forward parity (tiny train).** Wire the full net + PSQT. Train 1 superbatch,
   export float weights, transfer into STKNet, run make_net's `stage_export` +
   `stage_verify`. Engine==reference must hold (the existing bit-exact check).
3. **Full run.** Convert `runs/bulk/data/*.txt` with bullet's own tool, train
   ~8-10 epochs (500M positions ~= bullet's canonical 4-5B visits) at lr 1e-3
   cosine -> 3e-5, batch 16384, WDL proportion ~0.5-0.75.
4. **SPRT** the resulting `.nnue` vs runs/v2 with match.py. Only that verdict counts.

## Build notes (target machine)
- Needs Rust (stable) + the CUDA toolkit (not just torch's runtime).
- `git clone https://github.com/jw1912/bullet`; put our example under
  `examples/` and register it in `bullet_lib`'s Cargo.toml (per docs/2).
- Data convert + train + save-format details: bullet `docs/3-data.md`,
  `docs/4-saved-networks.md`, `crates/bullet_lib/src/value/save.rs`.

## Fallback
If the CUDA build fights us, `make_net.py` still trains this exact data
(`--input runs/bulk/data`, which now reads the raw 3-field shards and blends).
Cut it to `--epochs 10 --batch-size 32768` for ~1 day instead of ~3.
