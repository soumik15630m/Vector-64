#!/usr/bin/env python3
"""
Pure RL self-play trainer for Vector-64 NNUE.

This script performs:
  1) Self-play game generation using the current value network
  2) Replay-buffer storage of (state, Monte Carlo target) samples
  3) Gradient updates from replay

It writes checkpoints compatible with tools/nnue/export_vector64.py.

Dependencies:
  - torch
  - numpy
  - python-chess
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train_vector64 import TOTAL_FEATURES, Vector64NNUE


PIECE_BUCKET = {
    1: 0,  # pawn
    2: 1,  # knight
    3: 2,  # bishop
    4: 3,  # rook
    5: 4,  # queen
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure RL self-play trainer for Vector-64 NNUE.")
    p.add_argument("--out-checkpoint", required=True, help="Final checkpoint path (.pt)")
    p.add_argument("--resume", default=None, help="Optional checkpoint to resume from")

    p.add_argument("--games", type=int, default=2000, help="Number of self-play games")
    p.add_argument("--max-game-plies", type=int, default=240, help="Truncate games after this many plies")
    p.add_argument("--gamma", type=float, default=1.0, help="Return discount factor in [0,1]")
    p.add_argument("--result-scale", type=float, default=600.0, help="Map win/loss target to centipawns")

    p.add_argument("--epsilon-start", type=float, default=0.25, help="Exploration epsilon at game 1")
    p.add_argument("--epsilon-end", type=float, default=0.05, help="Exploration epsilon at decay end")
    p.add_argument(
        "--epsilon-decay-games",
        type=int,
        default=1500,
        help="Linear epsilon decay horizon in games",
    )
    p.add_argument("--inference-batch", type=int, default=256, help="Batch size for move scoring inference")

    p.add_argument("--replay-capacity", type=int, default=250_000)
    p.add_argument("--warmup-samples", type=int, default=8192)
    p.add_argument("--updates-per-game", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")

    p.add_argument("--max-features", type=int, default=64, help="Padded feature slots per position (>=30)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--log-every-games", type=int, default=25)
    p.add_argument("--save-every-games", type=int, default=250)
    p.add_argument("--bootstrap-truncated", action="store_true", help="Bootstrap truncated games from final state")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_decay(start: float, end: float, step: int, horizon: int) -> float:
    if horizon <= 0:
        return end
    t = min(max(step, 0), horizon) / float(horizon)
    return start + (end - start) * t


def outcome_white_score(outcome) -> float:
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if bool(outcome.winner) else -1.0


def encode_halfkp(board, max_features: int):
    import chess

    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    if white_king_sq is None or black_king_sq is None:
        raise ValueError("Invalid board without both kings.")

    white_idx = np.full((max_features,), -1, dtype=np.int32)
    black_idx = np.full((max_features,), -1, dtype=np.int32)

    k = 0
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        bucket_base = PIECE_BUCKET.get(piece.piece_type)
        if bucket_base is None:
            continue
        bucket = bucket_base + (5 if piece.color == chess.BLACK else 0)
        if k >= max_features:
            raise ValueError(
                f"max_features={max_features} is too small for this position; increase --max-features."
            )

        white_idx[k] = (((white_king_sq * 10 + bucket) * 64 + sq) * 2) + 0
        black_idx[k] = (((black_king_sq * 10 + bucket) * 64 + sq) * 2) + 1
        k += 1

    stm = 0 if board.turn else 1
    return white_idx, black_idx, stm


def model_infer_cp(
    model: torch.nn.Module,
    device: torch.device,
    white_idx: np.ndarray,
    black_idx: np.ndarray,
    stm: np.ndarray,
    amp_enabled: bool,
) -> np.ndarray:
    with torch.no_grad():
        w = torch.from_numpy(white_idx).to(device, non_blocking=True).long()
        b = torch.from_numpy(black_idx).to(device, non_blocking=True).long()
        s = torch.from_numpy(stm).to(device, non_blocking=True).long()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            pred = model(w, b, s)
    return pred.detach().cpu().numpy().astype(np.float32, copy=False)


def choose_selfplay_move(
    model: torch.nn.Module,
    board,
    *,
    epsilon: float,
    max_features: int,
    result_scale: float,
    device: torch.device,
    amp_enabled: bool,
    inference_batch: int,
):
    import chess

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    if len(legal_moves) == 1 or random.random() < epsilon:
        return random.choice(legal_moves)

    stm_is_white = board.turn == chess.WHITE
    move_scores = np.zeros((len(legal_moves),), dtype=np.float32)

    pending_idx = []
    pending_w = []
    pending_b = []
    pending_stm = []

    for i, mv in enumerate(legal_moves):
        board.push(mv)
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            white_score = outcome_white_score(outcome)
            move_scores[i] = white_score * result_scale if stm_is_white else -white_score * result_scale
        else:
            w, b, s = encode_halfkp(board, max_features)
            pending_idx.append(i)
            pending_w.append(w)
            pending_b.append(b)
            pending_stm.append(s)
        board.pop()

    if pending_idx:
        w_arr = np.stack(pending_w, axis=0)
        b_arr = np.stack(pending_b, axis=0)
        s_arr = np.asarray(pending_stm, dtype=np.int64)

        all_next_values = []
        stride = max(1, inference_batch)
        for start in range(0, w_arr.shape[0], stride):
            end = min(start + stride, w_arr.shape[0])
            next_cp = model_infer_cp(
                model,
                device,
                w_arr[start:end],
                b_arr[start:end],
                s_arr[start:end],
                amp_enabled=amp_enabled,
            )
            all_next_values.append(next_cp)

        next_cp = np.concatenate(all_next_values, axis=0)
        current_cp = -next_cp
        for j, midx in enumerate(pending_idx):
            move_scores[midx] = current_cp[j]

    best = float(np.max(move_scores))
    best_indices = np.flatnonzero(move_scores >= (best - 1e-6))
    chosen = int(np.random.choice(best_indices))
    return legal_moves[chosen]


@dataclass
class GeneratedGame:
    white_idx: np.ndarray
    black_idx: np.ndarray
    stm: np.ndarray
    target_cp: np.ndarray
    white_score: float
    plies: int
    terminated: bool


def generate_selfplay_game(
    model: torch.nn.Module,
    *,
    epsilon: float,
    max_game_plies: int,
    gamma: float,
    result_scale: float,
    max_features: int,
    device: torch.device,
    amp_enabled: bool,
    inference_batch: int,
    bootstrap_truncated: bool,
) -> GeneratedGame:
    import chess

    board = chess.Board()
    states_w = []
    states_b = []
    states_stm = []

    for _ in range(max_game_plies):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            break

        w, b, s = encode_halfkp(board, max_features)
        states_w.append(w)
        states_b.append(b)
        states_stm.append(s)

        mv = choose_selfplay_move(
            model,
            board,
            epsilon=epsilon,
            max_features=max_features,
            result_scale=result_scale,
            device=device,
            amp_enabled=amp_enabled,
            inference_batch=inference_batch,
        )
        if mv is None:
            break
        board.push(mv)

    outcome = board.outcome(claim_draw=True)
    terminated = outcome is not None
    white_score = outcome_white_score(outcome)

    if (not terminated) and bootstrap_truncated and states_stm:
        w, b, s = encode_halfkp(board, max_features)
        cp = float(
            model_infer_cp(
                model,
                device,
                w.reshape(1, -1),
                b.reshape(1, -1),
                np.asarray([s], dtype=np.int64),
                amp_enabled=amp_enabled,
            )[0]
        )
        # Convert STM-centric cp estimate to white-centric score in [-1, 1].
        score_white = cp / max(result_scale, 1e-6)
        if s == 1:
            score_white = -score_white
        white_score = float(np.clip(score_white, -1.0, 1.0))

    if not states_stm:
        empty_i = np.empty((0, max_features), dtype=np.int32)
        empty_s = np.empty((0,), dtype=np.int64)
        empty_t = np.empty((0,), dtype=np.float32)
        return GeneratedGame(
            white_idx=empty_i,
            black_idx=empty_i.copy(),
            stm=empty_s,
            target_cp=empty_t,
            white_score=white_score,
            plies=0,
            terminated=terminated,
        )

    white_arr = np.stack(states_w, axis=0).astype(np.int32, copy=False)
    black_arr = np.stack(states_b, axis=0).astype(np.int32, copy=False)
    stm_arr = np.asarray(states_stm, dtype=np.int64)

    signed = np.where(stm_arr == 0, white_score, -white_score).astype(np.float32)
    if gamma != 1.0:
        # Earlier plies have larger distance to terminal reward.
        dist_to_end = np.arange(stm_arr.shape[0] - 1, -1, -1, dtype=np.float32)
        signed *= np.power(np.float32(gamma), dist_to_end)

    target_cp = signed * np.float32(result_scale)
    return GeneratedGame(
        white_idx=white_arr,
        black_idx=black_arr,
        stm=stm_arr,
        target_cp=target_cp.astype(np.float32, copy=False),
        white_score=white_score,
        plies=int(stm_arr.shape[0]),
        terminated=terminated,
    )


class ReplayBuffer:
    def __init__(self, capacity: int, max_features: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.max_features = int(max_features)
        self.white_idx = np.full((capacity, max_features), -1, dtype=np.int32)
        self.black_idx = np.full((capacity, max_features), -1, dtype=np.int32)
        self.stm = np.zeros((capacity,), dtype=np.int64)
        self.target_cp = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        white_idx: np.ndarray,
        black_idx: np.ndarray,
        stm: np.ndarray,
        target_cp: np.ndarray,
    ) -> None:
        n = int(target_cp.shape[0])
        if n <= 0:
            return

        if n >= self.capacity:
            white_idx = white_idx[-self.capacity :]
            black_idx = black_idx[-self.capacity :]
            stm = stm[-self.capacity :]
            target_cp = target_cp[-self.capacity :]
            n = self.capacity

        first = min(n, self.capacity - self.pos)
        second = n - first

        end = self.pos + first
        self.white_idx[self.pos : end] = white_idx[:first]
        self.black_idx[self.pos : end] = black_idx[:first]
        self.stm[self.pos : end] = stm[:first]
        self.target_cp[self.pos : end] = target_cp[:first]

        if second > 0:
            self.white_idx[0:second] = white_idx[first:]
            self.black_idx[0:second] = black_idx[first:]
            self.stm[0:second] = stm[first:]
            self.target_cp[0:second] = target_cp[first:]

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, device: torch.device):
        if self.size < 1:
            raise RuntimeError("Cannot sample from empty replay buffer.")
        bs = min(batch_size, self.size)
        idx = np.random.randint(0, self.size, size=bs, dtype=np.int64)
        return {
            "white_idx": torch.from_numpy(self.white_idx[idx]).to(device, non_blocking=True).long(),
            "black_idx": torch.from_numpy(self.black_idx[idx]).to(device, non_blocking=True).long(),
            "stm": torch.from_numpy(self.stm[idx]).to(device, non_blocking=True).long(),
            "target_cp": torch.from_numpy(self.target_cp[idx]).to(device, non_blocking=True),
        }


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    replay: ReplayBuffer,
    *,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    scaler: Optional[torch.amp.GradScaler],
) -> Tuple[float, float]:
    model.train(True)
    batch = replay.sample(batch_size, device)
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        pred = model(batch["white_idx"], batch["black_idx"], batch["stm"])
        loss = F.mse_loss(pred, batch["target_cp"])

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    mae = torch.mean(torch.abs(pred.detach() - batch["target_cp"])).item()
    return float(loss.detach().item()), float(mae)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    game_idx: int,
    replay_size: int,
    avg_loss: float,
    avg_mae: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": {
            "project": "Vector-64",
            "mode": "pure-rl-selfplay",
            "game": game_idx,
            "replay_size": replay_size,
            "avg_loss": avg_loss,
            "avg_mae_cp": avg_mae,
            "args": vars(args),
            "saved_at_unix": time.time(),
        },
    }
    torch.save(payload, path)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be in [0, 1]")
    if args.max_features < 30:
        raise ValueError("--max-features must be >= 30 for full chess material.")
    if args.epsilon_decay_games < 0:
        raise ValueError("--epsilon-decay-games must be >= 0")
    if not (0.0 <= args.epsilon_start <= 1.0 and 0.0 <= args.epsilon_end <= 1.0):
        raise ValueError("epsilon values must be in [0,1]")
    if args.max_game_plies <= 0:
        raise ValueError("--max-game-plies must be > 0")
    if args.games <= 0:
        raise ValueError("--games must be > 0")

    try:
        import chess  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "python-chess is required for RL self-play. Install with: pip install python-chess"
        ) from exc

    set_seed(args.seed)
    device = choose_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    model = Vector64NNUE().to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Resumed model from: {args.resume}")
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_updates = max(1, args.games * max(args.updates_per_game, 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
        eta_min=args.min_lr,
    )
    replay = ReplayBuffer(args.replay_capacity, args.max_features)

    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Device={device} amp={amp_enabled} games={args.games} "
        f"batch={args.batch_size} replay={args.replay_capacity}"
    )

    running_losses = []
    running_maes = []
    running_results = []
    running_plies = []
    started = time.time()

    for game in range(1, args.games + 1):
        epsilon = linear_decay(
            args.epsilon_start,
            args.epsilon_end,
            step=game - 1,
            horizon=args.epsilon_decay_games,
        )

        generated = generate_selfplay_game(
            model,
            epsilon=epsilon,
            max_game_plies=args.max_game_plies,
            gamma=args.gamma,
            result_scale=args.result_scale,
            max_features=args.max_features,
            device=device,
            amp_enabled=amp_enabled,
            inference_batch=args.inference_batch,
            bootstrap_truncated=args.bootstrap_truncated,
        )
        replay.add_batch(
            generated.white_idx,
            generated.black_idx,
            generated.stm,
            generated.target_cp,
        )

        running_results.append(generated.white_score)
        running_plies.append(generated.plies)

        if len(replay) >= args.warmup_samples and args.updates_per_game > 0:
            for _ in range(args.updates_per_game):
                loss, mae = train_step(
                    model,
                    optimizer,
                    scheduler,
                    replay,
                    batch_size=args.batch_size,
                    device=device,
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                )
                running_losses.append(loss)
                running_maes.append(mae)
            model.eval()

        if args.log_every_games > 0 and game % args.log_every_games == 0:
            elapsed = max(time.time() - started, 1e-6)
            gps = game / elapsed
            avg_loss = float(np.mean(running_losses[-200:])) if running_losses else float("nan")
            avg_mae = float(np.mean(running_maes[-200:])) if running_maes else float("nan")
            avg_res = float(np.mean(running_results[-200:])) if running_results else 0.0
            avg_plies = float(np.mean(running_plies[-200:])) if running_plies else 0.0
            lr = scheduler.get_last_lr()[0]
            print(
                f"[game {game:5d}/{args.games}] eps={epsilon:.3f} replay={len(replay):7d} "
                f"loss={avg_loss:9.3f} mae_cp={avg_mae:8.2f} "
                f"result_w={avg_res:+.3f} plies={avg_plies:6.1f} lr={lr:.6g} gps={gps:.2f}"
            )

        if args.save_every_games > 0 and game % args.save_every_games == 0:
            snap = out_path.with_name(f"{out_path.stem}.g{game}{out_path.suffix}")
            avg_loss = float(np.mean(running_losses[-200:])) if running_losses else float("nan")
            avg_mae = float(np.mean(running_maes[-200:])) if running_maes else float("nan")
            save_checkpoint(
                snap,
                model,
                optimizer,
                args,
                game_idx=game,
                replay_size=len(replay),
                avg_loss=avg_loss,
                avg_mae=avg_mae,
            )
            print(f"Saved snapshot: {snap}")

    final_loss = float(np.mean(running_losses[-500:])) if running_losses else float("nan")
    final_mae = float(np.mean(running_maes[-500:])) if running_maes else float("nan")
    save_checkpoint(
        out_path,
        model,
        optimizer,
        args,
        game_idx=args.games,
        replay_size=len(replay),
        avg_loss=final_loss,
        avg_mae=final_mae,
    )
    print(f"Saved final checkpoint: {out_path}")

    meta = {
        "checkpoint": str(out_path),
        "games": args.games,
        "replay_size": len(replay),
        "final_avg_loss": final_loss,
        "final_avg_mae_cp": final_mae,
        "device": str(device),
        "amp": amp_enabled,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
