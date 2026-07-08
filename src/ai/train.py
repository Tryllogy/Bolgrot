"""Train a PPO agent to play the game headlessly.

Run with::

    uv run --extra ai python -m src.ai.train --iterations 300

The best policy (by evaluation win rate, then mean reward) is saved to
``src/ai/ppo.pt`` as it improves, so an interrupted run still leaves the best
policy on disk. A resumable training checkpoint (weights + optimizer + progress)
is saved to ``src/ai/ppo_ckpt.pt``; pass ``--resume`` to continue from it::

    uv run --extra ai python -m src.ai.train --iterations 300 --resume

Watch a trained policy afterwards with ``python -m src.ai.play``.
"""
from __future__ import annotations
import argparse
import os

import torch

from .ppo import PPOConfig, train

# Defaults next to this file so they are found regardless of CWD.
POLICY_PATH = os.path.join(os.path.dirname(__file__), "ppo.pt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "ppo_ckpt.pt")

# Default difficulty ramp for --curriculum: (num_waves, flames_per_wave,
# spawn_radius) stages from a trivially winnable start (1 flame spawned right
# next to the player) up to the full game (last stage: all None = the true
# spawn patterns and wave count). Flames spawn within `spawn_radius` of the
# player early on because the real difficulty is spawn *distance*, not count.
# The agent is promoted a stage each time it reliably wins the current one, so
# PPO always has a reachable win to learn from.
DEFAULT_CURRICULUM = (
    (1, 1, 2),          # trivial: one flame at the player's elbow
    (1, 1, 5),          # one flame, farther (~11% winnable by random)
    (1, 2, 4),          # two flames in one wave (~2%)
    (1, 3, 5),          # three flames in one wave (~0.3%)
    (2, 2, 6),          # a second wave: flames start to accumulate
    (3, 3, 8),          # more waves + flames
    (6, None, None),    # the full game: true patterns, all waves
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument("--iterations", type=int, default=300,
                        help="rollout+update cycles to run this session")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128,
                        help="env steps collected per game per rollout")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--out", default=POLICY_PATH,
                        help="where to save the best-by-eval policy")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="where to save the resumable training state")
    parser.add_argument(
        "--resume", nargs="?", const="", default=None,
        help="continue training from a checkpoint (defaults to --checkpoint "
             "when given without a path)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument(
        "--curriculum", action="store_true",
        help="ramp difficulty from a tiny winnable game up to the full one, "
             "promoting a stage each time eval win rate hits --promote-win")
    parser.add_argument("--promote-win", type=float, default=0.7,
                        help="eval win rate needed to advance a curriculum "
                             "stage")
    args = parser.parse_args()

    # Resolve --resume: absent -> no resume; bare flag -> the --checkpoint path;
    # explicit value -> that path.
    if args.resume is None:
        resume_from = ""
    elif args.resume == "":
        resume_from = args.checkpoint
    else:
        resume_from = args.resume

    cfg = PPOConfig(
        iterations=args.iterations,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        seed=args.seed,
        eval_every=args.eval_every,
        out_path=args.out,
        checkpoint_path=args.checkpoint,
        resume_from=resume_from,
        device=args.device,
        curriculum=DEFAULT_CURRICULUM if args.curriculum else (),
        promote_win_rate=args.promote_win,
    )
    mode = f"resuming from {resume_from}" if resume_from else "fresh run"
    print(f"Training PPO ({mode}): {cfg.iterations} iters, {cfg.num_envs} "
          f"envs x {cfg.num_steps} steps = {cfg.batch_size} samples/rollout, "
          f"device={cfg.device}", flush=True)
    train(cfg)
    print(f"Done. Best policy: {args.out} | resumable checkpoint: "
          f"{args.checkpoint}")


if __name__ == "__main__":
    main()
