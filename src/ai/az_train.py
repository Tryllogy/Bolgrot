"""Train an AlphaZero-style agent (PUCT self-play) to play the game.

Run with::

    uv run --extra ai python -m src.ai.az_train --iterations 100 --curriculum

Self-play uses the network's policy prior and value head inside a PUCT search
(:mod:`src.ai.alphazero`), then regresses the net onto the search's visit
distributions and the games' return-to-go. The best-by-eval policy is saved to
``src/ai/az.pt`` (loadable by ``play``/``baselines`` like any policy) and a
resumable checkpoint to ``src/ai/az_ckpt.pt`` (``--resume`` to continue).

Watch it afterwards with ``python -m src.ai.play --policy src/ai/az.pt``
(slower than PPO — each move runs a search).
"""
from __future__ import annotations
import argparse
import os

import torch

from .alphazero import AZConfig, train
from .train import AZ_CURRICULUM

POLICY_PATH = os.path.join(os.path.dirname(__file__), "az.pt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "az_ckpt.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AlphaZero agent.")
    parser.add_argument("--iterations", type=int, default=100,
                        help="self-play + update cycles this session")
    parser.add_argument("--games-per-iter", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=80,
                        help="PUCT sims per move during self-play")
    parser.add_argument("--eval-simulations", type=int, default=120)
    parser.add_argument("--bootstrap-games", type=int, default=0,
                        help="seed the replay buffer with this many MCTS games "
                             "before training (gives the net winning lines to "
                             "imitate; self-play never wins the full game)")
    parser.add_argument("--bootstrap-simulations", type=int, default=300,
                        help="MCTS sims/move for the bootstrap games")
    parser.add_argument("--value-target",
                        choices=("shaped", "winloss", "blend"),
                        default="shaped",
                        help="value-head target: 'shaped' (discounted shaped "
                             "return, the historical recipe), 'winloss' "
                             "(discounted game outcome ±1 — collapses on the "
                             "full game, sparse), or 'blend' (dense shaped "
                             "return + a large terminal ±bonus, the A/B aimed "
                             "at the ~50%% plateau). Use a distinct "
                             "--out/--checkpoint so a run never clobbers az.pt; "
                             "a checkpoint stores no mode, so --resume it with "
                             "the same --value-target.")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64,
                        help="MLP hidden width (raise to 128 for more "
                             "capacity; a new width needs a fresh run — it "
                             "cannot --resume a differently-sized checkpoint)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--out", default=POLICY_PATH,
                        help="where to save the best-by-eval policy")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="where to save the resumable training state")
    parser.add_argument(
        "--resume", nargs="?", const="", default=None,
        help="continue from a checkpoint (defaults to --checkpoint)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--curriculum", action="store_true",
                        help="ramp difficulty up to the full game, promoting "
                             "a stage when eval win rate hits --promote-win")
    parser.add_argument("--promote-win", type=float, default=0.6)
    args = parser.parse_args()

    if args.resume is None:
        resume_from = ""
    elif args.resume == "":
        resume_from = args.checkpoint
    else:
        resume_from = args.resume

    cfg = AZConfig(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        simulations=args.simulations,
        eval_simulations=args.eval_simulations,
        bootstrap_games=args.bootstrap_games,
        bootstrap_simulations=args.bootstrap_simulations,
        c_puct=args.c_puct,
        learning_rate=args.lr,
        value_mode=args.value_target,
        hidden=args.hidden,
        seed=args.seed,
        eval_every=args.eval_every,
        out_path=args.out,
        checkpoint_path=args.checkpoint,
        resume_from=resume_from,
        device=args.device,
        curriculum=AZ_CURRICULUM if args.curriculum else (),
        promote_win_rate=args.promote_win,
    )
    mode = f"resuming from {resume_from}" if resume_from else "fresh run"
    print(f"Training AlphaZero ({mode}): {cfg.iterations} iters, "
          f"{cfg.games_per_iter} games/iter x {cfg.simulations} sims/move, "
          f"value-target={cfg.value_mode}, device={cfg.device}", flush=True)
    train(cfg)
    print(f"Done. Best policy: {args.out} | checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
