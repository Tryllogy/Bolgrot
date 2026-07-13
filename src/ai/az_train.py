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
                        choices=("shaped", "winloss", "blend", "twohead"),
                        default="shaped",
                        help="value-head target: 'shaped' (discounted shaped "
                             "return, the historical recipe), 'winloss' "
                             "(discounted game outcome ±1 — collapses on the "
                             "full game, sparse), 'blend' (dense shaped "
                             "return + a large terminal ±bonus — avoids the "
                             "collapse but 0/6, the terminal spike poisons the "
                             "value near death), or 'twohead' (separate shaped "
                             "+ P(win) value heads — the ranking without the "
                             "spike; needs a fresh run, value_dim=2). Use a "
                             "distinct --out/--checkpoint so a run never "
                             "clobbers az.pt; a checkpoint stores no mode, so "
                             "--resume it with the same --value-target.")
    parser.add_argument("--reward-win", type=float, default=None,
                        help="override constant.REWARD_WIN for this run (a "
                             "FULL retrain — use a distinct --out/--checkpoint, "
                             "and benchmark with the same value on "
                             "`baselines --reward-win`). Raising it (e.g. 800) "
                             "lifts a win's *discounted* return-to-go clear of "
                             "the near-win band that shaped's value target "
                             "cannot separate (the +100 win, discounted over "
                             "~60 plies, is only ~+55 in an early state and is "
                             "swamped by clear-count noise). It is the "
                             "reward-side version of the ranking fix, and "
                             "asymmetric (REWARD_DEATH untouched), so unlike "
                             "'blend' it does not poison the value near a "
                             "death. Do NOT also raise REWARD_CLEAR_FLAME: it "
                             "widens the very near-win band this fights.")
    parser.add_argument("--value-scale", type=float, default=None,
                        help="override alphazero.VALUE_SCALE (default 100). "
                             "MUST scale with --reward-win: the value target "
                             "is divided by VALUE_SCALE to keep the value loss "
                             "O(1), so a larger REWARD_WIN needs a proportional "
                             "VALUE_SCALE (e.g. --reward-win 800 --value-scale "
                             "800) or the value gradient explodes and crushes "
                             "the policy head (p_loss rises, prior de-sharpens). "
                             "It is pure normalisation — it does NOT change the "
                             "win/clear ranking, which lives in the raw reward.")
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

    # Reward-shaping override. game.step reads constant.REWARD_WIN dynamically,
    # so setting it here (before any Game is built) applies to self-play, the
    # MCTS bootstrap and evaluation alike. A distinct value = a distinct reward
    # scale = a full retrain; never point --out/--checkpoint at az.pt.
    if args.reward_win is not None:
        from .. import constant
        constant.REWARD_WIN = args.reward_win
    # VALUE_SCALE is a module global read at call time in alphazero's search /
    # update, so overriding it here applies everywhere. Pair it with
    # --reward-win to keep the (now larger) value target O(1).
    if args.value_scale is not None:
        from . import alphazero
        alphazero.VALUE_SCALE = args.value_scale

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
    rwin = "" if args.reward_win is None else f", REWARD_WIN={args.reward_win}"
    if args.value_scale is not None:
        rwin += f", VALUE_SCALE={args.value_scale}"
    print(f"Training AlphaZero ({mode}): {cfg.iterations} iters, "
          f"{cfg.games_per_iter} games/iter x {cfg.simulations} sims/move, "
          f"value-target={cfg.value_mode}{rwin}, device={cfg.device}",
          flush=True)
    train(cfg)
    print(f"Done. Best policy: {args.out} | checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
