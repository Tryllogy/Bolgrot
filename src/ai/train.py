"""Train a NEAT population to play Bolgrot.

Run with::

    uv run python -m src.ai.train --generations 100

The best genome is pickled to ``winner.pkl`` (next to this file by default) so
``python -m src.ai.play`` can load and watch it.
"""
from __future__ import annotations
import argparse
import os
import pickle

import neat

from ..game import Game
from .policy import evaluate

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config-neat.txt")
WINNER_PATH = os.path.join(os.path.dirname(__file__), "winner.pkl")

# Fixed set of flame layouts every genome is scored against each generation.
EVAL_SEEDS = [0, 1, 2, 3, 4]


def _load_config(path: str) -> neat.Config:
    """Load the NEAT config and check it matches the game's I/O sizes."""
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        path,
    )
    obs, act = Game.observation_size(), Game.num_actions()
    if config.genome_config.num_inputs != obs:
        raise ValueError(
            f"config num_inputs={config.genome_config.num_inputs} but the "
            f"game observation is {obs} floats. Fix num_inputs in {path}.")
    if config.genome_config.num_outputs != act:
        raise ValueError(
            f"config num_outputs={config.genome_config.num_outputs} but the "
            f"game has {act} actions. Fix num_outputs in {path}.")
    return config


def eval_genomes(genomes, config: neat.Config) -> None:
    """NEAT fitness callback: score every genome on the eval seeds."""
    for _genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = evaluate(net, EVAL_SEEDS)


def train(generations: int, config_path: str, winner_path: str) -> None:
    """Run NEAT for ``generations`` and pickle the best genome."""
    config = _load_config(config_path)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(
        generation_interval=25,
        filename_prefix=os.path.join(
            os.path.dirname(winner_path), "neat-checkpoint-")))

    winner = pop.run(eval_genomes, generations)

    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nBest genome saved to {winner_path}")
    print(f"Fitness: {winner.fitness:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NEAT on Bolgrot.")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--out", default=WINNER_PATH)
    args = parser.parse_args()
    train(args.generations, args.config, args.out)


if __name__ == "__main__":
    main()
