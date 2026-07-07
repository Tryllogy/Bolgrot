"""Watch a trained NEAT genome play, using the normal game renderer.

Run with::

    uv run python -m src.ai.play --delay 400

Loads ``winner.pkl`` (produced by ``train.py``), rebuilds its network and lets
it drive the game while rendering each step so you can see the strategy.
"""
from __future__ import annotations
import argparse
import os
import pickle

import neat
import pygame

from ..entity import Player
from .. import constant
from ..game import Game
from ..renderer import Renderer
from .policy import choose_action
from .train import CONFIG_PATH, WINNER_PATH, _load_config


def watch(
    genome_path: str, config_path: str, seed: int, delay_ms: int,
) -> None:
    """Render one episode driven by the genome at ``genome_path``."""
    config = _load_config(config_path)
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    font_title = pygame.font.Font(None, 50)
    font_txt = pygame.font.Font(None, 25)

    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed)
    renderer = Renderer(screen, font_title, font_txt, game.map.cases)

    running = True
    last_step = pygame.time.get_ticks()
    while running:
        screen.fill((0, 0, 0))
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE):
                running = False

        now = pygame.time.get_ticks()
        if not game.done and now - last_step >= delay_ms:
            game.step(choose_action(net, game))
            last_step = now

        renderer.draw_map(mouse_x, mouse_y, game.map.cases,
                          game.previsualiation, game.spawn_pattern)
        renderer.draw_entities(game.map.cases)
        renderer.draw_spells(mouse_x, mouse_y, game.player.spells)
        renderer.draw_hp_player(game.player)
        renderer.draw_ap_player(game.player)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Victory: all flames cleared!" if game.won
          else f"Game over — waves survived: {game.waves_spawned}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a NEAT genome play.")
    parser.add_argument("--genome", default=WINNER_PATH)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=int, default=400,
                        help="milliseconds between AI moves")
    args = parser.parse_args()
    if not os.path.exists(args.genome):
        raise SystemExit(
            f"No genome at {args.genome}. Run `python -m src.ai.train` first.")
    watch(args.genome, args.config, args.seed, args.delay)


if __name__ == "__main__":
    main()
