import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from src import constant
from src.entity.flame import Flame
from src.entity.player import Player
from src.game import Game

_ref_game = Game(Player(*constant.BASE_PLAYER_POS))
MAP_CELLS = [{"x": c.x, "y": c.y} for c in _ref_game.map.cases.values()]
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"]
)
GAMES: dict[str, Game] = {}


def view(game: Game) -> dict:
    """View a game."""
    game_view: dict = {
        "player": {
            "x": game.player.pos_x,
            "y": game.player.pos_y,
            "hp": game.player.hp,
            "pa": game.player.pa
        },
        "bolgrot": {
            "x": game.bolgrot.pos_x,
            "y": game.bolgrot.pos_y
        },
        "next_spawn": game.spawn_pattern,
        "flames": [{"x": case.x, "y": case.y}
                   for case in game.map.cases.values()
                   if isinstance(case.entity, Flame)],
        "spells": [{
            "name": s.name,
            "cost": s.cost,
            "castable": s.is_castable(game.player),
            "uses_left": s.max_use - s.time_used
        } for s in game.player.spells],
        "waves": {
            "spawned": game.waves_spawned,
            "total": game.num_waves
        },
        "turn": game.turn,
        "done": game.done,
        "won": game.won
    }
    return game_view


@app.post("/games")
def create_game(seed: int | None = None) -> dict:
    game: Game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed)
    id_game: str = uuid4().hex
    GAMES.update({id_game: game})
    return {"game_id": id_game, "state": view(game)}


@app.get("/games/{game_id}")
def get_game(game_id: str) -> dict:
    if game_id not in GAMES:
        print("not in games")
        raise HTTPException(status_code=404, detail="game not found")
    return view(GAMES[game_id])


@app.post("/games/{game_id}/step")
def step_game(game_id: str, action_index: int) -> dict:
    game: Game = GAMES[game_id]
    reward, done = game.step(action_index=action_index)
    return {
        "state": view(game),
        "reward": reward,
        "done": done
    }


@app.get("/map")
def get_map() -> dict:
    return {
        "cells": MAP_CELLS
    }
