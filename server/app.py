import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from src.game import Game
from src.entity.player import Player
from src.entity.flame import Flame
from src import constant
from fastapi import FastAPI, HTTPException
from uuid import uuid4

_ref_game = Game(Player(*constant.BASE_PLAYER_POS))
MAP_CELLS = [{"x": c.x, "y": c.y} for c in _ref_game.map.cases.values()]


# --- AZ hint worker pool ----------------------------------------------------
# The AZ search is CPU-bound pure-Python => the GIL serialises it. We run it in
# separate processes (each has its own GIL / core) so concurrent hints don't
# queue behind each other and the async event loop is never blocked.
_WORKER_NET = None   # one net copy per worker process


def _worker_init():
    """Runs once when a worker process starts: load az.pt a single time."""
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from src.ai.policy import load_policy
    global _WORKER_NET
    _WORKER_NET = load_policy("src/ai/az.pt")


def _worker_hint(state_dict: dict, sims: int) -> int:
    """Reconstruct the game from a picklable state and return the AZ best move."""
    from src.ai.alphazero import az_action_fn
    game = Game(Player(*constant.BASE_PLAYER_POS))
    game.set_state(state_dict)
    return az_action_fn(_WORKER_NET, simulations=sims)(game)


_POOL: ProcessPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _POOL
    _POOL = ProcessPoolExecutor(max_workers=4, initializer=_worker_init)
    yield
    _POOL.shutdown()


app = FastAPI(lifespan=lifespan)
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
            "uses_left": s.max_use - s.time_used,
            "sprite": s.sprite,
            "description": s.description,
            "effects": s.effects
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


@app.get("/games/{game_id}/legal")
def legal(game_id: str) -> dict:
    if game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")
    game = GAMES[game_id]
    mask = game.legal_actions()
    actions = []
    for i, ok in enumerate(mask):
        if ok:
            spell_index, (dx, dy) = Game.ACTIONS[i]
            actions.append(
                {"index": i, "spell": spell_index, "dx": dx, "dy": dy})
    return {"actions": actions}


@app.post("/games/{game_id}/hint")
async def hint(game_id: str, sims: int = 1000) -> dict:
    if game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")
    game = GAMES[game_id]
    state_dict = game.get_state()
    loop = asyncio.get_running_loop()
    index = await loop.run_in_executor(_POOL, _worker_hint, state_dict, sims)
    spell_index, (dx, dy) = Game.ACTIONS[index]
    return {"index": index, "spell": spell_index, "dx": dx, "dy": dy}
