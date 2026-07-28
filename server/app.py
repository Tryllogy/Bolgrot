import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import asyncio
import random
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from src.game import Game
from src.entity.player import Player
from src.entity.flame import Flame
from src.case.case import CaseType
from src import constant
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4

_ref_game = Game(Player(*constant.BASE_PLAYER_POS))
MAP_CELLS = [{"x": c.x, "y": c.y} for c in _ref_game.map.cases.values()]


# --- AZ hint worker pool ----------------------------------------------------
# The AZ search is CPU-bound pure-Python => the GIL serialises it. We run it in
# separate processes (each has its own GIL / core) so concurrent hints don't
# queue behind each other and the async event loop is never blocked.
# Two hint engines the frontend can pick between: "az" — the flat MLP az.pt,
# fast (a hint's cost is dominated by single-threaded Python game-stepping, and
# the MLP forward is tiny); "cnn" — the grid CNN az_cnn_final, stronger per sim
# but ~4x slower in prod (the heavier conv can't hide behind the GIL-bound
# stepping and doesn't parallelise). Each worker holds BOTH so a request just
# names one. Paths overridable by env for a future swap.
_ENGINE_PATHS = {
    "az": os.environ.get("BOLGROT_HINT_NET", "src/ai/az.pt"),
    "cnn": os.environ.get("BOLGROT_HINT_NET_CNN", "src/ai/az_cnn_final.pt"),
}
_WORKER_NETS: dict = {}   # {engine: net} — one copy per worker process


def _worker_init():
    """Runs once when a worker process starts: load both hint nets a single time."""
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from src.ai.policy import load_policy
    global _WORKER_NETS
    _WORKER_NETS = {k: load_policy(p) for k, p in _ENGINE_PATHS.items()}


def _worker_hint(state_dict: dict, num_waves: int, sims: int,
                 engine: str = "az") -> int:
    """Reconstruct the game from a picklable state and return the AZ best move.

    ``num_waves`` is passed explicitly because ``get_state`` does not carry it;
    a custom puzzle uses ``num_waves=1`` and the worker must match, or the
    search would keep spawning waves and never see the win. The obs layout MUST
    match the chosen net (a CNN needs the 2D grid, the MLP the flat vector), so
    we set ``constant.OBS_MODE`` from the net's own ``net_type`` before the
    search — safe because a worker runs one task at a time (no OBS_MODE race).
    """
    from src.ai.alphazero import az_action_fn
    from src import constant
    net = _WORKER_NETS.get(engine) or _WORKER_NETS["az"]
    constant.OBS_MODE = (
        "grid" if getattr(net, "net_type", "mlp") == "cnn" else "flat")
    game = Game(Player(*constant.BASE_PLAYER_POS), num_waves=num_waves)
    game.set_state(state_dict)
    return az_action_fn(net, simulations=sims)(game)


_POOL: ProcessPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _POOL
    # Un uvicorn worker suffit : le parallélisme des recherches AZ (CPU-bound)
    # vient de ce pool de process, pas des workers uvicorn (qui, eux,
    # multiplieraient les copies du réseau en RAM). Taille réglable en prod.
    workers = int(os.environ.get("AZ_POOL_WORKERS", "4"))
    _POOL = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
    yield
    _POOL.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"]
)
# --- Sessions : source de vérité rejouable + cache mémoire ------------------
# Le jeu est DÉTERMINISTE : une partie est entièrement définie par sa seed et la
# suite ordonnée d'événements (coups joués + vagues dessinées en mode perso).
# On stocke donc ça (`Session`, entièrement JSON-sérialisable) comme source de
# vérité, et on garde à côté un cache d'objets `Game` reconstruits par rejeu. Le
# cache peut être vidé/évincé à tout moment (redémarrage, autre process) sans
# rien perdre : `_get_game` rejoue la session à la demande.


@dataclass
class Session:
    seed: int
    is_custom: bool = False
    initial_spawns: list | None = None   # perso : tuiles de la 1re vague
    # événements ordonnés : ["step", action_index] | ["spawn", [[x, y], ...]]
    events: list = field(default_factory=list)


SESSIONS: dict[str, Session] = {}
_CACHE: dict[str, Game] = {}


def _build_game(s: Session) -> Game:
    """Reconstruit un Game depuis sa session en rejouant ses événements."""
    game = Game(Player(*constant.BASE_PLAYER_POS), seed=s.seed)
    if s.is_custom:
        game.waves_spawned = 0
        game.spawn_pattern = [tuple(t) for t in (s.initial_spawns or [])]
    for kind, payload in s.events:
        if kind == "step":
            game.step(action_index=payload)
        else:  # "spawn" : override de la vague suivante (mode perso)
            game.spawn_pattern = [tuple(t) for t in payload]
    return game


def _get_game(game_id: str) -> Game:
    """Game depuis le cache, reconstruit par rejeu de la session si absent."""
    if game_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="game not found")
    game = _CACHE.get(game_id)
    if game is None:
        game = _build_game(SESSIONS[game_id])
        _CACHE[game_id] = game
    return game


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
    # Seed concrète obligatoire pour rejouer (None => non reproductible) : on la
    # tire côté serveur si le client n'en fournit pas.
    if seed is None:
        seed = random.randrange(2 ** 63)
    id_game: str = uuid4().hex
    SESSIONS[id_game] = Session(seed=seed)
    game = _build_game(SESSIONS[id_game])  # live == chemin de rejeu (cohérence)
    _CACHE[id_game] = game
    return {"game_id": id_game, "state": view(game)}


@app.get("/games/{game_id}")
def get_game(game_id: str) -> dict:
    return view(_get_game(game_id))


@app.post("/games/{game_id}/step")
def step_game(game_id: str, action_index: int) -> dict:
    game = _get_game(game_id)
    reward, done = game.step(action_index=action_index)
    SESSIONS[game_id].events.append(["step", action_index])
    return {
        "state": view(game),
        "reward": reward,
        "done": done
    }


@app.get("/map")
def get_map() -> dict:
    return {
        "cells": MAP_CELLS,
        "player": list(constant.BASE_PLAYER_POS),
        "bolgrot": list(constant.BASE_BOLGROT_POS)
    }


def _validate_tiles(game: Game, tiles: list[tuple[int, int]]) \
        -> list[tuple[int, int]]:
    """Return the tiles if each is a free, unoccupied map cell, else 400."""
    out = []
    for x, y in tiles:
        case = game.map.cases.get((x, y))
        if case is None or case.case_type != CaseType.FREE \
                or case.entity is not None:
            raise HTTPException(
                status_code=400, detail=f"tuile invalide: {x},{y}")
        out.append((x, y))
    return out


class CustomGame(BaseModel):
    spawns: list[tuple[int, int]]


@app.post("/games/custom")
def create_custom_game(body: CustomGame) -> dict:
    # Full 6-wave game whose waves the PLAYER designs. This creates it with the
    # first wave's spawn points; the frontend sets each subsequent wave via
    # POST /games/{id}/spawn after every end-turn (num_waves stays the default).
    # Une seed concrète est tirée pour que la session soit rejouable (les vagues
    # étant toutes dessinées par le joueur, elle ne change rien au jeu, mais
    # rend tout tirage RNG résiduel reproductible).
    id_game = uuid4().hex
    seed = random.randrange(2 ** 63)
    probe = Game(Player(*constant.BASE_PLAYER_POS), seed=seed)
    spawns = _validate_tiles(probe, body.spawns)
    SESSIONS[id_game] = Session(
        seed=seed, is_custom=True,
        initial_spawns=[list(t) for t in spawns])
    game = _build_game(SESSIONS[id_game])
    _CACHE[id_game] = game
    return {"game_id": id_game, "state": view(game)}


class SpawnTiles(BaseModel):
    tiles: list[tuple[int, int]]


@app.post("/games/{game_id}/spawn")
def set_spawn(game_id: str, body: SpawnTiles) -> dict:
    # Override the next wave's spawn pattern with the player's chosen tiles
    # (used by the custom mode between turns; the auto-generated random wave is
    # discarded). Rejects tiles already occupied by the player / a flame.
    game = _get_game(game_id)
    tiles = _validate_tiles(game, body.tiles)
    game.spawn_pattern = tiles
    SESSIONS[game_id].events.append(["spawn", [list(t) for t in tiles]])
    return {"state": view(game)}


@app.get("/games/{game_id}/legal")
def legal(game_id: str) -> dict:
    game = _get_game(game_id)
    mask = game.legal_actions()
    actions = []
    for i, ok in enumerate(mask):
        if ok:
            spell_index, (dx, dy) = Game.ACTIONS[i]
            actions.append(
                {"index": i, "spell": spell_index, "dx": dx, "dy": dy})
    return {"actions": actions}


@app.post("/games/{game_id}/hint")
async def hint(game_id: str, sims: int = 1000, engine: str = "az") -> dict:
    game = _get_game(game_id)
    state_dict = game.get_state()
    loop = asyncio.get_running_loop()
    index = await loop.run_in_executor(
        _POOL, _worker_hint, state_dict, game.num_waves, sims, engine)
    spell_index, (dx, dy) = Game.ACTIONS[index]
    return {"index": index, "spell": spell_index, "dx": dx, "dy": dy}


# --- Front statique ---------------------------------------------------------
# Le front buildé (`web/dist`, via `npm run build`) est servi par FastAPI lui-
# même : une seule origine en prod, plus besoin de CORS. Ce mount est monté en
# DERNIER pour que toutes les routes API ci-dessus aient la priorité, et il est
# conditionnel pour que le serveur démarre en dev même sans build.
_DIST = os.path.join(os.path.dirname(__file__), "..", "web", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
