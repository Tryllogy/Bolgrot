# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install      # install dependencies via uv sync
make run          # install + run the game (uv run python -m src)
make train        # train the PPO agent (make train ITERS=500 to set iterations)
make resume       # continue training the last checkpoint (make resume ITERS=300)
make play         # watch the trained policy play
make clean        # remove __pycache__, build artifacts

uv run python -m src                                    # run the game directly
uv run --extra ai python -m src.ai.train               # train PPO (see src/ai/)
uv run --extra ai python -m src.ai.train --resume      # continue from checkpoint
uv run --extra ai python -m src.ai.play                # watch the trained policy
```

Dependencies are managed with `uv` and locked in `uv.lock`. The project uses a `.venv` local virtual environment. The PPO agent lives behind the optional `ai` dependency group (`torch`), installed with `uv sync --extra ai` (the `make train`/`make play` targets do this for you). There are no tests yet; `pytest` is listed as a dev dependency for future use.

## Architecture

This is a pygame turn-based isometric grid game. The player tries to survive on a diamond-shaped map while Bolgrot (the boss enemy) and flame tiles threaten them. Each turn lasts 120 seconds; at the end of a turn, a new flame spawn pattern is placed on the map.

### Module responsibilities

- `src/game.py` — `Game` class: owns all game state and exposes both the interactive action API (`select_spell`, `play_selected_spell`, `clear_previsu`, `end_turn`, `reset`) and the **RL environment API** (`get_observation`, `legal_actions`, `step`, plus the `ACTIONS` table and the `observation_size()`/`num_actions()` static helpers). **No pygame dependency in its own logic** — it can be imported and driven headlessly (set `SDL_VIDEODRIVER=dummy` since it transitively imports pygame via `case`/`spells`).
- `src/renderer.py` — `Renderer` class: holds rendering state (screen offset, end-turn button, spell render list). All drawing methods receive game state as arguments and never mutate it. `_compute_map_offset()` (static method) calculates the isometric origin that centers the map on screen.
- `src/__main__.py` — pygame event loop only. Translates user input into `Game` method calls; passes game state to `Renderer` methods. Keyboard: `SPACE` ends the turn; `1`/`2`/`3` select the player's spell at index 0/1/2 (via `game.select_spell(event.key - pygame.K_1)`); `ESC` quits.
- `src/constant.py` — all magic numbers including layout constants (`RIGHT_PANEL_W`, `BUTTON_W`, `BUTTON_H`, `SPELL_GAP`) and map config path (`MAP_CONF`).
- `src/actions.py` — stateless click/hover helpers: `on_previsu_click()`, `on_spell_hover()`, `on_button_end_turn_click()`.
- `src/BFS/bfs.py` — `BFS` class: 4-directional breadth-first pathfinding over `Map.cases` (walls excluded). `find_path(start, goal)` returns the **single next step** toward the goal (one coordinate tuple) or `None`. Each `Spells` instance holds a `BFS` (assigned in `Game.__init__`); used as the fallback movement when the greedy quadrant step is blocked.
- `src/case/case.py` — `Case` class (grid cell with `x`, `y`, `entity`, `case_type`) and `CaseType` enum (`FREE`, `WALL`, `EMPTY`). `Case.draw()` handles isometric rendering and `Case.contains()` does mouse hit-testing.
- `src/button/button.py` — `Button` class for the end-turn button.

### Map representation

`Map.cases` is a `dict[tuple[int, int], Case]` keyed by `(x, y)` — lookups (`Map.place_entity`, `Spells._find_case`, `BFS.is_valid_position`) are O(1) `cases.get(pos)`. Iterate values with `for case in cases.values()`. Each `Case` holds its grid coordinates (`x`, `y`), an optional `entity: Entity | None`, and a `case_type: CaseType`.

The map is loaded from the packaged `src/config/bolgrot.map` (path in `constant.MAP_CONF`) by `Map.parse_map()`. The file uses a symbol grid: `.` = FREE, `#` = WALL, `|` = EMPTY, `N` = skip (no cell). After parsing, `self.grid_max_x` and `self.grid_max_y` store the bounds.

### Packaging & resource paths

`src` is a regular package (`src/__init__.py`). Data files (`src/config/*.map`, `src/sprites_png/*.png`) are declared as `[tool.setuptools.package-data]` and resolved at runtime via `importlib.resources.files("src") / ...` in `constant.MAP_CONF` and `constant.SPRITES_DIR` — so they work regardless of the current working directory (not just from the repo root). The game has a console-script entry point `bolgrot = "src.__main__:main"`; `__main__.py` exposes `main()` (called both by the entry point and the `if __name__ == "__main__"` guard for `python -m src`).

### Coordinate system

Iso conversion is done inline wherever needed:
- `iso_x = (x - y) * (CASE_WIDTH / 2)`
- `iso_y = (x + y) * (CASE_HEIGHT / 2)`

All rendering uses `offset: tuple[int, int]` — the screen pixel position of iso origin (0,0), stored on the `Renderer` instance. The inverse (mouse → grid) is computed in `Case.contains()` and in `on_previsu_click()` in `actions.py`.

### Entity hierarchy

```
Entity (ABC)
├── blocks_sight: bool = True   (Player overrides to False)
├── killable: bool = True        (Bolgrot overrides to False)
├── Player   — pos_x/y, hp, base_PA, pa (current AP), list[Spells]
├── Bolgrot  — pos_x/y, killable=False
└── Flame    — pos_x/y
```

### Spell system

```
Spells (ABC)
├── ShortJump  ("Astral leap")  — LINE range 1, implemented play()
├── LongJump   ("Double leap")  — LINE range 2, implemented play(), tracks time_used
└── MoveFlames ("Inaction")     — DIAGONAL range 1, line_of_sight=False, attracts all flames toward the player
```

The `Spells.__init__` signature takes a `bfs: BFS` argument (after `type_spell`, before `line_of_sight`). `Game.__init__` builds one `BFS(self.map)` and assigns it to every spell.

Each spell defines `type_spell: list[tuple[TypeSpell, int]]` where `TypeSpell` is LINE/DIAGONAL/FULL and the int is range. `Spells.previsu()` computes reachable tiles using `is_in_map()`, `is_blocked_by_sight()`, and `is_entity_killable()` — line-of-sight IS enforced in `previsu()` via the `line_of_sight` bool. All three concrete spells have implemented `play()` methods.

Spells also carry `cost` (AP cost), `max_use`, `time_used` (base-class attr, default 0), `effects: list[str]`, and render a tooltip on hover via `_draw_tooltip()`. `Spells.is_castable(player)` returns `player.pa >= cost and time_used < max_use`. `Game.select_spell()` calls it first and refuses to build a previsualisation (clearing it) for an uncastable spell, so out-of-AP / maxed-out spells can't be selected, hovered into a previsu, or cast.

**Flame attraction** (`Spells.attract_flames`, shared by all three spells' `play()`): flames are sorted nearest-first (Manhattan) and each moves **one** tile toward the player along the axis of greatest distance — `|dx|>|dy|` steps in x, `|dy|>|dx|` steps in y, equal steps diagonally. This keeps each flame inside its quadrant relative to the player. If the chosen tile is a WALL or off-map, it falls back to `BFS.find_path`. A flame landing on the player or another flame sets `player.hp = 0`; Bolgrot blocks (flame stays). `push_flames` is currently a stub (`pass`).

**Dependency layering (acyclic).** Imports form a clean DAG: `constant` → `entity` → `case`/`map` → `BFS` → `spells` → `game`. `Player` is spell-agnostic — it holds `self.spells: list[Spells]` (empty by default; the `Spells` type is a `TYPE_CHECKING`-only import) and does **not** import `spells` or `BFS`. `Game.__init__` constructs the concrete spells (`ShortJump`/`LongJump`/`MoveFlames`) with a shared `BFS(self.map)` and assigns them to `player.spells`. Because `entity` no longer depends on the higher layers, every module imports cleanly regardless of entry point; `spells` modules import entity/case classes normally at module top level.

### Flame spawn patterns

`Patterns(seed=None)` holds `_all_patterns` (class-level list of patterns); each pattern is a list of grid coordinates. `Patterns.draw()` picks one at random (without replacement) using a seeded `random.Random` instance, making runs reproducible. Each turn, the drawn pattern is passed to `Game.end_turn()` which calls `Map.place_entity()` per position. `Patterns.reset()` restores the full pool.

### Sprite assets

Sprites are stored as both `.xcf` (GIMP source) and `.png` in `src/sprites_png/`. Spell classes reference `.png` filenames in their `sprite` field. Images are lazy-loaded on first access via the `image` property.

### RL environment / PPO agent

`src/ai/` (optional `ai` dependency group, `torch`) trains a **PPO** agent to play the game headlessly. It drives `Game` purely through the environment API below — no game logic lives in `src/ai/`.

**Environment API on `Game`** (all in `src/game.py`):
- `get_observation() -> list[float]` — fixed-length egocentric vector. Layout: 11 scalars `[px, py, hp, pa, waves, flame_count, boss_dx, boss_dy, castable_0..2]`, then the `K_NEAREST_FLAMES` nearest flames and `K_NEAREST_SPAWN` nearest next-spawn tiles, each `(present, dx, dy)`, all normalised to ~`[-1, 1]`. Length is `observation_size()` (currently **53**). `K_NEAREST_FLAMES`/`K_NEAREST_SPAWN` live in `constant.py`.
- `ACTIONS` (module-level `_ACTIONS`, aliased as `Game.ACTIONS`) — the fixed discrete action space, **17** entries. Each is `(spell_index | None, (dx, dy))`; the target tile is the player's position plus the offset. Because every spell targets one of 4 orthogonal / 4 diagonal directions at range 1–2, the space collapses to ShortJump×4 + LongJump×8 (range 1 and 2) + MoveFlames×4 + one end-turn (`(None, (0, 0))`, the last index). This makes the target position-independent so the policy never has to output an arbitrary tile.
- `legal_actions() -> list[bool]` — mask over `ACTIONS`. End-turn is always legal while live; a spell action is legal only when the spell `is_castable` **and** its target tile is in the spell's `previsu()`. All-illegal once `done`.
- `step(action_index) -> (reward, done)` — applies one action. The reward is shaped so **active play beats sitting still** (all weights are `REWARD_*` constants in `constant.py`): clearing a flame is the dominant term (`REWARD_CLEAR_FLAME`, default `2.0`); surviving a turn pays only a token `REWARD_SURVIVE_TURN` (`0.1`) so "always end turn" is *not* optimal; `REWARD_HP_DELTA` (`0.02 × ΔHP`) mildly discourages waste; terminal `REWARD_WIN` (`+100`) / `REWARD_DEATH` (`-10`); illegal actions are no-ops (`REWARD_ILLEGAL`, masked out in practice). **Only flame *reductions* are rewarded** — the wave spawned on an end-turn adds flames the agent cannot prevent, so `flames_before - flames_after` is clamped to positive (`if cleared > 0`). Episodes always terminate because HP/PA are finite (every spell consumes both). Note: every non-winning game ends in death (HP drains 1/turn after all waves spawn), so the passive "end turn forever" strategy scores ~`-6.3`, below any policy that safely clears flames.

**`src/ai/` modules** (nothing here contains game logic; keep `__init__.py` import-free so the base game never needs `torch`):
- `network.py` — `ActorCritic`, a small shared-trunk MLP (`obs_size → 64 → 64 → {policy logits, value}`, orthogonal init). Illegal-action masking is centralised here: `get_action_and_value(obs, mask, action=None)` fills masked logits with `MASK_FILL` (`-1e8`) before the `Categorical`, so the policy can only pick legal moves. Rebuild dims from `Game.observation_size()` / `Game.num_actions()`.
- `policy.py` — env↔net glue: `choose_action(net, game, deterministic)` (greedy argmax over masked logits, or sampled), `run_episode`/`evaluate` (mean reward + win rate over the fixed `EVAL_SEEDS`), and `save_policy`/`load_policy` (a checkpoint stores the state dict plus `obs_size`/`num_actions`/`hidden` so it rebuilds standalone). `MAX_STEPS` is a safety cap.
- `ppo.py` — the algorithm. `PPOConfig` holds hyperparameters (CleanRL-style defaults). `_VecGames` steps `num_envs` independent games in lockstep with auto-reset. `_compute_gae` does GAE over the `[num_steps, num_envs]` rollout; `train(cfg)` runs rollout → GAE → clipped-surrogate + value + entropy update, and saves the best-by-eval policy to `cfg.out_path`. **The env is not a `gym` env** — it's driven directly through `get_observation`/`legal_actions`/`step`, and the legal-action mask is stored per step so the PPO update re-masks identically.
- `train.py` — `python -m src.ai.train [--iterations N] [--num-envs N] [--num-steps N] [--lr] [--seed] [--out ppo.pt] [--checkpoint ppo_ckpt.pt] [--resume [PATH]]`. Saves the best-by-eval **policy** to `src/ai/ppo.pt` and a resumable **training checkpoint** (weights + optimizer + `iteration`/`global_step`/`best_score`) to `src/ai/ppo_ckpt.pt` (both git-ignored). `--iterations` is *per session*; `--resume` restores the checkpoint and runs that many more iterations (each session anneals its own LR — a warm restart), continuing the step/iteration counters and carrying `best_score` so a worse policy never overwrites the saved best. Bare `--resume` uses the `--checkpoint` path.
- `play.py` — `python -m src.ai.play [--policy ppo.pt] [--seed S] [--delay MS]`. Loads a policy and drives the game through the real `Renderer` (needs a display).

**If you change the observation layout or `ACTIONS`, no config edit is needed** — the network sizes itself from `observation_size()`/`num_actions()` at construction, but existing `ppo.pt` checkpoints become incompatible and must be retrained.
