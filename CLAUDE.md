# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install      # install dependencies via uv sync
make run          # install + run the game (uv run python -m src)
make train        # train the PPO agent (make train ITERS=500 to set iterations)
make curriculum   # train with the difficulty-ramp curriculum (make curriculum PROMOTE=0.8)
make resume       # continue training the last checkpoint (make resume ITERS=300)
make curriculum-resume  # resume a curriculum run (restores the stage reached)
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

**Flame attraction** (`Spells.attract_flames`, shared by all three spells' `play()`): flames are sorted nearest-first (Manhattan) and each moves **one** tile toward the player along the axis of greatest distance — `|dx|>|dy|` steps in x, `|dy|>|dx|` steps in y, equal steps diagonally. This keeps each flame inside its quadrant relative to the player. If the chosen tile is a WALL or off-map, it falls back to `BFS.find_path`. A flame landing on the player or another flame sets `player.hp = 0`; Bolgrot blocks (flame stays).

**Flame pushback** (`Spells.push_flames`): the counterpart to attraction, run by `ShortJump`/`LongJump` `play()` **only after a flame is killed** and **before** `attract_flames`. It takes the flames **adjacent** to the player (Chebyshev distance ≤ 1, diagonals included) and steps each **one tile away** along its dominant axis (same axis/`sign` logic as attraction, WALL/off-map falling back to `BFS.find_path`). A pushed flame landing on **another flame or the player** sets `player.hp = 0`; Bolgrot blocks (flame stays). This is intended base-game behaviour — chaining a flame into another flame is a real way to die, not a bug. Net effect when you land on a flame: adjacent flames are pushed out one tile, then `attract_flames` pulls *all* flames (adjacent and not) back one tile, so only the non-adjacent ones make net progress toward you.

**Dependency layering (acyclic).** Imports form a clean DAG: `constant` → `entity` → `case`/`map` → `BFS` → `spells` → `game`. `Player` is spell-agnostic — it holds `self.spells: list[Spells]` (empty by default; the `Spells` type is a `TYPE_CHECKING`-only import) and does **not** import `spells` or `BFS`. `Game.__init__` constructs the concrete spells (`ShortJump`/`LongJump`/`MoveFlames`) with a shared `BFS(self.map)` and assigns them to `player.spells`. Because `entity` no longer depends on the higher layers, every module imports cleanly regardless of entry point; `spells` modules import entity/case classes normally at module top level.

### Flame spawn patterns

`Patterns(seed=None)` holds `_all_patterns` (class-level list of patterns); each pattern is a list of grid coordinates. `Patterns.draw()` picks one at random (without replacement) using a seeded `random.Random` instance, making runs reproducible. Each turn, the drawn pattern is passed to `Game.end_turn()` which calls `Map.place_entity()` per position. `Patterns.reset()` restores the full pool.

### Sprite assets

Sprites are stored as both `.xcf` (GIMP source) and `.png` in `src/sprites_png/`. Spell classes reference `.png` filenames in their `sprite` field. Images are lazy-loaded on first access via the `image` property.

### RL environment / PPO agent

`src/ai/` (optional `ai` dependency group, `torch`) trains a **PPO** agent to play the game headlessly. It drives `Game` purely through the environment API below — no game logic lives in `src/ai/`.

**Winnability / the flame-clear mechanic (read this before tuning rewards).** The mechanics are the finished base game and are **fixed** — the goal is a policy that beats the game *as-is*, not a reward hack or a mechanic change. Why it's hard: a flame is removed **only by jumping onto its tile**; every jump also runs `push_flames` (adjacent flames shoved out one tile, and a flame chained onto another flame or the player is instant death) then `attract_flames`, dragging *all* other flames one tile toward the player, and a flame reaching the player is instant death. So clearing flames greedily is self-defeating — measured random-search win rates: 1 flame ~11%, 2 ~2%, 3 ~0.3%, a full 6-flame wave **0%**, and the full 6-wave game **0%** (32k games, min 32 flames left). Random/naive play essentially never wins, so `REWARD_WIN` is (almost) never hit by an untrained policy — but this is a **skill / exploration** problem, not an unreachable-goal one: reward-weight tuning alone won't manufacture a winner, and the curriculum (below) exists to bootstrap the policy from trivially-winnable shrunk games up to the full game rather than to sidestep an impossible task.

**Empirically confirmed (2026-07).** With the curriculum plus the fixes it took to make it learn (`--ent-coef 0.03` to break a position-blind canned-action collapse, `--lr-floor 0.1` so each session keeps learning instead of freezing when the LR anneals, and chained short `--resume` sessions), PPO trained **all the way from stage 0 up to the full game (stage 7/7)**, winning up to ~65% on the shrunk winnable stages — and *still* **0%** on the full 6-wave game. So a **competent trained agent** (not just random search) reconfirms the wall: *reaching* the full game is a solved training problem, *winning* it is not, absent a mechanic change. The trained policy plays it "as well as the mechanic allows" (`approach` reward strongly positive — it actively hunts flames — clearing a few before an attracted flame kills it).

**Overturned — the full game IS winnable (2026-07, via search).** The 0% above is a limit of *model-free* PPO, **not of the game**: PPO can only reinforce wins it randomly stumbles into, and the winning line is far too deep (~60 actions) for random exploration to ever hit. A **planning agent that does explicit lookahead on the deterministic model wins the full 6-wave game outright** — `src/ai/mcts.py` (MCTS) clears the flames and triggers `REWARD_WIN` on the true patterns: **83% win rate over 6 full-game seeds** at 300 sims/move (`clear ≈ +340`, `terminal ≈ +82`), versus **0%** for the trained PPO policy *and* every scripted baseline (`python -m src.ai.baselines --mcts-sims 300 --seeds 6`). So the binding constraint was the **algorithm class** (model-free vs planning), never the mechanic. Treat the "mechanic wall" language above as *PPO's* wall; the mechanic itself has winning lines a search agent finds.

**Curriculum knobs on `Game.__init__`** (`num_waves`, `flames_per_wave`, `spawn_radius`, all `None` = full difficulty): shrink the game to a winnable size so PPO has a reward gradient. `spawn_radius` (when set) replaces the fixed spawn *patterns* with `flames_per_wave` flames placed on random free tiles within that Manhattan distance of the player — the real difficulty is spawn *distance*, so a small radius + 1 flame is a trivial win. `reset()` preserves all three. `_make_wave()` applies them at spawn time.

**Environment API on `Game`** (all in `src/game.py`):
- `get_observation() -> list[float]` — fixed-length egocentric vector. Layout: 11 scalars `[px, py, hp, pa, waves, flame_count, boss_dx, boss_dy, castable_0..2]`, then the `K_NEAREST_FLAMES` nearest flames and `K_NEAREST_SPAWN` nearest next-spawn tiles, each `(present, dx, dy)`, all normalised to ~`[-1, 1]`. Length is `observation_size()` (currently **53**). `K_NEAREST_FLAMES`/`K_NEAREST_SPAWN` live in `constant.py`.
- `clone() -> Game` / `get_state() -> dict` / `set_state(dict)` — **state save/restore for search agents** (the game is deterministic, so a clone/snapshot can be stepped freely without touching the original). `clone()` is a headless-safe `deepcopy` (~4 ms; spell sprites are lazy so no pygame `Surface` is in the graph). `get_state`/`set_state` are a ~80x-cheaper fast path (~80/140 µs) capturing only the mutable fields (player pos/hp/pa, flame occupancy, per-turn spell `time_used`, wave/turn/done/won counters, both RNGs, `spawn_pattern`/`previsu`/`spell`/`last_reward`) — the map structure and never-moving Bolgrot are shared. `set_state` rebuilds flame + player occupancy while keeping the *same* `player` object (it is aliased from `Game.player` and its cell). Used by `mcts.py`; verified behaviourally identical to `clone`.
- `ACTIONS` (module-level `_ACTIONS`, aliased as `Game.ACTIONS`) — the fixed discrete action space, **17** entries. Each is `(spell_index | None, (dx, dy))`; the target tile is the player's position plus the offset. Because every spell targets one of 4 orthogonal / 4 diagonal directions at range 1–2, the space collapses to ShortJump×4 + LongJump×8 (range 1 and 2) + MoveFlames×4 + one end-turn (`(None, (0, 0))`, the last index). This makes the target position-independent so the policy never has to output an arbitrary tile.
- `legal_actions() -> list[bool]` — mask over `ACTIONS`. End-turn is always legal while live; a spell action is legal only when the spell `is_castable` **and** its target tile is in the spell's `previsu()`. All-illegal once `done`.
- `step(action_index) -> (reward, done)` — applies one action (all weights are `REWARD_*` constants in `constant.py`). A flame is removed **only by jumping onto its tile** (`ShortJump`/`LongJump` `play()` overwrite `dst.entity` and refund +1 HP); `attract_flames` merely repositions flames, so "kill all flames" means landing on each one before the waves overwhelm you. Reward terms: `REWARD_CLEAR_FLAME` (`10.0`) per flame removed — the dominant progress signal; `REWARD_SURVIVE_TURN` (`0.1`) token per end-turn; `REWARD_END_TURN_PER_FLAME` (`-0.5`) × flames still **alive on the board** at end-turn (the `"stall"` part, measured *before* the wave spawns so only flames the agent left uncleared count — this is the controllable anti-stall term: a clean board nets +`REWARD_SURVIVE_TURN`, an uncleared board goes net-negative; magnitude kept well under the death penalty so suicide never becomes preferable); `REWARD_HP_DELTA` (`0.02 × ΔHP`); terminal `REWARD_WIN` (`+100`); terminal death `REWARD_DEATH` (`-10`) **plus** `REWARD_DEATH_PER_WAVE_LEFT` (`-5.0`) × waves still unspawned, so dying early is far worse than dying on the brink (this is what makes surviving beat "clear a few then die"); `REWARD_ILLEGAL` for masked-out no-ops. **Only flame *reductions* pay** (`if cleared > 0`) — the end-turn wave adds flames the agent can't prevent. Two shaping terms: `REWARD_APPROACH` (`1.0` per tile closer to the nearest flame/next-spawn tile, measured old-vs-new player position against a fixed snapshot, so only the two jumps earn it), and a per-spell cast bias `REWARD_CAST` (now `(0.0, 0.0, -0.2)` — kept near zero because an *unconditional* per-cast reward is farmable; only the no-progress "Inaction" drag keeps a small negative). `step` also stores a per-term breakdown dict in `self.last_reward` (keys = `REWARD_PARTS`, sums to the returned scalar) so tooling can see *which* term a policy accumulates. Episodes always terminate because HP/PA are finite. **Changing any `REWARD_*` requires retraining** but *not* a dimension change, since the observation and action space are unchanged.

**`src/ai/` modules** (nothing here contains game logic; keep `__init__.py` import-free so the base game never needs `torch`):
- `network.py` — `ActorCritic`, a small shared-trunk MLP (`obs_size → 64 → 64 → {policy logits, value}`, orthogonal init). Illegal-action masking is centralised here: `get_action_and_value(obs, mask, action=None)` fills masked logits with `MASK_FILL` (`-1e8`) before the `Categorical`, so the policy can only pick legal moves. Rebuild dims from `Game.observation_size()` / `Game.num_actions()`.
- `policy.py` — env↔net glue: `choose_action(net, game, deterministic)` (greedy argmax over masked logits, or sampled), `run_episode`/`evaluate` (mean reward + win rate over the fixed `EVAL_SEEDS`), and `save_policy`/`load_policy` (a checkpoint stores the state dict plus `obs_size`/`num_actions`/`hidden` so it rebuilds standalone). `MAX_STEPS` is a safety cap. Reward-decomposition helpers live here too: the `ActionFn` type (`Game -> action index`) and `net_action_fn(net)` let scripted and neural policies share `run_episode_decomposed` (returns `(total, won, steps, parts)`); `evaluate_decomposed` averages those `parts` over `EVAL_SEEDS`, and `format_reward_parts` renders the compact `name +value` line the trainer logs each eval.
- `baselines.py` — offline reward-balance harness. `python -m src.ai.baselines [--policy src/ai/ppo.pt] [--mcts-sims N] [--seeds N]` scores scripted policies (`always_end_turn`, `random_legal`, `greedy_clear`) plus the trained net over `EVAL_SEEDS` and prints a **policy × reward-term** table (via `score_policy`/`benchmark`). `--mcts-sims N` adds an MCTS row at that per-move sim budget (slow — pair with `--seeds N` to shrink the seed set to the first N). Balancing `REWARD_*` is an *ordering* problem: read down the `return` column (a winning policy must beat clearing-then-dying must beat sitting still) and across a row (whichever term dominates a non-winning policy is the one being over-paid). No game logic — pure env-driven tooling.
- `ppo.py` — the algorithm. `PPOConfig` holds hyperparameters (CleanRL-style defaults). `_VecGames` steps `num_envs` independent games in lockstep with auto-reset. `_compute_gae` does GAE over the `[num_steps, num_envs]` rollout; `train(cfg)` runs rollout → GAE → clipped-surrogate + value + entropy update, and saves the best-by-eval policy to `cfg.out_path`. **The env is not a `gym` env** — it's driven directly through `get_observation`/`legal_actions`/`step`, and the legal-action mask is stored per step so the PPO update re-masks identically.
- `train.py` — `python -m src.ai.train [--iterations N] [--num-envs N] [--num-steps N] [--lr] [--lr-floor F] [--ent-coef E] [--seed] [--out ppo.pt] [--checkpoint ppo_ckpt.pt] [--resume [PATH]] [--curriculum] [--promote-win R]`. `--lr-floor` (fraction of `--lr`, default 0.0; **use `0.1` for curriculum**) anneals the LR toward that floor instead of 0 so a session keeps learning to its end; `--ent-coef` (default `0.01`, **use `0.03` for curriculum**) is the entropy bonus — raise it if the policy collapses to a position-blind canned action sequence instead of tracking the flame. The `make curriculum` / `make curriculum-resume` targets pass `ENT=0.03 LRFLOOR=0.1` by default. Saves the best-by-eval **policy** to `src/ai/ppo.pt` and a resumable **training checkpoint** (weights + optimizer + `iteration`/`global_step`/`best_score`/`stage_idx`) to `src/ai/ppo_ckpt.pt` (both git-ignored). `--iterations` is *per session*; `--resume` restores the checkpoint and runs that many more iterations (each session anneals its own LR — a warm restart), continuing the step/iteration counters and carrying `best_score` so a worse policy never overwrites the saved best. Bare `--resume` uses the `--checkpoint` path. `--curriculum` enables `DEFAULT_CURRICULUM` (a `(num_waves, flames_per_wave, spawn_radius)` ramp **ordered by declining achievable win rate**, from a trivially-winnable one-flame game up to the full game — each stage's comment lists the measured `greedy_clear` ceiling); the agent is promoted to the next stage once its eval win rate hits `--promote-win` (default `0.5`), and eval always runs at the current stage's difficulty. **The per-stage ceilings in the comments are `greedy_clear`'s, not PPO's** — a trained PPO policy substantially *exceeds* them (it promoted past `(3,1,1)`, greedy ceiling ~9%, and climbed the whole ramp to the full game, hitting ~65% on some shrunk stages). Use the greedy numbers only to *order* stages, not to predict PPO's reach, and don't lower `--promote-win` prematurely — with `--ent-coef 0.03` + `--lr-floor 0.1` PPO clears the `0.5` bar on every shrunk stage. The genuine wall is the **last** stage only, and only for *model-free* PPO: the full 6-wave game stays 0% for any PPO policy (the winning line is too deep for random exploration), so the curriculum's job ends at "reach stage 7", not "win it". (A lookahead **MCTS** agent, `mcts.py`, *does* win the full game — see the Winnability note above.) Best-policy score is `stage_idx * 1e6 + win_rate*1000 + mean_reward`, so a policy that reaches a harder stage always beats an easier-stage one (note: once the frontier reaches stage 7, `ppo.pt` is a full-game policy at 0% win — a lower-stage policy that actually *wins* scores lower and is overwritten). Curriculum stages are driven through `PPOConfig.curriculum` / `_VecGames.set_difficulty`; empty curriculum = one fixed full-difficulty stage (logs then omit the stage tag).
- `play.py` — `python -m src.ai.play [--policy ppo.pt] [--seed S] [--delay MS]`. Loads a policy and drives the game through the real `Renderer` (needs a display).
- `mcts.py` — **Monte-Carlo Tree Search planning agent** — the approach that actually **wins the full game** (PPO can't; see Winnability note). Exposes `mcts_action_fn(simulations, rollout_depth, c_ucb, rollout_policy, gamma, seed) -> ActionFn`, so it plugs into `evaluate`/`baselines`/`play` unchanged. It clones the live game **once per move** then runs all `simulations` on **one working game** via `Game.get_state`/`set_state` (snapshots, not per-node clones). Two design keys: (1) **shaped return** — it backs up the summed per-step `game.last_reward` (rollouts almost never win, so a win/loss leaf value is uninformative); (2) **`safe_greedy` rollout** — like `greedy_clear` but a 1-ply survival probe (step + `set_state` restore) rejects jumps that end in death, because plain `greedy_clear` suicides in ~3 steps (landing on a flame lets `attract_flames` pull a neighbour onto the player). Pure planning, **no training/network** for this version; the Phase-2 upgrade path is AlphaZero-style PUCT reusing `network.py` (policy prior + value leaf replacing rollouts).

**If you change the observation layout or `ACTIONS`, no config edit is needed** — the network sizes itself from `observation_size()`/`num_actions()` at construction, but existing `ppo.pt` checkpoints become incompatible and must be retrained.
