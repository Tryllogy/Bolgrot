from __future__ import annotations
import random
from . import constant
from .BFS import BFS
from .map import Map
from .case import CaseType
from .patterns import Patterns
from .spells import Spells, ShortJump, LongJump, MoveFlames
from .entity import Flame, Player, Bolgrot

# Fixed discrete action space for the AI. Every spell target is one of the
# 4 orthogonal / 4 diagonal directions at a fixed range, so the whole game
# reduces to a small position-independent action head plus "end turn".
# Each entry is ``(spell_index | None, (dx, dy))`` where the tile targeted is
# the player's position plus the offset; end-turn is ``(None, (0, 0))``.
_LINE_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ACTIONS: list[tuple[int | None, tuple[int, int]]] = [
    *[(0, d) for d in _LINE_DIRS],                       # ShortJump, r=1
    *[(1, (d[0] * r, d[1] * r))                          # LongJump,  r=1,2
      for r in (1, 2) for d in _LINE_DIRS],
    *[(2, d) for d in _DIAG_DIRS],                       # MoveFlames, r=1
    (None, (0, 0)),                                      # end turn
]

# Names of the additive reward terms produced by :meth:`Game.step`. Kept as an
# explicit ordered tuple so evaluators can decompose an episode's return into
# per-term sums (see ``src/ai/baselines.py``) and spot which term a policy is
# actually farming. Their sum is exactly the scalar reward ``step`` returns.
REWARD_PARTS: tuple[str, ...] = (
    "survive", "cast", "approach", "clear", "hp", "terminal", "illegal",
)


class Game:
    """Owns all game state and the turn/action API (no pygame dependency)."""

    ACTIONS = _ACTIONS

    @staticmethod
    def num_actions() -> int:
        """Number of discrete actions (length of :data:`Game.ACTIONS`)."""
        return len(Game.ACTIONS)

    def __init__(
        self,
        player: Player,
        seed: int | None = None,
        num_waves: int | None = None,
        flames_per_wave: int | None = None,
        spawn_radius: int | None = None,
    ) -> None:
        """Set up the map, player spells, boss, spawn pattern and counters.

        ``seed`` makes the flame spawn sequence reproducible. The rest are
        **curriculum knobs** that shrink the game to an easier, winnable size
        so an RL agent has a reward gradient to climb (``None`` each = full
        difficulty):

        - ``num_waves`` — waves before the win check (default ``NB_WAVES``).
        - ``flames_per_wave`` — cap flames spawned per wave.
        - ``spawn_radius`` — when set, replace the fixed spawn *patterns* with
          ``flames_per_wave`` flames placed on random free tiles within this
          Manhattan distance of the player. Since a lone flame is a trivial win
          (jump onto it, nothing left to attract), a small radius + few flames
          makes the early stages actually winnable — the real difficulty is
          spawn *distance*, not count. The final stage leaves this ``None`` to
          train on the true game patterns.
        """
        self.map: Map = Map()
        self.player: Player = player
        self._rng: random.Random = random.Random(seed)
        self.num_waves: int = (num_waves if num_waves is not None
                               else constant.NB_WAVES)
        self.flames_per_wave: int | None = flames_per_wave
        self.spawn_radius: int | None = spawn_radius
        bfs: BFS = BFS(self.map)
        self.player.spells = [
            ShortJump(bfs=bfs),
            LongJump(bfs=bfs),
            MoveFlames(bfs=bfs),
        ]
        self.map.place_entity([constant.BASE_PLAYER_POS], self.player)
        self.bolgrot: Bolgrot = Bolgrot(*constant.BASE_BOLGROT_POS)
        self.map.place_entity([constant.BASE_BOLGROT_POS], self.bolgrot)
        self.patterns: Patterns = Patterns(seed=seed)
        self.spawn_pattern: list[tuple[int, int]] = \
            self._make_wave(self.patterns.draw_opening())
        self.waves_spawned: int = 0
        self.previsualiation: list[tuple[int, int]] = []
        self.spell: Spells | None = None
        self.turn: int = 0
        self.done: bool = False
        self.won: bool = False
        # Per-step reward decomposition, refreshed by every ``step`` call. Sums
        # to the scalar reward returned. Lets tooling see the breakdown.
        self.last_reward: dict[str, float] = self._zero_reward()

    def reset(self, seed: int | None = None) -> None:
        """Re-initialise all game state, keeping the current difficulty."""
        self.__init__(  # type: ignore[misc]
            self.player, seed=seed,
            num_waves=self.num_waves, flames_per_wave=self.flames_per_wave,
            spawn_radius=self.spawn_radius)

    def _make_wave(
        self, pattern: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Shape a drawn spawn wave for the current curriculum difficulty.

        With ``spawn_radius`` set, ignore ``pattern`` and place the flames on
        random free tiles near the player instead (the easy, winnable regime);
        otherwise just cap the real pattern to ``flames_per_wave``.
        """
        if self.spawn_radius is not None:
            return self._near_spawn(self.flames_per_wave or 1)
        if self.flames_per_wave is None:
            return pattern
        return pattern[:self.flames_per_wave]

    def _near_spawn(self, k: int) -> list[tuple[int, int]]:
        """``k`` random free tiles within ``spawn_radius`` of the player."""
        px, py = self.player.pos_x, self.player.pos_y
        occupied = self._occupied_tiles()
        candidates = [
            pos for pos, case in self.map.cases.items()
            if case.case_type == CaseType.FREE
            and pos not in occupied
            and 0 < abs(pos[0] - px) + abs(pos[1] - py) <= self.spawn_radius
        ]
        self._rng.shuffle(candidates)
        return candidates[:k]

    def select_spell(self, spell_index: int) -> None:
        """Select a spell and compute its previsualisation tiles.

        Does nothing for an out-of-range index; clears the previsualisation
        if the spell is not castable (not enough AP or uses exhausted).
        """
        if spell_index < 0 or spell_index >= len(self.player.spells):
            return
        spell: Spells = self.player.spells[spell_index]
        if not spell.is_castable(self.player):
            self.clear_previsu()
            return
        self.spell = spell
        self.previsualiation = spell.previsu(
            (self.player.pos_x, self.player.pos_y), self.map.cases)

    def play_selected_spell(
            self,
            tile_clicked: tuple[int, int] | None = None,
    ) -> None:
        """Play the selected spell on the clicked tile, if valid."""
        if not self.previsualiation or tile_clicked is None:
            return
        if self.spell is None:
            return
        self.spell.play(self.map, self.player, tile_clicked)
        self.clear_previsu()
        self._check_end()

    def clear_previsu(self) -> None:
        """Discard the current previsualisation tiles."""
        self.previsualiation = []

    def end_turn(self) -> None:
        """Advance one turn: refresh AP/spells and spawn the next wave.

        Exactly ``self.num_waves`` distinct waves spawn over a game (one per
        end-turn until the quota is reached, none after). Once every wave has
        spawned, clearing all flames from the map wins the game.
        """
        self.player.pa = self.player.base_PA
        for spell in self.player.spells:
            spell.next_turn()
        if self.spawn_pattern:
            for pos in self.spawn_pattern:
                self.map.place_entity([pos], Flame(pos[0], pos[1]))
            self.waves_spawned += 1
            self.turn += 1
            self.spawn_pattern = (
                self._make_wave(self.patterns.draw(self._occupied_tiles()))
                if self.waves_spawned < self.num_waves
                else []
            )
        if self.waves_spawned >= self.num_waves:
            self.player.hp -= 1
        self.clear_previsu()
        self._check_end()

    def _occupied_tiles(self) -> set[tuple[int, int]]:
        """Tiles currently holding a flame or the player."""
        return {
            pos for pos, case in self.map.cases.items()
            if isinstance(case.entity, (Flame, Player))
        }

    def _flames_remain(self) -> bool:
        """Whether any flame is still on the map."""
        return any(isinstance(case.entity, Flame)
                   for case in self.map.cases.values())

    def _check_end(self) -> None:
        """Resolve win/loss: dead player loses; all waves cleared wins."""
        if self.player.hp <= 0:
            self.done = True
        elif self.waves_spawned >= self.num_waves and \
                not self._flames_remain():
            self.done = True
            self.won = True

    # ------------------------------------------------------------------
    # RL environment interface: fixed-size observation, discrete action space.
    # ------------------------------------------------------------------

    def _flame_positions(self) -> list[tuple[int, int]]:
        """Grid positions of every flame currently on the map."""
        return [pos for pos, case in self.map.cases.items()
                if isinstance(case.entity, Flame)]

    @staticmethod
    def _nearest_target_dist(
        targets: list[tuple[int, int]],
        pos: tuple[int, int],
    ) -> int:
        """Manhattan distance from ``pos`` to the nearest tile in ``targets``.

        Returns 0 when there are no targets, so the approach reward vanishes
        rather than rewarding an arbitrary default.
        """
        if not targets:
            return 0
        px, py = pos
        return min(abs(x - px) + abs(y - py) for x, y in targets)

    def _nearest_relative(
        self,
        positions: list[tuple[int, int]],
        k: int,
    ) -> list[float]:
        """Encode the ``k`` nearest ``positions`` relative to the player.

        Returns ``3 * k`` floats: for each slot a presence flag (1/0) then the
        normalised ``dx, dy`` toward the player. Empty slots are all zeros.
        """
        px, py = self.player.pos_x, self.player.pos_y
        gx = self.map.grid_max_x or 1
        gy = self.map.grid_max_y or 1
        nearest = sorted(
            positions,
            key=lambda p: abs(p[0] - px) + abs(p[1] - py),
        )[:k]
        out: list[float] = []
        for i in range(k):
            if i < len(nearest):
                x, y = nearest[i]
                out.extend((1.0, (x - px) / gx, (y - py) / gy))
            else:
                out.extend((0.0, 0.0, 0.0))
        return out

    def get_observation(self) -> list[float]:
        """Return the fixed-length float vector fed to the neural network.

        Layout (length ``Game.observation_size()``):
        ``[px, py, hp, pa, waves, flame_count, boss_dx, boss_dy,
        castable_0, castable_1, castable_2]`` followed by the
        ``K_NEAREST_FLAMES`` nearest flames and ``K_NEAREST_SPAWN`` nearest
        next-spawn tiles, each as ``(present, dx, dy)``. All values are
        normalised to roughly ``[-1, 1]``.
        """
        p = self.player
        gx = self.map.grid_max_x or 1
        gy = self.map.grid_max_y or 1
        flames = self._flame_positions()
        scalars: list[float] = [
            p.pos_x / gx,
            p.pos_y / gy,
            p.hp / constant.MAX_HP,
            p.pa / p.base_PA if p.base_PA else 0.0,
            self.waves_spawned / self.num_waves,
            len(flames) / 20.0,
            (self.bolgrot.pos_x - p.pos_x) / gx,
            (self.bolgrot.pos_y - p.pos_y) / gy,
        ]
        scalars.extend(
            1.0 if spell.is_castable(p) else 0.0 for spell in p.spells
        )
        obs = scalars
        obs += self._nearest_relative(flames, constant.K_NEAREST_FLAMES)
        obs += self._nearest_relative(self.spawn_pattern,
                                      constant.K_NEAREST_SPAWN)
        return obs

    @staticmethod
    def observation_size() -> int:
        """Length of the vector returned by :meth:`get_observation`."""
        return (11 + 3 * constant.K_NEAREST_FLAMES
                + 3 * constant.K_NEAREST_SPAWN)

    def legal_actions(self) -> list[bool]:
        """Boolean mask over :data:`Game.ACTIONS` of the currently legal moves.

        End-turn is always legal while the game is live; a spell action is
        legal only if the spell is castable and its target tile is in the
        spell's previsualisation. All actions are illegal once ``done``.
        """
        mask = [False] * len(self.ACTIONS)
        if self.done:
            return mask
        px, py = self.player.pos_x, self.player.pos_y
        previs: dict[int, set[tuple[int, int]]] = {}
        for idx, spell in enumerate(self.player.spells):
            if spell.is_castable(self.player):
                previs[idx] = set(spell.previsu((px, py), self.map.cases))
        for i, (spell_index, (dx, dy)) in enumerate(self.ACTIONS):
            if spell_index is None:
                mask[i] = True
            elif spell_index in previs and \
                    (px + dx, py + dy) in previs[spell_index]:
                mask[i] = True
        return mask

    @staticmethod
    def _zero_reward() -> dict[str, float]:
        """A fresh reward-breakdown dict with every term at zero."""
        return {part: 0.0 for part in REWARD_PARTS}

    def step(self, action_index: int) -> tuple[float, bool]:
        """Apply one discrete action and return ``(reward, done)``.

        ``action_index`` indexes :data:`Game.ACTIONS`. Illegal actions are
        no-ops with a small penalty. The reward is built so that *active*
        play beats sitting still: clearing flames is the dominant term (it is
        the only real progress toward a win), survival pays only a token
        amount, and flame *spawns* are never penalised — only genuine
        reductions in flame count earn ``REWARD_CLEAR_FLAME``.

        On top of those outcome rewards, two shaping terms guide *style*: an
        approach reward (``REWARD_APPROACH`` per tile the player moves closer to
        the nearest flame / next-spawn tile) and a per-spell cast bias
        (``REWARD_CAST[spell_index]`` — favouring the short jump over the long
        jump, and penalising the "Inaction" flame-drag). See ``constant.py``.
        """
        if self.done or not 0 <= action_index < len(self.ACTIONS):
            self.last_reward = self._zero_reward()
            return 0.0, self.done

        spell_index, (dx, dy) = self.ACTIONS[action_index]
        parts = self._zero_reward()
        if not self.legal_actions()[action_index]:
            parts["illegal"] = constant.REWARD_ILLEGAL
            self.last_reward = parts
            return constant.REWARD_ILLEGAL, self.done

        hp_before = self.player.hp
        flames_before = len(self._flame_positions())
        # Snapshot the approach targets (current flames + the next wave's spawn
        # tiles) and the player's distance to the nearest one. Measured against
        # a *fixed* snapshot with the player's old vs new position, so only the
        # player actually moving (the two jumps) earns the approach reward —
        # ending a turn or dragging flames leaves the player put.
        approach_targets = self._flame_positions() + list(self.spawn_pattern)
        dist_before = self._nearest_target_dist(
            approach_targets, (self.player.pos_x, self.player.pos_y))

        if spell_index is None:
            self.end_turn()
            parts["survive"] = constant.REWARD_SURVIVE_TURN
        else:
            target = (self.player.pos_x + dx, self.player.pos_y + dy)
            self.select_spell(spell_index)
            self.play_selected_spell(target)
            parts["cast"] = constant.REWARD_CAST[spell_index]

        dist_after = self._nearest_target_dist(
            approach_targets, (self.player.pos_x, self.player.pos_y))
        parts["approach"] = \
            constant.REWARD_APPROACH * (dist_before - dist_after)

        flames_after = len(self._flame_positions())
        # Reward only flame *reductions*: the wave spawned on an end-turn adds
        # flames the agent cannot prevent, so increases must not be punished.
        cleared = flames_before - flames_after
        if cleared > 0:
            parts["clear"] = constant.REWARD_CLEAR_FLAME * cleared
        parts["hp"] = constant.REWARD_HP_DELTA * (self.player.hp - hp_before)

        if self.done:
            if self.won:
                parts["terminal"] = constant.REWARD_WIN
            else:
                # Death: base cost plus a penalty per wave left unspawned, so
                # dying early (much game left) is far worse than dying on the
                # brink — clearing a few flames then dying must never pay off.
                waves_left = constant.NB_WAVES - self.waves_spawned
                parts["terminal"] = (
                    constant.REWARD_DEATH
                    + constant.REWARD_DEATH_PER_WAVE_LEFT * waves_left)
        self.last_reward = parts
        return sum(parts.values()), self.done
