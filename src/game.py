from __future__ import annotations
from . import constant
from .BFS import BFS
from .map import Map
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
    ) -> None:
        """Set up the map, player spells, boss, spawn pattern and counters.

        ``seed`` makes the flame spawn sequence reproducible.
        """
        self.map: Map = Map()
        self.player: Player = player
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
            self.patterns.draw_opening()
        self.waves_spawned: int = 0
        self.previsualiation: list[tuple[int, int]] = []
        self.spell: Spells | None = None
        self.turn: int = 0
        self.done: bool = False
        self.won: bool = False

    def reset(self, seed: int | None = None) -> None:
        """Re-initialise all game state, optionally with a new seed."""
        self.__init__(self.player, seed=seed)  # type: ignore[misc]

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

        Exactly ``constant.NB_WAVES`` distinct waves spawn over a game (one per
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
                self.patterns.draw(self._occupied_tiles())
                if self.waves_spawned < constant.NB_WAVES
                else []
            )
        if self.waves_spawned >= 6:
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
        elif self.waves_spawned >= constant.NB_WAVES and \
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
            self.waves_spawned / constant.NB_WAVES,
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

    def step(self, action_index: int) -> tuple[float, bool]:
        """Apply one discrete action and return ``(reward, done)``.

        ``action_index`` indexes :data:`Game.ACTIONS`. Illegal actions are
        no-ops with a small penalty. The reward is built so that *active*
        play beats sitting still: clearing flames is the dominant term (it is
        the only real progress toward a win), survival pays only a token
        amount, and flame *spawns* are never penalised — only genuine
        reductions in flame count earn ``REWARD_CLEAR_FLAME``. See the reward
        weights in ``constant.py``.
        """
        if self.done or not 0 <= action_index < len(self.ACTIONS):
            return 0.0, self.done

        spell_index, (dx, dy) = self.ACTIONS[action_index]
        if not self.legal_actions()[action_index]:
            return constant.REWARD_ILLEGAL, self.done

        hp_before = self.player.hp
        flames_before = len(self._flame_positions())

        reward = 0.0
        if spell_index is None:
            self.end_turn()
            reward += constant.REWARD_SURVIVE_TURN
        else:
            target = (self.player.pos_x + dx, self.player.pos_y + dy)
            self.select_spell(spell_index)
            self.play_selected_spell(target)

        flames_after = len(self._flame_positions())
        # Reward only flame *reductions*: the wave spawned on an end-turn adds
        # flames the agent cannot prevent, so increases must not be punished.
        cleared = flames_before - flames_after
        if cleared > 0:
            reward += constant.REWARD_CLEAR_FLAME * cleared
        reward += constant.REWARD_HP_DELTA * (self.player.hp - hp_before)

        if self.done:
            reward += (constant.REWARD_WIN if self.won
                       else constant.REWARD_DEATH)
        return reward, self.done
