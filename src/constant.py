from importlib.resources import files

# 37
GRID_MAX_X = 37
# 34
GRID_MAX_Y = 34

CASE_HEIGHT = 32
CASE_WIDTH = 64

# (14,15)
BASE_PLAYER_POS = (14, 15)
# (21, 10)
BASE_BOLGROT_POS = (21, 10)

# HEXA COLOR
CASE_COLOR_1 = (205, 178, 111)
CASE_COLOR_2 = (189, 185, 132)
SPAWN_COLOR_1 = (151, 13, 158)
PREVISU_COLOR = (9, 91, 158)
BACKGROUND_POPUP = (79, 79, 61)

RIGHT_PANEL_W = 400
BUTTON_W = 200
BUTTON_H = 60
SPELL_GAP = 10

# IN SECONDS
TIME_TURN = 120

# Number of flame waves that spawn over a game (exactly this many, no repeats).
NB_WAVES = 6

# Player starting hit points (used for observation normalisation).
MAX_HP = 40

# Observation encoding: how many nearest flames / next-spawn tiles the AI sees.
# Each contributes 3 floats (present flag, dx, dy). Changing these changes the
# observation vector length and therefore the policy network's input size.
K_NEAREST_FLAMES = 8
K_NEAREST_SPAWN = 6

# Reward weights for the AI (`Game.step`). The dominant term is REWARD_CLEAR:
# clearing flames is the only way to make real progress toward a win, and it
# must out-earn passively ending turns. Survival is kept deliberately tiny so
# "sit still and end turns" is not a winning strategy. Flame *spawns* are never
# penalised (the agent cannot prevent them) — only flame *reductions* pay out.
REWARD_CLEAR_FLAME = 10.0    # per flame removed from the board
REWARD_SURVIVE_TURN = 0.1   # per turn survived (small; anti-suicide only)
REWARD_HP_DELTA = 0.02      # per HP gained/lost (tiny nudge against waste)
REWARD_WIN = 100.0          # terminal: board cleared after all waves
REWARD_DEATH = -10.0        # terminal: player died (base cost)
# Extra death cost per wave still unspawned when the player dies. Dying early
# (many waves left) must hurt more than the handful of flames clearable before
# then, so "clear a few then die" can never beat surviving. See `Game.step`.
REWARD_DEATH_PER_WAVE_LEFT = -5.0
REWARD_ILLEGAL = -0.1       # picked an illegal action (guard; masked in play)

# Shaping on top of the outcome rewards above, to guide *how* the agent plays.
# Approach: reward the player for physically moving toward the nearest flame or
# next-spawn tile (only the two jump spells move the player).
REWARD_APPROACH = 1.0      # per tile closer to the nearest flame / spawn tile
# Per-spell cast bias: kept near zero on purpose. An *unconditional* per-cast
# reward is farmable (the agent banks it by casting uselessly until PA drains),
# so progress is paid through REWARD_CLEAR/REWARD_APPROACH instead. Only the
# costly no-progress "Inaction" flame-drag keeps a small negative nudge.
# Indexed by spell_index (0=ShortJump, 1=LongJump, 2=MoveFlames) in `step`.
REWARD_CAST_SHORTJUMP = 0.0     # "Astral leap"  — neutral (reward the outcome)
REWARD_CAST_LONGJUMP = 0.0      # "Double leap"  — neutral (reward the outcome)
REWARD_CAST_MOVEFLAMES = -0.2   # "Inaction"     — discouraged
REWARD_CAST = (
    REWARD_CAST_SHORTJUMP,
    REWARD_CAST_LONGJUMP,
    REWARD_CAST_MOVEFLAMES,
)

# Package-data paths resolved relative to the installed `src` package so they
# work regardless of the current working directory.
MAP_CONF = str(files("src") / "config" / "bolgrot.map")
SPRITES_DIR = str(files("src") / "sprites_png")
PATTERNS_CONF = str(files("src") / "patterns" / "patterns.json")
