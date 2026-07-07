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
# observation vector length and therefore `num_inputs` in the NEAT config.
K_NEAREST_FLAMES = 8
K_NEAREST_SPAWN = 6

# Reward weights for the AI (`Game.step`). The dominant term is REWARD_CLEAR:
# clearing flames is the only way to make real progress toward a win, and it
# must out-earn passively ending turns. Survival is kept deliberately tiny so
# "sit still and end turns" is not a winning strategy. Flame *spawns* are never
# penalised (the agent cannot prevent them) — only flame *reductions* pay out.
REWARD_CLEAR_FLAME = 2.0    # per flame removed from the board
REWARD_SURVIVE_TURN = 0.1   # per turn survived (small; anti-suicide only)
REWARD_HP_DELTA = 0.02      # per HP gained/lost (tiny nudge against waste)
REWARD_WIN = 100.0          # terminal: board cleared after all waves
REWARD_DEATH = -10.0        # terminal: player died
REWARD_ILLEGAL = -0.1       # picked an illegal action (guard; masked in play)

# Package-data paths resolved relative to the installed `src` package so they
# work regardless of the current working directory.
MAP_CONF = str(files("src") / "config" / "bolgrot.map")
SPRITES_DIR = str(files("src") / "sprites_png")
PATTERNS_CONF = str(files("src") / "patterns" / "patterns.json")
