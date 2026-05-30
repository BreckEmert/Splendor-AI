# Splendor/RL/__init__.py

from .model import RLAgent
from .inference_model import InferenceAgent
from .random_model import RandomAgent
from .rewards import BasicRewardEngine, SparseRewardEngine
# Backward-compat: `RewardEngine` was renamed to `BasicRewardEngine` but
# RL/__init__ and Environment/rl_game still import the old name. Alias it so
# the package imports cleanly. (Pre-existing breakage, surfaced 2026-05-30.)
RewardEngine = BasicRewardEngine
from .training import ddqn_loop
