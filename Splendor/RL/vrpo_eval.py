# Splendor/RL/vrpo_eval.py
"""
Head-to-head evaluation for VRPO.

Self-play game-length is NOT a strength metric (both seats can degrade
symmetrically). The real yardstick is win rate of the GREEDY VRPO policy
against fixed opponents - principally the existing DQN (`inference_model`,
described in the README as superhuman) and a random agent as a floor.

Games reuse the real env (Board / Player / apply_move) via a thin EvalGame
that plays without learning. Seats are alternated across games to cancel
Splendor's first-move advantage.
"""

import numpy as np
import tensorflow as tf
from keras.models import load_model

from Environment.rl_game import RLGame
from .rewards import SparseRewardEngine


class KerasGreedyOpponent:
    """Greedy wrapper around a saved Keras Q-model, for use as a fixed eval
    opponent (e.g. the DQN inference_model). Loads with compile=False so custom
    optimizer/LR-schedule objects (ScheduleWithWarmup) don't need registration -
    inference needs only the forward pass. Exposes get_predictions(state, mask)
    so Player.choose_move can argmax it, exactly like InferenceAgent.
    """
    def __init__(self, model_path):
        self.model = load_model(model_path, compile=False)

    def get_predictions(self, state, legal_mask):
        qs = self.model(state[None, :], training=False)[0]
        return tf.where(legal_mask, qs, tf.fill(tf.shape(qs), -tf.float32.max))

    # No-op learning interface (never used during eval, but mirrors agents).
    def remember(self, *_): pass
    def replay(self): pass


class _NullModel:
    """Stand-in for RLGame's `model` during eval: no learning, but carries a
    one-element memory so apply_move's end-of-game poke
    (memory[-1][2] += loser_reward; memory[-1][5] = True) has a harmless target.
    """
    def __init__(self):
        self.memory = [[None, 0, 0.0, None, None, False]]

    def remember(self, _entry):
        pass


class EvalGame(RLGame):
    def __init__(self, players, max_half_turns=300):
        super().__init__(players, _NullModel())
        self.rewards = SparseRewardEngine(self)   # rewards irrelevant; just legal
        self.max_half_turns = max_half_turns

    def turn(self):
        state = self.to_state()
        move = self.active_player.choose_move(self.board, state)   # greedy argmax
        self.move_idx = move
        self.rewards._cache.clear()
        self.apply_move(move)
        self.half_turns += 1


def _play_one(vrpo_agent, opp_agent, vrpo_seat, max_half_turns):
    """Returns True if VRPO won, False if opponent won, None on timeout."""
    if vrpo_seat == 0:
        players = [('VRPO', vrpo_agent, 0), ('OPP', opp_agent, 1)]
    else:
        players = [('OPP', opp_agent, 0), ('VRPO', vrpo_agent, 1)]

    game = EvalGame(players, max_half_turns=max_half_turns)
    while not game.victor and game.half_turns < max_half_turns:
        game.turn()

    if not game.victor:
        return None
    winner = next(p for p in game.players if p.victor)
    return winner.agent is vrpo_agent


def evaluate(vrpo_agent, opp_agent, n_games=40, max_half_turns=300):
    """Play n_games (seats alternated). Returns (win_rate, n_wins, n_draws).

    win_rate counts timeouts as non-wins (conservative): wins / n_games.
    """
    wins = draws = 0
    for i in range(n_games):
        result = _play_one(vrpo_agent, opp_agent, i % 2, max_half_turns)
        if result is None:
            draws += 1
        elif result:
            wins += 1
    return wins / n_games, wins, draws
