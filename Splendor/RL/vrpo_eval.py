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

    def predict_batch(self, states, masks):
        """Greedy argmax action for a BATCH of (state, mask). One forward pass.
        Matches the single-row path: argmax over masked Q-values.
        """
        S = tf.convert_to_tensor(states, dtype=tf.float32)
        M = tf.convert_to_tensor(masks, dtype=tf.bool)
        qs = self.model(S, training=False)
        masked = tf.where(M, qs, tf.fill(tf.shape(qs), -tf.float32.max))
        return tf.argmax(masked, axis=1, output_type=tf.int32).numpy()

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
        self.step_move(move)

    def step_move(self, move):
        """Apply an externally chosen move (used by the vectorized evaluator)."""
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
    Sequential reference implementation; evaluate_vectorized is the fast path.
    """
    wins = draws = 0
    for i in range(n_games):
        result = _play_one(vrpo_agent, opp_agent, i % 2, max_half_turns)
        if result is None:
            draws += 1
        elif result:
            wins += 1
    return wins / n_games, wins, draws


def _greedy_actions(agent, states, masks):
    """Batched greedy actions for `agent`. Uses predict_batch when available
    (the NN agents), else falls back to per-row get_predictions+argmax (e.g.
    RandomAgent, which has no NN to batch and is already cheap).
    """
    if hasattr(agent, 'predict_batch'):
        return agent.predict_batch(states, masks)
    out = np.empty(len(states), dtype=np.int32)
    for i, (s, m) in enumerate(zip(states, masks)):
        q = agent.get_predictions(s, m)
        out[i] = int(np.argmax(q))
    return out


def evaluate_vectorized(vrpo_agent, opp_agent, n_games=80, max_half_turns=300,
                        batch=None):
    """Same result distribution as evaluate(), but runs games in lockstep and
    batches each side's policy forward pass (one call per side per step instead
    of one per turn). Seats alternated to cancel first-move advantage.

    Both agents are deterministic (greedy), so this only changes SPEED, not the
    decision rule: each move is still argmax over the masked outputs, identical
    to the sequential path.
    """
    batch = batch or n_games
    wins = draws = 0
    played = 0

    while played < n_games:
        k = min(batch, n_games - played)
        games = []
        for j in range(k):
            if (played + j) % 2 == 0:
                players = [('VRPO', vrpo_agent, 0), ('OPP', opp_agent, 1)]
            else:
                players = [('OPP', opp_agent, 0), ('VRPO', vrpo_agent, 1)]
            games.append(EvalGame(players, max_half_turns=max_half_turns))

        live = list(games)
        while live:
            # Partition live games by whose turn it is.
            v_idx, v_S, v_M = [], [], []
            o_idx, o_S, o_M = [], [], []
            for idx, g in enumerate(live):
                s = g.to_state()
                m = g.active_player.get_legal_moves(g.board)
                if g.active_player.agent is vrpo_agent:
                    v_idx.append(idx); v_S.append(s); v_M.append(m)
                else:
                    o_idx.append(idx); o_S.append(s); o_M.append(m)

            moves = {}
            if v_S:
                a = _greedy_actions(vrpo_agent,
                                    np.asarray(v_S, dtype=np.float32),
                                    np.asarray(v_M, dtype=bool))
                moves.update(zip(v_idx, (int(x) for x in a)))
            if o_S:
                a = _greedy_actions(opp_agent,
                                    np.asarray(o_S, dtype=np.float32),
                                    np.asarray(o_M, dtype=bool))
                moves.update(zip(o_idx, (int(x) for x in a)))

            still = []
            for idx, g in enumerate(live):
                g.step_move(moves[idx])
                if g.victor or g.half_turns >= max_half_turns:
                    if not g.victor:
                        draws += 1
                    else:
                        winner = next(p for p in g.players if p.victor)
                        if winner.agent is vrpo_agent:
                            wins += 1
                else:
                    still.append(g)
            live = still

        played += k

    return wins / n_games, wins, draws
