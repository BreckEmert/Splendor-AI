# Splendor/RL/vrpo_league.py
"""
League / past-self opponents for VRPO self-play.

Plain self-play (both seats = the current live policy) can localize into a
narrow equilibrium and "forget" how to beat older or different strategies. A
league periodically snapshots the current actor into a frozen pool and has a
fraction of rollout games pit the LEARNER (current policy) against a random
frozen snapshot.

On-policy correctness: only the learner's transitions are collected for the
update; the frozen opponent is just part of the environment, so the learner's
data stays on-policy (its own actions still come from the current policy with
recorded log-probs). See collect_vectorized in vrpo_game.py.
"""

import numpy as np
import tensorflow as tf
from collections import deque

from .vrpo_model import build_actor, masked_log_softmax


class FrozenActor:
    """Immutable actor snapshot that SAMPLES moves in batch. Mirrors
    VRPOAgent.act_batch so the rollout loop can treat it uniformly. Never
    trained; weights are copied in at construction.
    """
    def __init__(self, weights, state_dim, action_dim, layer_sizes):
        self._model = build_actor(state_dim, action_dim, layer_sizes)
        self._model.set_weights(weights)

    @tf.function(reduce_retracing=True)
    def _sample(self, S, M):
        logp_all = masked_log_softmax(self._model(S, training=False), M)
        actions = tf.random.categorical(logp_all, 1)[:, 0]
        logps = tf.gather(logp_all, actions, batch_dims=1)
        return actions, logps

    def act_batch(self, states, masks):
        S = tf.convert_to_tensor(states, dtype=tf.float32)
        M = tf.convert_to_tensor(masks, dtype=tf.bool)
        a, lp = self._sample(S, M)
        return a.numpy().astype(np.int32), lp.numpy().astype(np.float32)


class LeaguePool:
    """Bounded FIFO pool of frozen actor snapshots."""
    def __init__(self, max_size, state_dim, action_dim, layer_sizes):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer_sizes = layer_sizes
        self._pool = deque(maxlen=max_size)

    def __len__(self):
        return len(self._pool)

    def add(self, actor):
        """Snapshot the given (live) actor into the pool."""
        self._pool.append(FrozenActor(actor.get_weights(), self.state_dim,
                                      self.action_dim, self.layer_sizes))

    def sample(self):
        """Uniformly pick a frozen opponent. Caller must ensure len>0."""
        return self._pool[np.random.randint(len(self._pool))]
