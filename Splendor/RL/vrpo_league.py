# Splendor/RL/vrpo_league.py
"""
League / past-self opponents for VRPO self-play, with PFSP sampling.

Plain self-play (both seats = the current live policy) can localize into a
narrow equilibrium and "forget" how to beat older or different strategies. A
league periodically snapshots the current actor into a frozen pool and has a
fraction of rollout games pit the LEARNER (current policy) against a pooled
frozen snapshot.

PFSP (Prioritized Fictitious Self-Play, AlphaStar-style): rather than sampling
pooled opponents uniformly, weight each by how hard the learner currently finds
it. With the 'hard' weighting w(opp) = (1 - p)^pow, where p is the learner's
recent win-rate vs that opponent:
  - opponents the learner LOSES to (p->0) get weight ->1 (focus here)
  - opponents the learner has MASTERED (p->1) get weight ->0 (stop wasting games)
A small eps floor keeps a trickle of games against everyone (anti-forgetting),
and never-played opponents get max priority so they get measured. This points
compute where learning actually happens. pow=0 / VRPO_PFSP=0 recovers uniform.

On-policy correctness is unchanged: only the learner's transitions train; the
frozen opponent is just environment. See collect_vectorized in vrpo_game.py.
"""

import numpy as np
import tensorflow as tf
from collections import deque

from .vrpo_model import build_actor, masked_log_softmax


class FrozenActor:
    """Immutable actor snapshot that SAMPLES moves in batch. Mirrors
    VRPOAgent.act_batch so the rollout loop can treat it uniformly. Never
    trained; weights are copied in at construction. Carries its own PFSP stats
    (EMA of the learner's win-rate vs this snapshot) so they're evicted with it.
    """
    def __init__(self, weights, state_dim, action_dim, layer_sizes):
        self._model = build_actor(state_dim, action_dim, layer_sizes)
        self._model.set_weights(weights)
        self.lwr = None    # EMA learner win-rate vs this opp (None = unmeasured)
        self.games = 0     # games the learner has played vs this opp

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
    """Bounded FIFO pool of frozen actor snapshots with PFSP sampling."""
    def __init__(self, max_size, state_dim, action_dim, layer_sizes,
                 pfsp=True, pow=1.0, eps=0.05, ema_alpha=0.1):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer_sizes = layer_sizes
        self.pfsp = pfsp
        self.pow = pow              # PFSP 'hard' exponent (0 -> uniform)
        self.eps = eps             # min sampling weight floor (anti-forgetting)
        self.ema_alpha = ema_alpha  # responsiveness of per-opp win-rate EMA
        self._pool = deque(maxlen=max_size)

    def __len__(self):
        return len(self._pool)

    def add(self, actor):
        """Snapshot the given (live) actor into the pool."""
        self._pool.append(FrozenActor(actor.get_weights(), self.state_dim,
                                      self.action_dim, self.layer_sizes))

    def _weights(self):
        """PFSP sampling weights over the current pool (len>0)."""
        if not self.pfsp or self.pow == 0:
            return np.ones(len(self._pool), dtype=np.float64)
        w = np.empty(len(self._pool), dtype=np.float64)
        for i, opp in enumerate(self._pool):
            if opp.lwr is None:
                w[i] = 1.0                         # max priority until measured
            else:
                # 'hard' weighting: focus on opponents the learner loses to.
                w[i] = max((1.0 - opp.lwr) ** self.pow, self.eps)
        return w

    def sample(self):
        """Pick a frozen opponent by PFSP weight. Caller ensures len>0."""
        w = self._weights()
        p = w / w.sum()
        idx = int(np.random.choice(len(self._pool), p=p))
        return self._pool[idx]

    def record_result(self, opp, learner_won):
        """Update an opponent's EMA learner-win-rate after a league game."""
        o = 1.0 if learner_won else 0.0
        if opp.lwr is None:
            opp.lwr = o
        else:
            a = self.ema_alpha
            opp.lwr = (1.0 - a) * opp.lwr + a * o
        opp.games += 1

    def stats(self):
        """(pool_size, mean_measured_learner_wr, n_unmeasured) for logging."""
        measured = [o.lwr for o in self._pool if o.lwr is not None]
        mean_wr = float(np.mean(measured)) if measured else float('nan')
        n_unmeasured = sum(1 for o in self._pool if o.lwr is None)
        return len(self._pool), mean_wr, n_unmeasured
