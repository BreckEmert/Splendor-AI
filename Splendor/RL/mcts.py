# Splendor/RL/mcts.py
"""
PUCT search (AlphaZero-style MCTS) at decision time, on top of the trained
VRPO actor + critic.

WHY: every training-time lever in the campaign (kl grid, bigger critic,
league/PFSP, reward anneal, LR schedules) plateaued in the same ~0.6 band -
the signature of a function-class limit, not a tuning limit. The agent decides
with ONE forward pass; Splendor endgames are tactical races and blocking
sequences that are depth problems, not representation problems. Search spends
test-time compute on the exact decision at hand, using the nets we already
have: the actor as a move prior, the critic as a leaf evaluator.

Design notes (each is a deliberate choice):
- SIMULATION on the real engine: we snapshot the live game, replay candidate
  lines on a scratch EvalGame via apply_move, and restore between simulations.
  Card/Noble objects are immutable data, so snapshots share their references
  and copy only the mutable containers (cheap).
- DETERMINIZATION: deck order is the only hidden state, and it is hidden from
  BOTH players (symmetric), so this is a stochastic perfect-information game,
  not poker. Each simulation reshuffles the scratch decks so search never
  peeks at the true future draws.
- OPEN-LOOP TREE: nodes are keyed by the action path from the root, because
  determinization means the same path can reach slightly different states
  (different refill cards). Priors are those computed at first expansion; at
  every descent step selection is restricted to actions legal in the CURRENT
  determinization, so we never apply an illegal move.
- NEGAMAX BACKUP: Splendor strictly alternates movers, so values flip sign
  each ply. Node values are from the perspective of the player to move at
  that node. A terminal reached by a move means the MOVER just won (game ends
  only on the winner's own buy), so the player to move at the terminal gets
  -1.
- LEAF VALUE: V = sum_a pi(a|s) * Q(s,a) from the critic (the same policy
  expectation VRPO trains), squashed with tanh(V / value_scale) into [-1, 1].
  The critic was trained on shaped rewards, so this is a heuristic ordering,
  not a calibrated win probability - fine for search, which mostly needs
  relative comparisons plus exact terminals.
"""

import random

import numpy as np
import tensorflow as tf
from keras.models import load_model

from .vrpo_eval import EvalGame


# --------------------------------------------------------------------------- #
# State snapshot / restore (Card and Noble objects are immutable: share refs)
# --------------------------------------------------------------------------- #
def snapshot(game):
    b = game.board
    return {
        'gems': b.gems.copy(),
        'cards': [list(tier) for tier in b.cards],
        'decks': [list(d.cards) for d in b.decks],
        'nobles': list(b.nobles),
        'players': [{
            'gems': p.gems.copy(),
            'cards': p.cards.copy(),
            'reserved': list(p.reserved_cards),
            'card_ids': [list(x) for x in p.card_ids],
            'points': p.points,
            'victor': p.victor,
        } for p in game.players],
        'half_turns': game.half_turns,
        'victor': game.victor,
    }


def restore(game, snap):
    b = game.board
    b.gems = snap['gems'].copy()
    b.cards = [list(tier) for tier in snap['cards']]
    for d, saved in zip(b.decks, snap['decks']):
        d.cards = list(saved)
    b.nobles = list(snap['nobles'])
    for p, ps in zip(game.players, snap['players']):
        p.gems = ps['gems'].copy()
        p.cards = ps['cards'].copy()
        p.reserved_cards = list(ps['reserved'])
        p.card_ids = [list(x) for x in ps['card_ids']]
        p.points = ps['points']
        p.victor = ps['victor']
    game.half_turns = snap['half_turns']
    game.victor = snap['victor']


# --------------------------------------------------------------------------- #
# PUCT search
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ('P', 'N', 'W', 'legal')

    def __init__(self, priors, legal):
        self.P = priors                      # (A,) prior over actions
        self.N = np.zeros_like(priors)       # visit counts
        self.W = np.zeros_like(priors)       # total backed-up value
        self.legal = legal                   # mask at first expansion


class SearchAgent:
    """Chooses moves by PUCT search using a trained actor (prior) + critic
    (leaf value). choose_move(game) reads the live game, never mutates it.
    """

    def __init__(self, actor_path, critic_path, sims=150, c_puct=2.0,
                 value_scale=5.0, max_half_turns=300, seed=None):
        self.actor = load_model(actor_path, compile=False)
        self.critic = load_model(critic_path, compile=False)
        self.sims = sims
        self.c_puct = c_puct
        self.value_scale = value_scale
        self.max_half_turns = max_half_turns
        self._rng = random.Random(seed)
        # Scratch game for simulations, restored from a snapshot every sim.
        # Player agents are never consulted (we drive apply_move directly).
        self._sim = EvalGame([('S0', None, 0), ('S1', None, 1)],
                             max_half_turns=max_half_turns)

    # -- NN helpers (single-row; search is sequential by nature) -------- #
    def _evaluate(self, game):
        """Priors over legal moves + tanh-squashed value, for the player to
        move in `game`. Returns (priors, legal_mask, value)."""
        state = game.to_state()
        legal = game.active_player.get_legal_moves(game.board)
        s = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        logits = self.actor(s, training=False).numpy()[0]
        q = self.critic(s, training=False).numpy()[0]

        z = np.where(legal, logits, -np.inf)
        z = z - z.max()
        e = np.exp(z, where=np.isfinite(z), out=np.zeros_like(z))
        priors = e / e.sum()

        v = float(np.tanh((priors * q).sum() / self.value_scale))
        return priors, legal, v

    # -- core ------------------------------------------------------------ #
    def choose_move(self, game):
        root_snap = snapshot(game)
        tree = {}

        # Root expansion (root state is fully known; decks only affect draws).
        restore(self._sim, root_snap)
        priors, legal, _ = self._evaluate(self._sim)
        tree[()] = _Node(priors, legal)

        for _ in range(self.sims):
            restore(self._sim, root_snap)
            for d in self._sim.board.decks:          # determinize hidden order
                self._rng.shuffle(d.cards)

            path = ()
            edges = []                                # [(node, action), ...]
            v = None
            while True:
                node = tree[path]
                # Restrict to moves legal in THIS determinization.
                cur_legal = self._sim.active_player.get_legal_moves(self._sim.board)
                avail = node.legal & cur_legal
                if not avail.any():
                    avail = cur_legal                # tree mask useless here
                a = self._select(node, avail)
                edges.append((node, a))

                self._sim.rewards._cache.clear()
                self._sim.apply_move(a)
                self._sim.half_turns += 1

                if self._sim.victor:
                    v = -1.0                          # mover won; to-move lost
                    break
                if self._sim.half_turns >= self.max_half_turns:
                    *_, v = self._evaluate(self._sim)
                    break

                path = path + (a,)
                if path not in tree:
                    priors, legal, v = self._evaluate(self._sim)
                    tree[path] = _Node(priors, legal)
                    break

            # Negamax backup: v is from the perspective of the player to move
            # at the leaf; flip once per edge walking back to the root.
            w = v
            for node, a in reversed(edges):
                w = -w
                node.W[a] += w
                node.N[a] += 1

        root = tree[()]
        return int(np.argmax(root.N))                # most-visited move

    def _select(self, node, avail):
        idx = np.flatnonzero(avail)
        n, w, p = node.N[idx], node.W[idx], node.P[idx]
        q = np.where(n > 0, w / np.maximum(n, 1), 0.0)
        u = self.c_puct * p * np.sqrt(node.N.sum() + 1.0) / (1.0 + n)
        return int(idx[np.argmax(q + u)])
