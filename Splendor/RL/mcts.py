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
(Measured: same-weights search vs greedy = 63% @25 sims, 83% @150, 97% @400;
search @150 sims beat the original DQN 29-1.)

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
  Backed-up search values are therefore ALWAYS in [-1, 1] (tanh leaves, exact
  +/-1 terminals) regardless of the critic's output scale - which is what
  makes root W/N usable as scale-consistent critic targets for the AlphaZero
  training loop (RL/azero.py).
- BATCHED LEAF EVALUATION: profiling showed ~all per-sim cost was the two
  single-row NN calls per leaf, not the engine. We run `eval_batch` descents
  to their leaves, then evaluate every pending leaf in ONE batched actor call
  + ONE batched critic call. To stop in-batch descents from piling down the
  identical path, each descent applies a VIRTUAL LOSS (N+1, W-1) along its
  edges, undone exactly when its real value is backed up. eval_batch=1
  reproduces the fully sequential algorithm exactly. Measured: 0.75s ->
  0.18s/move (batch 8) at 150 sims; strength preserved (9-3 vs greedy).
- ROOT DIRICHLET NOISE (training-data generation only): root_noise=True mixes
  Dir(alpha) over legal moves into the root prior, AlphaZero-style, so
  self-play data explores beyond the actor's current preferences. Eval play
  leaves it off.
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
    (leaf value). Accepts .keras paths or live in-memory Keras models (the
    AlphaZero loop passes the models it is training). choose_move(game) reads
    the live game and never mutates it.
    """

    def __init__(self, actor, critic, sims=150, c_puct=2.0,
                 value_scale=5.0, max_half_turns=300, eval_batch=8,
                 root_noise=False, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                 seed=None):
        self.actor = load_model(actor, compile=False) if isinstance(actor, str) else actor
        self.critic = load_model(critic, compile=False) if isinstance(critic, str) else critic
        self.sims = sims
        self.c_puct = c_puct
        self.value_scale = value_scale
        self.max_half_turns = max_half_turns
        self.eval_batch = max(1, eval_batch)
        self.root_noise = root_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        # Scratch game for simulations, restored from a snapshot every sim.
        # Player agents are never consulted (we drive apply_move directly).
        self._sim = EvalGame([('S0', None, 0), ('S1', None, 1)],
                             max_half_turns=max_half_turns)

    # -- NN helpers ------------------------------------------------------ #
    def _priors_value(self, logits, q, legal):
        """Masked-softmax priors + tanh-squashed policy-expectation value."""
        z = np.where(legal, logits, -np.inf)
        z = z - z.max()
        e = np.exp(z, where=np.isfinite(z), out=np.zeros_like(z))
        priors = e / e.sum()
        v = float(np.tanh((priors * q).sum() / self.value_scale))
        return priors, v

    def _evaluate_single(self, game):
        """Single-row eval (root only). Returns (priors, legal, value)."""
        state = game.to_state()
        legal = game.active_player.get_legal_moves(game.board)
        s = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        logits = self.actor(s, training=False).numpy()[0]
        q = self.critic(s, training=False).numpy()[0]
        priors, v = self._priors_value(logits, q, legal)
        return priors, legal, v

    # -- public API ------------------------------------------------------ #
    def choose_move(self, game):
        """Eval play: most-visited root move (no root noise unless set)."""
        root = self._run_sims(game)
        return int(np.argmax(root.N))

    def search_policy(self, game):
        """Training-data generation: run the same search and expose the root
        statistics. Returns (pi, N, Q, legal) where pi = N / sum(N) is the
        visit distribution (the actor's training target), and Q[a] = W[a]/N[a]
        for visited actions (0 elsewhere) are the search-backed action values
        in [-1, 1] (the critic's training targets). Caller handles temperature
        sampling vs argmax.
        """
        root = self._run_sims(game)
        n_sum = root.N.sum()
        pi = root.N / n_sum if n_sum > 0 else root.P
        q = np.divide(root.W, root.N, out=np.zeros_like(root.W),
                      where=root.N > 0)
        return pi, root.N.copy(), q, root.legal.copy()

    # -- core ------------------------------------------------------------ #
    def _run_sims(self, game):
        root_snap = snapshot(game)
        tree = {}

        # Root expansion (root state is fully known; decks only affect draws).
        restore(self._sim, root_snap)
        priors, legal, _ = self._evaluate_single(self._sim)
        if self.root_noise:
            idx = np.flatnonzero(legal)
            noise = self._np_rng.dirichlet(
                np.full(len(idx), self.dirichlet_alpha))
            priors = priors * (1.0 - self.dirichlet_eps)
            priors[idx] += self.dirichlet_eps * noise
        tree[()] = _Node(priors, legal)

        done = 0
        while done < self.sims:
            k = min(self.eval_batch, self.sims - done)

            # Phase 1: run k descents, applying virtual loss along each path.
            pending = []
            for _ in range(k):
                restore(self._sim, root_snap)
                for d in self._sim.board.decks:      # determinize hidden order
                    self._rng.shuffle(d.cards)

                path = ()
                edges = []                            # [(node, action), ...]
                v = None
                state = mask = None
                expand = False
                while True:
                    node = tree[path]
                    cur_legal = self._sim.active_player.get_legal_moves(
                        self._sim.board)
                    avail = node.legal & cur_legal
                    if not avail.any():
                        avail = cur_legal             # tree mask useless here
                    a = self._select(node, avail)
                    edges.append((node, a))
                    node.N[a] += 1.0                  # virtual loss on
                    node.W[a] -= 1.0

                    self._sim.rewards._cache.clear()
                    self._sim.apply_move(a)
                    self._sim.half_turns += 1

                    if self._sim.victor:
                        v = -1.0                      # mover won; to-move lost
                        break
                    if self._sim.half_turns >= self.max_half_turns:
                        state = self._sim.to_state()  # value only, no expand
                        mask = self._sim.active_player.get_legal_moves(
                            self._sim.board)
                        break

                    path = path + (a,)
                    if path not in tree:
                        state = self._sim.to_state()
                        mask = self._sim.active_player.get_legal_moves(
                            self._sim.board)
                        expand = True
                        break

                pending.append({'path': path, 'edges': edges, 'v': v,
                                'state': state, 'mask': mask,
                                'expand': expand})

            # Phase 2: ONE batched actor + critic call for all pending leaves.
            need = [p for p in pending if p['state'] is not None]
            if need:
                S = tf.convert_to_tensor(
                    np.stack([p['state'] for p in need]), dtype=tf.float32)
                logits = self.actor(S, training=False).numpy()
                qs = self.critic(S, training=False).numpy()
                for row, p in enumerate(need):
                    pri, v = self._priors_value(logits[row], qs[row], p['mask'])
                    p['v'] = v
                    # Two in-batch descents can reach the same new path; the
                    # first expansion wins, the second still backs up its value.
                    if p['expand'] and p['path'] not in tree:
                        tree[p['path']] = _Node(pri, p['mask'])

            # Phase 3: undo virtual loss and apply the real negamax backup.
            for p in pending:
                w = p['v']
                for node, a in reversed(p['edges']):
                    node.N[a] -= 1.0                  # virtual loss off
                    node.W[a] += 1.0
                    w = -w
                    node.W[a] += w
                    node.N[a] += 1.0

            done += k

        return tree[()]

    def _select(self, node, avail):
        idx = np.flatnonzero(avail)
        n, w, p = node.N[idx], node.W[idx], node.P[idx]
        q = np.where(n > 0, w / np.maximum(n, 1), 0.0)
        u = self.c_puct * p * np.sqrt(node.N.sum() + 1.0) / (1.0 + n)
        return int(idx[np.argmax(q + u)])
