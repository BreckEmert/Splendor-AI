# Splendor/RL/vrpo_game.py
"""
Self-play rollout for VRPO.

VRPOGame subclasses the existing RLGame and changes ONLY how a turn is taken
and collected:
  - moves are SAMPLED from the policy (not argmax'd) and the behavior log-prob
    is recorded, because PPO needs the rollout policy to form its ratio;
  - the acting-state legal mask is stored (the DQN stored the *next* player's
    mask, which PPO doesn't want);
  - the reward engine defaults to BasicRewardEngine (full shaping) instead of
    RLGame's SparseRewardEngine.

Everything else - board mechanics, apply_move (including its end-of-game
loser-reward poke), state encoding, legal-move generation - is reused verbatim.

PERFORMANCE: rollout was 96% of wall-clock because act() ran one single-row
Keras call per turn (~2048 tiny inferences/iter, dominated by call overhead).
collect_vectorized() runs G games in lockstep and does ONE batched act() of G
rows per step, collapsing those calls ~G-fold. Each game keeps its own memory
list; the shared agent.memory pointer is aimed at the current game's list right
before apply_move so apply_move's end-of-game poke (memory[-1]) still lands
correctly. Single-threaded, so the pointer swap is race-free.
"""

import numpy as np

from Environment.rl_game import RLGame
from .rewards import BasicRewardEngine


class VRPOGame(RLGame):
    def __init__(self, players, agent, reward_cls=BasicRewardEngine,
                 max_half_turns=200):
        super().__init__(players, agent)
        self.rewards = reward_cls(self)        # override RLGame's SparseRewardEngine
        self.max_half_turns = max_half_turns
        self.mem = []

    def reset(self):
        super().reset()
        self.mem = []

    # ---- single-game path (kept for tests / clarity) ----------------- #
    def turn(self):
        state = self.to_state()
        seat = self.half_turns % 2
        mask = self.active_player.get_legal_moves(self.board)

        action, logp = self.model.act(state, mask)
        self._commit(state, mask, seat, action, logp)

    def play_game(self):
        self.model.memory = self.mem = []
        self.reset()
        while not self.victor and self.half_turns < self.max_half_turns:
            self.turn()
        return self._split_seats()

    # ---- vectorized path -------------------------------------------- #
    def observe(self):
        """Capture (state, mask) for the active player; return them for the
        batched forward pass. Stored so _commit can reuse the exact state/mask.
        """
        self._pending_state = self.to_state()
        self._pending_mask = self.active_player.get_legal_moves(self.board)
        return self._pending_state, self._pending_mask

    def step_external(self, action, logp):
        """Apply an externally-chosen (already-sampled) action + its logp."""
        seat = self.half_turns % 2
        self._commit(self._pending_state, self._pending_mask, seat, action, logp)

    def _commit(self, state, mask, seat, action, logp):
        self.model.memory = self.mem          # aim apply_move's poke at our list
        self.move_idx = action
        self.rewards._cache.clear()
        reward = self.apply_move(action)
        reward -= self.rewards.constant_penalty
        self.half_turns += 1
        next_state = self.to_state()
        done = bool(self.victor)
        # [state, action, reward, next_state, mask, done, logp, seat]
        self.mem.append([state, action, reward, next_state, mask, done, logp, seat])

    def _split_seats(self):
        seat0 = [e for e in self.mem if e[7] == 0]
        seat1 = [e for e in self.mem if e[7] == 1]
        return seat0, seat1


def collect_vectorized(agent, players_template, n_parallel, rollout_size,
                       max_half_turns):
    """Run n_parallel VRPOGames in lockstep, batching the policy forward pass.

    players_template: list of (name, agent, pos) reused for every game.
    Returns (trajectories, game_lengths) where trajectories is a list of
    (seat0, seat1) tuples for each FINISHED game.
    """
    games = [VRPOGame(players_template, agent, max_half_turns=max_half_turns)
             for _ in range(n_parallel)]
    for g in games:
        g.reset()

    trajectories = []
    game_lengths = []
    collected = 0

    while collected < rollout_size:
        # 1) Gather observations from all live games.
        states = np.empty((n_parallel, agent.state_dim), dtype=np.float32)
        masks = np.empty((n_parallel, agent.action_dim), dtype=bool)
        for i, g in enumerate(games):
            s, m = g.observe()
            states[i] = s
            masks[i] = m

        # 2) ONE batched policy forward + sample for all games.
        actions, logps = agent.act_batch(states, masks)

        # 3) Apply each action to its game; finalize + recycle finished slots.
        for i, g in enumerate(games):
            g.step_external(int(actions[i]), float(logps[i]))
            if g.victor or g.half_turns >= max_half_turns:
                game_lengths.append(g.half_turns)
                if not g.victor:
                    agent._truncated_games += 1
                trajectories.append(g._split_seats())
                collected += len(g.mem)
                g.reset()                      # recycle the slot

    return trajectories, game_lengths
