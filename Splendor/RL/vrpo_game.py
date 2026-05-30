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
loser-reward poke), state encoding, legal-move generation - is reused
verbatim. apply_move pokes self.model.memory[-1][2] and [5]; our collected
entries keep reward at index 2 and done at index 5 so that still works.
"""

from Environment.rl_game import RLGame
from .rewards import BasicRewardEngine


class VRPOGame(RLGame):
    def __init__(self, players, agent, reward_cls=BasicRewardEngine,
                 max_half_turns=200):
        super().__init__(players, agent)
        # Override RLGame's hardcoded SparseRewardEngine.
        self.rewards = reward_cls(self)
        self.max_half_turns = max_half_turns

    def turn(self):
        state = self.to_state()
        seat = self.half_turns % 2
        mask = self.active_player.get_legal_moves(self.board)

        action, logp = self.model.act(state, mask)
        self.move_idx = action

        self.rewards._cache.clear()
        reward = self.apply_move(action)         # may set victor + poke memory[-1]
        reward -= self.rewards.constant_penalty
        self.half_turns += 1

        next_state = self.to_state()
        done = bool(self.victor)
        # Layout: [state, action, reward, next_state, mask, done, logp, seat]
        # (first six mirror the DQN memory so reused apply_move stays valid)
        entry = [state, action, reward, next_state, mask, done, logp, seat]
        self.model.remember(entry)

    def play_game(self):
        """Play one full self-play game; return (seat0_traj, seat1_traj),
        each a list of transitions in that seat's own decision order.
        """
        self.model.memory = []          # fresh per-game buffer
        self.reset()

        while not self.victor and self.half_turns < self.max_half_turns:
            self.turn()

        if not self.victor:
            # Stalled game hit the safety cap; count it for diagnostics.
            self.model._truncated_games += 1

        mem = self.model.memory
        seat0 = [e for e in mem if e[7] == 0]
        seat1 = [e for e in mem if e[7] == 1]
        return seat0, seat1
