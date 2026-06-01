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
                       max_half_turns, league=None, league_prob=0.0):
    """Run n_parallel VRPOGames in lockstep, batching the policy forward pass.

    players_template: list of (name, agent, pos) reused for every game.
    Returns (trajectories, game_lengths): trajectories is a list of
    (seat0_traj, seat1_traj) per finished game (one side empty for league games).

    League play: if `league` is given and non-empty, each game independently has
    probability `league_prob` of being a LEARNER-vs-FROZEN game - one seat is the
    current agent (learner), the other a random frozen snapshot. Only the
    learner's trajectory is returned for those games (the frozen opponent is just
    environment), keeping the update on-policy. The remaining games are ordinary
    self-play (both seats = agent, both seats collected). When league is None/
    empty this is exactly the original pure-self-play loop.
    """
    games = [VRPOGame(players_template, agent, max_half_turns=max_half_turns)
             for _ in range(n_parallel)]
    use_league = league is not None and len(league) > 0

    def assign(g, counter):
        """(Re)start a game and assign its per-seat policies + learner seat."""
        g.reset()
        if use_league and np.random.rand() < league_prob:
            learner = counter % 2                 # alternate learner seat
            opp = league.sample()
            g.learner_seat = learner
            g.opp_ref = opp                       # for PFSP record_result
            g.seat_policies = [None, None]
            g.seat_policies[learner] = agent
            g.seat_policies[1 - learner] = opp
        else:
            g.learner_seat = None                 # pure self-play: collect both
            g.opp_ref = None
            g.seat_policies = [agent, agent]

    for i, g in enumerate(games):
        assign(g, i)

    trajectories = []
    game_lengths = []
    collected = 0
    counter = n_parallel                          # for learner-seat alternation

    # Fast path: no league -> single batched forward over all games (original).
    if not use_league:
        while collected < rollout_size:
            states = np.empty((n_parallel, agent.state_dim), dtype=np.float32)
            masks = np.empty((n_parallel, agent.action_dim), dtype=bool)
            for i, g in enumerate(games):
                s, m = g.observe()
                states[i] = s; masks[i] = m
            actions, logps = agent.act_batch(states, masks)
            for i, g in enumerate(games):
                g.step_external(int(actions[i]), float(logps[i]))
                if g.victor or g.half_turns >= max_half_turns:
                    game_lengths.append(g.half_turns)
                    if not g.victor:
                        agent._truncated_games += 1
                    trajectories.append(g._split_seats())
                    collected += len(g.mem)
                    g.reset()
        return trajectories, game_lengths

    # League path: group active games by the policy whose turn it is, batch each.
    while collected < rollout_size:
        groups = {}                               # id(pol) -> (pol, [idx...])
        obs = {}
        for idx, g in enumerate(games):
            s, m = g.observe()
            obs[idx] = (s, m)
            pol = g.seat_policies[g.half_turns % 2]
            groups.setdefault(id(pol), (pol, []))[1].append(idx)

        decided = {}                              # idx -> (action, logp)
        for pol, idxs in groups.values():
            S = np.asarray([obs[i][0] for i in idxs], dtype=np.float32)
            M = np.asarray([obs[i][1] for i in idxs], dtype=bool)
            a, lp = pol.act_batch(S, M)
            for k, i in enumerate(idxs):
                decided[i] = (int(a[k]), float(lp[k]))

        for idx, g in enumerate(games):
            act, lp = decided[idx]
            g.step_external(act, lp)
            if g.victor or g.half_turns >= max_half_turns:
                game_lengths.append(g.half_turns)
                if not g.victor:
                    agent._truncated_games += 1
                if g.learner_seat is None:
                    s0, s1 = g._split_seats()
                    trajectories.append((s0, s1))
                    collected += len(s0) + len(s1)
                else:
                    learner_traj = [e for e in g.mem if e[7] == g.learner_seat]
                    trajectories.append((learner_traj, []))
                    collected += len(learner_traj)
                    # PFSP bookkeeping: record whether the learner beat this opp
                    # (a truncated/no-victor game counts as a non-win, which is
                    # the conservative choice). Drives prioritized sampling.
                    if g.opp_ref is not None:
                        won = bool(g.victor and g.players[g.learner_seat].victor)
                        league.record_result(g.opp_ref, won)
                assign(g, counter)
                counter += 1

    return trajectories, game_lengths
