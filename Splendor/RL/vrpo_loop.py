# Splendor/RL/vrpo_loop.py
"""
VRPO training loop. Each iteration gathers a fresh batch of self-play
transitions (vectorized), converts them to Q-boosting advantages / critic
targets, runs the clipped policy update + critic regression, and periodically
EVALUATES the greedy policy head-to-head vs the fixed DQN inference model -
the one true strength signal (self-play game length only correlates while it is
shrinking, so we do not rely on it).
"""

import numpy as np

from .vrpo_trainer import VRPOAgent
from .vrpo_game import collect_vectorized
from .vrpo_eval import evaluate_vectorized, KerasGreedyOpponent
from .vrpo_league import LeaguePool


def _load_dqn_opponent(paths):
    """Load the fixed DQN eval opponent, or None if unavailable."""
    dqn_path = paths.get('dqn_eval_path')
    if not dqn_path:
        return None
    try:
        opp = KerasGreedyOpponent(dqn_path)
        print(f"Eval opponent (DQN) loaded: {dqn_path}")
        return opp
    except Exception as e:
        print(f"WARNING: could not load DQN eval opponent ({e}). "
              f"Skipping eval.")
        return None


def vrpo_loop(paths, iterations=5000, save_every=50, log_every=None,
              rollout_size=None):
    agent = VRPOAgent(paths)
    if rollout_size:
        agent.rollout_size = rollout_size
    # Metrics logging cadence: explicit arg wins, else the agent's (env) value.
    if log_every is None:
        log_every = agent.log_every

    players = [('Player1', agent, 0), ('Player2', agent, 1)]

    dqn_opp = _load_dqn_opponent(paths) if agent.eval_every else None

    # League: bounded pool of frozen past-self snapshots. Seed immediately so
    # there is an opponent from iter 0 (useful when resuming a strong model).
    league = None
    if agent.league_on:
        league = LeaguePool(agent.league_pool, agent.state_dim,
                            agent.action_dim, paths['layer_sizes'])
        league.add(agent.actor)
        print(f"League ON: pool<= {agent.league_pool}, prob={agent.league_prob}, "
              f"snapshot every {agent.snapshot_every} iters.")

    print(f"Starting VRPO: {iterations} iterations, "
          f"~{agent.rollout_size} transitions/iter.")

    for it in range(iterations):
        agent.iteration = it

        # Periodically snapshot the current actor into the league pool.
        if league is not None and it > 0 and it % agent.snapshot_every == 0:
            league.add(agent.actor)

        # --- Collect an on-policy rollout (vectorized: G games in lockstep) ---
        batch_S, batch_A, batch_LP, batch_MK, batch_ADV, batch_QT = \
            [], [], [], [], [], []
        trajectories, game_lengths = collect_vectorized(
            agent, players, agent.parallel_games,
            agent.rollout_size, agent.max_half_turns,
            league=league, league_prob=(agent.league_prob if league else 0.0))
        # Batch all trajectories' forward passes into 2 GPU calls (vs 2 per
        # trajectory) - see VRPOAgent.process_trajectories.
        flat = [t for seat0, seat1 in trajectories for t in (seat0, seat1) if t]
        for st, ac, lp, mk, adv, qt in agent.process_trajectories(flat):
            batch_S.append(st); batch_A.append(ac); batch_LP.append(lp)
            batch_MK.append(mk); batch_ADV.append(adv); batch_QT.append(qt)

        S = np.concatenate(batch_S)
        A = np.concatenate(batch_A)
        LP = np.concatenate(batch_LP)
        MK = np.concatenate(batch_MK)
        ADV = np.concatenate(batch_ADV)
        QT = np.concatenate(batch_QT)

        # --- Update ---
        metrics = agent.update(S, A, LP, MK, ADV, QT)

        if log_every and it % log_every == 0:
            agent.log_iteration(metrics, game_lengths, ADV, QT)

        # --- Evaluate (the real strength signal: vs the fixed DQN) ---
        if dqn_opp is not None and it % agent.eval_every == 0:
            wr_dqn, _, _ = evaluate_vectorized(
                agent, dqn_opp, agent.eval_games, agent.max_half_turns)
            agent.log_eval(wr_dqn)
            agent.maybe_save_best(wr_dqn)
            print(f"[iter {it}] EVAL vs_dqn={wr_dqn:.3f} "
                  f"(best {agent.best_eval:.3f} @ {agent.best_eval_iter})")

        if it % 10 == 0:
            print(f"[iter {it}] pg={metrics['pg_loss']:.4f} "
                  f"kl={metrics['kl_to_uniform']:.4f} "
                  f"ent={metrics['entropy']:.4f} "
                  f"critic={metrics['critic_loss']:.4f} "
                  f"ratio={metrics['mean_ratio']:.3f} "
                  f"clipfrac={metrics['clip_fraction']:.3f} "
                  f"lr={agent._current_lr():.2e} "
                  f"avg_turns={np.mean(game_lengths)/2:.1f}"
                  + (f" league={len(league)}" if league is not None else ""))

        if save_every and it > 0 and it % save_every == 0:
            agent.save_model()

    agent.save_model()
    print("VRPO training complete.")
