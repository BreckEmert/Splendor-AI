# Splendor/RL/vrpo_loop.py
"""
VRPO training loop. Each iteration gathers a fresh batch of self-play
transitions, converts them to Q-boosting advantages / critic targets, runs the
clipped policy update + critic regression, and periodically EVALUATES the greedy
policy head-to-head vs the DQN and vs random (the real strength signal).
"""

import numpy as np

from .vrpo_trainer import VRPOAgent
from .vrpo_game import VRPOGame
from .vrpo_eval import evaluate, KerasGreedyOpponent


def _load_opponents(paths):
    """Best-effort load of eval opponents; returns (dqn_agent_or_None, random_agent)."""
    from .random_model import RandomAgent
    random_opp = RandomAgent(paths)

    dqn_opp = None
    dqn_path = paths.get('dqn_eval_path')
    if dqn_path:
        try:
            dqn_opp = KerasGreedyOpponent(dqn_path)
            print(f"Eval opponent (DQN) loaded: {dqn_path}")
        except Exception as e:
            print(f"WARNING: could not load DQN eval opponent ({e}). "
                  f"Skipping vs-DQN eval.")
    return dqn_opp, random_opp


def vrpo_loop(paths, iterations=5000, save_every=50, log_every=1,
              rollout_size=None):
    agent = VRPOAgent(paths)
    if rollout_size:
        agent.rollout_size = rollout_size

    players = [('Player1', agent, 0), ('Player2', agent, 1)]
    game = VRPOGame(players, agent, max_half_turns=agent.max_half_turns)

    dqn_opp, random_opp = (None, None)
    if agent.eval_every:
        dqn_opp, random_opp = _load_opponents(paths)

    print(f"Starting VRPO: {iterations} iterations, "
          f"~{agent.rollout_size} transitions/iter.")

    for it in range(iterations):
        agent.iteration = it

        # --- Collect an on-policy rollout ---
        batch_S, batch_A, batch_LP, batch_MK, batch_ADV, batch_QT = \
            [], [], [], [], [], []
        game_lengths = []
        collected = 0
        while collected < agent.rollout_size:
            seat0, seat1 = game.play_game()
            game_lengths.append(game.half_turns)
            for traj in (seat0, seat1):
                if not traj:
                    continue
                st, ac, lp, mk, adv, qt = agent.process_trajectory(traj)
                batch_S.append(st); batch_A.append(ac); batch_LP.append(lp)
                batch_MK.append(mk); batch_ADV.append(adv); batch_QT.append(qt)
                collected += len(st)

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

        # --- Evaluate (the real strength signal) ---
        if agent.eval_every and it % agent.eval_every == 0:
            wr_dqn = dr_dqn = float('nan')
            if dqn_opp is not None:
                wr_dqn, _, draws = evaluate(agent, dqn_opp, agent.eval_games,
                                            agent.max_half_turns)
                dr_dqn = draws
            wr_rnd, _, draws_r = evaluate(agent, random_opp, agent.eval_games,
                                          agent.max_half_turns)
            agent.log_eval(wr_dqn, dr_dqn, wr_rnd, draws_r)
            print(f"[iter {it}] EVAL vs_dqn={wr_dqn:.3f} vs_random={wr_rnd:.3f}")

            # Track best model by vs_dqn (the goal metric), falling back to
            # vs_random when no DQN opponent is available.
            best_metric = wr_dqn if dqn_opp is not None else wr_rnd
            agent.maybe_save_best(best_metric)

        if it % 10 == 0:
            print(f"[iter {it}] pg={metrics['pg_loss']:.4f} "
                  f"kl={metrics['kl_to_uniform']:.4f} "
                  f"ent={metrics['entropy']:.4f} "
                  f"critic={metrics['critic_loss']:.4f} "
                  f"ratio={metrics['mean_ratio']:.3f} "
                  f"clipfrac={metrics['clip_fraction']:.3f} "
                  f"avg_turns={np.mean(game_lengths)/2:.1f}")

        if save_every and it > 0 and it % save_every == 0:
            agent.save_model()

    agent.save_model()
    print("VRPO training complete.")
