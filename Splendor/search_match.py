# Splendor/search_match.py
"""
Head-to-head: PUCT search vs a greedy net - the decisive "is search worth
test-time compute?" experiment.

The default matchup is SAME WEIGHTS on both sides: the tournament-champion
actor playing greedily (one forward pass, argmax) vs the identical actor
wrapped in search (RL/mcts.py, actor as prior + critic as leaf value). Any
gap between them is therefore attributable purely to search.

Usage (from Splendor/, container or host):
    python search_match.py                       # defaults: 150 sims, 20 games
    python search_match.py --sims 300 --games 40
    python search_match.py --opponent RL/trained_agents/old_inference_models/PRE_VRPO_dqn_inference_model.keras

Games run sequentially (search is inherently sequential), seats alternate,
draws/timeouts credit neither side. Expect roughly ~0.5-2s per search move
depending on --sims and hardware.
"""

import os
import sys
import time
import argparse

import numpy as np

from RL.mcts import SearchAgent
from RL.vrpo_eval import EvalGame, KerasGreedyOpponent

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = os.path.join(HERE, "RL", "trained_agents")
# Tournament #1 (the PFSP-league champion) - actor is the prior, critic the
# leaf evaluator; both saved by best-checkpointing during its run.
CHAMP_ACTOR = os.path.join(
    AGENTS, "05-31-21-54__vrpo_512-512-256__kl0.06_alr3e-4_clr1.5e-4_cbuf4_best_actor.keras")
CHAMP_CRITIC = CHAMP_ACTOR.replace("_actor.keras", "_critic.keras")


def play_one(search, opp, search_seat, max_half_turns):
    """One game; returns +1 search win, -1 opp win, 0 draw/timeout."""
    game = EvalGame([('A', None, 0), ('B', None, 1)],
                    max_half_turns=max_half_turns)
    while not game.victor and game.half_turns < max_half_turns:
        seat = game.half_turns % 2
        if seat == search_seat:
            a = search.choose_move(game)
        else:
            state = game.to_state()
            mask = game.active_player.get_legal_moves(game.board)
            a = int(np.argmax(opp.get_predictions(state, mask)))
        game.step_move(a)

    if not game.victor:
        return 0
    winner_seat = 0 if game.players[0].victor else 1
    return 1 if winner_seat == search_seat else -1


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--c-puct", type=float, default=2.0)
    ap.add_argument("--eval-batch", type=int, default=8,
                    help="leaves evaluated per batched NN call (1 = sequential)")
    ap.add_argument("--actor", default=CHAMP_ACTOR)
    ap.add_argument("--critic", default=CHAMP_CRITIC)
    ap.add_argument("--aznet", default=None,
                    help="two-headed AZ net .keras; overrides --actor/--critic "
                         "for the search side")
    ap.add_argument("--opponent", default=None,
                    help="opponent .keras (default: same actor, greedy)")
    ap.add_argument("--max-half-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    opp_path = args.opponent or args.actor
    search_desc = args.aznet or args.actor
    print(f"search : {os.path.basename(search_desc)}  ({args.sims} sims/move)")
    print(f"greedy : {os.path.basename(opp_path)}")

    if args.aznet:
        search = SearchAgent(args.aznet, None, sims=args.sims,
                             c_puct=args.c_puct, eval_batch=args.eval_batch,
                             max_half_turns=args.max_half_turns, seed=args.seed)
    else:
        search = SearchAgent(args.actor, args.critic, sims=args.sims,
                             c_puct=args.c_puct, eval_batch=args.eval_batch,
                             max_half_turns=args.max_half_turns, seed=args.seed)
    opp = KerasGreedyOpponent(opp_path)

    wins = losses = draws = 0
    t0 = time.perf_counter()
    for g in range(args.games):
        r = play_one(search, opp, g % 2, args.max_half_turns)
        wins += r == 1
        losses += r == -1
        draws += r == 0
        elapsed = time.perf_counter() - t0
        print(f"game {g+1:3d}/{args.games}: {'W' if r==1 else 'L' if r==-1 else 'D'}"
              f"   running {wins}-{losses}-{draws}"
              f"   ({elapsed/ (g+1):.0f}s/game)")

    n = max(wins + losses, 1)
    print(f"\nFINAL search vs greedy: {wins}-{losses}-{draws}"
          f"   winrate(decisive)={wins/n:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
