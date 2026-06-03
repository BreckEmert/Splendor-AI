# Splendor/tournament.py
"""
Round-robin tournament across saved Splendor agents -> win matrix + ratings.

WHY: self-play game-length and even win-rate-vs-one-fixed-DQN can't rank a set
of champions against each OTHER (the metric we actually care about once we have
many strong models, e.g. to pick the webapp model or judge whether a league /
reward-anneal agent is genuinely stronger). This plays every agent vs every
other, greedily, seats alternated, and fits Bradley-Terry ratings from the
pairwise results.

All agents are loaded as KerasGreedyOpponent (argmax over the model's output):
- a DQN .keras -> argmax over Q-values (its normal greedy policy);
- a VRPO actor .keras -> argmax over policy logits (its greedy policy).
Both are the correct greedy decision rule, so they're directly comparable.

Usage (from Splendor/, in the container):
    python tournament.py                  # all *_best_actor.keras + inference_model
    python tournament.py --games 100      # games per ordered pair (default 60)
    python tournament.py --glob '*cbuf4_best_actor.keras' --include-dqn
    python tournament.py --agents a.keras b.keras c.keras
Reads model files on disk; no training. Writes a CSV + prints the leaderboard.
"""

import os
import sys
import glob
import argparse

import numpy as np

from RL.vrpo_eval import KerasGreedyOpponent, evaluate_vectorized

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(HERE, "RL", "trained_agents")


def _discover(pattern, include_dqn):
    paths = sorted(glob.glob(os.path.join(AGENTS_DIR, pattern)))
    if include_dqn:
        dqn = os.path.join(AGENTS_DIR, "inference_model.keras")
        if os.path.exists(dqn) and dqn not in paths:
            paths.append(dqn)
    return paths


def _short(path):
    name = os.path.basename(path)
    name = name.replace("__vrpo_512-512-256__", "_").replace("_best_actor.keras", "")
    return name.replace(".keras", "")


def _bradley_terry(wins, games, iters=500):
    """Fit BT strengths from win/game matrices. Returns ratings (mean 0, in
    'logit' units) then scaled to a familiar Elo-like spread for readability."""
    n = len(wins)
    s = np.ones(n)
    W = wins.sum(axis=1)                       # total wins per player
    for _ in range(iters):
        s_new = np.empty(n)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                nij = games[i, j] + games[j, i]
                if nij > 0:
                    denom += nij / (s[i] + s[j])
            s_new[i] = (W[i] + 1e-9) / (denom + 1e-9)
        s_new /= np.exp(np.mean(np.log(s_new)))   # normalize geometric mean -> 1
        if np.max(np.abs(np.log(s_new) - np.log(s))) < 1e-9:
            s = s_new
            break
        s = s_new
    logits = np.log(s)
    elo = 400.0 / np.log(10) * logits             # logit -> Elo-like points
    return elo


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60,
                    help="games per ORDERED pair (total per pair = 2x, seats alternate)")
    ap.add_argument("--glob", default="*_best_actor.keras")
    ap.add_argument("--include-dqn", action="store_true", default=True)
    ap.add_argument("--no-dqn", dest="include_dqn", action="store_false")
    ap.add_argument("--agents", nargs="*", default=None,
                    help="explicit list of .keras paths (overrides --glob)")
    ap.add_argument("--max-half-turns", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(HERE, "tournament_results.csv"))
    args = ap.parse_args(argv)

    paths = args.agents if args.agents else _discover(args.glob, args.include_dqn)
    if len(paths) < 2:
        print(f"Need >=2 agents; found {len(paths)} for pattern '{args.glob}'.")
        return
    names = [_short(p) for p in paths]
    print(f"Loading {len(paths)} agents...")
    agents = [KerasGreedyOpponent(p) for p in paths]

    n = len(agents)
    wins = np.zeros((n, n))      # wins[i,j] = games i won vs j
    games = np.zeros((n, n))
    # Each unordered pair: evaluate_vectorized already alternates seats.
    for i in range(n):
        for j in range(i + 1, n):
            wr, w, d = evaluate_vectorized(agents[i], agents[j], args.games,
                                           args.max_half_turns)
            wi = w                          # i's wins
            wj = args.games - w - d         # j's wins (draws/timeouts to neither)
            wins[i, j] = wi
            wins[j, i] = wj
            games[i, j] = games[j, i] = args.games   # symmetric total exposure
            print(f"  {names[i][:28]:28s} vs {names[j][:28]:28s}  "
                  f"{wi:3.0f}-{wj:3.0f}  (draws {d})")

    elo = _bradley_terry(wins, games)
    order = np.argsort(-elo)

    # Leaderboard
    print("\n=== LEADERBOARD (Bradley-Terry, Elo-scaled, mean=0) ===")
    total_games = games.sum(axis=1)
    total_wins = wins.sum(axis=1)
    for rank, idx in enumerate(order, 1):
        wr = total_wins[idx] / max(total_games[idx], 1)
        print(f"{rank:2d}. {elo[idx]:+7.1f}  wr={wr:.3f}  {names[idx]}")

    # CSV win-matrix
    with open(args.out, "w") as f:
        f.write("agent," + ",".join(names) + ",elo,overall_wr\n")
        for i in range(n):
            row = [f"{wins[i,j]:.0f}/{games[i,j]:.0f}" if i != j else "-"
                   for j in range(n)]
            wr = total_wins[i] / max(total_games[i], 1)
            f.write(f"{names[i]}," + ",".join(row) + f",{elo[i]:.1f},{wr:.3f}\n")
    print(f"\nWrote win matrix -> {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
