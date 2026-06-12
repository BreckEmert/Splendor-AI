# Splendor/RL/azero.py
"""
AlphaZero-style training loop: search generates the data, the nets learn from
search, stronger nets make search stronger. This is the compounding step the
campaign was missing - search alone added ~+590 Elo at decision time, but
those gains lived only in the tree; this loop bakes them into the weights.

Per generation:
  1. SELF-PLAY: games where BOTH seats move by PUCT search (RL/mcts.py) with
     root Dirichlet noise; for the first `temp_moves` half-turns the move is
     SAMPLED from the visit distribution (diverse openings), then argmax.
  2. TARGETS per visited state (always from the mover's perspective):
       pi    = root visit distribution      -> actor cross-entropy target
       q[a]  = root W[a]/N[a] per visited a -> critic regression target, with
               the PLAYED action's target replaced by the true outcome z.
     Search-backed q targets are the deliberate deviation from vanilla
     AlphaZero (which regresses a scalar V on z): our critic is a Q-VECTOR,
     and z-only targets would leave unplayed actions at the old shaped-reward
     scale, corrupting the sum(pi*Q) leaf values search relies on. Root W/N is
     already in [-1, 1] (tanh leaves, exact terminals), so the critic becomes
     scale-consistent from the first generation.
  3. TRAIN actor (masked CE vs pi) + critic (masked MSE vs q) over a sliding
     window of the last `window_gens` generations of data.
  4. EVAL the new weights RAW (greedy, no search) vs the frozen starting
     champion - the loop's success metric is whether the WEIGHTS get stronger,
     since search-time strength is already a solved dial.
  5. Checkpoint every generation; track the best by raw-greedy winrate.

Note on value_scale: generation 0 evaluates leaves with the warm-start critic
(shaped-reward scale, ~+/-10), so it uses value_scale=5; once the critic has
been trained on [-1,1] search targets, later generations use value_scale=1.
"""

import os
import time
from collections import deque

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam

from .mcts import SearchAgent
from .vrpo_eval import EvalGame, evaluate_vectorized
from .vrpo_model import masked_log_softmax


def _envf(name, default):
    return float(os.getenv(name, default))


def _envi(name, default):
    return int(os.getenv(name, default))


class _GreedyNet:
    """Minimal predict_batch wrapper so evaluate_vectorized can pit raw
    in-memory actors against each other (greedy argmax over masked logits)."""
    def __init__(self, model):
        self.model = model

    def predict_batch(self, states, masks):
        S = tf.convert_to_tensor(states, dtype=tf.float32)
        logits = self.model(S, training=False).numpy()
        logits[~np.asarray(masks, dtype=bool)] = -np.inf
        return np.argmax(logits, axis=1).astype(np.int32)


class AZeroLoop:
    def __init__(self, paths):
        self.paths = paths
        self.gens = _envi('AZ_GENS', 10)
        self.games_per_gen = _envi('AZ_GAMES_PER_GEN', 200)
        self.sims = _envi('AZ_SIMS', 120)
        self.eval_batch = _envi('AZ_EVAL_BATCH', 8)
        self.temp_moves = _envi('AZ_TEMP_MOVES', 16)     # half-turns of sampling
        self.window_gens = _envi('AZ_WINDOW', 4)
        self.epochs = _envi('AZ_EPOCHS', 3)
        self.batch_size = _envi('AZ_BATCH', 512)
        self.lr = _envf('AZ_LR', 1e-4)
        self.eval_games = _envi('AZ_EVAL_GAMES', 40)
        self.max_half_turns = _envi('AZ_MAX_HALF_TURNS', 300)
        self.warm_critic_scaled = _envi('AZ_WARM_CRITIC_SHAPED', 1)

        print(f"AZero config: gens={self.gens} games/gen={self.games_per_gen} "
              f"sims={self.sims} temp_moves={self.temp_moves} "
              f"window={self.window_gens} epochs={self.epochs} lr={self.lr}")

        # Live nets being trained, warm-started from the champion (or, on
        # resume, from the latest generation checkpoint via AZ_WARM_ACTOR).
        self.actor = load_model(paths['warm_actor'], compile=False)
        self.critic = load_model(paths['warm_critic'], compile=False)
        # Frozen baseline for the per-gen raw-strength eval. Deliberately
        # SEPARATE from the warm start so resumed runs keep measuring against
        # the same original champion and winrates stay comparable across runs.
        self.baseline = _GreedyNet(
            load_model(paths.get('baseline', paths['warm_actor']),
                       compile=False))

        self.actor_opt = Adam(learning_rate=self.lr, clipnorm=1.0)
        self.critic_opt = Adam(learning_rate=self.lr, clipnorm=1.0)

        self.window = deque(maxlen=self.window_gens)
        self.tb = tf.summary.create_file_writer(paths['tensorboard_dir'])
        self.best_wr = -1.0
        self._np_rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ #
    # 1) self-play data generation
    # ------------------------------------------------------------------ #
    def generate(self, gen):
        # Gen 0 leaves are valued by the shaped-scale warm-start critic.
        vscale = 5.0 if (gen == 0 and self.warm_critic_scaled) else 1.0
        agent = SearchAgent(self.actor, self.critic, sims=self.sims,
                            eval_batch=self.eval_batch, value_scale=vscale,
                            max_half_turns=self.max_half_turns,
                            root_noise=True, seed=gen)

        S, PI, QT, QM, LG = [], [], [], [], []
        seat_idx, played = [], []
        lengths = []
        t0 = time.perf_counter()
        for g in range(self.games_per_gen):
            if g and g % 25 == 0:               # heartbeat for long generations
                pace = (time.perf_counter() - t0) / g
                eta = pace * (self.games_per_gen - g) / 60.0
                print(f"  gen {gen}: {g}/{self.games_per_gen} games, "
                      f"{pace:.1f}s/game, ~{eta:.0f}min left", flush=True)
            game = EvalGame([('A', None, 0), ('B', None, 1)],
                            max_half_turns=self.max_half_turns)
            game_rows = []                      # (row_index, seat)
            while not game.victor and game.half_turns < self.max_half_turns:
                pi, n, q, legal = agent.search_policy(game)
                if game.half_turns < self.temp_moves:
                    a = int(self._np_rng.choice(len(pi), p=pi))
                else:
                    a = int(np.argmax(n))

                S.append(game.to_state())
                PI.append(pi.astype(np.float32))
                QT.append(q.astype(np.float32))
                QM.append((n > 0))
                LG.append(legal)
                seat_idx.append(game.half_turns % 2)
                played.append(a)
                game_rows.append(len(S) - 1)

                game.step_move(a)

            lengths.append(game.half_turns)
            if game.victor:
                winner = 0 if game.players[0].victor else 1
            else:
                winner = None                   # truncated: z = 0 for both
            for r in game_rows:
                z = 0.0 if winner is None else (1.0 if seat_idx[r] == winner
                                                else -1.0)
                # Ground-truth outcome overrides the search value for the
                # action actually played.
                QT[r][played[r]] = z
                QM[r][played[r]] = True

        data = {
            'S': np.asarray(S, dtype=np.float32),
            'PI': np.asarray(PI, dtype=np.float32),
            'QT': np.asarray(QT, dtype=np.float32),
            'QM': np.asarray(QM, dtype=bool),
            'LG': np.asarray(LG, dtype=bool),
        }
        self.window.append(data)
        return len(S), float(np.mean(lengths)) / 2.0

    # ------------------------------------------------------------------ #
    # 2) training on the sliding window
    # ------------------------------------------------------------------ #
    @tf.function(reduce_retracing=True)
    def _train_step(self, S, PI, QT, QM, LG):
        with tf.GradientTape() as tape:
            logp = masked_log_softmax(self.actor(S, training=True), LG)
            pi_loss = -tf.reduce_mean(tf.reduce_sum(PI * logp, axis=1))
        g = tape.gradient(pi_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(g, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            q = self.critic(S, training=True)
            m = tf.cast(QM, tf.float32)
            q_loss = tf.reduce_sum(m * tf.square(q - QT)) / \
                tf.maximum(tf.reduce_sum(m), 1.0)
        g = tape.gradient(q_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(g, self.critic.trainable_variables))
        return pi_loss, q_loss

    def train(self):
        S = np.concatenate([d['S'] for d in self.window])
        PI = np.concatenate([d['PI'] for d in self.window])
        QT = np.concatenate([d['QT'] for d in self.window])
        QM = np.concatenate([d['QM'] for d in self.window])
        LG = np.concatenate([d['LG'] for d in self.window])
        n = len(S)

        pi_l = q_l = 0.0
        steps = 0
        for _ in range(self.epochs):
            order = self._np_rng.permutation(n)
            for start in range(0, n, self.batch_size):
                b = order[start:start + self.batch_size]
                pl, ql = self._train_step(
                    tf.convert_to_tensor(S[b]), tf.convert_to_tensor(PI[b]),
                    tf.convert_to_tensor(QT[b]), tf.convert_to_tensor(QM[b]),
                    tf.convert_to_tensor(LG[b]))
                pi_l += float(pl); q_l += float(ql); steps += 1
        return pi_l / steps, q_l / steps

    # ------------------------------------------------------------------ #
    # 3) the loop
    # ------------------------------------------------------------------ #
    def run(self):
        for gen in range(self.gens):
            n_samples, avg_turns = self.generate(gen)
            pi_loss, q_loss = self.train()

            # Raw-weights strength: greedy new actor vs frozen champion.
            wr, _, _ = evaluate_vectorized(_GreedyNet(self.actor),
                                           self.baseline, self.eval_games,
                                           self.max_half_turns)

            with self.tb.as_default():
                tf.summary.scalar('AZero/pi_loss', pi_loss, step=gen)
                tf.summary.scalar('AZero/q_loss', q_loss, step=gen)
                tf.summary.scalar('AZero/raw_winrate_vs_champ', wr, step=gen)
                tf.summary.scalar('AZero/avg_turns', avg_turns, step=gen)
                tf.summary.scalar('AZero/samples', n_samples, step=gen)
            self.tb.flush()

            print(f"[gen {gen}] samples={n_samples} turns={avg_turns:.1f} "
                  f"pi_loss={pi_loss:.4f} q_loss={q_loss:.4f} "
                  f"raw_wr_vs_champ={wr:.3f}")

            self.actor.save(self.paths['gen_actor'].format(gen=gen))
            self.critic.save(self.paths['gen_critic'].format(gen=gen))
            if wr > self.best_wr:
                self.best_wr = wr
                self.actor.save(self.paths['best_actor'])
                self.critic.save(self.paths['best_critic'])
                print(f"  ** new best raw winrate {wr:.3f} -> saved best ckpt")

        print(f"AZero loop complete. Best raw winrate vs champion: "
              f"{self.best_wr:.3f}")
