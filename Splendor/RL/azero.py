# Splendor/RL/azero.py
"""
AlphaZero-style training loop, v2: textbook two-headed network.

v1 kept the VRPO actor + Q-vector critic and regressed search-backed per-action
values; it distilled fast (raw winrate vs champion 0.375 -> 0.625 by gen ~2)
then saturated. The Q-vector's variance-reduction rationale doesn't apply under
AlphaZero training (search-backed/outcome targets are already low-noise), and
search only ever consumes a SCALAR at the leaf - so v2 is the canonical
architecture: one shared trunk, a policy head (141 logits) and a scalar value
head (tanh, [-1,1]), trained jointly:

    loss = CE(root visit distribution || policy)  +  c_v * MSE(z, value)

Warm start is WEIGHT SURGERY: the trunk + policy head copy the champion
actor's Dense weights layer-for-layer (identical shapes), so the net begins
playing exactly like the champion; only the small value head starts fresh.
Because that head is untrained at gen 0, generation 0 produces its data with
the proven legacy actor+critic search agent; from gen 1 the two-headed net
drives search itself - at HALF the NN calls per leaf (one call returns both
heads).

Per generation: self-play with both seats on PUCT search (root Dirichlet
noise, temperature sampling for the first AZ_TEMP_MOVES half-turns) -> train
on a sliding window -> raw-greedy eval vs the frozen original champion (the
loop's success metric) -> checkpoint with printed path + size proof.
"""

import os
import time
from collections import deque

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, LeakyReLU
from keras.initializers import HeNormal
from keras.models import load_model, Model
from keras.optimizers import Adam

from .mcts import SearchAgent
from .vrpo_eval import EvalGame, evaluate_vectorized
from .vrpo_model import masked_log_softmax


def _envf(name, default):
    return float(os.getenv(name, default))


def _envi(name, default):
    return int(os.getenv(name, default))


# --------------------------------------------------------------------------- #
# Two-headed net + warm-start surgery
# --------------------------------------------------------------------------- #
def build_azero_net(state_dim, action_dim, layer_sizes):
    """Shared trunk -> policy logits head + scalar tanh value head."""
    s = Input(shape=(state_dim,))
    x = s
    for i, n in enumerate(layer_sizes):
        x = Dense(n, kernel_initializer=HeNormal(), name=f'az_dense{i+1}')(x)
        x = LeakyReLU(negative_slope=0.3)(x)
    logits = Dense(action_dim, kernel_initializer=HeNormal(),
                   name='az_policy')(x)
    h = Dense(64, kernel_initializer=HeNormal(), name='az_value_hidden')(x)
    h = LeakyReLU(negative_slope=0.3)(h)
    value = Dense(1, activation='tanh', kernel_initializer=HeNormal(),
                  name='az_value')(h)
    return Model(inputs=s, outputs=[logits, value], name='azero_net')


def warm_start_from_actor(net, actor):
    """Copy the champion actor's Dense weights (trunk + policy head) into the
    two-headed net so its policy starts IDENTICAL to the champion. Returns
    True on success, False on architecture mismatch (net then stays fresh)."""
    src = [l for l in actor.layers if isinstance(l, Dense)]
    n_trunk = len(src) - 1                      # last Dense = policy_logits
    dst = [net.get_layer(f'az_dense{i+1}') for i in range(n_trunk)]
    dst.append(net.get_layer('az_policy'))
    for s_l, d_l in zip(src, dst):
        sw, dw = s_l.get_weights(), d_l.get_weights()
        if any(a.shape != b.shape for a, b in zip(sw, dw)):
            return False
        d_l.set_weights(sw)
    return True


class _GreedyNet:
    """predict_batch wrapper for evaluate_vectorized: greedy argmax over the
    masked policy logits of either a single-output actor or a two-headed net."""
    def __init__(self, model, two_headed=False):
        self.model = model
        self.two_headed = two_headed

    def predict_batch(self, states, masks):
        S = tf.convert_to_tensor(states, dtype=tf.float32)
        out = self.model(S, training=False)
        logits = out[0].numpy() if self.two_headed else out.numpy()
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
        self.vloss_coef = _envf('AZ_VLOSS', 1.0)
        self.eval_games = _envi('AZ_EVAL_GAMES', 40)
        self.max_half_turns = _envi('AZ_MAX_HALF_TURNS', 300)
        self.layers = paths.get('layers', [512, 512, 256])

        print(f"AZero v2 config: gens={self.gens} games/gen={self.games_per_gen} "
              f"sims={self.sims} layers={self.layers} temp_moves={self.temp_moves} "
              f"window={self.window_gens} epochs={self.epochs} lr={self.lr} "
              f"vloss={self.vloss_coef}")

        # The two-headed net being trained.
        if paths.get('warm_aznet'):                      # resume a prior v2 run
            self.net = load_model(paths['warm_aznet'], compile=False)
            self.resumed = True
            print(f"Resumed aznet <- {paths['warm_aznet']}")
        else:
            self.net = build_azero_net(251, 141, self.layers)
            champ_actor = load_model(paths['warm_actor'], compile=False)
            if warm_start_from_actor(self.net, champ_actor):
                print("Warm start: champion actor weights copied into trunk "
                      "+ policy head (value head fresh).")
            else:
                print("WARNING: layer shapes differ from champion actor; "
                      "net starts from scratch.")
            self.resumed = False

        # Legacy actor+critic pair drives generation 0 (proven data quality
        # while the fresh value head is untrained). Skipped on resume.
        self.legacy_actor = load_model(paths['warm_actor'], compile=False)
        self.legacy_critic = load_model(paths['warm_critic'], compile=False)

        # Frozen baseline for the per-gen raw-strength eval - the ORIGINAL
        # champion, independent of warm start, so winrates stay comparable.
        self.baseline = _GreedyNet(
            load_model(paths.get('baseline', paths['warm_actor']),
                       compile=False))

        self.opt = Adam(learning_rate=self.lr, clipnorm=1.0)
        self.window = deque(maxlen=self.window_gens)
        self.tb = tf.summary.create_file_writer(paths['tensorboard_dir'])
        self.best_wr = -1.0
        self._np_rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ #
    # 1) self-play data generation
    # ------------------------------------------------------------------ #
    def generate(self, gen):
        if gen == 0 and not self.resumed:
            agent = SearchAgent(self.legacy_actor, self.legacy_critic,
                                sims=self.sims, eval_batch=self.eval_batch,
                                value_scale=5.0,
                                max_half_turns=self.max_half_turns,
                                root_noise=True, seed=gen)
            print("  gen 0 data: legacy actor+critic search "
                  "(value head not yet trained)")
        else:
            agent = SearchAgent(self.net, None, sims=self.sims,
                                eval_batch=self.eval_batch,
                                max_half_turns=self.max_half_turns,
                                root_noise=True, seed=gen)

        S, PI, LG, seat_idx = [], [], [], []
        Z = []                                    # filled per game
        lengths = []
        t0 = time.perf_counter()
        for g in range(self.games_per_gen):
            if g and g % 25 == 0:                 # heartbeat
                pace = (time.perf_counter() - t0) / g
                eta = pace * (self.games_per_gen - g) / 60.0
                print(f"  gen {gen}: {g}/{self.games_per_gen} games, "
                      f"{pace:.1f}s/game, ~{eta:.0f}min left", flush=True)

            game = EvalGame([('A', None, 0), ('B', None, 1)],
                            max_half_turns=self.max_half_turns)
            rows = []
            while not game.victor and game.half_turns < self.max_half_turns:
                pi, n, _q, legal = agent.search_policy(game)
                if game.half_turns < self.temp_moves:
                    a = int(self._np_rng.choice(len(pi), p=pi))
                else:
                    a = int(np.argmax(n))
                S.append(game.to_state())
                PI.append(pi.astype(np.float32))
                LG.append(legal)
                seat_idx.append(game.half_turns % 2)
                rows.append(len(S) - 1)
                game.step_move(a)

            lengths.append(game.half_turns)
            winner = (0 if game.players[0].victor else 1) if game.victor else None
            Z.extend([0.0] * len(rows))
            for r in rows:
                if winner is not None:
                    Z[r] = 1.0 if seat_idx[r] == winner else -1.0

        self.window.append({
            'S': np.asarray(S, dtype=np.float32),
            'PI': np.asarray(PI, dtype=np.float32),
            'LG': np.asarray(LG, dtype=bool),
            'Z': np.asarray(Z, dtype=np.float32),
        })
        return len(S), float(np.mean(lengths)) / 2.0

    # ------------------------------------------------------------------ #
    # 2) joint training on the sliding window
    # ------------------------------------------------------------------ #
    @tf.function(reduce_retracing=True)
    def _train_step(self, S, PI, LG, Z):
        with tf.GradientTape() as tape:
            logits, v = self.net(S, training=True)
            logp = masked_log_softmax(logits, LG)
            pi_loss = -tf.reduce_mean(tf.reduce_sum(PI * logp, axis=1))
            v_loss = tf.reduce_mean(tf.square(v[:, 0] - Z))
            loss = pi_loss + self.vloss_coef * v_loss
        g = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(g, self.net.trainable_variables))
        return pi_loss, v_loss

    def train(self):
        S = np.concatenate([d['S'] for d in self.window])
        PI = np.concatenate([d['PI'] for d in self.window])
        LG = np.concatenate([d['LG'] for d in self.window])
        Z = np.concatenate([d['Z'] for d in self.window])
        n = len(S)

        pi_l = v_l = 0.0
        steps = 0
        for _ in range(self.epochs):
            order = self._np_rng.permutation(n)
            for start in range(0, n, self.batch_size):
                b = order[start:start + self.batch_size]
                pl, vl = self._train_step(
                    tf.convert_to_tensor(S[b]), tf.convert_to_tensor(PI[b]),
                    tf.convert_to_tensor(LG[b]), tf.convert_to_tensor(Z[b]))
                pi_l += float(pl); v_l += float(vl); steps += 1
        return pi_l / steps, v_l / steps

    # ------------------------------------------------------------------ #
    # 3) the loop
    # ------------------------------------------------------------------ #
    def run(self):
        for gen in range(self.gens):
            n_samples, avg_turns = self.generate(gen)
            pi_loss, v_loss = self.train()

            wr, _, _ = evaluate_vectorized(
                _GreedyNet(self.net, two_headed=True), self.baseline,
                self.eval_games, self.max_half_turns)

            with self.tb.as_default():
                tf.summary.scalar('AZero/pi_loss', pi_loss, step=gen)
                tf.summary.scalar('AZero/v_loss', v_loss, step=gen)
                tf.summary.scalar('AZero/raw_winrate_vs_champ', wr, step=gen)
                tf.summary.scalar('AZero/avg_turns', avg_turns, step=gen)
                tf.summary.scalar('AZero/samples', n_samples, step=gen)
            self.tb.flush()

            print(f"[gen {gen}] samples={n_samples} turns={avg_turns:.1f} "
                  f"pi_loss={pi_loss:.4f} v_loss={v_loss:.4f} "
                  f"raw_wr_vs_champ={wr:.3f}")

            gen_path = self.paths['gen_net'].format(gen=gen)
            self.net.save(gen_path)
            assert os.path.exists(gen_path), f"checkpoint missing: {gen_path}"
            print(f"  saved {os.path.abspath(gen_path)} "
                  f"({os.path.getsize(gen_path)/1e6:.1f} MB)", flush=True)
            if wr > self.best_wr:
                self.best_wr = wr
                self.net.save(self.paths['best_net'])
                print(f"  ** new best raw winrate {wr:.3f} -> saved "
                      f"{os.path.abspath(self.paths['best_net'])}", flush=True)

        print(f"AZero v2 complete. Best raw winrate vs champion: "
              f"{self.best_wr:.3f}")
