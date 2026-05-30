# Splendor/RL/vrpo_trainer.py
"""
VRPO agent: holds the actor + critic, samples moves during self-play, turns
collected trajectories into Q-boosting advantages / critic targets, and runs
the clipped policy update plus the Q-critic regression.

Faithful to the paper's algorithm:
  - Q-boosting advantage from a multi-step Expected SARSA(lambda) trace,
    with V^pi(s) = sum_a pi(a|s) Q(s,a) (a policy expectation, not a sample).
  - PPO clipped surrogate, ratio measured against the behavior (rollout) policy.
  - KL(pi || Uniform) regularization to promote exploration (the paper's L^reg).
    Minimizing KL(pi||U) == maximizing policy entropy, so kl_coef IS the
    exploration knob.

Documented deviations (pragmatic, for a single 16GB GPU; each is an isolated
swap point):
  - Adam instead of the paper's Muon optimizer.
  - Advantages/targets are computed once per iteration at the rollout policy
    (standard PPO practice) rather than re-deriving the policy-expectation
    terms inside every actor minibatch.
  - The critic trains on the current rollout only; the paper's cyclic critic
    replay buffer is left as a clearly marked TODO below.

Hyperparameters are overridable via environment variables (for cheap grid
search without editing code), e.g.:  VRPO_KL_COEF=0.05 python train_vrpo.py
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from collections import deque

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

from keras.models import load_model

from .vrpo_model import (
    build_actor, build_critic, NEG_INF, WarmupDecaySchedule,
    masked_log_softmax, masked_softmax,
)

# Indices into a collected transition (see vrpo_game.VRPOGame.turn).
# First six match the existing DQN memory layout so apply_move's end-of-game
# poke (memory[-1][2] += loser_reward; memory[-1][5] = True) still works.
S_STATE, S_ACTION, S_REWARD, S_NEXT, S_MASK, S_DONE, S_LOGP, S_SEAT = range(8)


def _envf(name, default):
    return float(os.getenv(name, default))


def _envi(name, default):
    return int(os.getenv(name, default))


class VRPOAgent:
    def __init__(self, paths):
        print("Making a new VRPOAgent.")
        self.paths = paths

        # Dimensions (match the env / existing DQN)
        self.state_dim = 251
        self.action_dim = 141

        # --- Hyperparameters (env-overridable) ---
        self.gamma = _envf('VRPO_GAMMA', 0.99)        # discount per own-decision step
        self.lam = _envf('VRPO_LAMBDA', 0.95)         # Expected SARSA(lambda) trace
        self.clip_eps = _envf('VRPO_CLIP', 0.2)       # PPO clip coefficient
        self.kl_coef = _envf('VRPO_KL_COEF', 0.03)    # exploration knob (KL-to-uniform)
        self.actor_epochs = _envi('VRPO_ACTOR_EPOCHS', 4)    # K_actor
        self.critic_epochs = _envi('VRPO_CRITIC_EPOCHS', 4)  # K_critic
        self.minibatches = _envi('VRPO_MINIBATCHES', 4)      # M
        self.rollout_size = _envi('VRPO_ROLLOUT', 2048)      # transitions/iter
        self.parallel_games = _envi('VRPO_PARALLEL_GAMES', 32)  # vectorized rollout width
        self.max_half_turns = _envi('VRPO_MAX_HALF_TURNS', 200)
        self.actor_lr = _envf('VRPO_ACTOR_LR', 3e-4)
        self.critic_lr = _envf('VRPO_CRITIC_LR', 3e-4)

        # LR schedule (warmup + gentle exponential decay), on by default - the
        # DQN uses an analogous warmup/decay schedule; VRPO previously ran flat
        # Adam. Steps are optimizer steps (~minibatches*epochs per iteration).
        self.lr_schedule_on = _envi('VRPO_LR_SCHEDULE', 1)
        self.lr_warmup_steps = _envi('VRPO_LR_WARMUP_STEPS', 200)
        self.lr_decay_steps = _envi('VRPO_LR_DECAY_STEPS', 4000)
        self.lr_decay_rate = _envf('VRPO_LR_DECAY_RATE', 0.5)
        self.lr_min_frac = _envf('VRPO_LR_MIN_FRAC', 0.1)

        # Cyclic critic replay buffer (the paper's design; previously a TODO).
        # Retains (state, action, q_target) from the last N rollouts so the
        # critic trains on more, decorrelated data instead of a single rollout
        # it immediately discards. 1 == old behaviour (current rollout only).
        # Targets from older rollouts are mildly stale (computed by a recent
        # critic), which is the standard accuracy/variance trade-off; keeping N
        # small (~4) bounds that staleness.
        self.critic_buffer_rollouts = _envi('VRPO_CRITIC_BUFFER', 4)
        self._critic_buf = deque(maxlen=self.critic_buffer_rollouts)

        # Evaluation cadence (0 disables in-loop eval). eval_games is a binomial
        # sample over RANDOM BOARDS (decks reshuffle each game), so more games
        # reduces metric noise even though both policies are deterministic.
        self.eval_every = _envi('VRPO_EVAL_EVERY', 25)
        self.eval_games = _envi('VRPO_EVAL_GAMES', 80)
        # Best-checkpoint tracking: periodic saves capture the LAST iter, which
        # may be past the peak (the policy can over-sharpen and regress). Track
        # the best eval so we never lose the strongest model.
        self.best_eval = -1.0
        self.best_eval_iter = -1

        # Config snapshot for logging / run naming
        self.config = {
            'gamma': self.gamma, 'lam': self.lam, 'clip_eps': self.clip_eps,
            'kl_coef': self.kl_coef, 'actor_epochs': self.actor_epochs,
            'critic_epochs': self.critic_epochs, 'minibatches': self.minibatches,
            'rollout_size': self.rollout_size, 'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'critic_buffer_rollouts': self.critic_buffer_rollouts,
        }

        layer_sizes = paths['layer_sizes']
        print("Building VRPO actor & critic with layer sizes", layer_sizes)
        print("VRPO config:", self.config)
        self.actor = build_actor(self.state_dim, self.action_dim, layer_sizes)
        self.critic = build_critic(self.state_dim, self.action_dim, layer_sizes)

        # Optionally resume weights from a prior run (weights-only, compile=False
        # so we never deserialize a saved optimizer/schedule). VRPO_RESUME points
        # at the actor .keras; the critic path is derived by name.
        resume = os.getenv('VRPO_RESUME')
        if resume:
            self._resume_weights(resume)

        actor_lr = self._make_lr(self.actor_lr)
        critic_lr = self._make_lr(self.critic_lr)
        self.actor_opt = Adam(learning_rate=actor_lr, clipnorm=1.0)
        self.critic_opt = Adam(learning_rate=critic_lr, clipnorm=1.0)

        # Collection interface used by VRPOGame + reused apply_move.
        self.memory: list = []

        self.tensorboard = tf.summary.create_file_writer(paths['tensorboard_dir'])
        self._log_config_text()
        self.iteration = 0
        self._truncated_games = 0

    def _make_lr(self, peak):
        if not self.lr_schedule_on:
            return peak
        return WarmupDecaySchedule(
            peak_lr=peak, warmup_steps=self.lr_warmup_steps,
            decay_steps=self.lr_decay_steps, decay_rate=self.lr_decay_rate,
            min_lr=peak * self.lr_min_frac)

    def _resume_weights(self, actor_path):
        critic_path = actor_path.replace('_actor.keras', '_critic.keras')
        a = load_model(actor_path, compile=False)
        self.actor.set_weights(a.get_weights())
        print(f"Resumed actor weights <- {actor_path}")
        if os.path.exists(critic_path):
            c = load_model(critic_path, compile=False)
            self.critic.set_weights(c.get_weights())
            print(f"Resumed critic weights <- {critic_path}")
        else:
            print(f"WARNING: critic checkpoint not found at {critic_path}; "
                  f"critic starts fresh.")

    def _log_config_text(self):
        text = "\n".join(f"{k}: {v}" for k, v in self.config.items())
        with self.tensorboard.as_default():
            tf.summary.text('VRPO/config', text, step=0)

    # ------------------------------------------------------------------ #
    # Collection interface (apply_move calls self.model.memory / remember)
    # ------------------------------------------------------------------ #
    def remember(self, entry) -> None:
        self.memory.append(entry)

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def act(self, state, mask):
        """SAMPLE a legal move from the current policy; return (action, logp).
        Single-row path (kept for the non-vectorized game loop / tests).
        """
        s = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        m = tf.convert_to_tensor(mask[None, :], dtype=tf.bool)
        logits = self.actor(s, training=False)
        logp = masked_log_softmax(logits, m)[0]
        action = int(tf.random.categorical(logp[None, :], 1)[0, 0].numpy())
        return action, float(logp[action].numpy())

    @tf.function(reduce_retracing=True)
    def _act_batch_tf(self, S, M):
        logits = self.actor(S, training=False)
        logp_all = masked_log_softmax(logits, M)             # (G, A)
        actions = tf.random.categorical(logp_all, 1)[:, 0]   # (G,)
        logps = tf.gather(logp_all, actions, batch_dims=1)   # (G,)
        return actions, logps

    def act_batch(self, states, masks):
        """SAMPLE one legal move per game for a BATCH of states/masks.
        Returns (actions[int32], logps[float32]) as numpy. This is the call
        that makes the vectorized rollout fast: one forward pass for G games.
        """
        S = tf.convert_to_tensor(states, dtype=tf.float32)
        M = tf.convert_to_tensor(masks, dtype=tf.bool)
        actions, logps = self._act_batch_tf(S, M)
        return actions.numpy().astype(np.int32), logps.numpy().astype(np.float32)

    def get_predictions(self, state, mask):
        """GREEDY interface compatible with Player.choose_move (which argmaxes).
        Returns masked logits so argmax == best legal action. Used for
        evaluation and could back an inference/webapp agent later.
        """
        s = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        m = tf.convert_to_tensor(mask[None, :], dtype=tf.bool)
        logits = self.actor(s, training=False)[0]
        return tf.where(m, logits, tf.fill(tf.shape(logits), NEG_INF))

    @tf.function(reduce_retracing=True)
    def _greedy_batch_tf(self, S, M):
        logits = self.actor(S, training=False)
        masked = tf.where(M, logits, tf.fill(tf.shape(logits), NEG_INF))
        return tf.argmax(masked, axis=1, output_type=tf.int32)

    def predict_batch(self, states, masks):
        """Batched GREEDY argmax actions (for the vectorized evaluator). Same
        rule as get_predictions+argmax, one forward pass for many states.
        """
        S = tf.convert_to_tensor(states, dtype=tf.float32)
        M = tf.convert_to_tensor(masks, dtype=tf.bool)
        return self._greedy_batch_tf(S, M).numpy()

    # ------------------------------------------------------------------ #
    # Q-boosting: turn one seat's trajectory into advantages + Q targets
    # ------------------------------------------------------------------ #
    def process_trajectory(self, traj):
        states = np.asarray([e[S_STATE] for e in traj], dtype=np.float32)
        actions = np.asarray([e[S_ACTION] for e in traj], dtype=np.int32)
        rewards = np.asarray([e[S_REWARD] for e in traj], dtype=np.float32)
        masks = np.asarray([e[S_MASK] for e in traj], dtype=bool)
        dones = np.asarray([e[S_DONE] for e in traj], dtype=np.float32)
        logps = np.asarray([e[S_LOGP] for e in traj], dtype=np.float32)

        S = tf.convert_to_tensor(states)
        M = tf.convert_to_tensor(masks)
        Q = self.critic(S, training=False).numpy()                       # (L, A)
        pi = masked_softmax(self.actor(S, training=False), M).numpy()     # (L, A)
        V = np.sum(pi * Q, axis=1)                                        # (L,) V^pi
        Qsa = Q[np.arange(len(actions)), actions]                        # (L,)

        # Backward Expected SARSA(lambda) trace:
        #   delta_t = r_t + gamma*(1-done)*V^pi(s_{t+1}) - Q(s_t,a_t)
        #   G_t     = delta_t + (lambda*gamma)*(1-done)*G_{t+1}
        #   A_t     = (Q(s_t,a_t) - V^pi(s_t)) + G_t
        #   Qtarget = Q(s_t,a_t) + G_t
        L = len(traj)
        G = np.zeros(L, dtype=np.float32)
        gl = self.gamma * self.lam
        next_G = 0.0
        for i in range(L - 1, -1, -1):
            nonterminal = 1.0 - dones[i]
            v_next = V[i + 1] if (i + 1 < L) else 0.0
            delta = rewards[i] + self.gamma * nonterminal * v_next - Qsa[i]
            G[i] = delta + gl * nonterminal * next_G
            next_G = G[i]

        advantages = (Qsa - V) + G
        q_targets = Qsa + G
        return states, actions, logps, masks, advantages.astype(np.float32), \
            q_targets.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Update: K_actor clipped-policy epochs, then K_critic regression epochs
    # ------------------------------------------------------------------ #
    def update(self, states, actions, logp_old, masks, adv, q_target):
        N = len(states)
        S = tf.convert_to_tensor(states, tf.float32)
        A = tf.convert_to_tensor(actions, tf.int32)
        LP = tf.convert_to_tensor(logp_old, tf.float32)
        MK = tf.convert_to_tensor(masks, tf.bool)
        ADV = tf.convert_to_tensor(adv, tf.float32)
        QT = tf.convert_to_tensor(q_target, tf.float32)

        # Normalize advantages across the batch (standard PPO).
        ADV = (ADV - tf.reduce_mean(ADV)) / (tf.math.reduce_std(ADV) + 1e-8)

        mb = max(1, N // self.minibatches)

        # Accumulate over ALL minibatch steps for stable logging (not just last).
        acc = {'pg': 0.0, 'kl': 0.0, 'ent': 0.0, 'ratio': 0.0,
               'clipfrac': 0.0, 'approx_kl': 0.0}
        a_steps = 0
        for _ in range(self.actor_epochs):
            order = tf.random.shuffle(tf.range(N))
            for start in range(0, N, mb):
                b = order[start:start + mb]
                pg, kl, ent, ratio, clipfrac, approx_kl = self._actor_step(
                    tf.gather(S, b), tf.gather(A, b), tf.gather(LP, b),
                    tf.gather(MK, b), tf.gather(ADV, b)
                )
                acc['pg'] += float(pg); acc['kl'] += float(kl)
                acc['ent'] += float(ent); acc['ratio'] += float(ratio)
                acc['clipfrac'] += float(clipfrac)
                acc['approx_kl'] += float(approx_kl)
                a_steps += 1

        # --- Critic phase: train over the cyclic replay buffer ---
        # Add this rollout's (state, action, q_target) to the buffer, then
        # regress the critic across ALL retained rollouts. This gives the critic
        # several rollouts' worth of data per iteration instead of one.
        self._critic_buf.append((states, actions, q_target))
        cb_S = tf.convert_to_tensor(
            np.concatenate([r[0] for r in self._critic_buf]), tf.float32)
        cb_A = tf.convert_to_tensor(
            np.concatenate([r[1] for r in self._critic_buf]), tf.int32)
        cb_QT = tf.convert_to_tensor(
            np.concatenate([r[2] for r in self._critic_buf]), tf.float32)
        cb_N = int(cb_S.shape[0])
        cb_mb = max(1, cb_N // self.minibatches)

        critic_sum = 0.0
        c_steps = 0
        for _ in range(self.critic_epochs):
            order = tf.random.shuffle(tf.range(cb_N))
            for start in range(0, cb_N, cb_mb):
                b = order[start:start + cb_mb]
                c = self._critic_step(tf.gather(cb_S, b), tf.gather(cb_A, b),
                                      tf.gather(cb_QT, b))
                critic_sum += float(c); c_steps += 1

        return {
            'pg_loss': acc['pg'] / a_steps,
            'kl_to_uniform': acc['kl'] / a_steps,
            'entropy': acc['ent'] / a_steps,
            'mean_ratio': acc['ratio'] / a_steps,
            'clip_fraction': acc['clipfrac'] / a_steps,
            'approx_kl': acc['approx_kl'] / a_steps,
            'critic_loss': critic_sum / c_steps,
        }

    @tf.function
    def _actor_step(self, S, A, LP_old, MK, ADV):
        with tf.GradientTape() as tape:
            logits = self.actor(S, training=True)
            logp_all = masked_log_softmax(logits, MK)
            idx = tf.stack([tf.range(tf.shape(A)[0]), A], axis=1)
            logp = tf.gather_nd(logp_all, idx)

            ratio = tf.exp(logp - LP_old)
            unclipped = ratio * ADV
            clipped = tf.clip_by_value(ratio, 1.0 - self.clip_eps,
                                       1.0 + self.clip_eps) * ADV
            pg_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

            # KL(pi || Uniform_over_legal) = -H(pi) + log(n_legal).
            pi = tf.exp(logp_all)
            n_legal = tf.reduce_sum(tf.cast(MK, tf.float32), axis=1)
            neg_entropy = tf.reduce_sum(
                tf.where(MK, pi * logp_all, tf.zeros_like(pi)), axis=1)
            entropy = -neg_entropy
            kl = neg_entropy + tf.math.log(n_legal)
            kl_loss = tf.reduce_mean(kl)

            loss = pg_loss + self.kl_coef * kl_loss

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        # Diagnostics (no grad)
        clip_frac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(ratio - 1.0), self.clip_eps), tf.float32))
        approx_kl = tf.reduce_mean(LP_old - logp)   # behavior -> current
        return (pg_loss, kl_loss, tf.reduce_mean(entropy),
                tf.reduce_mean(ratio), clip_frac, approx_kl)

    @tf.function
    def _critic_step(self, S, A, QT):
        with tf.GradientTape() as tape:
            Q = self.critic(S, training=True)
            idx = tf.stack([tf.range(tf.shape(A)[0]), A], axis=1)
            Qsa = tf.gather_nd(Q, idx)
            loss = 0.5 * tf.reduce_mean(tf.square(Qsa - QT))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss

    # ------------------------------------------------------------------ #
    # Logging / saving
    # ------------------------------------------------------------------ #
    def log_iteration(self, metrics, game_lengths, adv, q_target):
        step = self.iteration
        with self.tensorboard.as_default():
            for key in ('pg_loss', 'kl_to_uniform', 'entropy', 'critic_loss',
                        'mean_ratio', 'clip_fraction', 'approx_kl'):
                tf.summary.scalar(f'VRPO/{key}', metrics[key], step=step)
            tf.summary.scalar('VRPO/adv_mean', float(np.mean(adv)), step=step)
            tf.summary.scalar('VRPO/adv_std', float(np.std(adv)), step=step)
            tf.summary.scalar('VRPO/q_target_mean', float(np.mean(q_target)), step=step)
            if game_lengths:
                tf.summary.scalar('VRPO/game_length',
                                  float(np.mean(game_lengths)) / 2.0, step=step)
            tf.summary.scalar('VRPO/truncated_games_total',
                              self._truncated_games, step=step)
            # Current LR (resolve schedule at the optimizer's step count).
            lr = self.actor_opt.learning_rate
            if callable(lr):
                lr = lr(self.actor_opt.iterations)
            tf.summary.scalar('VRPO/actor_lr', float(lr), step=step)
        self.tensorboard.flush()

    def log_eval(self, wr_dqn, draws_dqn, wr_random, draws_random):
        step = self.iteration
        with self.tensorboard.as_default():
            tf.summary.scalar('Eval/winrate_vs_dqn', wr_dqn, step=step)
            tf.summary.scalar('Eval/draws_vs_dqn', draws_dqn, step=step)
            tf.summary.scalar('Eval/winrate_vs_random', wr_random, step=step)
            tf.summary.scalar('Eval/draws_vs_random', draws_random, step=step)
        self.tensorboard.flush()

    def save_model(self) -> None:
        self.actor.save(self.paths['actor_save_path'])
        self.critic.save(self.paths['critic_save_path'])
        print(f"Saved actor -> {self.paths['actor_save_path']}")
        print(f"Saved critic -> {self.paths['critic_save_path']}")

    def maybe_save_best(self, eval_value) -> bool:
        """Save to the stable *_best.keras paths if this is the best eval so
        far. Returns True if a new best was saved. NaN values are ignored.
        """
        if eval_value != eval_value:          # NaN
            return False
        if eval_value <= self.best_eval:
            return False
        self.best_eval = eval_value
        self.best_eval_iter = self.iteration
        self.actor.save(self.paths['actor_best_path'])
        self.critic.save(self.paths['critic_best_path'])
        with self.tensorboard.as_default():
            tf.summary.scalar('Eval/best_winrate', self.best_eval,
                              step=self.iteration)
        self.tensorboard.flush()
        print(f"  ** new best eval={eval_value:.3f} @ iter {self.iteration} "
              f"-> saved {os.path.basename(self.paths['actor_best_path'])}")
        return True
