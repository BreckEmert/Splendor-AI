# Splendor/RL/vrpo_trainer.py
"""
VRPO agent: holds the actor + critic, samples moves during self-play, turns
collected trajectories into Q-boosting advantages / critic targets, and runs
the clipped policy update plus the Q-critic regression.

Faithful to the paper's algorithm:
  - Q-boosting advantage from a multi-step Expected SARSA(lambda) trace,
    with V^pi(s) = sum_a pi(a|s) Q(s,a) (a policy expectation, not a sample).
  - PPO clipped surrogate, ratio measured against the behavior (rollout) policy.
  - KL(pi || Uniform) regularization (the paper's L^reg). Minimizing KL(pi||U)
    == maximizing entropy, so kl_coef IS the exploration knob. Splendor punishes
    "falling off" (resources clog), so sustained-but-low exploration is right;
    the policy also sharpens naturally as it learns. kl_coef can ANNEAL from a
    start to an end value so exploration eases off late (see kl_coef_end).

Documented deviations (pragmatic, single 16GB GPU; each an isolated swap point):
  - Adam instead of the paper's Muon optimizer.
  - Advantages/targets computed once per iteration at the rollout policy
    (standard PPO) rather than re-derived inside every actor minibatch.

Hyperparameters are env-overridable for grid search, e.g.:
    VRPO_KL_COEF=0.06 VRPO_KL_COEF_END=0.01 python train_vrpo.py
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from collections import deque

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import load_model

from .vrpo_model import (
    build_actor, build_critic, NEG_INF, WarmupCosineSchedule,
    masked_log_softmax, masked_softmax,
)

# Indices into a collected transition (see vrpo_game.VRPOGame._commit).
# First six match the DQN memory layout so apply_move's end-of-game poke
# (memory[-1][2] += loser_reward; memory[-1][5] = True) still works.
S_STATE, S_ACTION, S_REWARD, S_NEXT, S_MASK, S_DONE, S_LOGP, S_SEAT = range(8)


def _envf(name, default):
    return float(os.getenv(name, default))


def _envi(name, default):
    return int(os.getenv(name, default))


def _parse_layers(s, default):
    """Parse '1024-1024-512' or '1024,1024,512' -> [1024,1024,512]. Empty/None
    falls back to `default` (the actor layer sizes)."""
    if not s:
        return list(default)
    return [int(x) for x in s.replace(',', '-').split('-') if x.strip()]


class VRPOAgent:
    def __init__(self, paths):
        print("Making a new VRPOAgent.")
        self.paths = paths

        # Dimensions (match the env / existing DQN)
        self.state_dim = 251
        self.action_dim = 141

        # --- Core hyperparameters (env-overridable) ---
        self.gamma = _envf('VRPO_GAMMA', 0.99)        # discount per own-decision step
        self.lam = _envf('VRPO_LAMBDA', 0.95)         # Expected SARSA(lambda) trace
        self.clip_eps = _envf('VRPO_CLIP', 0.2)       # PPO clip coefficient
        self.actor_epochs = _envi('VRPO_ACTOR_EPOCHS', 4)    # K_actor
        self.critic_epochs = _envi('VRPO_CRITIC_EPOCHS', 4)  # K_critic
        self.minibatches = _envi('VRPO_MINIBATCHES', 4)      # M
        self.rollout_size = _envi('VRPO_ROLLOUT', 2048)      # transitions/iter
        self.parallel_games = _envi('VRPO_PARALLEL_GAMES', 32)  # vectorized rollout width
        self.max_half_turns = _envi('VRPO_MAX_HALF_TURNS', 200)
        self.actor_lr = _envf('VRPO_ACTOR_LR', 3e-4)
        self.critic_lr = _envf('VRPO_CRITIC_LR', 3e-4)

        # Exploration: KL-to-uniform coefficient. Default 0.06 is the proven
        # baseline (held vs_dqn without regression in tuning). Optionally anneal
        # from kl_coef -> kl_coef_end over the run so it sharpens late; default
        # end == start (constant) so behaviour is unchanged unless opted in.
        self.kl_coef_start = _envf('VRPO_KL_COEF', 0.06)
        self.kl_coef_end = _envf('VRPO_KL_COEF_END', self.kl_coef_start)
        self.kl_coef_var = tf.Variable(self.kl_coef_start, trainable=False,
                                       dtype=tf.float32)

        # Reward shaping anneal (shaping -> sparse). shaping_alpha scales every
        # shaped reward component; the +/-10 win/loss stays full strength. Read
        # live by BlendedRewardEngine each move. Annealed start->end over
        # [shape_anneal_start_frac, shape_anneal_end_frac] of the schedule
        # horizon. Defaults: start=1 end=1 => constant full shaping (no-op unless
        # the run opts into BlendedRewardEngine AND sets an end<1).
        self.shaping_alpha = _envf('VRPO_SHAPE_ALPHA', 1.0)   # live value
        self.shape_alpha_start = _envf('VRPO_SHAPE_ALPHA', 1.0)
        self.shape_alpha_end = _envf('VRPO_SHAPE_ALPHA_END', self.shape_alpha_start)
        self.shape_anneal_start_frac = _envf('VRPO_SHAPE_ANNEAL_START', 0.2)
        self.shape_anneal_end_frac = _envf('VRPO_SHAPE_ANNEAL_END', 0.8)

        # Total planned iterations (for LR/KL horizons). Read from env so it
        # matches train_vrpo's VRPO_ITERS; loop may pass fewer (tests) - then
        # the schedules just stay near their start, which is harmless.
        self.total_iters = _envi('VRPO_ITERS', 5000)
        # Schedule horizon (for LR + KL anneal), DECOUPLED from run length so a
        # long overnight run can reach its LR/KL floors by VRPO_SCHED_ITERS and
        # then HOLD there for the remaining iters. Defaults to the run length.
        self.sched_iters = _envi('VRPO_SCHED_ITERS', self.total_iters)
        self.steps_per_iter = self.minibatches * self.actor_epochs

        # LR schedule (warmup + horizon-matched cosine decay), on by default.
        self.lr_schedule_on = _envi('VRPO_LR_SCHEDULE', 1)
        self.lr_warmup_steps = _envi('VRPO_LR_WARMUP_STEPS', 150)
        self.lr_min_frac = _envf('VRPO_LR_MIN_FRAC', 0.2)

        # Cyclic critic replay buffer (the paper's design). Retains the last N
        # rollouts of (state, action, q_target) so the critic trains on more,
        # decorrelated data instead of one rollout it discards. 1 == old single-
        # rollout behaviour. Older targets are mildly stale; small N bounds that.
        self.critic_buffer_rollouts = _envi('VRPO_CRITIC_BUFFER', 4)
        self._critic_buf = deque(maxlen=self.critic_buffer_rollouts)

        # League / past-self opponents (opt-in). When on, a fraction of rollout
        # games pit the learner vs a frozen snapshot from a bounded pool; only
        # the learner's transitions train. Counters self-play localization.
        self.league_on = _envi('VRPO_LEAGUE', 0)
        self.league_prob = _envf('VRPO_LEAGUE_PROB', 0.5)
        self.league_pool = _envi('VRPO_LEAGUE_POOL', 5)
        self.snapshot_every = _envi('VRPO_SNAPSHOT_EVERY', 50)
        # PFSP: prioritized opponent sampling. pfsp_pow>0 weights pooled
        # opponents by (1-learner_winrate)^pow (focus on hard ones); 0 == uniform.
        self.pfsp = _envi('VRPO_PFSP', 1)
        self.pfsp_pow = _envf('VRPO_PFSP_POW', 1.0)
        self.pfsp_eps = _envf('VRPO_PFSP_EPS', 0.05)

        # Evaluation cadence (0 disables). Strength = greedy win-rate vs the
        # fixed DQN inference model. The eval is the expensive periodic step
        # (eval_games self-play games each), so it runs sparsely. eval_games is
        # a binomial sample over random boards, so more games => less noise.
        self.eval_every = _envi('VRPO_EVAL_EVERY', 350)
        self.eval_games = _envi('VRPO_EVAL_GAMES', 80)
        # Cheap scalar metrics (critic_loss, entropy, lr, ...) logging cadence.
        self.log_every = _envi('VRPO_LOG_EVERY', 100)

        # Best-checkpoint tracking (periodic saves capture the last iter, which
        # may be past a peak; keep the best-by-vs_dqn model separately).
        self.best_eval = -1.0
        self.best_eval_iter = -1

        # Config snapshot for logging / reproducibility.
        self.config = {
            'gamma': self.gamma, 'lam': self.lam, 'clip_eps': self.clip_eps,
            'kl_coef_start': self.kl_coef_start, 'kl_coef_end': self.kl_coef_end,
            'actor_epochs': self.actor_epochs, 'critic_epochs': self.critic_epochs,
            'minibatches': self.minibatches, 'rollout_size': self.rollout_size,
            'actor_lr': self.actor_lr, 'critic_lr': self.critic_lr,
            'critic_buffer_rollouts': self.critic_buffer_rollouts,
            'total_iters': self.total_iters, 'sched_iters': self.sched_iters,
            'eval_every': self.eval_every, 'log_every': self.log_every,
            'league_on': self.league_on, 'league_prob': self.league_prob,
            'league_pool': self.league_pool, 'snapshot_every': self.snapshot_every,
        }

        layer_sizes = paths['layer_sizes']
        print("Building VRPO actor & critic with layer sizes", layer_sizes)
        # Critic can be sized independently of the actor (VRPO_CRITIC_LAYERS).
        # The critic has the harder job (Q for every action) and critic_loss has
        # been the standing bottleneck (~13, never improving), so a bigger/deeper
        # critic is worth testing while the actor stays small. Defaults to the
        # actor's layer sizes (unchanged behaviour when unset).
        self.critic_layers = _parse_layers(os.getenv('VRPO_CRITIC_LAYERS'),
                                           layer_sizes)
        self.config['critic_layers'] = self.critic_layers

        print("VRPO config:", self.config)
        self.actor = build_actor(self.state_dim, self.action_dim, layer_sizes)
        self.critic = build_critic(self.state_dim, self.action_dim,
                                   self.critic_layers)

        # Optionally resume weights from a prior run (weights-only, compile=False
        # so we never deserialize a saved optimizer/schedule). VRPO_RESUME points
        # at the actor .keras; the critic path is derived by name.
        resume = os.getenv('VRPO_RESUME')
        if resume:
            self._resume_weights(resume)

        # Build LR schedules (kept as handles so logging can resolve current LR
        # reliably across Keras versions - opt.learning_rate readback is flaky).
        self._actor_sched = self._make_lr(self.actor_lr)
        self._critic_sched = self._make_lr(self.critic_lr)
        self.actor_opt = Adam(learning_rate=self._actor_sched, clipnorm=1.0)
        self.critic_opt = Adam(learning_rate=self._critic_sched, clipnorm=1.0)

        # Collection interface used by VRPOGame + reused apply_move.
        self.memory: list = []

        self.tensorboard = tf.summary.create_file_writer(paths['tensorboard_dir'])
        self._log_config_text()
        self.iteration = 0
        self._truncated_games = 0

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _make_lr(self, peak):
        if not self.lr_schedule_on:
            return peak
        total_steps = max(self.sched_iters * self.steps_per_iter,
                          self.lr_warmup_steps + 1)
        return WarmupCosineSchedule(
            peak_lr=peak, warmup_steps=self.lr_warmup_steps,
            total_steps=total_steps, min_lr=peak * self.lr_min_frac)

    def _resume_weights(self, actor_path):
        critic_path = actor_path.replace('_actor.keras', '_critic.keras')
        a = load_model(actor_path, compile=False)
        self.actor.set_weights(a.get_weights())
        print(f"Resumed actor weights <- {actor_path}")
        if os.path.exists(critic_path):
            try:
                c = load_model(critic_path, compile=False)
                self.critic.set_weights(c.get_weights())
                print(f"Resumed critic weights <- {critic_path}")
            except Exception as e:
                # Expected when VRPO_CRITIC_LAYERS changes the critic arch: the
                # actor resumes, the (now bigger) critic just starts fresh.
                print(f"WARNING: critic arch differs from checkpoint "
                      f"({type(e).__name__}); critic starts fresh.")
        else:
            print(f"WARNING: critic checkpoint not found at {critic_path}; "
                  f"critic starts fresh.")

    def _log_config_text(self):
        text = "\n".join(f"{k}: {v}" for k, v in self.config.items())
        with self.tensorboard.as_default():
            tf.summary.text('VRPO/config', text, step=0)

    def _current_kl_coef(self):
        """Linearly anneal kl_coef start->end across the schedule horizon, then
        HOLD at the end value for the remaining iters (frac clamps to 1.0)."""
        if self.kl_coef_end == self.kl_coef_start or self.sched_iters <= 1:
            return self.kl_coef_start
        frac = min(1.0, self.iteration / float(self.sched_iters))
        return self.kl_coef_start + (self.kl_coef_end - self.kl_coef_start) * frac

    def _current_shaping_alpha(self):
        """Anneal shaping alpha start->end over a [start_frac, end_frac] window
        of the schedule horizon: hold at start before the window, linearly ramp
        within it, hold at end after. Lets the agent learn fast under shaping
        early, then transition to (near-)sparse late to discover beyond-designer
        strategy. No-op when start==end."""
        if self.shape_alpha_end == self.shape_alpha_start or self.sched_iters <= 1:
            return self.shape_alpha_start
        p = self.iteration / float(self.sched_iters)
        lo, hi = self.shape_anneal_start_frac, self.shape_anneal_end_frac
        if p <= lo:
            w = 0.0
        elif p >= hi:
            w = 1.0
        else:
            w = (p - lo) / max(hi - lo, 1e-6)
        return self.shape_alpha_start + (self.shape_alpha_end - self.shape_alpha_start) * w

    def update_schedules(self):
        """Refresh per-iteration annealed values that the ROLLOUT depends on.
        Must be called before collecting the rollout (the reward engine reads
        shaping_alpha during the games). kl_coef is set in update() since it only
        affects the actor loss."""
        self.shaping_alpha = self._current_shaping_alpha()

    def remember(self, entry) -> None:
        self.memory.append(entry)

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def act(self, state, mask):
        """SAMPLE a legal move (single-row path; tests / non-vectorized loop)."""
        s = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        m = tf.convert_to_tensor(mask[None, :], dtype=tf.bool)
        logp = masked_log_softmax(self.actor(s, training=False), m)[0]
        action = int(tf.random.categorical(logp[None, :], 1)[0, 0].numpy())
        return action, float(logp[action].numpy())

    @tf.function(reduce_retracing=True)
    def _act_batch_tf(self, S, M):
        logp_all = masked_log_softmax(self.actor(S, training=False), M)
        actions = tf.random.categorical(logp_all, 1)[:, 0]
        logps = tf.gather(logp_all, actions, batch_dims=1)
        return actions, logps

    def act_batch(self, states, masks):
        """SAMPLE one legal move per game for a BATCH (fast vectorized rollout)."""
        S = tf.convert_to_tensor(states, dtype=tf.float32)
        M = tf.convert_to_tensor(masks, dtype=tf.bool)
        actions, logps = self._act_batch_tf(S, M)
        return actions.numpy().astype(np.int32), logps.numpy().astype(np.float32)

    def get_predictions(self, state, mask):
        """GREEDY interface compatible with Player.choose_move (argmax)."""
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
        """Batched GREEDY argmax actions (for the vectorized evaluator)."""
        S = tf.convert_to_tensor(states, dtype=tf.float32)
        M = tf.convert_to_tensor(masks, dtype=tf.bool)
        return self._greedy_batch_tf(S, M).numpy()

    # ------------------------------------------------------------------ #
    # Q-boosting: one seat's trajectory -> advantages + Q targets
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
        #   A_t     = (Q(s_t,a_t) - V^pi(s_t)) + G_t ;  Qtarget = Q(s_t,a_t) + G_t
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

    def _trace(self, rewards, dones, V, Qsa):
        """Backward Expected SARSA(lambda) trace for one trajectory (numpy).
        Returns (advantages, q_targets). Pure CPU; no GPU calls."""
        L = len(rewards)
        G = np.zeros(L, dtype=np.float32)
        gl = self.gamma * self.lam
        next_G = 0.0
        for i in range(L - 1, -1, -1):
            nonterminal = 1.0 - dones[i]
            v_next = V[i + 1] if (i + 1 < L) else 0.0
            delta = rewards[i] + self.gamma * nonterminal * v_next - Qsa[i]
            G[i] = delta + gl * nonterminal * next_G
            next_G = G[i]
        return (Qsa - V) + G, Qsa + G

    def process_trajectories(self, trajs):
        """Batched equivalent of process_trajectory over MANY trajectories.

        Runs ONE critic forward + ONE actor forward over every state across all
        trajectories (instead of a pair per trajectory), then the cheap
        per-trajectory Expected SARSA(lambda) trace in numpy. Mathematically
        identical to calling process_trajectory on each trajectory - the trace
        respects per-trajectory boundaries (v_next=0 at each trajectory's end) -
        but collapses ~2*N_traj tiny GPU round-trips into 2. `trajs` must be
        non-empty trajectories. Returns a list of per-traj tuples
        (states, actions, logps, masks, advantages, q_targets).
        """
        if not trajs:
            return []
        lengths = [len(t) for t in trajs]
        states = np.concatenate(
            [np.asarray([e[S_STATE] for e in t], dtype=np.float32) for t in trajs])
        masks = np.concatenate(
            [np.asarray([e[S_MASK] for e in t], dtype=bool) for t in trajs])

        # The only two GPU calls, over the whole rollout's states at once.
        S = tf.convert_to_tensor(states)
        M = tf.convert_to_tensor(masks)
        Q_all = self.critic(S, training=False).numpy()
        pi_all = masked_softmax(self.actor(S, training=False), M).numpy()
        V_all = np.sum(pi_all * Q_all, axis=1)

        out = []
        off = 0
        for t, L in zip(trajs, lengths):
            sl = slice(off, off + L)
            off += L
            actions = np.asarray([e[S_ACTION] for e in t], dtype=np.int32)
            rewards = np.asarray([e[S_REWARD] for e in t], dtype=np.float32)
            dones = np.asarray([e[S_DONE] for e in t], dtype=np.float32)
            logps = np.asarray([e[S_LOGP] for e in t], dtype=np.float32)
            Qsa = Q_all[sl][np.arange(L), actions]
            adv, qt = self._trace(rewards, dones, V_all[sl], Qsa)
            out.append((states[sl], actions, logps, masks[sl],
                        adv.astype(np.float32), qt.astype(np.float32)))
        return out

    # ------------------------------------------------------------------ #
    # Update: K_actor clipped-policy epochs, then K_critic regression epochs
    # ------------------------------------------------------------------ #
    def update(self, states, actions, logp_old, masks, adv, q_target):
        # Set the (possibly annealed) KL coefficient for this iteration.
        self.kl_coef_var.assign(self._current_kl_coef())

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

        acc = {'pg': 0.0, 'kl': 0.0, 'ent': 0.0, 'ratio': 0.0,
               'clipfrac': 0.0, 'approx_kl': 0.0}
        a_steps = 0
        for _ in range(self.actor_epochs):
            order = tf.random.shuffle(tf.range(N))
            for start in range(0, N, mb):
                b = order[start:start + mb]
                pg, kl, ent, ratio, clipfrac, approx_kl = self._actor_step(
                    tf.gather(S, b), tf.gather(A, b), tf.gather(LP, b),
                    tf.gather(MK, b), tf.gather(ADV, b))
                acc['pg'] += float(pg); acc['kl'] += float(kl)
                acc['ent'] += float(ent); acc['ratio'] += float(ratio)
                acc['clipfrac'] += float(clipfrac); acc['approx_kl'] += float(approx_kl)
                a_steps += 1

        # --- Critic phase over the cyclic replay buffer ---
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

    @tf.function(reduce_retracing=True)
    def _actor_step(self, S, A, LP_old, MK, ADV):
        with tf.GradientTape() as tape:
            logp_all = masked_log_softmax(self.actor(S, training=True), MK)
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

            loss = pg_loss + self.kl_coef_var * kl_loss

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        clip_frac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(ratio - 1.0), self.clip_eps), tf.float32))
        approx_kl = tf.reduce_mean(LP_old - logp)
        return (pg_loss, kl_loss, tf.reduce_mean(entropy),
                tf.reduce_mean(ratio), clip_frac, approx_kl)

    @tf.function(reduce_retracing=True)
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
    def _current_lr(self):
        if callable(self._actor_sched):
            return float(self._actor_sched(self.actor_opt.iterations))
        return float(self._actor_sched)

    def log_iteration(self, metrics, game_lengths, adv, q_target):
        step = self.iteration
        with self.tensorboard.as_default():
            for key in ('pg_loss', 'kl_to_uniform', 'entropy', 'critic_loss',
                        'mean_ratio', 'clip_fraction', 'approx_kl'):
                tf.summary.scalar(f'VRPO/{key}', metrics[key], step=step)
            tf.summary.scalar('VRPO/adv_mean', float(np.mean(adv)), step=step)
            tf.summary.scalar('VRPO/adv_std', float(np.std(adv)), step=step)
            tf.summary.scalar('VRPO/q_target_mean', float(np.mean(q_target)), step=step)
            tf.summary.scalar('VRPO/kl_coef', float(self.kl_coef_var.numpy()), step=step)
            tf.summary.scalar('VRPO/shaping_alpha', float(self.shaping_alpha), step=step)
            tf.summary.scalar('VRPO/actor_lr', self._current_lr(), step=step)
            if game_lengths:
                # Secondary proxy only: game length tracks strength while it is
                # shrinking, but decouples once it stagnates. vs_dqn is truth.
                tf.summary.scalar('VRPO/game_length',
                                  float(np.mean(game_lengths)) / 2.0, step=step)
            tf.summary.scalar('VRPO/truncated_games_total',
                              self._truncated_games, step=step)
        self.tensorboard.flush()

    def log_eval(self, wr_dqn):
        with self.tensorboard.as_default():
            tf.summary.scalar('Eval/winrate_vs_dqn', wr_dqn, step=self.iteration)
        self.tensorboard.flush()

    def maybe_save_best(self, eval_value) -> bool:
        """Save *_best.keras if this is the best vs_dqn so far. NaN ignored."""
        if eval_value != eval_value or eval_value <= self.best_eval:
            return False
        self.best_eval = eval_value
        self.best_eval_iter = self.iteration
        self.actor.save(self.paths['actor_best_path'])
        self.critic.save(self.paths['critic_best_path'])
        with self.tensorboard.as_default():
            tf.summary.scalar('Eval/best_winrate', self.best_eval,
                              step=self.iteration)
        self.tensorboard.flush()
        print(f"  ** new best vs_dqn={eval_value:.3f} @ iter {self.iteration} "
              f"-> saved {os.path.basename(self.paths['actor_best_path'])}")
        return True

    def save_model(self) -> None:
        self.actor.save(self.paths['actor_save_path'])
        self.critic.save(self.paths['critic_save_path'])
        print(f"Saved actor -> {self.paths['actor_save_path']}")
        print(f"Saved critic -> {self.paths['critic_save_path']}")
