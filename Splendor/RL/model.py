# Splendor/RL/model.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Stop NUMA warnings

import pickle
import numpy as np
from collections import deque
from random import sample

import tensorflow as tf
from keras.config import enable_unsafe_deserialization                 
from keras.layers import Input, Dense, LeakyReLU
from keras.losses import Huber
from keras import ops
from keras.models import load_model
from keras.initializers import HeNormal
from keras.optimizers import Adam


class RLAgent:
    def __init__(self, paths):
        print("Making a new RLAgent.")
        enable_unsafe_deserialization()
        self.paths = paths
        self.huber = Huber()

        # Dimensions
        self.state_dim = 326
        self.action_dim = 141
        self.batch_size = 512

        # DQN
        # self.gamma = 0.98
        # # self.gamma = 0.999
        # self.gamma_max = 0.999
        # self.gamma_accum = 1.5e-4
        # Overnight:
        self.gamma = 1.0
        self.gamma_max = 1.0
        self.gamma_accum = 2e-5

        self.epsilon = 0.002
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99996
        self.exploration_temp = tf.Variable(0.5, trainable=False, dtype=tf.float32)

        self.tau = 0.001
        self._target_update_interval = 1

        # Initial memory, note batch size/replay_freq is samples per memory
        # 10k/50 = 200 games but memories are correlated as both 
        # current and enemy player are stored
        self.replay_buffer_size = 50_000
        self.memory = self._load_memory()

        # Learning rate
        # Regular:
        # lr_schedule = ScheduleWithWarmup(
        #     warmup_init_lr = 1e-5, 
        #     decay_init_lr = 8e-4,
        #     warmup_steps = 500, 
        #     decay_steps = 20_000, 
        #     decay_rate = 0.1
        # )
        # Finetuning:
        override_schedule = True
        lr_schedule = ScheduleWithWarmup(
            warmup_init_lr = 1e-7, 
            decay_init_lr = 8e-5, 
            warmup_steps = 4_000, 
            decay_steps = 30_000, 
            decay_rate = 0.1
        )
        # Overnight:
        # lr_schedule = ScheduleWithWarmup(
        #     warmup_init_lr = 5e-7,
        #     decay_init_lr = 1.5e-4,
        #     warmup_steps = 4_000,
        #     decay_steps = 50_000,
        #     decay_rate = 0.5
        # )

        # Model
        model_from_path = paths['model_from_path']
        if model_from_path:
            if override_schedule:
                self.model = load_model(model_from_path, compile=False)
                self.target_model = load_model(model_from_path, compile=False)
                optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)
                self.model.compile(loss='mse', optimizer=optimizer)
            else:
                print("Loading existing model")
                self.model: tf.keras.Model = load_model(model_from_path)
                self.target_model: tf.keras.Model = load_model(model_from_path)
        else:
            self.model: tf.keras.Model = self._build_model(lr_schedule)
            self.model._name = "policy_model"
            self.target_model: tf.keras.Model = self._build_model(lr_schedule)
            self.target_model._name = "target_model"
            self.target_model.set_weights(self.model.get_weights())

        # Tensorboard logging
        self.tensorboard = tf.summary.create_file_writer(paths['tensorboard_dir'])
        self._get_action_indices()
        self.step = 0

    def _get_action_indices(self):
        """Helper function for TensorBoard so that we can 
        plot specific types of actions.
        """
        # Take indices
        self.i_take_3 = tf.range(0, 40, 4)
        self.i_take_2 = tf.range(40, 55, 3)
        combined = tf.concat([self.i_take_3, self.i_take_2], axis=0)
        self.i_other_takes = tf.sets.difference(
            tf.constant([range(70)]),
            tf.expand_dims(combined, axis=0)
        ).values

        # Buy indices
        self.i_buy_tier1 = tf.range(95, 103)
        self.i_buy_tier2 = tf.range(103, 111)
        self.i_buy_tier3 = tf.range(111, 119)
        self.i_buy_reserved = tf.range(119, 125)

        # Reserve indices
        self.i_reserve = tf.range(125, 140)


        # Maps buy actions to the associated point value in the state
        action_offsets = [-1] * self.action_dim
        for buy_action in range(24):
            a = 95 + buy_action
            card_idx = buy_action // 2
            offset = 7 + card_idx*11 + 5
            action_offsets[a] = offset
        # = 232    # start of active player block
        #  + 12    # skip gems (6), gem_sum(1), cards(5)
        #  + i*11  # skip i entire reserved-card blocks
        #  + 5     # offset to the 'points' element within the card

        for buy_reserved_action in range(6):
            a = 119 + buy_reserved_action
            i = buy_reserved_action // 2
            offset = 232 + 12 + i*11 + 5
            action_offsets[a] = offset
        #  232 = start of active player block

        self.buyIdx_to_pointIdx = tf.constant(action_offsets, dtype=tf.int32)

    def _build_model(self, lr_schedule):
        """Builds a nn for both policy and target network."""
        layer_sizes = self.paths['layer_sizes']
        print("Building a new model with layer sizes", layer_sizes)
        
        # Input
        state_input = Input(shape=(self.state_dim, ))

        # Dense
        x = state_input
        for i, layer_size in enumerate(self.paths['layer_sizes']):
            x = Dense(layer_size, kernel_initializer=HeNormal(), name=f'dense{i+1}')(x)
            x = LeakyReLU(negative_slope=0.3)(x)

        # Dueling heads
        v = Dense(1, kernel_initializer=HeNormal(), name='value')(x)
        a = Dense(self.action_dim, kernel_initializer=HeNormal(), name='advantage')(x)
        action = a - ops.mean(a, axis=1, keepdims=True) + v
        
        # Model
        model = tf.keras.Model(inputs=state_input, outputs=action)
        optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)
        model.compile(loss='mse', optimizer=optimizer)

        return model
    
    def _load_memory(self):
        """Loads existing memory into the buffer, or creates 
        a single fake memory so that memory[-1] works.
        """
        if self.paths['memory_buffer_path']:
            with open(self.paths['memory_buffer_path'], 'rb') as f:
                flattened_memory = pickle.load(f)
            loaded_memory = [mem for mem in flattened_memory]
            print(f"Loading {len(loaded_memory)} memories")
        else:
            print("Warning, training will be garbage with random memory.")
            dummy_state = np.zeros(self.state_dim, dtype=np.float32)
            dummy_mask = np.ones(self.action_dim, dtype=bool)
            loaded_memory = [
                [dummy_state, 1, 1, dummy_state, dummy_mask, 1]
                for _ in range(self.replay_buffer_size)
            ]

        return deque(loaded_memory, maxlen=self.replay_buffer_size)
    
    def write_memory(self) -> None:
        memory_path = os.path.join(self.paths['saved_files_dir'], "memory.pkl")

        # Get the old memories if we don't want to overwrite them
        if self.paths['model_from_path']:
            with open(memory_path, 'rb') as f:
                existing_memory = pickle.load(f)
            print(f"Loaded {len(existing_memory)} existing memories.")
            existing_memory.extend(self.memory)
            memory = existing_memory
        else:
            memory = self.memory

        with open(memory_path, 'wb') as f:
            pickle.dump(memory, f)

        print(f"Wrote {len(memory)} memories to {memory_path}")

    @tf.function
    def _update_target_model(self):
        tau = self.tau
        for v, tv in zip(self.model.trainable_variables,
                         self.target_model.trainable_variables):
            tv.assign(tau * v + (1.0 - tau) * tv)

    # @tf.function
    # def get_predictions(self, state, legal_mask):
    #     """Returns q-values (random if we explore)"""
    #     r = tf.random.uniform(())
    #     qs = tf.cond(
    #         r <= self.epsilon,
    #         lambda: tf.random.uniform([self.action_dim]),  # Exploration
    #         lambda: self.model(state[None, :], training=False)[0]  # Exploitation
    #     )
    #     # Set illegal moves' q to -inf
    #     return tf.where(legal_mask, qs, tf.fill(qs.shape, -tf.float32.max))

    @tf.function
    def get_predictions(self, state, legal_mask):
        def _exploit():
            q = self.model(state[None, :], training=False)[0]
            return tf.where(legal_mask, q, tf.fill(tf.shape(q), -1e9))

        def _explore():
            """Boltzmann sampling"""
            q = self.model(state[None, :], training=False)[0]
            q = tf.where(legal_mask, q, tf.fill(tf.shape(q), -1e9))
            u = tf.random.uniform(tf.shape(q), 1e-6, 1.0 - 1e-6)
            g = -tf.math.log(-tf.math.log(u))  # Gumbel(0,1)
            return q / self.exploration_temp + g

        qs = tf.cond(tf.random.uniform(()) <= self.epsilon, _explore, _exploit)
        return tf.where(legal_mask, qs, tf.fill(tf.shape(qs), -tf.float32.max))

    def remember(self, memory) -> None:
        self.memory.append(memory)

    @tf.function
    def _batch_train(self, states, actions, rewards, next_states, 
                     legal_masks, dones) -> tuple[tf.Tensor, tf.Tensor]:
        """The sole training function"""
        # Predict next turn's actions with primary model
        next_actions = self.model(next_states, training=False)
        next_actions = tf.where(legal_masks, next_actions, tf.fill(next_actions.shape, -np.inf))
        next_actions = tf.argmax(next_actions, axis=1, output_type=tf.int32)

        # Calculate next turn's qs with target model
        next_qs = self.target_model(next_states, training=False)
        next_batch_indices = tf.range(tf.shape(next_actions)[0], dtype=tf.int32)
        next_actions_indices = tf.stack([next_batch_indices, next_actions], axis=1)
        selected_next_qs = tf.gather_nd(next_qs, next_actions_indices)

        # Prepare to bootstrap the next state's qs outside of tape
        # Note this is negamax because both players share one memory
        targets = rewards - (1.0-dones) * self.gamma * selected_next_qs
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        actions_indices = tf.stack([batch_indices, actions], axis=1)

        # Fit
        with tf.GradientTape() as tape:
            # Calculate this turn's qs with primary model
            qs = self.model(states, training=True)
            # Replace chosen actions with target values
            target_qs = tf.tensor_scatter_nd_update(tf.stop_gradient(qs), actions_indices, targets)
            loss: tf.Tensor = self.huber(target_qs, qs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return qs, loss

    def _tensorboard(self, states, actions, rewards, qs, loss):
        step = self.step

        with self.tensorboard.as_default():
            # Training metrics
            ###################################################################
            current_lr = self.model.optimizer.learning_rate
            tf.summary.scalar('Training Metrics/learning_rate', current_lr, step=step)
            tf.summary.scalar('Training Metrics/gamma', self.gamma, step=step)
            tf.summary.scalar('Training Metrics/batch_loss', tf.reduce_mean(loss), step=step)
            tf.summary.scalar('Training Metrics/epsilon', self.epsilon, step=step)
            tf.summary.scalar('Training Metrics/batch_avg_reward', tf.reduce_mean(rewards), step=step)
            tf.summary.histogram('Training Metrics/action_hist', actions, step=step)


            # Q-values
            ###################################################################
            folder = "Average Q-Values (normalized by global avg)"
            legal_qs = tf.where(tf.math.is_finite(qs), qs, tf.zeros_like(qs))
            avg_q = tf.reduce_mean(legal_qs)  # Global average q
            tf.summary.scalar(f'{folder}/_avg_q', avg_q, step=step)

            # Q-values of take moves
            ###########################
            # qs
            take_3_qs = tf.gather(legal_qs, self.i_take_3, axis=1)
            take_2_qs = tf.gather(legal_qs, self.i_take_2, axis=1)
            other_takes_qs = tf.gather(legal_qs, self.i_other_takes, axis=1)
            # avg and normalize
            take_3_qs = tf.reduce_mean(take_3_qs) - avg_q
            take_2_qs = tf.reduce_mean(take_2_qs) - avg_q
            other_takes_qs = tf.reduce_mean(other_takes_qs) - avg_q
            # log
            tf.summary.scalar(f'{folder}/take_3', take_3_qs, step=step)
            tf.summary.scalar(f'{folder}/take_2', take_2_qs, step=step)
            tf.summary.scalar(f'{folder}/take_other', other_takes_qs, step=step)
            
            # Q-values of buy moves
            ###########################
            # qs
            buy_tier1_qs = tf.gather(legal_qs, self.i_buy_tier1, axis=1)
            buy_tier2_qs = tf.gather(legal_qs, self.i_buy_tier2, axis=1)
            buy_tier3_qs = tf.gather(legal_qs, self.i_buy_tier3, axis=1)
            buy_reserved_qs = tf.gather(legal_qs, self.i_buy_reserved, axis=1)
            # avg and normalize
            buy_tier1_qs = tf.reduce_mean(buy_tier1_qs) - avg_q
            buy_tier2_qs = tf.reduce_mean(buy_tier2_qs) - avg_q
            buy_tier3_qs = tf.reduce_mean(buy_tier3_qs) - avg_q
            buy_reserved_qs = tf.reduce_mean(buy_reserved_qs) - avg_q
            # log
            tf.summary.scalar(f'{folder}/.buy_tier1', buy_tier1_qs, step=step)
            tf.summary.scalar(f'{folder}/.buy_tier2', buy_tier2_qs, step=step)
            tf.summary.scalar(f'{folder}/.buy_tier3', buy_tier3_qs, step=step)
            tf.summary.scalar(f'{folder}/_buy_reserved', buy_reserved_qs, step=step)

            # Q-values of reserve moves
            ###########################
            # Reserve actions
            reserve_qs = tf.gather(legal_qs, self.i_reserve, axis=1)
            reserve_qs = tf.reduce_mean(reserve_qs) - avg_q
            tf.summary.scalar(f'{folder}/_reserve', reserve_qs, step=step)

            # Q-values of buys, by point value
            ###########################
            # Get the action indices of buy moves
            offsets = tf.gather(self.buyIdx_to_pointIdx, actions)
            is_buy = offsets >= 0
            buy_rows = tf.where(is_buy)[:, 0]

            # Get the points of the cards
            gather_pts_idx = tf.stack([
                tf.cast(buy_rows, tf.int32), 
                tf.gather(offsets, buy_rows)
            ], axis=1)
            points_f = tf.gather_nd(states, gather_pts_idx) * 15.0

            # Get the q-values of the actions
            gather_q_idx = tf.stack([
                tf.cast(buy_rows, tf.int32), 
                tf.gather(actions, buy_rows)
            ], axis=1)
            buy_qs = tf.gather_nd(qs, gather_q_idx)

            # Bucket the q-values by points
            num_buckets = 6
            points_i = tf.cast(tf.round(points_f), tf.int32)
            sum_by_pts = tf.math.unsorted_segment_sum(buy_qs, points_i, num_buckets)
            count_by_pts = tf.math.unsorted_segment_sum(tf.ones_like(buy_qs), points_i, num_buckets)
            mean_by_pts = sum_by_pts / (count_by_pts + 1e-9)

            # Log each point bucket
            for p in range(num_buckets):
                tf.summary.scalar(f'BuyQ/points_{p}', mean_by_pts[p] - avg_q, step=step)


            # Model weights
            ###################################################################
            for layer in self.model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    weights = layer.kernel
                    tf.summary.histogram('Model Weights/' + layer.name + '_weights', weights, step=step)

        # self.tensorboard.flush()

    def log_game_lengths(self, avg):
        with self.tensorboard.as_default():
            tf.summary.scalar('Training Metrics/game lengths', avg, step=self.step)

    def replay(self) -> None:
        # Get a batch and convert it to tf.tensors
        batch = sample(self.memory, self.batch_size)

        states = tf.convert_to_tensor([mem[0] for mem in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([mem[1] for mem in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([mem[2] for mem in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([mem[3] for mem in batch], dtype=tf.float32)
        legal_masks = tf.convert_to_tensor([mem[4] for mem in batch], dtype=tf.bool)
        dones = tf.convert_to_tensor([mem[5] for mem in batch], dtype=tf.float32)

        # Train and log
        qs, loss = self._batch_train(states, actions, rewards, next_states, legal_masks, dones)
        
        self.step += 1
        if self.step % 300 == 0:
            self._tensorboard(states, actions, rewards, qs, loss)
        
        # Update DQN parameters
        self.epsilon -= (self.epsilon - self.epsilon_min) * (1 - self.epsilon_decay)
        self.gamma += (self.gamma_max - self.gamma) * self.gamma_accum
        
        # Update target model
        if self.step % self._target_update_interval == 0:
            self._update_target_model()

    def save_model(self) -> None:
        self.model.save(self.paths['model_save_path'])
        print(f"Saved the model at {self.paths['model_save_path']}")

    def _batch_train_two_ply(
            self, states, actions, rewards, next_states, 
            legal_masks, dones, 
            next2_states, next2_legal_masks, next2_valid
        ) -> tuple[tf.Tensor, tf.Tensor]:
        """Trying this out as well.  Because we have alternating players
        in the memory, in order to get classical behavior you have to
        look ahead two steps to see your own next state.
        """
        # Predict next turn's actions with primary model
        next2_actions = self.model(next2_states, training=False)
        next2_actions = tf.where(next2_legal_masks, next2_actions, tf.fill(next2_actions.shape, -np.inf))
        next2_actions = tf.argmax(next2_actions, axis=1, output_type=tf.int32)

        # Calculate next turn's qs with target model
        next2_qs = self.target_model(next2_states, training=False)
        next2_batch_indices = tf.range(tf.shape(next2_actions)[0], dtype=tf.int32)
        next2_actions_indices = tf.stack([next2_batch_indices, next2_actions], axis=1)
        selected_next2_qs = tf.gather_nd(next2_qs, next2_actions_indices)

        # Prepare to bootstrap the next state's qs outside of tape
        # Note this is regular max over the next turn (skipped opponent)
        targets = rewards + next2_valid * self.gamma * selected_next2_qs
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        actions_indices = tf.stack([batch_indices, actions], axis=1)

        # Fit
        with tf.GradientTape() as tape:
            # Calculate this turn's qs with primary model
            qs = self.model(states, training=True)
            # Replace chosen actions with target values
            target_qs = tf.tensor_scatter_nd_update(tf.stop_gradient(qs), actions_indices, targets)
            loss: tf.Tensor = self.huber(target_qs, qs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return qs, loss

    def replay_two_ply(self) -> None:
        # Get a batch and convert it to tf.tensors
        mem = self.memory
        N = len(mem) - 1
        idxs = np.random.randint(0, N, size=self.batch_size)

        batch = [mem[i]   for i in idxs]
        next2 = [mem[i+1] for i in idxs]

        states = tf.convert_to_tensor([m[0] for m in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([m[1] for m in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([m[2] for m in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([m[3] for m in batch], dtype=tf.float32)
        legal_masks = tf.convert_to_tensor([m[4] for m in batch], dtype=tf.bool)
        dones = tf.convert_to_tensor([m[5] for m in batch], dtype=tf.float32)

        next2_states = tf.convert_to_tensor([m[3] for m in next2], dtype=tf.float32)
        next2_legal_masks = tf.convert_to_tensor([m[4] for m in next2], dtype=tf.bool)
        next2_dones = tf.convert_to_tensor([m[5] for m in next2], dtype=tf.float32)
        next2_valid = 1.0 - tf.maximum(dones, next2_dones)

        # Train and log
        qs, loss = self._batch_train_two_ply(
            states, actions, rewards, next_states, 
            legal_masks, dones, 
            next2_states, next2_legal_masks, next2_valid
        )
        
        self.step += 1
        if self.step % 300 == 0:
            self._tensorboard(states, actions, rewards, qs, loss)
        
        # Update DQN parameters
        self.epsilon -= (self.epsilon - self.epsilon_min) * (1 - self.epsilon_decay)
        self.gamma += (self.gamma_max - self.gamma) * self.gamma_accum
        
        # Update target model
        if self.step % self._target_update_interval == 0:
            self._update_target_model()

class ScheduleWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_init_lr, decay_init_lr, 
                 warmup_steps, decay_steps, decay_rate):
        super().__init__()
        self.warmup_init_lr = tf.cast(warmup_init_lr, tf.float32)
        self.decay_init_lr = tf.cast(decay_init_lr, tf.float32)
        self.warmup_slope = self.decay_init_lr - self.warmup_init_lr

        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)

    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)

        # During warmup, linear increase
        warmup_progress = step_float / self.warmup_steps
        warmup_lr = self.warmup_init_lr + self.warmup_slope * warmup_progress

        # After warmup, exponential decay
        decay_step = step_float - self.warmup_steps
        decay_factor = tf.pow(self.decay_rate, decay_step / self.decay_steps)
        decayed_lr = self.decay_init_lr * decay_factor

        # Return based on self.step
        return tf.cond(step_float < self.warmup_steps,
                       lambda: warmup_lr,
                       lambda: decayed_lr)

    def get_config(self):
        return {
            "warmup_init_lr": float(self.warmup_init_lr.numpy()),
            "decay_init_lr": float(self.decay_init_lr.numpy()),
            "warmup_steps": float(self.warmup_steps.numpy()),
            "decay_steps": float(self.decay_steps.numpy()),
            "decay_rate": float(self.decay_rate.numpy())
        }
