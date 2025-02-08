# Splendor/RL/model.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Stop NUMA warnings

import pickle
from collections import deque
from random import sample
import numpy as np

import tensorflow as tf
from keras.config import enable_unsafe_deserialization                 
from keras.layers import Input, Dense, LeakyReLU   
from keras.losses import mean_squared_error, Huber
from keras.models import load_model
from keras.initializers import GlorotNormal, HeNormal
from keras.optimizers import Adam
from keras.regularizers import l2


class RLAgent:
    def __init__(self, paths):
        print("Making a new RLAgent.")
        enable_unsafe_deserialization()
        self.paths = paths

        # Dimensions
        self.state_dim = 251
        self.action_dim = 141
        self.batch_size = 512

        # Huber loss
        self.huber = Huber()

        # DQN
        self.gamma = 0.93
        self.gamma_max = 0.995
        self.gamma_accum = 6e-5

        self.epsilon = 0.04
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.999_978

        # Initial memory, note batch size/replay_freq is samples per memory
        # 10k/50 = 200 games but memories are correlated as both 
        # current and enemy player are stored
        self.replay_buffer_size = 50_000
        self.memory = self._load_memory()

        # Learning rate
        lr_schedule = ScheduleWithWarmup(
            warmup_init_lr = 1e-5, 
            decay_init_lr = 1e-4, 
            warmup_steps = 3_000, 
            decay_steps = 100_000, 
            decay_rate = 0.3
        )
        self.tau = 0.002

        # Model
        model_from_path = paths['model_from_path']
        if model_from_path:
            print("Loading previous model")
            self.model = load_model(model_from_path)
            self.target_model = load_model(model_from_path)
        else:
            self.model = self._build_model(lr_schedule)
            self.model._name = "policy_model"
            self.target_model = self._build_model(lr_schedule)
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
        # = 157    # start of active player block
        #  + 12    # skip gems (6), gem_sum(1), cards(5)
        #  + i*11  # skip i entire reserved-card blocks
        #  + 5     # offset to the 'points' element within the card

        for buy_reserved_action in range(6):
            a = 119 + buy_reserved_action
            i = buy_reserved_action // 2
            offset = 157 + 12 + i*11 + 5
            action_offsets[a] = offset
        #  157 = start of active player block

        self.buyIdx_to_pointIdx = tf.constant(action_offsets, dtype=tf.int32)

    def _build_model(self, lr_schedule):
        """Builds a nn for both policy and target network."""
        layer_sizes = self.paths['layer_sizes']
        print("Building a new model with layer sizes", layer_sizes)
        
        # Input
        state_input = Input(shape=(self.state_dim, ))

        # Dense layers
        x = state_input
        for i, layer_size in enumerate(self.paths['layer_sizes']):
            x = Dense(layer_size, kernel_initializer=HeNormal(), 
                      name=f'dense{i+1}')(x)
            x = LeakyReLU(negative_slope=0.3)(x)

        # Action layer
        action = Dense(self.action_dim, activation='linear', 
                       kernel_initializer=HeNormal(),  #, kernel_regularizer=l2(0.015)
                       name='action')(x)

        # Final model
        model = tf.keras.Model(inputs=state_input, outputs=action)

        # Optimizer
        optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

        # Compile and return
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

    def _update_target_model(self) -> None:
        """Keeps the target model lagged behind the policy model"""
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        updated_weights = []
        for mw, tw in zip(model_weights, target_weights):
            updated_weights.append(self.tau * mw + (1 - self.tau) * tw)

        self.target_model.set_weights(updated_weights)

    @tf.function
    def get_predictions(self, state, legal_mask):
        """Returns q-values (random if we explore)"""
        r = tf.random.uniform(())
        qs = tf.cond(
            r <= self.epsilon,
            lambda: tf.random.uniform([self.action_dim]),  # Exploration
            lambda: self.model(state[None, :], training=False)[0]  # Exploitation
        )
        # Set illegal moves' q to -inf
        return tf.where(legal_mask, qs, tf.fill(qs.shape, -tf.float32.max))

    def remember(self, memory) -> None:
        self.memory.append(memory)

    @tf.function
    def _batch_train(self, states, actions, rewards, next_states, legal_masks, dones) -> None:
        """The sole training function"""
        # Calculate this turn's qs with primary model
        qs = self.model(states, training=False)

        # Predict next turn's actions with primary model
        next_actions = self.model(next_states, training=False)
        next_actions = tf.where(legal_masks, next_actions, tf.fill(next_actions.shape, -np.inf))
        next_actions = tf.argmax(next_actions, axis=1, output_type=tf.int32)

        # Calculate next turn's qs with target model
        next_qs = self.target_model(next_states, training=False)
        actions_indices = tf.stack([tf.range(len(next_actions)), next_actions], axis=1)
        selected_next_qs = tf.gather_nd(next_qs, actions_indices)

        # Ground qs with reward and value trajectory
        targets = rewards + (1.0-dones) * self.gamma * selected_next_qs
        actions_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
        target_qs = tf.tensor_scatter_nd_update(qs, actions_indices, targets)

        # Fit
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = self.huber(target_qs, predictions)
            # loss = mean_squared_error(target_qs, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return qs, loss

    def _tensorboard(self, states, actions, rewards, qs, loss):
        """Tensorboard logging.  Always active, but 
        you can increase the step between logs.
        """
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
            legal_qs = tf.where(tf.math.is_finite(qs), qs, tf.zeros_like(qs))  # Removes NaN and inf
            avg_q = tf.reduce_mean(legal_qs)
            tf.summary.scalar(f'{folder}/_avg_q', avg_q, step=step)  # Global average q

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
        """Standard off-policy replay"""
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
        if self.step % 250 == 0:
            self._tensorboard(states, actions, rewards, qs, loss)
        
        # Update DQN parameters
        self.epsilon -= (self.epsilon - self.epsilon_min) * (1 - self.epsilon_decay)
        self.gamma += (self.gamma_max - self.gamma) * self.gamma_accum
        
        # Update target model
        self._update_target_model()

    def save_model(self) -> None:
        self.model.save(self.paths['model_save_path'])
        print(f"Saved the model at {self.paths['model_save_path']}")


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
