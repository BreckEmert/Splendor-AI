# Splendor/RL/model.py

import os
import pickle
from collections import deque
from copy import deepcopy
from random import sample
import numpy as np

import tensorflow as tf
from keras.config import enable_unsafe_deserialization                 # type: ignore
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization   # type: ignore
from keras.losses import mean_squared_error                            # type: ignore
from keras.models import load_model                                    # type: ignore
from keras.initializers import GlorotNormal, HeNormal                  # type: ignore
from keras.optimizers import Adam                                      # type: ignore
from keras.optimizers.schedules import ExponentialDecay                # type: ignore
from keras.regularizers import l2                                      # type: ignore


class RLAgent:
    def __init__(self, paths):
        print("Making a new RLAgent.")
        enable_unsafe_deserialization()
        self.paths = paths

        self.state_size = 242
        self.action_size = 165
        self.batch_size = 128

        self.memory = self.load_memory()

        self.gamma = 0.9  # 0.1**(1/25)
        self.epsilon = 1.0
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.995
        self.lr = 0.01

        model_from_path = paths['model_from_path']
        if model_from_path:
            print("Loading previous model")
            self.model = load_model(model_from_path)
            self.target_model = load_model(model_from_path)
        else:
            print("Building a new model")
            self.model = self._build_model(paths['layer_sizes'])
            self.target_model = self._build_model(paths['layer_sizes'])
            self.update_target_model()

        self.tensorboard = tf.summary.create_file_writer(paths['tensorboard_dir'])
        self.step = 0

    def _build_model(self, layer_sizes):
        state_input = Input(shape=(self.state_size, ))

        dense1 = Dense(layer_sizes[0], kernel_initializer=HeNormal(), 
                       name='Dense1')(state_input)  # Can do # l2(0.001)
        dense1 = LeakyReLU(alpha=0.3)(dense1)
        # dense1 = BatchNormalization(name='dense1')(dense1)

        # dense2 = Dense(layer_sizes[1], kernel_initializer=HeNormal(), name='Dense2')(dense1)
        # dense2 = LeakyReLU(alpha=0.3)(dense2)
        # dense2 = BatchNormalization(name='Dense2')(dense2)

        action = Dense(self.action_size, activation='linear', 
                       kernel_initializer=HeNormal(), kernel_regularizer=l2(0.015), 
                       name='action')(dense1)

        model = tf.keras.Model(inputs=state_input, outputs=action)
        lr_schedule = ExponentialDecay(self.lr, decay_steps=15, decay_rate=0.98, staircase=False)
        model.compile(loss='mse', optimizer=Adam(learning_rate=lr_schedule, clipnorm=1.0))
        return model
    
    def load_memory(self):
        if self.paths['memory_buffer_path']:
            with open(self.paths['memory_buffer_path'], 'rb') as f:
                flattened_memory = pickle.load(f)
            loaded_memory = [mem for mem in flattened_memory]
            print(f"Loading {len(loaded_memory)} memories")
        else:
            # Should be run with preexisting memory from 
            # training.find_fastest_game because this memory is bad
            dummy_state = np.zeros(self.state_size, dtype=np.float32)
            dummy_mask = np.ones(self.action_size, dtype=bool)
            loaded_memory = [[dummy_state, 1, 1, dummy_state, 1, dummy_mask]]

        return deque(loaded_memory, maxlen=50_000)
    
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

    def update_target_model(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    @tf.function
    def get_predictions(self, state, legal_mask):
        if tf.random.uniform(()) <= self.epsilon:
            qs = tf.random.uniform([self.action_size])
        else:
            state = tf.reshape(state, [1, self.state_size])
            qs = self.model(state, training=False)[0]
        
        return tf.where(legal_mask, qs, tf.fill(qs.shape, -tf.float32.max))

    def remember(self, memory, legal_mask) -> None:
        self.memory[-1].append(legal_mask.copy())  # Avoids recalculation, but overall I regret this method
        self.memory.append(deepcopy(memory))

    @tf.function
    def _batch_train(self, batch) -> None:
        states = tf.convert_to_tensor([mem[0] for mem in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([mem[1] for mem in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([mem[2] for mem in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([mem[3] for mem in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([mem[4] for mem in batch], dtype=tf.float32)
        legal_masks = tf.convert_to_tensor([mem[5] for mem in batch], dtype=tf.bool)

        # Calculate this turn's qs with primary model
        qs = self.model(states, training=False)

        # Predict next turn's actions with primary model
        next_actions = self.model(next_states, training=False)
        next_actions = tf.where(legal_masks, next_actions, tf.fill(next_actions.shape, -np.inf))
        next_actions = tf.argmax(next_actions, axis=1, output_type=tf.int32)

        # Calculate next turn's qs with target model
        next_qs = self.target_model(next_states, training=False)
        selected_next_qs = tf.gather_nd(next_qs, tf.stack([tf.range(len(next_actions)), next_actions], axis=1))

        # Ground qs with reward and value trajectory
        targets = rewards + dones * self.gamma * selected_next_qs
        actions_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
        target_qs = tf.tensor_scatter_nd_update(qs, actions_indices, targets)

        # Fit
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = mean_squared_error(target_qs, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Log
        if self.tensorboard:
            self.step += 1
            step = self.step
            with self.tensorboard.as_default():
                # Training Metrics
                current_lr = self.model.optimizer.learning_rate
                tf.summary.scalar('Training Metrics/learning_rate', current_lr, step=step)
                tf.summary.histogram('Training Metrics/action_hist', actions, step=step)
                tf.summary.scalar('Training Metrics/batch_loss', tf.reduce_mean(loss), step=step)
                tf.summary.scalar('Training Metrics/epsilon', self.epsilon, step=step)
                tf.summary.scalar('Training Metrics/avg_reward', tf.reduce_mean(rewards), step=step)  # Global average reward


                # Q-Values
                legal_qs = tf.where(tf.math.is_finite(qs), qs, tf.zeros_like(qs))  # Removes NaN and inf
                tf.summary.scalar('Q-Values/avg_q', tf.reduce_mean(legal_qs), step=step)  # Global average q

                tf.summary.scalar('Q-Values/avg_take_1', tf.reduce_mean(legal_qs[:5]), step=step)  # Take a single token (really 3)
                tf.summary.scalar('Q-Values/avg_take_2', tf.reduce_mean(legal_qs[5:10]), step=step)  # Take two tokens of a single kind
                tf.summary.scalar('Q-Values/avg_discard', tf.reduce_mean(legal_qs[10:15]), step=step)  # Discard a single token

                tf.summary.scalar('Q-Values/avg_buy_tier_1', tf.reduce_mean(legal_qs[15:27]), step=step)  # Buying actions (each tier)
                tf.summary.scalar('Q-Values/avg_buy_tier_2', tf.reduce_mean(legal_qs[27:39]), step=step)
                tf.summary.scalar('Q-Values/avg_buy_tier_3', tf.reduce_mean(legal_qs[39:45]), step=step)

                tf.summary.scalar('Q-Values/avg_reserve', tf.reduce_mean(legal_qs[45:]), step=step)  # Reserve actions


                # Weights
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        weights = layer.kernel
                        tf.summary.histogram('Model Weights/' + layer.name + '_weights', weights, step=step)

    def replay(self) -> None:
        """Standard off-policy replay"""
        batch = sample(self.memory, self.batch_size)
        self._batch_train(batch)
        
        # Decrease exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self) -> None:
        self.model.save(self.paths['model_save_path'])
        print(f"Saved the model at {self.paths['model_save_path']}")
