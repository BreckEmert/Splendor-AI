# Splendor/RL/model.py

from collections import deque
import os
from random import sample

import numpy as np
import tensorflow as tf
from keras.config import enable_unsafe_deserialization
from keras.layers import Input, Dense, Concatenate, Lambda, LeakyReLU, BatchNormalization
from keras.models import load_model
from keras.initializers import GlorotNormal, HeNormal
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.regularizers import l2


class RLAgent:
    def __init__(self, model_path=None, from_model_path=None, layer_sizes=None, memory_path=None, tensorboard_dir=None):
        enable_unsafe_deserialization()
        self.state_size = 243 # Size of state vector
        self.action_size = 61 # Maximum number of actions 

        self.memory = self.load_memory(memory_path)
        self.batch_size = 128

        self.gamma = 0.99 # 0.1**(1/25)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.995
        self.lr = 0.01

        if from_model_path:
            self.model = load_model(from_model_path)
            self.target_model = load_model(from_model_path)

        else:
            print("Building a new model")
            self.model = self._build_model(layer_sizes)
            self.target_model = self._build_model(layer_sizes)
            self.update_target_model()

        if tensorboard_dir:
            self.tensorboard = tf.summary.create_file_writer(tensorboard_dir)
            self.action_counts = np.zeros(self.action_size)
            self.step = 0

    def _build_model(self, layer_sizes):
        state_input = Input(shape=(self.state_size, ))

        dense1 = Dense(layer_sizes[0], 
                       kernel_initializer=HeNormal(), # kernel_regularizer=l2(0.001), 
                       name='Dense1')(state_input)
        dense1 = LeakyReLU(alpha=0.3)(dense1)
        dense1 = BatchNormalization(name='dense1')(dense1)

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
    
    def load_memory(self, memory_path):
        if memory_path:
            import pickle
            with open(memory_path, 'rb') as f:
                flattened_memory = pickle.load(f)
            loaded_memory = [mem for mem in flattened_memory]
            print(f"Loading {len(loaded_memory)} memories")
        else:
            loaded_memory = [[0, 0, 0, 0, 0]]
        return deque(loaded_memory, maxlen=50_000)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_predictions(self, state, legal_mask):
        if np.random.rand() <= self.epsilon:
            qs = np.random.rand(self.action_size)  # Exploration
        else:
            state = tf.reshape(state, [1, self.state_size])
            qs = self.model.predict(state, verbose=0)[0]  # All actions

        # Illegal move filter
        qs = np.where(legal_mask, qs, -np.inf)

        return qs

    def remember(self, memory, legal_mask):
        # assert memory[2] >= 0, f'Reward is {memory[2]}'
        self.memory.append(memory)
        self.memory[-2].append(legal_mask)

    def _batch_train(self, batch):
        states = tf.convert_to_tensor([mem[0] for mem in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([mem[1] for mem in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([mem[2] for mem in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([mem[3] for mem in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([mem[4] for mem in batch], dtype=tf.float32)
        legal_masks = tf.convert_to_tensor([mem[5] for mem in batch], dtype=tf.bool)

        # assert np.all(rewards >= 0), 'rewards are lt 0'

        # Calculate this turn's qs with primary model
        qs = self.model.predict(states, verbose=0)

        # Predict next turn's actions with primary model
        next_actions = self.model.predict(next_states, verbose=0)
        next_actions = tf.where(legal_masks, next_actions, tf.fill(next_actions.shape, -np.inf))
        next_actions = tf.argmax(next_actions, axis=1, output_type=tf.int32)

        # Calculate next turn's qs with target model
        next_qs = self.target_model.predict(next_states, verbose=0)
        selected_next_qs = tf.gather_nd(next_qs, tf.stack([tf.range(len(next_actions)), next_actions], axis=1))

        # Ground qs with reward and value trajectory
        targets = rewards + dones * self.gamma * selected_next_qs
        actions_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
        target_qs = tf.tensor_scatter_nd_update(qs, actions_indices, targets)

        # Fit
        history = self.model.fit(states, target_qs, batch_size=self.batch_size, epochs=1, verbose=0)

        # Log
        if self.tensorboard:
            self.step += 1
            step = self.step
            with self.tensorboard.as_default():
                # Grouped cards
                tf.summary.histogram('Training Metrics/action_hist', actions, step=step)
                tf.summary.scalar('Training Metrics/batch_loss', history.history['loss'][0], step=step)
                tf.summary.scalar('Training Metrics/avg_reward', tf.reduce_mean(rewards), step=step)
                current_lr = self.model.optimizer.learning_rate.numpy()
                tf.summary.scalar('Training Metrics/learning_rate', current_lr, step=step)
                legal_qs = tf.where(tf.math.is_finite(qs), qs, tf.zeros_like(qs))
                tf.summary.scalar('Training Metrics/avg_q', tf.reduce_mean(legal_qs), step=step)
                
                # Weights
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        weights = layer.get_weights()[0]
                        tf.summary.histogram('Model Weights/'+ layer.name +'_weights', weights, step=step)

    def replay(self):
        batch = sample(self.memory, self.batch_size)
        self._batch_train(batch)
        
        # Decrease exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, base_path):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        model_path = os.path.join(base_path, 'model.keras')
        self.model.save(model_path)
        print(f"Saved the model at {model_path}")