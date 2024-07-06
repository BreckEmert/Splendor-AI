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
    def __init__(self, model_path=None, from_model_path=None, layer_sizes=None, memories_path=None, tensorboard_dir=None):
        enable_unsafe_deserialization()
        self.state_size = 241 # Size of state vector
        self.action_size = 61 # Maximum number of actions 

        self.memory = self.load_memories(memories_path)
        self.game_length = 0
        self.batch_size = 256

        self.gamma = 0.91 # 0.1**(1/25)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.99
        self.lr = 0.002

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

        # Split the state into players and board
        board = Lambda(lambda x: x[:, :150], name='board')(state_input)
        players = Lambda(lambda x: x[:, 150:], name='players')(state_input)

        # Process players and then combine with the state
        playerLayer = Dense(layer_sizes[0], kernel_initializer=HeNormal(), name='playerLayer')(players)
        playerLayer = LeakyReLU(alpha=0.3)(playerLayer)
        playerLayer = BatchNormalization(name='playerNorm')(playerLayer)
        playerOut = Dense(32, activation='tanh', kernel_initializer=GlorotNormal(), name='playerOut')(playerLayer)

        # Concatenate the processed part with the rest of the state
        stateCombined = Concatenate()([board, playerOut])

        # Process the entire game
        gameLayer1 = Dense(layer_sizes[1], kernel_initializer=HeNormal(), name='gameLayer1')(stateCombined)
        gameLayer1 = LeakyReLU(alpha=0.3)(gameLayer1)
        gameLayer1 = BatchNormalization(name='gameNorm1')(gameLayer1)

        gameLayer2 = Dense(layer_sizes[2], kernel_initializer=HeNormal(), name='gameLayer2')(gameLayer1)
        gameLayer2 = LeakyReLU(alpha=0.3)(gameLayer2)
        gameLayer2 = BatchNormalization(name='gameNorm2')(gameLayer2)

        move = Dense(self.action_size, activation='linear', 
                    kernel_initializer=HeNormal(), kernel_regularizer=l2(0.013), name='action')(gameLayer2)

        model = tf.keras.Model(inputs=state_input, outputs=move)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr, clipnorm=1.0))
        return model

    def concat_build_model(self, layer_sizes):
        state_input = Input(shape=(self.state_size, ))

        categorizer1 = Dense(layer_sizes[0], kernel_initializer=HeNormal(), name='categorizer1')(state_input)
        categorizer1 = LeakyReLU(alpha=0.3)(categorizer1)
        categorizer1 = BatchNormalization(name='categorizerNorm1')(categorizer1)
        # categorizer2 = Dense(layer_sizes[0], kernel_initializer=HeNormal(), name='categorizer2')(categorizer1)
        # categorizer2 = LeakyReLU(alpha=0.01)(categorizer2)
        # categorizer2 = BatchNormalization(name='categorizerNorm2')(categorizer2)
        category = Dense(3, activation='tanh', kernel_initializer=GlorotNormal(), name='category')(categorizer1)

        state_w_category = Concatenate()([state_input, category])

        # Reuse via categorizer1(state_w_category)?
        specific1 = Dense(layer_sizes[1], kernel_initializer=HeNormal(), name='specific1')(state_w_category)
        specific1 = LeakyReLU(alpha=0.3)(specific1)
        specific1 = BatchNormalization(name='specificNorm1')(specific1)
        # specific2 = Dense(layer_sizes[1], kernel_initializer=HeNormal(), name='specific2')(specific1)
        # specific2 = LeakyReLU(alpha=0.01)(specific2)
        # specific2 = BatchNormalization(name='specificNorm2')(specific2)
        move = Dense(self.action_size, activation='linear', 
                     kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01), name='move')(specific1)

        model = tf.keras.Model(inputs=state_input, outputs=move)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr, clipnorm=1.0))
        return model
    
    def load_memories(self, memories_path):
        if memories_path:
            import pickle
            with open(memories_path, 'rb') as f:
                flattened_memories = pickle.load(f)
            loaded_memories = [mem for mem in flattened_memories]
            print(f"Loading {len(loaded_memories)} memories")
        else:
            loaded_memories = [[0, 0, 0, 0, 0]]
        return deque(loaded_memories, maxlen=50_000)

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
        self.memory.append(memory)
        self.memory[-2].append(legal_mask)
        self.game_length += 1

    def _batch_train(self, batch):
        states = tf.convert_to_tensor([mem[0] for mem in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([mem[1] for mem in batch], dtype=tf.int32) # Int8 or 16?
        rewards = tf.convert_to_tensor([mem[2] for mem in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([mem[3] for mem in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([mem[4] for mem in batch], dtype=tf.float32)
        legal_masks = tf.convert_to_tensor([mem[5] for mem in batch], dtype=tf.bool)

        # Calculate this turn's qs with primary model
        qs = self.model.predict(states, verbose=0)

        # Predict next turn's actions with primary model
        next_actions = self.model.predict(next_states, verbose=0)
        next_actions = tf.where(legal_masks, next_actions, tf.fill(next_actions.shape, -np.inf))
        next_actions = tf.argmax(next_actions, axis=1, output_type=tf.int32)
        next_actions = tf.stack([tf.range(len(next_actions), dtype=tf.int32), next_actions], axis=1)

        # Calculate next turn's qs with target model
        next_qs = self.target_model.predict(next_states, verbose=0)
        next_qs = tf.where(legal_masks, next_qs, tf.fill(next_qs.shape, -np.inf))
        next_qs = tf.gather_nd(next_qs, next_actions)

        # Ground qs with reward and value trajectory
        targets = rewards + dones * self.gamma * next_qs
        actions_indices = tf.stack([tf.range(len(actions), dtype=tf.int32), actions], axis=1)
        target_qs = tf.tensor_scatter_nd_update(qs, actions_indices, targets)

        # Fit
        history = self.model.fit(states, target_qs, batch_size=256, epochs=1, verbose=0)

        # Log
        if self.tensorboard:
            self.step += 1
            step = self.step
            with self.tensorboard.as_default():
                # Grouped cards
                tf.summary.histogram('Training Metrics/action_hist', actions, step=step)
                tf.summary.scalar('Training Metrics/batch_loss', history.history['loss'][0], step=step)
                tf.summary.scalar('Training Metrics/avg_reward', tf.reduce_mean(rewards), step=step)
                tf.summary.scalar('Training Metrics/learning_rate', self.lr, step=step)
                legal_qs = tf.where(tf.math.is_finite(qs), qs, tf.zeros_like(qs))
                tf.summary.scalar('Training Metrics/avg_q', tf.reduce_mean(legal_qs), step=step)

                # Q-values over time
                # for action in range(self.action_size):
                #     average_qs = np.mean(legal_qs[:, action], axis=0)
                #     tf.summary.scalar(f"action_qs/action_{action}", average_qs, step=step)
                
                # Weights
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        weights = layer.get_weights()[0]
                        tf.summary.histogram('Model Weights/'+ layer.name +'_weights', weights, step=step)

    def replay(self):
        # Training twice for now
        batch = sample(self.memory, self.batch_size)
        self._batch_train(batch)
        
        # Decrease exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, game_length):
        batch = list(self.memory)[-game_length:]
        total_negative = total_positive = n_negative = n_positive = 0
        for mem in batch:
            reward = mem[2]
            if reward < 0:
                total_negative += reward
                n_negative += 1
            elif reward > 0:
                total_positive += reward
                n_positive += 1

        print("\nNegative Rewards:", total_negative, total_negative/n_negative)
        print("Positive Rewards:", total_positive, total_positive/n_positive, "\n")
        self._batch_train(batch)

    def save_model(self, base_path):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        model_path = os.path.join(base_path, 'model.keras')
        self.model.save(model_path)
        print(f"Saved the model at {model_path}")