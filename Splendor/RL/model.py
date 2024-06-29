# Splendor/RL/model.py

from collections import deque
from random import sample

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import load_model, Sequential
from keras.optimizers import Adam


class RLAgent:
    def __init__(self, model_path=None, layer_sizes=None, memories=None, tensorboard_dir=None):
        self.state_size = 241 # Size of state vector
        self.action_size = 61 # Maximum number of actions 

        self.memory = self.load_memories() if memories else deque(maxlen=10_000)
        self.game_length = 0
        self.batch_size = 128

        self.gamma = 0.99 # 0.1**(1/25)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.99
        self.lr = 0.001

        if model_path:
            self.model = load_model(model_path)
            self.target_model = load_model(model_path)
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

        categorizer1 = Dense(layer_sizes[0], activation='relu', name='categorizer1')(state_input)
        categorizer2 = Dense(layer_sizes[1], activation='relu', name='categorizer2')(categorizer1)
        category = Dense(3, activation='softmax', name='category')(categorizer2)

        state_w_category = tf.keras.layers.concatenate([state_input, category])

        # Reuse via categorizer1(state_w_category)?
        specific1 = Dense(layer_sizes[2], activation='relu', name='specific1')(state_w_category)
        specific2 = Dense(layer_sizes[3], activation='relu', name='specific2')(specific1)
        move = Dense(self.action_size, activation='linear', name='move')(specific2)

        model = tf.keras.Model(inputs=state_input, outputs=move)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model
    
    def load_memories(self):
        print("Loading existing memories")
        import pickle
        with open("/workspace/RL/memories.pkl", 'rb') as f:
            flattened_memories = pickle.load(f)
        loaded_memories = [mem for mem in flattened_memories]
        return deque(loaded_memories, maxlen=10_000)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_learning_rate(self, avg_game_length):
        # 0.01 schedule: y=\frac{\left(x-12\right)^{3.2}}{25000000} 25_000_000
        new_lr = max(min((avg_game_length-12)**3.6 / 1_000_000_000, 0.01), 0.0001) # y=\frac{\left(x-12\right)^{3.6}}{1000000000} 1_000_000_000
        self.model.optimizer.learning_rate.assign(new_lr)
        self.lr = new_lr

        if self.tensorboard:
            with self.tensorboard.as_default():
                tf.summary.scalar('Training Metrics/learning_rate', self.lr, step=self.step)

    def get_predictions(self, state, legal_mask):
        state = tf.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(self.action_size)  # Exploration
        else:
            act_values = self.model.predict(state, verbose=0)[0]  # All actions

        # Illegal move filter
        act_values = np.where(legal_mask, act_values, -np.inf) # Compatible with graph?

        return act_values

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
                tf.summary.scalar('Training Metrics/batch_loss', history.history['loss'][0], step=step)
                tf.summary.scalar('Training Metrics/avg_reward', tf.reduce_mean(rewards), step=step)
                legal_qs = tf.where(tf.math.is_finite(qs), qs, tf.zeros_like(qs))
                tf.summary.scalar('Training Metrics/avg_q', tf.reduce_mean(legal_qs), step=step)
                tf.summary.histogram('Training Metrics/action_hist', actions, step=step)

                # Q-values over time
                for action in range(self.action_size):
                    average_qs = np.mean(legal_qs[:, action], axis=0)
                    tf.summary.scalar(f"action_qs/action_{action}", average_qs, step=step)
                
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
        self._batch_train(batch)

    def save_model(self, model_path):
        if not model_path:
            model_path = "/workspace/RL/trained_agents/model.keras"
        self.model.save(model_path)
        print(f"Saved the model at {model_path}")