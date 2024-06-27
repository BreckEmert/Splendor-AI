# Splendor/RL/model.py

from collections import deque
from random import sample

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import load_model, Sequential
from keras.optimizers import Adam


class RLAgent:
    def __init__(self, layer_sizes, model_path=None):
        physical_devices = tf.config.list_physical_devices('GPU')
        print(tf.config.experimental.get_memory_growth(physical_devices[0]))
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU mem growth set")
        except:
            print("Failed to set GPU mem growth")
            pass

        self.state_size = 240 # Size of state vector
        self.action_size = 61 # Maximum number of actions 

        self.memory = deque(maxlen=10_000)
        self.batch_size = 128

        self.gamma = 0.99
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.99
        self.lr = 0.01

        self.layer_sizes = layer_sizes
        if model_path:
            self.model = load_model(model_path)
            self.target_model = load_model(model_path)
        else:
            print("Building a new model")
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.layer_sizes[0], input_dim=self.state_size, activation='relu'))
        for size in self.layer_sizes[1:]:
            model.add(Dense(size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_learning_rate(self, avg_game_length):
        new_lr = max(min(avg_game_length**2.14 / 1_000_000, 0.01), 0.0001) # y=\frac{x^{2.14}}{1000000}
        self.model.optimizer.learning_rate.assign(new_lr)
        self.lr = new_lr

    def get_predictions(self, state, legal_mask):
        state = tf.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(self.action_size)  # Exploration
        else:
            state = tf.convert_to_tensor(state, dtype=tf.float32) # Batch dimension
            act_values = self.model.predict(state, verbose=0)[0]  # All actions
            # if np.random.rand() < 0.0004:
            #     print(act_values[legal_mask==1])

        # Illegal move filter
        act_values = np.where(legal_mask, act_values, -np.inf) # Compatible with graph?
        return act_values

    def remember(self, state, action, reward, next_state, done):
        # print(type(state), type(action), type(reward), type(next_state), type(done))
        penalty = 0.05 # Incentivize quicker games
        # self.memory.append((tf.cast(state, dtype=tf.float32), action, reward-penalty, tf.cast(next_state, dtype=tf.float32), done))
        self.memory.append((state, action, reward-penalty, next_state, done))

    def _batch_train(self, batch):
        states = tf.convert_to_tensor([mem[0] for mem in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([mem[1] for mem in batch], dtype=tf.int32) # Int8 or 16?
        rewards = tf.convert_to_tensor([mem[2] for mem in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([mem[3] for mem in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([mem[4] for mem in batch], dtype=bool) # Int8 or 16?

        qs = self.model.predict(states, verbose=0)
        target_qs = qs.copy()
        next_qs = self.model.predict(next_states, verbose=0)
        next_targets = self.target_model.predict(next_states, verbose=0)

        # Set q-values
        for i in range(len(batch)):
            if dones[i]:
                target_qs[i][actions[i]] = rewards[i]
            else:
                best_next_q = tf.argmax(next_qs[i])
                target_qs[i][actions[i]] = rewards[i] + self.gamma*next_targets[i][best_next_q]
        # Fit
        self.model.fit(states, target_qs, epochs=1, verbose=0)

    def replay(self):
        # Training twice for now
        batch = sample(self.memory, self.batch_size)
        self._batch_train(batch)
        
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