# Splendor/RL/model.py

import numpy as np
import os
from collections import deque
from keras.models import load_model

class RLAgent:
    def __init__(self, layer_sizes, model_path=None):
        self.state_size = 247 # Size of state vector
        self.action_size = 303 # Maximum number of actions
        self.memory = deque(maxlen=500)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.layer_sizes = layer_sizes
        if model_path:
            self.model = load_model(model_path)
        else:
            print("Building a new model")
            self.model = self._build_model()

    def _build_model(self):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam

        model = Sequential()
        model.add(Dense(self.layer_sizes[0], input_dim=self.state_size, activation='relu'))
        for size in self.layer_sizes[1:]:
            model.add(Dense(size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def get_predictions(self, state, legal_mask):
        state = np.reshape(state, [1, self.state_size]) # In case of only 1 state
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(self.action_size)  # Exploration
        else:
            act_values = self.model.predict(state, verbose=0)[0]  # All actions

        # Filter out illegal moves
        act_values = [act_values[i] if legal_mask[i] == 1 else -np.inf for i in range(len(act_values))]
        return act_values

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = np.random.choice(len(self.memory), len(self.memory), replace=False)
        for i in minibatch:
            state, action, reward, next_state, done = self.memory[i]
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
                target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def train(self, state, action, reward, next_state, done):
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def save_model(self, model_dir, player_name):
        layer_sizes_str = '_'.join(map(str, self.layer_sizes))
        model_path = os.path.join(model_dir, f"{player_name}_{layer_sizes_str}")
        os.makedirs(model_path, exist_ok=True)
        self.model.save(os.path.join(model_path, 'model.keras'))

    def load_model(self, model_path):
        return load_model(model_path)