# Splendor/RL/model.py

import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam


class RLAgent:
    def __init__(self, layer_sizes, model_path=None):
        self.state_size = 240 # Size of state vector
        self.action_size = 61 # Maximum number of actions 

        self.state_memory = np.empty((0, self.state_size), dtype=float)
        self.action_memory = np.empty((0, ), dtype=int)

        self.epsilon = 1.0  # exploration rate
        self.lr = 0.01

        self.layer_sizes = layer_sizes
        if model_path:
            self.model = load_model(model_path)
        else:
            print("Building a new model")
            self.model = self._build_model()

    def _build_model(self):
        from keras.models import Sequential
        from keras.layers import Dense

        model = Sequential()
        model.add(Dense(self.layer_sizes[0], input_dim=self.state_size, activation='selu'))
        for size in self.layer_sizes[1:]:
            model.add(Dense(size, activation='selu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.lr))
        return model

    def update_parameters(self, epsilon, lr):
        self.epsilon = epsilon
        self.lr = lr
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.lr))
        
    def get_predictions(self, state, legal_mask):
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(self.action_size)  # Exploration
        else:
            state = np.expand_dims(state, axis=0) # Batch dimension
            act_values = self.model.predict(state, verbose=0)[0]  # All actions
            # if np.random.rand() < 0.0004:
            #     print(act_values[legal_mask==1])

        # Illegal move filter
        act_values = np.where(legal_mask, act_values, -np.inf)
        return act_values

    def remember(self, state, action):
        self.state_memory = np.vstack([self.state_memory, state])
        self.action_memory = np.append(self.action_memory, action)

    def train_batch(self, states, actions):
        example_index = np.random.randint(states.shape[0])
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = tf.keras.losses.categorical_crossentropy(actions, predictions)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_model(self, model_path):
        if not model_path:
            model_path = "/workspace/RL/trained_agents/model.keras"
        self.model.save(model_path)
        print(f"Saved the model at {model_path}")

    def load_model(self, model_path):
        return load_model(model_path)
    
if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("cuDNN version:")
    os.system("cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2")
