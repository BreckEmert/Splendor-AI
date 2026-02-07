# Splendor/RL/random_model.py

import numpy as np
import os
import pickle
from collections import deque
from copy import deepcopy


class RandomAgent:
    def __init__(self, paths):
        self.paths = paths
        self.state_size = 326
        self.action_size = 141
        self.memory = self.load_memory()
        self.step = 0

    def reset(self):
        self.memory = self.load_memory()

    def load_memory(self):
        dummy_state = np.zeros(self.state_size, dtype=np.float32)
        dummy_mask = np.ones(self.action_size, dtype=bool)
        loaded_memory = [[dummy_state, 0, 0.0, dummy_state, dummy_mask, False]]
        return deque(loaded_memory, maxlen=50_000)

    def write_memory(self):
        saved_files_dir = self.paths['saved_files_dir']
        os.makedirs(saved_files_dir, exist_ok=True)
        memory_path = os.path.join(saved_files_dir, "random_memory.pkl")
        with open(memory_path, 'wb') as f:
            pickle.dump(list(self.memory), f)
        print(f"Wrote {len(self.memory)} memories to {memory_path}")

    def get_predictions(self, state, legal_mask):
        logits = np.random.rand(legal_mask.shape[0]).astype(np.float32)
        return np.where(legal_mask, logits, -np.inf)

    def remember(self, memory):
        self.memory.append(memory)

    def replay(self): 
        self.step += 1

    def save_model(self): 
        pass
