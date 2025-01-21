# Splendor/RL/random_model.py

import numpy as np
import os
import pickle
from collections import deque
from copy import deepcopy


class RandomAgent:
    def __init__(self, paths):
        self.paths = paths
        self.state_size = 242
        self.action_size = 140
        # self.memory = self.load_memory()

    def reset(self):
        self.memory = self.load_memory()

    def load_memory(self):
        dummy_state = np.zeros(self.state_size, dtype=np.float32)
        dummy_mask = np.ones(self.action_size, dtype=bool)
        loaded_memory = [[dummy_state, 0, 0, dummy_state, 1, dummy_mask]]
        return deque(loaded_memory, maxlen=50_000)
    
    def write_memory(self, memory):
        saved_files_dir = self.paths['saved_files_dir']
        memory_path = os.path.join(saved_files_dir, "random_memory.pkl")

        # Write out the memories
        with open(memory_path, 'wb') as f:
            pickle.dump(memory, f)

        print(f"Wrote {len(memory)} memories to {memory_path}")

    def get_predictions(self, state, legal_mask):
        return np.where(legal_mask, np.random.rand(self.action_size), -np.inf)

    def remember(self, memory, legal_mask) -> None:
        self.memory.append(deepcopy(memory))
        self.memory[-2].append(legal_mask.copy())
