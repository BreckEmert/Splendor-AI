# Splendor/RL/random_model.py

import numpy as np
import os
import pickle
from collections import deque
from copy import deepcopy


class RandomAgent:
    def __init__(self):
        self.state_size = 241
        self.action_size = 61
        self.memory = self.load_memory()

    def load_memory(self):
        dummy_state = np.zeros(self.state_size, dtype=np.float32)
        dummy_mask = np.ones(self.action_size, dtype=bool)
        loaded_memory = [[dummy_state, 0, 0, dummy_state, 1, dummy_mask]]
        return deque(loaded_memory, maxlen=50_000)
    
    def write_memory(self, memory, base_dir, append_to_prev_mem):
        memory_path = os.path.join(base_dir, "RL", "random_memory.pkl")

        # Get the old memories if we don't want to overwrite them
        if append_to_prev_mem:
            with open(memory_path, 'rb') as f:
                existing_memory = pickle.load(f)
            print(f"Loaded {len(existing_memory)} existing memories.")
            existing_memory.extend(memory)
            memory = existing_memory

        # Write out the memories
        with open(memory_path, 'wb') as f:
            pickle.dump(memory, f)

        print(f"Wrote {len(memory)} memories to {memory_path}")
        print("Absolute memory path: ", os.path.abspath(memory_path))

    def get_predictions(self, state, legal_mask):
        return np.where(legal_mask, np.random.rand(self.action_size), -np.inf)

    def remember(self, memory, legal_mask):
        self.memory.append(deepcopy(memory))
        self.memory[-2].append(legal_mask.copy())