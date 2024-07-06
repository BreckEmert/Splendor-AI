# Splendor/RL/random_model.py

from collections import deque
import numpy as np


class RandomAgent:
    def __init__(self, memories_path=None):
        self.state_size = 241 # Size of state vector
        self.action_size = 61 # Maximum number of actions 

        self.memory = self.load_memories(memories_path)
    
    def load_memories(self, memories_path):
        if memories_path:
            import pickle
            with open(memories_path, 'rb') as f:
                flattened_memories = pickle.load(f)
            loaded_memories = [mem for mem in flattened_memories]
            print(f"Loading {len(loaded_memories)} memories")
        else:
            loaded_memories = []
        return deque(loaded_memories, maxlen=50_000)

    def get_predictions(self, state, legal_mask):
        return np.where(legal_mask, np.random.rand(self.action_size), -np.inf)

    def remember(self, memory, legal_mask):
        self.memory.append(memory)
        self.memory[-2].append(legal_mask)