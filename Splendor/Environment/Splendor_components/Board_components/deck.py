# Splendor/Environment/Splendor_components/Board_components/deck.py
import os
import random

import numpy as np
import pandas as pd


class Card:
    def __init__(self, id, tier, gem, points, cost):
        self.id: int = id
        self.tier: int = tier
        self.gem: int = gem
        self.gem_one_hot: np.ndarray = self.gem_to_one_hot(gem)
        self.points: float = points
        self.cost: np.ndarray = np.array(cost, dtype=int)  # List of gem costs

    def gem_to_one_hot(self, index):
        one_hot = np.zeros(5, dtype=int)
        one_hot[index] = 1
        return one_hot

    def to_vector(self, effective_gems):
        clipped_cost = np.maximum(self.cost - effective_gems, 0)
        return np.concatenate((self.gem_one_hot, [self.points/15], clipped_cost/4))
    
class Noble:
    def __init__(self, id, tier, gem, points, cost):
        self.id: int = id
        self.points: int = points  # always 3
        self.cost: np.ndarray = np.array(cost, dtype=int)
    
    def to_vector(self, effective_gems) -> np.ndarray:
        # Helps the model learn how far it is from the Noble
        relative_cost = np.maximum(self.cost - effective_gems, 0)
        return np.concatenate(([self.points], relative_cost)) / 4.0

class Deck:
    def __init__(self, tier):
        self.tier = tier
        self.cards = self.load_deck()

    def load_deck(self):
        workspace_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(workspace_dir, "Splendor_cards_numeric.xlsx")

        # path = '/workspace/Environment/Splendor_components/Board_components/Splendor_cards_numeric.xlsx'
        deck = pd.read_excel(path, sheet_name=self.tier)

        if self.tier == 'Noble':
            cards = [
                Noble(id=row[0], tier=self.tier, gem=row[1], points=row[2], 
                    cost=[row[3], row[4], row[5], row[6], row[7]])
                for row in deck.itertuples(index=False)
            ]
        else:
            cards = [
                Card(id=row[0], tier=self.tier, gem=row[1], points=row[2], 
                    cost=[row[3], row[4], row[5], row[6], row[7]])
                for row in deck.itertuples(index=False)
            ]
        
        random.shuffle(cards)  # in-place
        
        return cards

    def draw(self):
        return self.cards.pop() if self.cards else None
