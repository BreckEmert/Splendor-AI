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
        self.cost: np.ndarray = np.concatenate((cost, [0]))  # gem costs

    def gem_to_one_hot(self, index):
        one_hot = np.zeros(6, dtype=int)
        one_hot[index] = 1
        return one_hot

    def to_vector(self, effective_gems):
        # Subtracting helps the model learn how far it is from buying
        clipped_cost = np.maximum(self.cost - effective_gems, 0)[:5]
        return np.concatenate((self.gem_one_hot[:5], [self.points/15], clipped_cost/4))
    

class Noble:
    def __init__(self, id, tier, gem, points, cost):
        self.id: int = id
        self.points: int = points  # always 3
        self.cost: np.ndarray = np.concatenate((cost, [0]))  # visit gem requirement
    
    def to_vector(self, effective_gems) -> np.ndarray:
        # Subtracting helps the model learn how far it is from the Noble
        relative_cost = np.maximum(self.cost - effective_gems, 0)[:5]
        return np.concatenate(([self.points], relative_cost)) / 4.0


_PRELOADED_DECKS = None
def _preload_decks():
    # Get the excel file
    workspace_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(workspace_dir, "Splendor_cards_numeric.xlsx")

    # All deck names
    all_tiers = [0, 1, 2, 'Noble']

    # Read each tier's sheet into memory
    preloaded_decks = {}
    for tier in all_tiers:
        df = pd.read_excel(path, sheet_name=str(tier))
        if tier == 'Noble':
            cards = [
                Noble(id=row[0], tier=tier, gem=row[1], points=row[2], 
                    cost=[row[3], row[4], row[5], row[6], row[7]])
                for row in df.itertuples(index=False)
            ]
        else:
            cards = [
                Card(id=row[0], tier=tier, gem=row[1], points=row[2], 
                    cost=[row[3], row[4], row[5], row[6], row[7]])
                for row in df.itertuples(index=False)
            ]
        
        preloaded_decks[tier] = cards

    return preloaded_decks


class Deck:
    def __init__(self, tier):
        global _PRELOADED_DECKS
        if _PRELOADED_DECKS is None:
            _PRELOADED_DECKS = _preload_decks()

        self.tier = tier

        self.cards = list(_PRELOADED_DECKS[self.tier])
        random.shuffle(self.cards)  # (in-place)

    def draw(self):
        return self.cards.pop() if self.cards else None
