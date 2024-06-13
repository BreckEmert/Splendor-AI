# Splendor/Environment/Splendor_components/Board_components/deck.py

import numpy as np
import pandas as pd
import random


class Card:
    def __init__(self, id, tier, gem, points, cost):
        self.id: int = id
        self.tier: int = tier
        self.gem: int = gem
        self.points: int = points
        self.cost: np.ndarray = np.array(cost, dtype=int)  # List of gem costs
        self.vector: np.ndarray = self.to_vector()  # Vector representation

    def gem_to_one_hot(self, index):
        one_hot = np.zeros(5, dtype=int)
        one_hot[index] = 1
        return one_hot
    
    def to_vector(self):
        gem_one_hot = self.gem_to_one_hot(self.gem)
        return np.concatenate((gem_one_hot, [self.points], self.cost))

    
class Deck:
    def __init__(self, tier):
        self.tier = tier
        self.cards = self.load_deck()

    def load_deck(self):
        path = 'C:/Users/Public/Documents/Python_Files/Splendor/Environment/Splendor_components/Board_components/Splendor_cards_numeric.xlsx'
        deck = pd.read_excel(path, sheet_name=self.tier)

        cards = []
        for _, row in deck.iterrows():
            id, gem, points, white, blue, green, red, black = row
            cost = [white, blue, green, red, black]
            cards.append(Card(id=id, tier=self.tier, gem=gem, points=points, cost=cost))
            
        random.shuffle(cards)
        
        return cards

    def draw(self):
        return self.cards.pop() if self.cards else None
    

if __name__ == "__main__":
    import sys

    sys.path.append("C:/Users/Public/Documents/Python_Files/Splendor")

    tier1 = Deck(0)