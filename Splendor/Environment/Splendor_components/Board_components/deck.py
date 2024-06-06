# Splendor/Environment/Splendor_components/Board_components/deck.py

import random
import pandas as pd


class Card:
    def __init__(self, id, tier, gem, points, cost):
        self.id = id
        self.tier = tier
        self.gem = gem
        self.points = points
        self.cost = cost  # Dictionary of gem costs
        self.vector = self.to_vector()  # Vector representation

    def gem_to_one_hot(self, gem):
        gem_dict = {
            'white': [1, 0, 0, 0, 0],
            'blue': [0, 1, 0, 0, 0],
            'green': [0, 0, 1, 0, 0],
            'red': [0, 0, 0, 1, 0],
            'black': [0, 0, 0, 0, 1]}
        return gem_dict[gem]
    
    def to_vector(self):
        gem_one_hot = self.gem_to_one_hot(self.gem)
        return gem_one_hot + [self.points, 
                              self.cost['white'], self.cost['blue'], self.cost['green'], self.cost['red'], self.cost['black']]
    
    def to_dict(self):
        return {
            'id': self.id,
            'tier': self.tier,
            'gem': self.gem,
            'points': self.points,
            'cost': self.cost
        }

    
class Deck:
    def __init__(self, tier):
        self.tier = tier
        self.cards = self.load_deck()

    def load_deck(self):
        path = 'C:/Users/Public/Documents/Python_Files/Splendor/Environment/Splendor_components/Board_components/Splendor_Cards.xlsx'
        deck = pd.read_excel(path, sheet_name=self.tier)

        cards = []
        for _, row in deck.iterrows():
            id, gem, points, white, blue, green, red, black = row
            cards.append(Card(id = id, tier = self.tier, gem = gem, points = points, 
                              cost = {'white': white, 'blue': blue, 'green': green, 'red': red, 'black': black}))
            
        random.shuffle(cards)
        
        return cards

    def draw(self):
        return self.cards.pop() if self.cards else None
    
    def __len__(self):
        return len(self.cards)
    
if __name__ == "__main__":
    import sys

    sys.path.append("C:/Users/Public/Documents/Python_Files/Splendor")

    tier1 = Deck('nobles')