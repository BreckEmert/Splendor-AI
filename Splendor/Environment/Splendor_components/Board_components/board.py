# Splendor/Environment/Splendor_components/Board_components/board.py

import numpy as np

from .deck import Deck


class Board:
    def __init__(self):
        # Gems
        self.gems = np.array([10, 10, 10, 10, 10, 5], dtype=int)  # [white, blue, green, red, black, gold]

        # Decks
        self.tier1 = Deck(0)
        self.tier2 = Deck(1)
        self.tier3 = Deck(2)
        self.nobles = Deck(3)

        self.deck_mapping = {
            0: self.tier1, 
            1: self.tier2, 
            2: self.tier3
        }
        
        self.cards = [
            [self.tier1.draw() for _ in range(4)],
            [self.tier2.draw() for _ in range(4)],
            [self.tier3.draw() for _ in range(4)]
        ]

        self.nobles = [self.nobles.draw() for _ in range(3)]
                
    def take_gems(self, taken_gems): 
        self.gems -= np.pad(taken_gems, (0, 6-len(taken_gems)))
        assert np.all(self.gems >= 0), f"Illegal board gems {self.gems}, {taken_gems}"

    def return_gems(self, returned_gems):
        self.gems += np.pad(returned_gems, (0, 6-len(returned_gems)))
        # Remind me to change this back down to 10 after the model works
        assert np.all(self.gems <= 10), f"Illegal board gems {self.gems}, {returned_gems}"

    def take_card(self, tier, position):
        card = self.cards[tier][position]
        new_card = self.deck_mapping[tier].draw()
        self.cards[tier][position] = new_card if new_card else None
        return card
    
    def reserve(self, tier, position):
        # Gold is not subtracted by this function, as player can discard
        gold = np.zeros(6, dtype=int)
        if self.gems[5]:
            gold[5] = 1
        
        # Replace card
        card = self.take_card(tier, position)
        return card, gold
    
    def reserve_from_deck(self, tier):
        # Gold is not subtracted by this function, as player can discard
        gold = np.zeros(6, dtype=int)
        if self.gems[5]:
            gold[5] = 1
        
        # Remove card
        card = self.deck_mapping[tier].draw()
        return card, gold
        
    def to_state_vector(self):
        tier_vector = [
            card.vector if card else np.zeros(11)  # reward1hot, points, cost1hot = 11
            for tier in self.cards  # 3 tiers
            for card in tier  # 4 cards per tier
        ]  # 11*4*3 = 132
        
        nobles_vector = [ # 6*3
            card.vector[5:] if card else np.zeros(6)
            for card in self.nobles
        ]

        state_vector = np.concatenate((self.gems, *tier_vector, *nobles_vector))
        # UPDATE: including self.gems again
        # No longer including self.gems
        # Need to fully decide on this.
        # assert len(state_vector) == 156, f"board vector is {len(state_vector)}"
        return state_vector  # length 156
