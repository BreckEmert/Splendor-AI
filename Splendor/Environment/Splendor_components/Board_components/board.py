# Splendor/Environment/Splendor_components/Board_components/board.py

import numpy as np

from .deck import Deck


class Board:
    def __init__(self):
        # Gems - [white, blue, green, red, black, gold]
        self.gems = np.array([4, 4, 4, 4, 4, 5], dtype=int)

        # Decks
        self.tier1 = Deck(0)
        self.tier2 = Deck(1)
        self.tier3 = Deck(2)
        self.noble = Deck('Noble')

        self.decks = [
            self.tier1, 
            self.tier2, 
            self.tier3
        ]
        
        # Draw cards for the game
        self.cards = [
            [self.tier1.draw() for _ in range(4)],
            [self.tier2.draw() for _ in range(4)],
            [self.tier3.draw() for _ in range(4)]
        ]

        self.nobles = [self.noble.draw() for _ in range(3)]
                
    def take_gems(self, taken_gems): 
        self.gems -= np.pad(taken_gems, (0, 6-len(taken_gems)))

    def return_gems(self, returned_gems):
        self.gems += np.pad(returned_gems, (0, 6-len(returned_gems)))

    def take_card(self, tier, position):
        card = self.cards[tier][position]
        new_card = self.decks[tier].draw()
        self.cards[tier][position] = new_card if new_card else None
        return card
    
    def reserve(self, tier, position):
        """Gold is not subtracted by this function, as 
        player could potentially have discards.
        """
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
        card = self.decks[tier].draw()
        return card, gold
        
    def to_state(self, effective_gems):
        # All shop cards, size 3*4*11 = 132
        i = 0
        tier_vector = np.zeros(132, dtype=float)
        for tier in self.cards:  # 3 tiers
            for card in tier:    # 4 cards per tier
                if card:         # reward1hot, points, cost1hot = 11
                    tier_vector[i:i+11] = card.to_vector(effective_gems)
                i += 11

        # All nobles, size 3*6 = 18
        i = 0
        nobles_vector = np.zeros(18, dtype=float)
        for card in self.nobles:
            if card:
                nobles_vector[i:i+6] = card.to_vector(effective_gems)
            i += 6

        # Final state vector for the board
        state_vector = np.concatenate((
            self.gems[:5] / 4,      # 5
            [self.gems[5] / 5],     # 1
            [self.gems.sum() / 10], # 1
            tier_vector,            # 132
            nobles_vector           # 18
        ))
        return state_vector  # 157
