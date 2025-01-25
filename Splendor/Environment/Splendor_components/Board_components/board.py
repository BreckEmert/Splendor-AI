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

    @property
    def state_vector(self):
        return self._state_vector

    def take_gems(self, taken_gems): 
        self.gems -= taken_gems

    def return_gems(self, returned_gems):
        self.gems += returned_gems

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
        state_vector = np.zeros(157, dtype=np.float32)

        # Gems (6+1 = 7)
        state_vector[:6] = self.gems / 4.0
        state_vector[5] /= 1.25  # Normalize to 5
        state_vector[6] = self.gems.sum() / 10.0

        # Shop cards (3*4*11 = 132)
        start = 7
        for tier in self.cards:  # 3 tiers
            for card in tier:    # 4 cards per tier
                if card:         # reward1hot, points, cost1hot = 11
                    card_vector = card.to_vector(effective_gems)
                    state_vector[start:start+11] = card_vector
                start += 11

        # Nobles (3*6 = 18)
        start = 139
        for card in self.nobles:
            if card:
                card_vector = card.to_vector(effective_gems)
                state_vector[start:start+6] = card_vector
            start += 6

        return state_vector  # 157
