# Splendor/Environment/Splendor_components/Board_components/board.py

import numpy as np

from .deck import Deck


class Board:
    def __init__(self):
        # Gems
        self.gems = np.array([6, 6, 6, 6, 6, 5], dtype=int) # [white, blue, green, red, black, gold]

        # Decks
        self.tier1 = Deck(0)
        self.tier2 = Deck(1)
        self.tier3 = Deck(2)
        self.nobles = Deck(3)

        self.deck_mapping = {
            0: self.tier1, 
            1: self.tier2, 
            2: self.tier3, 
            3: self.nobles
        }
        
        self.cards = [
            [self.tier1.draw() for _ in range(4)],
            [self.tier2.draw() for _ in range(4)],
            [self.tier3.draw() for _ in range(4)], 
            [self.nobles.draw() for _ in range(3)]
        ]
                
    def take_or_return_gems(self, gems_to_change):
        self.gems -= np.pad(gems_to_change, (0, 6-len(gems_to_change)))
        assert np.all(self.gems >= 0), f"Illegal board gems {self.gems}"

    def take_card(self, tier, position):
        card = self.cards[tier][position]
        new_card = self.deck_mapping[tier].draw()
        self.cards[tier][position] = new_card if new_card else None
        return card
    
    def reserve(self, tier, position):
        # Give gold if available
        gold = 0
        if self.gems[5]:
            self.gems[5] -= 1
            gold = 1

        # Replace card
        card = self.take_card(tier, position)
        return card, gold
    
    def reserve_from_deck(self, tier):
        # Give gold if available
        gold = 0
        if self.gems[5]:
            self.gems[5] -= 1
            gold = 1

        # Remove card
        return self.deck_mapping[tier].draw(), gold
    
    def get_state(self):
        card_dict = {
            f"tier{tier_index+1}": [card.id if card else None for card in tier] 
            for tier_index, tier in enumerate(self.cards[:3])
        }
        card_dict['nobles'] = [card.id if card else None for card in self.cards[3]]
        return {'gems': self.gems.tolist(), 'cards': card_dict}
        
    def to_vector(self):
        tier_vector = [ # 11*4*3
            card.vector if card else np.zeros(11)
            for tier in self.cards[:3]
            for card in tier
        ]
        
        nobles_vector = [ # 6*3
            card.vector[5:] if card else np.zeros(6)
            for card in self.cards[3]
        ]

        state_vector = np.concatenate((*tier_vector, *nobles_vector)) # No longer including self.gems
        return state_vector # length 150, UPDATE STATE_OFFSET IF THIS CHANGES