# Splendor/Environment/Splendor_components/Board_components/board.py

import numpy as np

from .deck import Deck


class Board:
    def __init__(self, num_players):
        # Gems
        gems = 7 - (5-num_players)
        self.gems = np.array([gems, gems, gems, gems, gems, 5], dtype=int) # [white, blue, green, red, black, gold]

        # Decks
        self.tier1 = Deck(0)
        self.tier2 = Deck(1)
        self.tier3 = Deck(2)
        self.nobles = Deck(3)

        self.taken_cards = 0

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
            [self.nobles.draw() for _ in range(num_players+1)]
        ]
    
    def get_card_by_id(self, card_id):
        for tier in self.cards:
            for card in tier:
                if card and card.id == card_id:
                    return card
                
    def take_or_return_gems(self, gems_to_change):
        self.gems -= np.pad(gems_to_change, (0, 6-len(gems_to_change)))

    def take_card(self, tier, position):
        card = self.cards[tier][position]
        new_card = self.deck_mapping[tier].draw()
        if new_card:
            self.cards[tier][position] = new_card
        else:
            self.cards[tier][position] = None # Placeholder
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
        return {
            'gems': self.gems.tolist(), 
            'cards': [[card.id if card is not None else None for card in tier] for tier in self.cards]
        }
        
    def to_vector(self):
        state_vector = self.gems.copy() # length 6

        tier_vector = []
        for tier in self.cards[:3]: # length 11*3
            for card in tier:
                if card:
                    tier_vector.extend(card.vector)
                else:
                    tier_vector.extend([0] * 11)
            
        nobles_vector = []
        for card in self.cards[3]: # length 6*3
            if card:
                nobles_vector.extend(card.vector[5:]) # Don't need the gem reward
            else:
                nobles_vector.extend([0] * 6)

        state_vector = np.concatenate((self.gems, tier_vector, nobles_vector))

        assert len(state_vector) == 156, f"Board state vector is not 156, but {len(state_vector)}"
        return state_vector # length 156