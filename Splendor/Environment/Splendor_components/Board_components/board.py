# Splendor/Environment/Splendor_components/Board_components/board.py

from .deck import Deck


class Board:
    def __init__(self, num_players):
        # Gems
        gems = 7 - (5-num_players)
        self.gems = {'white': gems, 'blue': gems, 'green': gems, 'red': gems, 'black': gems, 'gold': 5}

        # Decks
        self.nobles = Deck('nobles')
        self.tier3 = Deck('tier3')
        self.tier2 = Deck('tier2')
        self.tier1 = Deck('tier1')

        self.taken_cards = 0

        self.deck_mapping = {
            'tier1': self.tier1,
            'tier2': self.tier2,
            'tier3': self.tier3
        }
        
        self.cards = {
            'nobles': [self.nobles.draw() for _ in range(num_players+1)], 
            'tier1': [self.tier1.draw() for _ in range(4)],
            'tier2': [self.tier2.draw() for _ in range(4)],
            'tier3': [self.tier3.draw() for _ in range(4)]
        }
    
    def get_card_by_id(self, card_id):
        for tier in ['tier1', 'tier2', 'tier3']:
            for card in self.cards[tier]:
                if card.id == card_id:
                    return card
                
    def change_gems(self, gems_to_change):
        for gem, amount in gems_to_change.items():
            self.gems[gem] += amount

    def take_card(self, tier, position):
        card = self.cards[tier][position]
        self.cards[tier].remove(position)
        new_card = self.deck_mapping[tier].draw()
        if new_card:
            self.cards[tier][position] = new_card
        return card
    
    def reserve(self, tier, position):
        # Give gold if available
        gold = 0
        if self.gems['gold']:
            self.gems['gold'] -= 1
            gold = 1

        # Replace card
        card = self.take_card(tier, position)
        return card, gold
    
    def reserve_from_deck(self, tier):
        # Give gold if available
        gold = 0
        if self.gems['gold']:
            self.gems['gold'] -= 1
            gold = 1

        # Remove card
        return self.deck_mapping[tier].draw(), gold
    
    def get_state(self):
        return {
            'gems': self.gems, 
            'cards': {
                tier: [card.id for card in cards]
                for tier, cards in self.cards.items()
            }
        }
        
    def to_vector(self):
        state_vector = list(self.gems.values()) # length 6

        for tier in ['tier1', 'tier2', 'tier3']: # length 11*3
            tier_vector = []
            for card in self.cards[tier]:
                tier_vector.extend(card.vector)
            tier_vector += [0] * (11 * (4 - len(self.cards[tier])))
            state_vector.extend(tier_vector)
            
        nobles_vector = []
        for card in self.cards['nobles']: # length 6*3
            nobles_vector += [card.points] + list(card.cost.values())
        nobles_vector += [0] * (6 * (3 - len(self.cards['nobles'])))
        state_vector.extend(nobles_vector)

        return state_vector # length 57


if __name__ == "__main__":
    import sys

    sys.path.append("C:/Users/Public/Documents/Python_Files/Splendor")

    from Environment.Splendor_components.Board_components.deck import Deck # type: ignore

    b1 = Board(2)