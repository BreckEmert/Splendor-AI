# Splendor/Environment/Splendor_components/player.py


from collections import defaultdict
from itertools import combinations
import numpy as np
from RL import RLAgent # type: ignore


class Player:
    def __init__(self, name, strategy, strategy_strength, layer_sizes, model_path=None):
        self.name: str = name
        self.gems: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0, 'gold': 0}
        self.cards: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0}
        self.reserved_cards: list = []
        self.points: int = 0

        self.cards_state = {'tier1': [], 'tier2': [], 'tier3': []}
        self.cards_state = {gem: {'tier1': [], 'tier2': [], 'tier3': []} for gem in self.cards}
        self.rl_model = RLAgent(layer_sizes, model_path)
        self.victor = False
        #self.strategy: strategy = strategy
        #self.strategy_strength: int = strategy_strength
    
    def change_gems(self, gems_to_change):
        for gem, amount in gems_to_change.items():
            self.gems[gem] -= amount

    def get_bought_card(self, card):
        self.cards[card.gem] += 1
        self.points += card.points
        self.cards_state[card.gem][card.tier].append(card.id)

    def get_legal_moves(self, board):
        legal_moves = []

        # Reserve card
        if len(self.reserved_cards) < 3:
            for tier in ['tier1', 'tier2', 'tier3']:
                for position, card in enumerate(board.cards[tier]):
                    if card:
                        # Also used as 'go for' moves
                        legal_moves.append('reserve', (tier, position))
                    if board.deck_mapping[tier]:
                        legal_moves.append(('reserve_top', (tier, None)))

        # Buy card
        for tier in ['tier1', 'tier2', 'tier3']:
            for position, card in enumerate(board.cards[tier]):
                if card:
                    can_afford = True
                    gold_needed = 0
                    gold_combinations = []

                for gem, amount in card.cost.items():
                    if self.gems[gem] < amount:
                        gold_needed += amount - self.gems[gem]
                        gold_combinations.append((gem, amount - self.gems[gem]))
                        if gold_needed > self.gems['gold']:
                            can_afford = False
                            break

                if can_afford:
                    legal_moves.append(('buy', (tier, position)))
                    if gold_combinations:
                        for comb in combinations(gold_combinations, min(gold_needed, len(gold_combinations))):
                            comb_dict = {gem: gold_amount for gem, gold_amount in comb}
                            total_cost = {gem: card.cost[gem] for gem in card.cost}
                            for gem, amount in comb_dict.items():
                                total_cost[gem] = total_cost.get(gem, 0) - amount
                            total_cost['gold'] = gold_needed
                            legal_moves.append(('buy_with_gold', ((tier, (position, total_cost)))))

        # Buy reserved card
        for position, card in enumerate(self.reserved_cards):
            # Go for moves
            legal_moves.append('go for', (None, position))

            can_afford = True
            gold_needed = 0
            gold_combinations = []

            for gem, amount in card.cost.items():
                if self.gems[gem] < amount:
                    gold_needed += amount - self.gems[gem]
                    gold_combinations.append((gem, amount - self.gems[gem]))
                    if gold_needed > self.gems['gold']:
                        can_afford = False
                        break

            if can_afford:
                legal_moves.append(('buy_reserved', card.id))
                if gold_combinations:
                    for comb in combinations(gold_combinations, min(gold_needed, len(gold_combinations))):
                        comb_dict = {gem: gold_amount for gem, gold_amount in comb}
                        total_cost = {gem: card.cost[gem] for gem in card.cost}
                        for gem, amount in comb_dict.items():
                            total_cost[gem] = total_cost.get(gem, 0) - amount
                        total_cost['gold'] = gold_needed
                        legal_moves.append(('buy_reserved_with_gold', (None, (position, total_cost))))

        return legal_moves
    
    def legal_to_vector(self, legal_moves):
        # 30 moves = 12+3 buy + 12+3 reserve
        # 303 moves = 90 buy + 90 buy reserved + 90 reserve + 3 reserve top
        moves_vector = [0] * 303
        for move, details in legal_moves:
            print(move, details)
            match move:
                case 'take':
                    for position, format_details in enumerate(format_vector):
                        if format_details == (move, details):
                            moves_vector[position] = 1
                            break
                case 'buy':
                    moves_vector[29 + details] = 1
                case 'buy with gold':
                    moves_vector[1]
                case 'buy_reserved':
                    moves_vector[119 + details] = 1
                case 'reserve':
                    moves_vector[209 + details] = 1
                case 'reserve_top':
                    moves_vector[299 + int(details[-1])] = 1 # Grabs n in 'tiern'
        print("-------------------------------")
        return moves_vector
    
    def vector_to_details(self, move_index):
        if move_index < 30:  # Go for card moves
            pass
        elif move_index < 120:  # Buy moves
            move = ('buy', move_index - 29) # Lowered to 29 because m_i=30 - 29 = 1
        elif move_index < 210:  # Buy reserved moves
            move = ('buy_reserved', move_index - 119)
        elif move_index < 300:  # Reserve moves
            move = ('reserve', move_index - 209)
        else:  # Reserve top moves
            move = ('reserve_top', 'tier' + str(move_index - 299))
        
        return move
    
    def choose_move(self, board, game_state):
        legal_moves = self.get_legal_moves(board)
        legal_mask = self.legal_to_vector(legal_moves)
        rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
        #strategic_moves = self.strategy.strategize(game_state, rl_moves, self.strategy_strength)
        self.move_index = np.argmax(rl_moves) # changing to self.move_index
        chosen_move = self.vector_to_details(self.move_index)
        return chosen_move
    
    def get_state(self):
        reserved_cards_state = {'tier1': [], 'tier2': [], 'tier3': []}
        for card in self.reserved_cards:
            reserved_cards_state[f'{card.tier}'].append(card.id)
        return {
            'gems': self.gems,
            'cards': self.cards_state,
            'reserved_cards': reserved_cards_state,
            'points': self.points
        }

    def to_vector(self):
        reserved_cards_vector = []
        for card in self.reserved_cards:
            reserved_cards_vector.extend(card.vector)
        reserved_cards_vector += [0] * (11 * (3-len(self.reserved_cards)))
        return (
            list(self.gems.values()) + 
            list(self.cards.values()) + 
            reserved_cards_vector + 
            [self.points])

if __name__ == "__main__":
    import sys
    sys.path.append("C:/Users/Public/Documents/Python_Files/Splendor")
    from Environment.Splendor_components.Player_components.strategy import BestStrategy # type: ignore
    from Environment.Splendor_components.Board_components.board import Board # type: ignore
    from RL.model import RLAgent # type: ignore

    bob = Player('Bob', BestStrategy(),  1)
    board = Board(2)
    moves = bob.get_legal_moves(board)