# Splendor/Environment/Splendor_components/player.py


from collections import defaultdict
from itertools import combinations
import numpy as np
from RL import RLAgent # type: ignore


class Player:
    def __init__(self, name, strategy, strategy_strength, layer_sizes, model_path=None):
        self.name: str = name
        self.gems: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0, 'gold': 0}
        self.gem_to_index = {'white': 0, 'blue': 1, 'green': 2, 'red': 3, 'black': 4, 'gold': 5}
        self.index_to_gem = {0: 'white', 1: 'blue', 2: 'green', 3: 'red', 4: 'black', 5: 'gold'}
        self.cards: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0}
        self.reserved_cards: list = []
        self.points: int = 0

        self.cards_state = {'tier1': [], 'tier2': [], 'tier3': []}
        self.cards_state = {gem: {'tier1': [], 'tier2': [], 'tier3': []} for gem in self.cards}
        self.rl_model = RLAgent(layer_sizes, model_path)
        self.local_memory = []
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

    def take_tokens_loop(self, game_state):
        self_gems = {gem: amount for gem, amount in self.gems.items() if gem != 'gold' and amount > 0}
        total_gems = sum(self_gems.values())
        chosen_move = [0] * 6

        takes_remaining = 3
        required_discards = max(0, total_gems + sum(chosen_move.values()) - 10)

        legal_mask = [0] * 61
        while takes_remaining:
            # Discard a gem to help with the logic
            if total_gems == 10 and takes_remaining % 2:
                for i, count, in enumerate(self_gems):
                    if count > 0:
                        legal_mask[i+10] == 1
                rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
                move_index = np.argmax(rl_moves)
                self.local_memory.append(game_state, move_index)

                required_discards -= 1
                game_state[move_index] += 1 # Update board
                game_state[move_index] -= 1 # Update player
                chosen_move[move_index] -= 1 # Update for apply_move()?

            # Take a gem
            rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
            move_index = np.argmax(rl_moves)
            self.local_memory.append((game_state, move_index))

            chosen_move[move_index-10] += 1
            game_state[move_index] += 1 # Calling this at the end in case we still need it (instead of at beginning)
        
        while required_discards:
            for i, count, in enumerate(self_gems): # Using cards because it doesn't have gold
                if count:
                    legal_mask[i+10] = 1

            rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
            move_index = np.argmax(rl_moves)
            self.local_memory.append(game_state, move_index)

            chosen_move[move_index-60] -= 1
            game_state[move_index] -= 1 # Calling this at the end in case we still need it (instead of at beginning)

        return chosen_move

    def buy_with_gold_loop(self, game_state, cost):
        # Need to confirm if its ok to be modifying game_state locally - that we don't need it elsewhere.
        # For now, we'll just use the same five action slots as discarding, since neither are chosen moves but are just required for certain moves
        # This is nice because these nodes shouldn't have been punished and remain in a similar reward space
        legal_mask = [0] * 61 # Action vector size
        remaining_cost = sum(cost.values())
        while remaining_cost:
            # Enable spending gold as a legal move
            for index, cost in enumerate(cost.values()): # gem is not cost.values but which did i mean
                if self.gems[self.index_to_gem(index)] > cost:
                    legal_mask[index+10] = 1
                if self.gems['gold'] >= cost:
                    legal_mask[60] = 1 # is this discarding or taking gold though need to make sure we don't need 2

            rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
            move_index = np.argmax(rl_moves)
            self.local_memory.append(game_state, move_index)

            chosen_move[move_index] -= 1 # Need to make sure all these negatives/positives work correctly with change_gems

            cost[move_index] -= 1
            # Predict gems to discard and have a discard gold move which can only be called once legal to purchase with only gold from there on out.  Then just subtract remaining cost with gold
            
            # Calling this at the end in case we still need it (instead of at beginning)
            if move_index == 60:
                i = 6
            game_state[i] += 1 # Board gains the gem
            game_state[i+some_other_index] -= 1 # Player spends the gem

            return chosen_move

    def get_legal_moves(self, board):
        legal_moves = []

        # Take gems
        for gem, amount in board.gems.items():
            if amount > 0:
                legal_moves.append('take', (gem, 1))
                if amount >= 4:
                    legal_moves.append('take', (gem, 2))

        # Reserve card
        if len(self.reserved_cards) < 3:
            for tier in ['tier1', 'tier2', 'tier3']:
                for position, card in enumerate(board.cards[tier]):
                    if card:
                        legal_moves.append('reserve', (tier, position))
                    if board.deck_mapping[tier]:
                        legal_moves.append(('reserve_top', (tier, 4))) # Setting position to 4 to be used later

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
                legal_moves.append(('buy_reserved', (4, position)))
                if gold_combinations:
                    for comb in combinations(gold_combinations, min(gold_needed, len(gold_combinations))):
                        comb_dict = {gem: gold_amount for gem, gold_amount in comb}
                        total_cost = {gem: card.cost[gem] for gem in card.cost}
                        for gem, amount in comb_dict.items():
                            total_cost[gem] = total_cost.get(gem, 0) - amount
                        total_cost['gold'] = gold_needed
                        legal_moves.append(('buy_reserved_with_gold', (4, (position, total_cost))))

        return legal_moves
    
    def legal_to_vector(self, legal_moves):
        moves_vector = [0] * 60
        for move, details in legal_moves:
            tier, position = details
            match move:
                case 'take':
                    gem, amount = details # Overriding tier and position
                    if amount == 1:
                        moves_vector[self.gem_to_index[gem]] = 1
                    elif amount == 2:
                        moves_vector[self.gem_to_index[gem]*2] = 1
                    elif amount == -1:
                        pass # Discards are only handled by take_tokens_loop
                case 'buy' | 'buy_reserved':
                    moves_vector[15 + 5*tier + position] = 1
                case 'buy with gold' | 'buy reserved with gold':
                    moves_vector[29 + 5*tier + position] = 1
                case 'reserve' | 'reserve top':
                    moves_vector[44 + 5*tier + position] = 1
        return moves_vector
    
    def vector_to_details(self, move_index):
        # Maybe we can include this as part of the process if we're sure it doesn't need called a second time
        tier = move_index % 15 // 5
        position = tier % 4

        if move_index < 15:  # Take (includes discarding a gem)
            if move_index < 5:
                move = ('take', (self.index_to_gem[move_index], 1))
            elif move_index < 10:
                move = ('take', (self.index_to_gem[move_index], 2))
            else:
                move = ('take', (self.index_to_gem[move_index], -1))

        elif move_index < 45: # Buy
            if move_index < 27:
                move = ('buy', (tier, position))
            elif move_index < 30:
                move = ('buy reserved', (tier, position))
            elif move_index < 42:
                move = ('buy with gold', (tier, position))
            else:
                move = ('buy reserved with gold', (tier, position))

        elif move_index < 60: # Reserve
            if move_index < 57:
                move = ('reserve', (tier, position))
            elif move_index < 60:
                move = ('reserve top', (tier, position))
        
        return move
    
    def choose_move(self, board, game_state):
        legal_moves = self.get_legal_moves(board)
        legal_mask = self.legal_to_vector(legal_moves)
        rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
        #strategic_moves = self.strategy.strategize(game_state, rl_moves, self.strategy_strength)
        self.move_index = np.argmax(rl_moves)

        # If the move takes tokens, make a secondary prediction
        if self.move_index < 10:
            # We want to remember during this process, but we need to remember each sub-move within it
            chosen_move = self.take_tokens_loop(game_state)
        else:
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
            list(self.gems.values()) + # length 6
            list(self.cards.values()) + # length 5
            reserved_cards_vector + # length 11*3 = 33
            [self.points] # length 1
        ) # length 45
    

if __name__ == "__main__":
    import sys
    sys.path.append("C:/Users/Public/Documents/Python_Files/Splendor")
    from Environment.Splendor_components.Player_components.strategy import BestStrategy # type: ignore
    from Environment.Splendor_components.Board_components.board import Board # type: ignore
    from RL.model import RLAgent # type: ignore

    bob = Player('Bob', BestStrategy(),  1)
    board = Board(2)
    moves = bob.get_legal_moves(board)