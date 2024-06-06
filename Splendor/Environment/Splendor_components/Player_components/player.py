# Splendor/Environment/Splendor_components/player.py


from itertools import combinations
import numpy as np
from RL import RLAgent # type: ignore


class Player:
    def __init__(self, name, strategy, strategy_strength):
        self.name: str = name
        self.gems: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0, 'gold': 0}
        self.cards: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0}
        self.reserved_cards: list = []
        self.points: int = 0

        self.cards_state = {'tier1': [], 'tier2': [], 'tier3': []}
        self.cards_state = {gem: {'tier1': [], 'tier2': [], 'tier3': []} for gem in self.cards}
        self.rl_model = RLAgent()
        self.victor = False
        #self.strategy: strategy = strategy
        #self.strategy_strength: int = strategy_strength
    
    def change_gems(self, gems_to_change):
        for gem, amount in gems_to_change.items():
            self.gems[gem] += amount

    def get_bought_card(self, card):
        self.cards[card.gem] += 1
        self.points += card.points
        self.cards_state[card.gem][card.tier].append(card.id)

    def reserve_card(self, card):
        self.reserved_cards.append(card)

    def get_legal_moves(self, board):
            non_zero_gems = [gem for gem in board.gems if gem != 'gold' and board.gems[gem] > 0]
            len_nzg = len(non_zero_gems)
            total_gems = sum(self.gems.values())
            take_2_count = take_3_count = 0
            take_1 = take_2 = take_2_diff = take_3 = []

            # Take 3 different gems
            for i in range(len_nzg):
                for j in range(i + 1, len_nzg):
                    for k in range(j + 1, len_nzg):
                        take_3.append(('take', {non_zero_gems[i]: -1, non_zero_gems[j]: -1, non_zero_gems[k]: -1}))
            legal_moves = self.handle_discards(take_3, total_gems, max_gems=10)

            # Take 2 of the same gem if at least 4 available
            for gem, count in board.gems.items():
                if gem != 'gold' and count >= 4:
                    take_2.append(('take', {gem: -2}))
            legal_moves += self.handle_discards(take_2, total_gems, max_gems=10)

            # Take 2 different gems if no legal take 3s
            if take_3_count == 0:
                for i in range(len_nzg):
                    for j in range(i + 1, len_nzg):
                        take_2_diff.append(('take', {non_zero_gems[i]: -1, non_zero_gems[j]: -1}))
                        take_2_count += 1
            legal_moves += self.handle_discards(take_2_diff, total_gems, max_gems=10)

            # Take 1 gem if no legal takes
            if take_2_count == 0:
                for gem, count in board.gems.items():
                    if gem != 'gold' and count > 0:
                        take_1.append(('take', {gem: -1}))
            legal_moves += self.handle_discards(take_1, total_gems, max_gems=10)
            
            # Reserve card
            if len(self.reserved_cards) < 3:
                for tier in ['tier1', 'tier2', 'tier3']:
                    for card in board.cards[tier]:
                        legal_moves.append(('reserve', card.id))
                    legal_moves.append(('reserve_top', tier)) # Reserve unknown top of decks

            # Buy card
            for tier in ['tier1', 'tier2', 'tier3']:
                for card in board.cards[tier]:
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
                        legal_moves.append(('buy', card.id))
                        if gold_combinations:
                            for comb in combinations(gold_combinations, min(gold_needed, len(gold_combinations))):
                                comb_dict = {gem: gold_amount for gem, gold_amount in comb}
                                total_cost = {gem: card.cost[gem] for gem in card.cost}
                                for gem, amount in comb_dict.items():
                                    total_cost[gem] = total_cost.get(gem, 0) - amount
                                total_cost['gold'] = gold_needed
                                legal_moves.append(('buy_with_gold', {'card_id': card.id, 'cost': total_cost}))
            
            # Buy reserved card
            for card in self.reserved_cards:
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
                            legal_moves.append(('buy_reserved_with_gold', {'card_id': card.id, 'cost': total_cost}))

            return legal_moves

    def handle_discards(self, moves, total_gems, max_gems):
        legal_moves = []
        for move in moves:
            if total_gems + sum(move[1].values()) <= max_gems:
                legal_moves.append(move)
            else:
                num_discards = total_gems + sum(move[1].values()) - max_gems
                discard_combinations = self.generate_discard_combinations(self.gems.keys(), num_discards, move[1])
                for discard_comb in discard_combinations:
                    combined_move = {**move[1], **discard_comb}
                    legal_moves.append(('take', combined_move)) # Discard takes are still take
        return legal_moves

    def generate_discard_combinations(self, gems_keys, num_discards, taken_gems):
        discard_combinations = []
        discardable_gems = list(gems_keys) + list(taken_gems.keys())
        for discard_comb in combinations(discardable_gems, num_discards):
            discard_dict = {}
            for gem in discard_comb:
                if gem in discard_dict:
                    discard_dict[gem] -= 1 # Discards are negative takes
                else:
                    discard_dict[gem] = -1 #
            discard_combinations.append(discard_dict)
        return discard_combinations
    
    def legal_to_vector(self, legal_moves):
        format_vector = [
            ('take', {'white': -1, 'blue': -1, 'green': -1}), # Take 1*3 of 5 gems
            ('take', {'white': -1, 'blue': -1, 'red': -1}),
            ('take', {'white': -1, 'blue': -1, 'black': -1}),
            ('take', {'white': -1, 'green': -1, 'red': -1}),
            ('take', {'white': -1, 'green': -1, 'black': -1}),
            ('take', {'white': -1, 'red': -1, 'black': -1}),
            ('take', {'blue': -1, 'green': -1, 'red': -1}),
            ('take', {'blue': -1, 'green': -1, 'black': -1}),
            ('take', {'blue': -1, 'red': -1, 'black': -1}),
            ('take', {'green': -1, 'red': -1, 'black': -1}), 
            ('take', {'white': -2}), # Take 2*1 of 5 gems
            ('take', {'blue': -2}), 
            ('take', {'green': -2}), 
            ('take', {'red': -2}), 
            ('take', {'black': -2}), 
            ('take', {'white': -1, 'blue': -1}), # Take 1*2 of 2 of 5 gems
            ('take', {'white': -1, 'green': -1}), 
            ('take', {'white': -1, 'red': -1}), 
            ('take', {'white': -1, 'black': -1}), 
            ('take', {'blue': -1, 'green': -1}), 
            ('take', {'blue': -1, 'red': -1}), 
            ('take', {'blue': -1, 'black': -1}), 
            ('take', {'green': -1, 'red': -1}), 
            ('take', {'green': -1, 'black': -1}), 
            ('take', {'red': -1, 'black': -1}), 
            ('take', {'white': -1}), # Take 1 of 1 of 5 gems
            ('take', {'blue': -1}), 
            ('take', {'green': -1}),
            ('take', {'red': -1}),
            ('take', {'black': -1}), # IMPLEMENT RESERVING/BUYING BASED ON SIMPLE ID, DONT HAVE LONG FORMAT
        ]

        # 60 moves = 30 take + 12+3 buy + 12+3 reserve
        # 303 moves = 30 take + 90 buy + 90 buy reserved + 90 reserve + 3 reserve top
        moves_vector = [0] * 303
        for move, details in legal_moves:
            match move:
                case 'take':
                    for index, format_details in enumerate(format_vector):
                        if format_details == (move, details):
                            moves_vector[index] = 1
                            break
                case 'buy':
                    moves_vector[29 + details] = 1
                case 'buy_reserved':
                    moves_vector[119 + details] = 1
                case 'reserve':
                    moves_vector[209 + details] = 1
                case 'reserve_top':
                    moves_vector[299 + int(details[-1])] = 1 # Grabs n in 'tiern'
        return moves_vector
    
    def vector_to_details(self, move_index):
        format_vector = [
            ('take', {'white': -1, 'blue': -1, 'green': -1}), # Take 1*3 of 5 gems
            ('take', {'white': -1, 'blue': -1, 'red': -1}),
            ('take', {'white': -1, 'blue': -1, 'black': -1}),
            ('take', {'white': -1, 'green': -1, 'red': -1}),
            ('take', {'white': -1, 'green': -1, 'black': -1}),
            ('take', {'white': -1, 'red': -1, 'black': -1}),
            ('take', {'blue': -1, 'green': -1, 'red': -1}),
            ('take', {'blue': -1, 'green': -1, 'black': -1}),
            ('take', {'blue': -1, 'red': -1, 'black': -1}),
            ('take', {'green': -1, 'red': -1, 'black': -1}), 
            ('take', {'white': -2}), # Take 2*1 of 5 gems
            ('take', {'blue': -2}), 
            ('take', {'green': -2}), 
            ('take', {'red': -2}), 
            ('take', {'black': -2}), 
            ('take', {'white': -1, 'blue': -1}), # Take 1*2 of 2 of 5 gems
            ('take', {'white': -1, 'green': -1}), 
            ('take', {'white': -1, 'red': -1}), 
            ('take', {'white': -1, 'black': -1}), 
            ('take', {'blue': -1, 'green': -1}), 
            ('take', {'blue': -1, 'red': -1}), 
            ('take', {'blue': -1, 'black': -1}), 
            ('take', {'green': -1, 'red': -1}), 
            ('take', {'green': -1, 'black': -1}), 
            ('take', {'red': -1, 'black': -1}), 
            ('take', {'white': -1}), # Take 1 of 1 of 5 gems
            ('take', {'blue': -1}), 
            ('take', {'green': -1}),
            ('take', {'red': -1}),
            ('take', {'black': -1}), # IMPLEMENT RESERVING/BUYING BASED ON SIMPLE ID, DONT HAVE LONG FORMAT
        ]

        if move_index < 30:  # Take moves
            move = format_vector[move_index]
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
            reserved_cards_vector.extend(card.to_vector())
        reserved_cards_vector += [0] * (11 * (3-len(self.reserved_cards)))
        # print ('printing gems.values', 
        #     list(self.gems.values()), '\nprinting cards.values',
        #     list(self.cards.values()), '\nprinting reserved_cards',
        #     [card.to_vector() for card in self.reserved_cards], '\nprinting not reserved 0s',
        #     [0] * (11 * (3-len(self.reserved_cards))), '\nprinting points',# Constant len state_vector, 11=len(card.to_vector)
        #     [self.points])
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