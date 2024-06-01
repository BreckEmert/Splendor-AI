# Splendor/Environment/Splendor_components/player.py

import numpy as np
from RL import RLAgent # type: ignore


class Player:
    def __init__(self, name, strategy, strategy_strength):
        self.name: str = name
        self.gems: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0, 'gold': 0}
        self.cards: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0}
        self.reserved_cards: list = []
        self.points: int = 0

        self.rl_model = RLAgent()
        #self.strategy: strategy = strategy
        #self.strategy_strength: int = strategy_strength
    
    def take_gems(self, gems_to_take):
        for gem, amount in gems_to_take.items():
            self.gems[gem] += amount

    def buy_card(self, card):
        for gem, amount in card.cost.items():
            self.gems[gem] -= amount
        self.cards[card.gem] += 1
        self.points += card.points

    def reserve_card(self, card):
        self.reserved_cards.append(card)

    def get_legal_moves(self, board):
        #print('board.state()', board.get_state())
        best_moves = []

        # Take gems moves
        total_gems = sum(self.gems.values())
        take_1_count = take_2_count = take_3_count = 0
        if total_gems <= 8:  # Max of 10 gems in inventory
            non_zero_gems = [gem for gem in board.gems if gem != 'gold' and board.gems[gem] > 0]
            len_nzg = len(non_zero_gems)

            if total_gems < 7:
                # Taking 3 different gems if available
                for i in range(len_nzg):
                    for j in range(i + 1, len_nzg):
                        for k in range(j + 1, len_nzg):
                            best_moves.append(('take', {non_zero_gems[i]: 1,
                                                        non_zero_gems[j]: 1, 
                                                        non_zero_gems[k]: 1}))
                            take_3_count += 1

            # Taking 2 of the same gem if at least 4 available
            for gem, count in board.gems.items():
                if gem != 'gold' and count >= 4:
                    best_moves.append(('take', {gem: 2}))

        worse_moves = []
        if take_3_count == 0 and total_gems < 9:
            # Taking 2 different gems if available
            for i in range(len_nzg):
                for j in range(i + 1, len_nzg):
                    worse_moves.append(('take', {non_zero_gems[i]: 1,
                                                 non_zero_gems[j]: 1}))
                    take_2_count += 1

        if take_2_count == 0 and total_gems < 10:
            # Taking 1 gem if available
            for gem, count in board.gems.items():
                if gem != 'gold' and count > 0:
                    worse_moves.append(('take', {gem: 1}))
                    take_1_count += 1
        
        # Reserve card moves
        if len(self.reserved_cards) < 3:
            for tier in ['tier1', 'tier2', 'tier3']:
                for card in board.cards[tier]:
                    best_moves.append(('reserve', card.id))
                best_moves.append(('reserve_top', tier)) # Reserve unknown top of decks

        # Buy card moves
        for tier in ['tier1', 'tier2', 'tier3']:
            for card in board.cards[tier]:
                if all(self.gems[gem] >= amount for gem, amount in card.cost.items()):
                    best_moves.append(('buy', card.id))
        
        # Buy reserved card moves
        # This adds on a lot of action space, think of ways to remove
        for card in self.reserved_cards:
            if all(self.gems[gem] >= amount for gem, amount in card.cost.items()):
                    best_moves.append(('buy_reserved', card.id))

        # print('self.state()', self.get_state())
        # print('printing legal moves', best_moves + worse_moves)
        print(take_1_count + take_2_count + take_3_count)
        print(best_moves + worse_moves)
        return best_moves + worse_moves

    def legal_to_vector(self, legal_moves):
        format_vector = [
            ('take', {'white': 1, 'blue': 1, 'green': 1}), # Take 1*3 of 5 gems
            ('take', {'white': 1, 'blue': 1, 'red': 1}),
            ('take', {'white': 1, 'blue': 1, 'black': 1}),
            ('take', {'white': 1, 'green': 1, 'red': 1}),
            ('take', {'white': 1, 'green': 1, 'black': 1}),
            ('take', {'white': 1, 'red': 1, 'black': 1}),
            ('take', {'blue': 1, 'green': 1, 'red': 1}),
            ('take', {'blue': 1, 'green': 1, 'black': 1}),
            ('take', {'blue': 1, 'red': 1, 'black': 1}),
            ('take', {'green': 1, 'red': 1, 'black': 1}), 
            ('take', {'white': 2}), # Take 2*1 of 5 gems
            ('take', {'blue': 2}), 
            ('take', {'green': 2}), 
            ('take', {'red': 2}), 
            ('take', {'black': 2}), 
            ('take', {'white': 1, 'blue': 1}), # Take 1*2 of 2 of 5 gems
            ('take', {'white': 1, 'green': 1}), 
            ('take', {'white': 1, 'red': 1}), 
            ('take', {'white': 1, 'black': 1}), 
            ('take', {'blue': 1, 'green': 1}), 
            ('take', {'blue': 1, 'red': 1}), 
            ('take', {'blue': 1, 'black': 1}), 
            ('take', {'green': 1, 'red': 1}), 
            ('take', {'green': 1, 'black': 1}), 
            ('take', {'red': 1, 'black': 1}), 
            ('take', {'white': 1}), # Take 1 of 1 of 5 gems
            ('take', {'blue': 1}), 
            ('take', {'green': 1}),
            ('take', {'red': 1}),
            ('take', {'black': 1}), # IMPLEMENT RESERVING/BUYING BASED ON SIMPLE ID, DONT HAVE LONG FORMAT
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
            ('take', {'white': 1, 'blue': 1, 'green': 1}), # Take 1*3 of 5 gems
            ('take', {'white': 1, 'blue': 1, 'red': 1}),
            ('take', {'white': 1, 'blue': 1, 'black': 1}),
            ('take', {'white': 1, 'green': 1, 'red': 1}),
            ('take', {'white': 1, 'green': 1, 'black': 1}),
            ('take', {'white': 1, 'red': 1, 'black': 1}),
            ('take', {'blue': 1, 'green': 1, 'red': 1}),
            ('take', {'blue': 1, 'green': 1, 'black': 1}),
            ('take', {'blue': 1, 'red': 1, 'black': 1}),
            ('take', {'green': 1, 'red': 1, 'black': 1}), 
            ('take', {'white': 2}), # Take 2*1 of 5 gems
            ('take', {'blue': 2}), 
            ('take', {'green': 2}), 
            ('take', {'red': 2}), 
            ('take', {'black': 2}), 
            ('take', {'white': 1, 'blue': 1}), # Take 1*2 of 2 of 5 gems
            ('take', {'white': 1, 'green': 1}), 
            ('take', {'white': 1, 'red': 1}), 
            ('take', {'white': 1, 'black': 1}), 
            ('take', {'blue': 1, 'green': 1}), 
            ('take', {'blue': 1, 'red': 1}), 
            ('take', {'blue': 1, 'black': 1}), 
            ('take', {'green': 1, 'red': 1}), 
            ('take', {'green': 1, 'black': 1}), 
            ('take', {'red': 1, 'black': 1}), 
            ('take', {'white': 1}), # Take 1 of 1 of 5 gems
            ('take', {'blue': 1}), 
            ('take', {'green': 1}),
            ('take', {'red': 1}),
            ('take', {'black': 1}), # IMPLEMENT RESERVING/BUYING BASED ON SIMPLE ID, DONT HAVE LONG FORMAT
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
        return {
            'gems': self.gems, 
            'cards': self.cards, 
            'reserved_cards': self.reserved_cards, 
            'points:': self.points
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
    moves_vector = bob.legal_to_vector(moves)