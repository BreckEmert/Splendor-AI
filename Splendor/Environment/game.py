# Splendor/Environment/game.py

import numpy as np

from Environment.Splendor_components.Board_components.board import Board
from Environment.Splendor_components.Player_components.player import Player


class Game:
    def __init__(self, players):
        self.num_players = len(players)

        self.board = Board(self.num_players)
        self.players: list = [Player(name, strategy, strategy_strength, rl_model, turn_order_index) 
                              for name, strategy, strategy_strength, rl_model, turn_order_index in players]

        self.reward = 0
        self.active_player = 0
        self.half_turns: int = 0
        self.is_final_turn: bool = 0
        self.victor = 0
    
    def turn(self):
        if self.is_final_turn:
            self.victor = self.get_victor()
            self.active_player.victor = True

        self.reward = 0
        self.active_player = self.players[self.half_turns % self.num_players]
        prev_state = self.to_vector()

        # Apply primary move
        chosen_move = self.active_player.choose_move(self.board, prev_state)
        self.apply_move(chosen_move)

        self.check_noble_visit()
        if self.active_player.points >= 15:
            self.is_final_turn = 1

        self.half_turns += 1

    def apply_move(self, move): # WE DON'T ACTUALLY NEED 'WITH GOLD' MOVES
        action, (tier, card_index) = move
        match action:
            case 'take':
                gems_to_take = tier

                self.board.take_or_return_gems(gems_to_take) # Confused about what we want here.  What does move contain, can it contain multiple?
                self.active_player.take_or_spend_gems(gems_to_take)

                self.reward -= 1
            case 'buy':
                bought_card = self.board.take_card(tier, card_index)
                self.active_player.get_bought_card(bought_card)

                self.board.take_or_return_gems(bought_card.cost)
                self.active_player.take_or_spend_gems(bought_card.cost)

                self.reward += bought_card.points 
            case 'buy reserved':
                bought_card = self.active_player.reserved_cards.pop(card_index)
                self.active_player.get_bought_card(bought_card)

                self.board.take_or_return_gems(bought_card.cost)
                self.active_player.take_or_spend_gems(bought_card.cost)
            case 'buy with gold':
                spent_gems = card_index
                tier, card_index = tier
                bought_card = self.board.take_card(tier, card_index)
                self.active_player.get_bought_card(bought_card)

                self.board.take_or_return_gems(spent_gems)
                self.active_player.take_or_spend_gems(spent_gems)
            case 'buy reserved with gold':
                card_index, spent_gems = tier, card_index
                bought_card = self.active_player.reserved_cards.pop(card_index)
                self.active_player.get_bought_card(bought_card)

                self.board.take_or_return_gems(spent_gems)
                self.active_player.take_or_spend_gems(spent_gems)
            case 'reserve':

                reserved_card, gold = self.board.reserve(tier, card_index)
                self.active_player.reserved_cards.append(reserved_card)

                self.active_player.gems[5] += gold
            case 'reserve top':
                reserved_card, gold = self.board.reserve_from_deck(tier)
                self.active_player.reserved_cards.append(reserved_card)

                self.active_player.gems[5] += gold

    def check_noble_visit(self):
        for index, noble in enumerate(self.board.cards[3]):
            if noble and np.all(self.active_player.cards >= noble.cost):
                self.reward += noble.points
                self.active_player.points += noble.points
                self.board.cards[3][index] = None
                break # Implement logic to choose the noble if tied

    def get_victor(self):
        victor = max(self.players, key=lambda p: p.points)
        return victor
   
    def get_state(self):
        return {
            'board': self.board.get_state(),
            'players': {player.name: player.get_state() for player in self.players},
            'current_half_turn': self.half_turns
        }

    def to_vector(self):
        state_vector = self.board.to_vector() # length 156
        player_vectors = [player.to_vector() for player in self.players] # length 45*2  90

        state_vector = np.concatenate((state_vector, *player_vectors, [self.is_final_turn])) # plus length 1

        assert len(state_vector) == 247, f"Game state vector is not 247, but {len(state_vector)}"
        return state_vector # length 247