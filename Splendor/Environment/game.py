# Splendor/Environment/game.py

import numpy as np
from Environment.Splendor_components import Board # type: ignore
from Environment.Splendor_components import Player # type: ignore


class Game:
    def __init__(self, players):
        self.num_players = len(players)

        self.board = Board(self.num_players)
        self.players: list = [Player(name, strategy, strategy_strength, layer_sizes, model_path) 
                              for name, strategy, strategy_strength, layer_sizes, model_path in players]

        self.reward = 0
        self.active_player = 0
        self.half_turns: int = 0
        self.is_final_turn: bool = False
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
        print(self.half_turns, chosen_move)
        self.apply_move(chosen_move)

        self.check_noble_visit()
        if self.active_player.points >= 15:
            self.is_final_turn = True

        self.half_turns += 1

    def apply_move(self, move):
        action, (tier, position) = move
        match action:
            case 'take':
                gem, amount = tier, position
                self.board.change_gems({gem: amount})
                self.active_player.change_gems({gem: amount})
                self.reward -= 1
            case 'buy':
                bought_card = self.board.take_card(tier, position)
                self.active_player.get_bought_card(bought_card)

                self.board.change_gems(bought_card.cost)
                self.active_player.change_gems(bought_card.cost)
                self.reward += bought_card.points 
            case 'buy_reserved':
                bought_card = self.active_player.reserved_cards.pop(position)
                self.active_player.get_bought_card(bought_card)

                self.board.change_gems(bought_card.cost)
                self.active_player.change_gems(bought_card.cost)
            case 'buy_with_gold':
                position, spent_gems = position
                bought_card = self.board.take_card(tier, position)
                self.active_player.get_bought_card(bought_card)

                self.board.change_gems(spent_gems)
                self.active_player.change_gems(spent_gems)
            case 'buy_reserved_with_gold':
                position, spent_gems = position
                bought_card = self.active_player.reserved_cards.pop(position)
                self.active_player.get_bought_card(bought_card)

                self.board.change_gems(spent_gems)
                self.active_player.change_gems(spent_gems)
            case 'reserve':
                reserved_card, gold = self.board.reserve(tier, position)
                self.active_player.reserve_card(reserved_card)

                self.active_player.gems['gold'] += gold
            case 'reserve_top':
                reserved_card, gold = self.board.reserve_from_deck(tier)
                self.active_player.reserve_card(reserved_card)

                self.active_player.gems['gold'] += gold

    def check_noble_visit(self):
        for noble in self.board.cards['nobles']:
            if all(self.active_player.cards[gem] >= amount for gem, amount in noble.cost.items()):
                self.reward += noble.points
                self.active_player.points += noble.points
                self.board.cards['nobles'].remove(noble)
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
        state_vector = self.board.to_vector() # length 57
        for player in self.players: # length 45*2 = 90
            state_vector.extend(player.to_vector())
        state_vector.append(int(self.is_final_turn)) # length 1
        return state_vector # length 148