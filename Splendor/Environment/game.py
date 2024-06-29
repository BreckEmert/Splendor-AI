# Splendor/Environment/game.py

import numpy as np

from Environment.Splendor_components.Board_components.board import Board # type: ignore
from Environment.Splendor_components.Player_components.player import Player # type: ignore


class Game:
    def __init__(self, players):
        self.players: list = [Player(name, rl_model) for name, rl_model in players]
        self.reset()

    def reset(self):
        self.board = Board()
        for player in self.players:
            player.reset()
        self.half_turns: int = 0
        self.victor: bool = False
    
    def turn(self):
        self.active_player = self.players[self.half_turns % 2]
        game_state = self.to_vector()

        # Apply primary move
        chosen_move = self.active_player.choose_move(self.board, game_state)
        if chosen_move:
            self.apply_move(chosen_move)
        else:
            self.victor = True

        self.half_turns += 1

    def apply_move(self, move):
        action, (tier, card_index) = move
        match action:
            case 'take':
                gems_to_take = tier
                self.board.take_or_return_gems(gems_to_take)
                self.active_player.take_or_spend_gems(gems_to_take)
            case 'buy':
                bought_card = self.board.take_card(tier, card_index)
                self.active_player.get_bought_card(bought_card)

                cost = np.maximum(bought_card.cost - self.active_player.cards, 0)
                self.board.take_or_return_gems(-cost)
                self.active_player.take_or_spend_gems(-cost)
            case 'buy reserved':
                bought_card = self.active_player.reserved_cards.pop(card_index)
                self.active_player.get_bought_card(bought_card)

                cost = np.maximum(bought_card.cost - self.active_player.cards, 0)
                self.board.take_or_return_gems(-cost)
                self.active_player.take_or_spend_gems(-cost)
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

                if sum(self.active_player.gems) < 10:
                    self.active_player.gems[5] += gold
                else:
                    discard, _ = self.active_player.choose_discard(self.to_vector())
                    self.active_player.take_or_spend_gems(-discard)
                    self.active_player.gems[5] += gold
            case 'reserve top': # OTHER PLAYERS CAN'T ACTUALLY SEE THIS CARD
                reserved_card, gold = self.board.reserve_from_deck(tier)
                self.active_player.reserved_cards.append(reserved_card)

                if sum(self.active_player.gems) < 10:
                    self.active_player.gems[5] += gold
                else:
                    discard, _ = self.active_player.choose_discard(self.to_vector())
                    self.active_player.take_or_spend_gems(-discard)
                    self.active_player.gems[5] += gold

    def get_state(self):
        return {
            'board': self.board.get_state(),
            'players': {player.name: player.get_state() for player in self.players},
            'current_half_turn': self.half_turns
        }

    def to_vector(self):
        board_vector = self.board.to_vector() # length 150 !change player.state_offset if this changes!
        active_player = self.active_player.to_vector() # length 45
        enemy_player = self.players[(self.half_turns+1) % 2].to_vector() # length 45

        return np.concatenate((board_vector, active_player, enemy_player)).astype(np.float32)