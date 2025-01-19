# Splendor/Environment/game.py

import numpy as np

from Environment.Splendor_components.Board_components.board import Board  # type: ignore
from Environment.Splendor_components.Player_components.player import Player  # type: ignore


class Game:
    def __init__(self, players):
        """Note: rest of init is performed by reset."""
        self.players = [Player(name, rl_model) for name, rl_model in players]
        self.reset()

    def reset(self):
        self.board = Board()

        for player in self.players:
            player.reset()    

        self.half_turns: int = 0
        self.victor: bool = False
    
    @property
    def active_player(self):
        return self.players[self.half_turns % 2]

    def turn(self):
        # Apply primary move
        chosen_move_index = self.active_player.choose_move(self.board, self.to_state_vector())
        self.apply_move(chosen_move_index)
        # DONT FORGET TO DO self.victor IN self.apply_move

        self.half_turns += 1

    def apply_move(self, chosen_move_index):
        reward = 0
        player, board = self.active_player, self.board

        # Take gems moves
        if chosen_move_index < 400:  # Still guessing on this
            gems_to_take = tier
            board.take_or_return_gems(gems_to_take)
            player.take_or_spend_gems(gems_to_take)
        # Buy card moves
        elif chosen_move_index < 430:
            bought_card = board.take_card(tier, card_index)  # Buy from regular tier
            bought_card = player.reserved_cards.pop(card_index)  # Buy reserved
            player.get_bought_card(bought_card)

            cost = np.maximum(bought_card.cost - player.cards, 0)
            board.take_or_return_gems(-cost)
            player.take_or_spend_gems(-cost)

            reward = bought_card.cost
        # Reserve card moves
        elif chosen_move_index < 460:
            reserved_card, gold = board.reserve(tier, card_index)  # Reserve from regular tier
            reserved_card, gold = board.reserve_from_deck(tier)  # Reserve top
            player.reserved_cards.append(reserved_card)

            if sum(player.gems) < 10:
                player.gems[5] += gold
            else:
                discard, _ = player.choose_discard(self.to_state_vector(), player.gems)
                player.take_or_spend_gems(discard)
                player.gems[5] += gold

        return reward

    def check_noble_visit(self):
        for index, noble in enumerate(self.board.cards[3]):
            if noble and np.all(player.cards >= noble.cost):
                self.board.cards[3][index] = None
                return True  # No logic to tie-break, seems too insignificant for training
        return False
    
    def to_state_vector(self):
        board_vector = self.board.to_state_vector()  # length 150, change player.state_offset if this changes
        active_player = self.active_player.to_state_vector()  # length 46
        enemy_player = self.players[(self.half_turns+1) % 2].to_state_vector()  # length 46

        # Adds on [0.0] which indicates progression through loop
        vector = np.concatenate((board_vector, active_player, [0.0], enemy_player))
        # assert len(vector) == 243, f"Game vector is length {len(vector)}"
        return vector.astype(np.float32)
    