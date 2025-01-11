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
        player, board = self.active_player, self.board

        match action:
            case 'take':
                gems_to_take = tier
                board.take_or_return_gems(gems_to_take)
                player.take_or_spend_gems(gems_to_take)
            case 'buy':
                bought_card = board.take_card(tier, card_index)
                player.get_bought_card(bought_card)

                cost = np.maximum(bought_card.cost - player.cards, 0)
                board.take_or_return_gems(-cost)
                player.take_or_spend_gems(-cost)
            case 'buy reserved':
                bought_card = player.reserved_cards.pop(card_index)
                player.get_bought_card(bought_card)

                cost = np.maximum(bought_card.cost - player.cards, 0)
                board.take_or_return_gems(-cost)
                player.take_or_spend_gems(-cost)
            case 'buy with gold':
                spent_gems = card_index
                tier, card_index = tier
                bought_card = board.take_card(tier, card_index)
                player.get_bought_card(bought_card)

                board.take_or_return_gems(spent_gems)
                player.take_or_spend_gems(spent_gems)
            case 'buy reserved with gold':
                card_index, spent_gems = tier, card_index
                bought_card = player.reserved_cards.pop(card_index)
                player.get_bought_card(bought_card)

                board.take_or_return_gems(spent_gems)
                player.take_or_spend_gems(spent_gems)
            case 'reserve':
                reserved_card, gold = board.reserve(tier, card_index)
                player.reserved_cards.append(reserved_card)

                if sum(player.gems) < 10:
                    player.gems[5] += gold
                else:
                    discard, _ = player.choose_discard(
                        self.to_vector(), player.gems, reward=-1/30)
                    player.take_or_spend_gems(discard)
                    player.gems[5] += gold
            case 'reserve top':  # OTHER PLAYERS CAN'T ACTUALLY SEE THIS CARD
                reserved_card, gold = board.reserve_from_deck(tier)
                player.reserved_cards.append(reserved_card)

                if sum(player.gems) < 10:
                    player.gems[5] += gold
                else:
                    discard, _ = player.choose_discard(
                        self.to_vector(), player.gems, reward=-1/30)
                    player.take_or_spend_gems(discard)
                    player.gems[5] += gold

    def get_state(self):
        return {
            'board': self.board.get_state(),
            'players': {player.name: player.get_state() for player in self.players}
        }

    def to_vector(self):
        board_vector = self.board.to_vector()  # length 150, change player.state_offset if this changes
        active_player = self.active_player.to_vector()  # length 46
        enemy_player = self.players[(self.half_turns+1) % 2].to_vector()  # length 46

        # Adds on [0.0] which indicates progression through loop
        vector = np.concatenate((board_vector, active_player, [0.0], enemy_player))
        assert len(vector) == 243, f"Game vector is length {len(vector)}"
        return vector.astype(np.float32)
    