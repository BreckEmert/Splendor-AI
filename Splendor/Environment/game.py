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
        if chosen_move_index < 120:
            if chosen_move_index < 40: # all_takes_3*(4 discard permutations)
                gems_to_take = player.all_takes_3[chosen_move_index]
            elif chosen_move_index < 80: # all_takes_2_diff*4
                gems_to_take = player.all_takes_2_diff[chosen_move_index]
            elif chosen_move_index < 100: # all_takes_2_same*4
                gems_to_take = player.all_takes_2_same[chosen_move_index]
            else: # < 120, all_takes_1
                gems_to_take = player.all_takes_1[chosen_move_index]

            net_gems = player._auto_discard(gems_to_take)
            board.take_or_return_gems(net_gems)
        # Buy card moves
        elif chosen_move_index < 150:
            if chosen_move_index < 144:  # Buy from a tier, 120 + 12*2
                idx = (chosen_move_index-120) // 2  chatgpt says there's 4 tiers here?
                bought_card = board.take_card(idx//4, idx%3)  # Tier, card idx
            else:  # Buy reserved, 3*2
                card_index = (chosen_move_index - 144) % 3
                bought_card = player.reserved_cards.pop(card_index)
            
            spent_gems = player._auto_spend(bought_card)  # Spends in-place
            player.get_bought_card(bought_card)
            board.take_or_return_gems(spent_gems)

            """Noble visit and end-of-game"""
            reward = bought_card.cost
            if self.check_noble_visit(player):
                reward += 3
            if player.points >= 15:
                self.victor = True
                player.victor = True
                # reward += 1  # Should we only do -1 for the loser?
        # Reserve card moves
        elif chosen_move_index < 165:
            tier = (chosen_move_index - 150) // 5
            card_index = (chosen_move_index - 150) % 5

            if card_index < 4:  # Reserve from regular tier
                reserved_card, gold = board.reserve(tier, card_index)  # DO WE NEED RESERVE FOR GOLD REWARD AND RESERVE FOR NOT?
                # JUST GET STATISTICS ON % TIME RESERVED WHEN AVAILABLE GOLD VS NOT
            else:  # Reserve top
                reserved_card, gold = board.reserve_from_deck(tier)

            player.reserved_cards.append(reserved_card)
            player.gems[5] += gold

        return reward

    def check_noble_visit(self, player):
        for index, noble in enumerate(self.board.cards[3]):
            if noble and np.all(player.cards >= noble.cost):
                self.board.cards[3][index] = None
                player.points += 3
                return True  # No logic to tie-break, seems too insignificant for training
        return False
    
    def to_state_vector(self):
        board_vector = self.board.to_state_vector()  # length 150
        active_player = self.active_player.to_state_vector()  # length 46
        enemy_player = self.players[(self.half_turns+1) % 2].to_state_vector()  # length 46

        vector = np.concatenate((board_vector, active_player, enemy_player))
        assert len(vector) == 242, f"Game vector is length {len(vector)}"
        return vector.astype(np.float32)
    