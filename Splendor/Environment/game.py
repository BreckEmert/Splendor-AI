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

        self.half_turns += 1  # Increments even if .victor...

    def apply_move(self, chosen_move_index):
        reward = 0
        player, board = self.active_player, self.board

        # Take gems moves
        if chosen_move_index < 95:
            if chosen_move_index < 40: # all_takes_3
                # n_discards = chosen_move_index % 4
                gems_to_take = player.all_takes_3[chosen_move_index % 4]
            elif chosen_move_index < 70: # all_takes_2_diff
                # n_discards = (chosen_move_index-40) % 3
                gems_to_take = player.all_takes_2_diff[(chosen_move_index-40) % 3]
            elif chosen_move_index < 85: # all_takes_2_same
                # n_discards = (chosen_move_index-70) % 3
                gems_to_take = player.all_takes_2_same[(chosen_move_index-70) % 3]
            else: # chosen_move_index < 95, all_takes_1
                # n_discards = (chosen_move_index-85) % 2
                gems_to_take = player.all_takes_1[(chosen_move_index-85) % 2]

            net_gems = player._auto_discard(gems_to_take)
            board.take_gems(net_gems)
        # Buy card moves
        elif chosen_move_index < 125:
            if chosen_move_index < 119:  # Buy from a tier, 95 + 12*2
                idx = (chosen_move_index-95) // 2
                bought_card = board.take_card(idx//4, idx%4)  # Tier, card idx
            else:  # Buy reserved, 3*2
                card_index = (chosen_move_index - 119) % 3
                bought_card = player.reserved_cards.pop(card_index)
            
            # Player spends the tokens
            if not chosen_move_index % 2:  # All odd indices are gold spends
                player._auto_spend_gold(bought_card.cost)  # Spends in-place
            else:
                player.gems[:5] -= bought_card.cost

            # Board gets them back
            board.return_gems(-bought_card.cost)
            
            # Player gets the card
            player.get_bought_card(bought_card)

            """Noble visit and end-of-game"""
            reward = bought_card.points
            if self.check_noble_visit(player):
                reward += 3
            if player.points >= 15:
                self.victor = True
                player.victor = True
                # reward += 1  # Should we only do -1 for the loser?
        # Reserve card moves
        else:  #  chosen_move_index < 140
            tier = (chosen_move_index - 125) // 5
            card_index = (chosen_move_index - 125) % 5

            if card_index < 4:  # Reserve from regular tier
                reserved_card, gold = board.reserve(tier, card_index)  # DO WE NEED RESERVE FOR GOLD REWARD AND RESERVE FOR NOT?
                # JUST GET STATISTICS ON % TIME RESERVED WHEN AVAILABLE GOLD VS NOT
            else:  # Reserve top
                reserved_card, gold = board.reserve_from_deck(tier)

            player.reserved_cards.append(reserved_card)
            player.gems[5] += gold

        return reward

    def check_noble_visit(self, player):
        visited = False
        for index, noble in enumerate(self.board.nobles):
            if noble and np.all(player.cards >= noble.cost):
                self.board.nobles[index] = None
                player.points += 3
                visited = True  # No logic to tie-break, seems too insignificant for training
        return visited
    
    def to_state_vector(self):
        board_vector = self.board.to_state_vector()  # length 150
        active_player = self.active_player.to_state_vector()  # length 46
        enemy_player = self.players[(self.half_turns+1) % 2].to_state_vector()  # length 46

        vector = np.concatenate((board_vector, active_player, enemy_player))
        assert len(vector) == 242, f"Game vector is length {len(vector)}"
        return vector.astype(np.float32)
    