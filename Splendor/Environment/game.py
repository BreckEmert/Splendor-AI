# Splendor/Environment/game.py

import numpy as np

from Environment.Splendor_components.Board_components.board import Board
from Environment.Splendor_components.Player_components.player import Player


class Game:
    def __init__(self, players, model):
        """Note: rest of init is performed by reset()"""
        self.players = [Player(name, rl_model) for name, rl_model in players]
        self.model = model
        self.reset()
        
        self.turn_penalty: float = -0.0  # Testing 0?
        self.discard_penalty: float = -0.3  # < 25 moves / 15 gems / 3 gems
        self.final_reward: float = 5.0

    def reset(self):
        self.board = Board()

        for player in self.players:
            player.reset()

        self.half_turns: int = 0
        self.move_index: int = 0
        self.victor: bool = False
    
    @property
    def active_player(self):
        return self.players[self.half_turns % 2]

    def turn(self):
        # Log previous state for model memory
        state = self.to_state_vector()

        # Apply primary move
        move_index = self.active_player.choose_move(self.board, state)
        self.move_index = move_index
        reward = self.apply_move(move_index)

        # assert np.all(self.board.gems >= 0), "Board gems lt0"
        # assert np.all(self.board.gems[:5] <= 4), "Board gems gt10"
        # assert self.active_player.gems.sum() >= 0, "Player gems lt0"
        # assert self.active_player.gems.sum() <= 10, "Player gems gt10"

        # Remember
        next_state = self.to_state_vector()
        legal_mask = self.active_player.get_legal_moves(self.board)
        sarsld = [state, move_index, reward, 
                  next_state, legal_mask, 
                  self.victor]
        self.model.remember(sarsld)

        self.half_turns += 1

    def apply_move(self, chosen_move_index):
        player, board = self.active_player, self.board

        # Take gems moves
        if chosen_move_index < player.take_dim:
            if chosen_move_index < 40: # all_takes_3; 10 * 4discards
                gems_to_take = player.all_takes_3[chosen_move_index // 4]
            elif chosen_move_index < 55: # all_takes_2_same; 5 * 3discards
                gems_to_take = player.all_takes_2_same[(chosen_move_index-40) // 3]
            elif chosen_move_index < 85: # all_takes_2_diff; 10 * 3discards
                gems_to_take = player.all_takes_2_diff[(chosen_move_index-55) // 3]
            else: # chosen_move_index < 95  # all_takes_1; 5 * 2discards
                gems_to_take = player.all_takes_1[(chosen_move_index-85) // 2]

            taken_gems, n_discards = player.auto_take(gems_to_take)
            board.take_gems(taken_gems)

            reward = self.discard_penalty * n_discards
            return reward + self.turn_penalty

        # Buy card moves
        chosen_move_index -= player.take_dim
        if chosen_move_index < player.buy_dim:
            if chosen_move_index < 24:  # 12 cards * w&w/o gold
                idx = chosen_move_index // 2
                bought_card = board.take_card(idx//4, idx%4)  # Tier, card idx
            else:  # Buy reserved, 3 cards* w&w/o gold
                card_index = (chosen_move_index-24) // 2
                bought_card = player.reserved_cards.pop(card_index)

            # Player spends the tokens
            with_gold = chosen_move_index % 2  # All odd indices are gold spends
            spent_gems = player._auto_spend(bought_card.cost, with_gold=with_gold)

            # Board gets them back
            board.return_gems(spent_gems)
            
            # Player gets the card
            player.get_bought_card(bought_card)

            """Noble visit and end-of-game"""
            # Base reward value
            reward = bought_card.points
            reward += 3 * self._check_noble_visit(player)  # UNCOMMENT THIS LATER

            # Capping any points past 15
            original_points = player.points - bought_card.points  # player already got points so need to take them back
            reward = min(reward, 15 - original_points)  / 2

            if player.points >= 15:
                self.victor = True
                player.victor = True
                reward += self.final_reward
                self.model.memory[-1][2] -= self.final_reward  # Loser reward
                self.model.memory[-1][5] = True  # Mark loser's memory as done
            
            return reward + self.turn_penalty
        
        # Reserve card moves
        chosen_move_index -= player.buy_dim
        if chosen_move_index < player.reserve_dim:
            tier = chosen_move_index // 5  # 4 cards + top of deck
            card_index = chosen_move_index % 5

            if card_index < 4:  # Reserve from regular tier
                reserved_card, gold = board.reserve(tier, card_index)  # DO WE NEED RESERVE FOR GOLD REWARD AND RESERVE FOR NOT?
                # JUST GET STATISTICS ON % TIME RESERVED WHEN AVAILABLE GOLD VS NOT
                #### ALSO CAN JUST DO ONE RESERVE OPERATION AND CHECK IF CARD INDEX IS 5
            else:  # Reserve top
                reserved_card, gold = board.reserve_from_deck(tier)

            n_discards = 0
            player.reserved_cards.append(reserved_card)
            if gold[5]:
                discard_if_gt10, n_discards = player.auto_take(gold)
                board.take_gems(discard_if_gt10)

            reward = self.discard_penalty * n_discards
            return reward + self.turn_penalty

    def _check_noble_visit(self, player):
        visited = 0
        for index, noble in enumerate(self.board.nobles):
            if noble and np.all(player.cards >= noble.cost):
                self.board.nobles[index] = None
                player.points += 3
                visited += 1
        return visited
    
    def to_state_vector(self):
        board_vector = self.board.to_state_vector()  # 157
        active_player = self.active_player.to_state_vector()  # 46
        enemy_player = self.players[(self.half_turns+1) % 2].to_state_vector()  # 46

        vector = np.concatenate((board_vector, active_player, enemy_player))
        return vector.astype(np.float32)  # 249
    