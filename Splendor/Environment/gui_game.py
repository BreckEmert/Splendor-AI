# Splendor/Environment/gui_game.py
"""Mirror of rl_game.py with trimmed functionality."""

import numpy as np

from Environment.Splendor_components.Board_components import Board
from Environment.Splendor_components.Player_components import Player


class GUIGame:
    def __init__(self, players, model):
        """Note: rest of init is performed by reset()"""
        self.players = [Player(name, agent, pos) for name, agent, pos in players]
        self.model = model
        self.reset()
    
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

    def turn(self) -> None:
        move_index = self.active_player.choose_move(self.board, self.to_state())
        self.move_index = move_index
        self.apply_move(move_index)

        assert np.all(self.board.gems >= 0), "Board gems lt0"
        assert np.all(self.board.gems[:5] <= 4), "Board gems gt4"
        assert self.active_player.gems.sum() >= 0, "Player gems lt0"
        assert self.active_player.gems.sum() <= 10, "Player gems gt10"
        
        self.half_turns += 1

    def apply_move(self, chosen_move_index) -> None:
        """Deeply sorry for the magic numbers approach."""
        player, board = self.active_player, self.board
        gems_to_take: np.ndarray | None = None

        # Take gems moves
        if chosen_move_index < player.take_dim:
            if chosen_move_index < 40: # all_takes_3; 10 * 4discards
                gems_to_take = player.all_takes_3[chosen_move_index // 4]
            elif chosen_move_index < 55: # all_takes_2_same; 5 * 3discards
                gems_to_take = player.all_takes_2_same[(chosen_move_index-40) // 3]
            elif chosen_move_index < 85: # all_takes_2_diff; 10 * 3discards
                gems_to_take = player.all_takes_2_diff[(chosen_move_index-55) // 3]
            elif chosen_move_index < 95:  # all_takes_1; 5 * 2discards
                gems_to_take = player.all_takes_1[(chosen_move_index-85) // 2]
            else:  # All else is illegal, discard
                legal_discards = np.where(player.gems > 0)[0]
                discard_idx = np.random.choice(legal_discards)
                player.gems[discard_idx] -= 1
                board.gems[discard_idx] += 1

            taken_gems, _ = player.auto_take(gems_to_take)
            board.take_gems(taken_gems)

        # Buy card moves
        chosen_move_index -= player.take_dim
        if chosen_move_index < player.buy_dim:
            if chosen_move_index < 24:  # 12 cards * w&w/o gold
                idx = chosen_move_index // 2
                bought_card = board.take_card(idx//4, idx%4)  # Tier, card idx
            else:  # Buy reserved, 3 cards * w&w/o gold
                card_index = (chosen_move_index-24) // 2
                bought_card = player.reserved_cards.pop(card_index)

            # Spend the tokens
            if getattr(player.agent, "pending_spend", None):
                spent_gems = player.manual_spend(player.agent.pending_spend)  # type: ignore
                player.agent.pending_spend = None  # type: ignore
            else:
                with_gold = chosen_move_index % 2  # All odd indices are gold spends
                spent_gems = player.auto_spend(bought_card.cost, with_gold=with_gold)

            board.return_gems(spent_gems)
            player.get_bought_card(bought_card)

            """Noble visit and end-of-game"""
            self._check_noble_visit(player)

            if player.points >= 15:
                self.victor = True
                player.victor = True
        
        # Reserve card moves
        chosen_move_index -= player.buy_dim
        if chosen_move_index < player.reserve_dim:
            tier = chosen_move_index // 5  # 4 cards + top of deck
            card_index = chosen_move_index % 5

            if card_index < 4:  # Reserve from regular tier
                reserved_card, gold = board.reserve(tier, card_index)
            else:  # Reserve top of deck
                reserved_card, gold = board.reserve_from_deck(tier)

            player.reserved_cards.append(reserved_card)
            if gold[5]:
                discard_if_gt10, _ = player.auto_take(gold)
                board.take_gems(discard_if_gt10)

    def _check_noble_visit(self, player) -> None:
        for index, noble in enumerate(self.board.nobles):
            if noble and np.all(player.cards >= noble.cost):
                self.board.nobles[index] = None
                player.points += 3
    
    def to_state(self) -> np.ndarray:
        cur_player = self.active_player
        enemy_player = self.players[(self.half_turns+1) % 2]

        board_vector = self.board.to_state(cur_player.effective_gems)        # 157
        hero_vector = self.active_player.to_state()                          # 47
        enemy_vector = enemy_player.to_state()                               # 47

        vector = np.concatenate((board_vector, hero_vector, enemy_vector))   # 251
        return vector.astype(np.float32)
