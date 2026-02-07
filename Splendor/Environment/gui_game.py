# Splendor/Environment/gui_game.py
"""Mirror of rl_game.py with trimmed functionality."""

import numpy as np
from typing import TYPE_CHECKING

from Environment import Board, Player
from RL import BasicRewardEngine, SparseRewardEngine
if TYPE_CHECKING:
    from Play.common_types import GUIMove


class GUIGame:
    def __init__(self, players, model):
        """Note: rest of init is performed by reset()"""
        self.players = [Player(name, agent, pos) for name, agent, pos in players]
        self.model = model
        self.rewards = BasicRewardEngine(self)
        self.reset()
    
    def reset(self):
        self.board = Board()

        for player in self.players:
            player.reset()

        self.half_turns: int = 0
        self.move_idx: int | None = 0
        self.victor: bool = False
    
    @property
    def active_player(self):
        return self.players[self.half_turns % 2]

    @property
    def inactive_player(self):
        return self.players[(self.half_turns + 1) % 2]
    
    def turn(self) -> None:
        move = self.active_player.choose_move(self.board, self.to_state())
        if isinstance(move, int):
            self.move_idx = move
            self.apply_ai_move(move)
        else:
            self.move_idx = None
            self.apply_human_move(move)

        assert np.all(self.board.gems >= 0), "Board gems lt0"
        assert np.all(self.board.gems[:5] <= 4), "Board gems gt4"
        assert self.active_player.gems.sum() >= 0, "Player gems lt0"
        assert self.active_player.gems.sum() <= 10, "Player gems gt10"
        
        self.half_turns += 1

    def apply_human_move(self, move: "GUIMove") -> None:
        """Handles moves sent from the GUI.
        Note that these moves are always complete, with
        full spend information needed to not require any
        _auto method or calculation.
        """
        player, board = self.active_player, self.board

        if move.kind == "take":
            player.gems += move.take
            board.gems  -= move.take

            if move.discard is not None:
                player.gems -= move.discard
                board.gems  += move.discard

        elif move.kind == "buy":
            ft = move.source
            assert ft is not None, "buy GUIMove has no FocusTarget"

            if ft.kind == "shop":
                bought = self.board.take_card(ft.tier, ft.pos)
            else:  # reserved
                assert ft.reserve_idx is not None, "reserve ft has no reserve_index"
                bought = player.reserved_cards.pop(ft.reserve_idx)

            player.gems -= move.spend
            self.board.return_gems(move.spend)
            player.get_bought_card(bought)

            # End of game check
            self._check_noble_visit(player)
            if player.points >= 15:
                self.victor = player.victor = True

        elif move.kind == "reserve":
            ft = move.source  # FocusTarget
            assert ft is not None, "reserve GUIMove has no FocusTarget"
            if ft.kind == "shop":
                reserved, gold = board.reserve(ft.tier, ft.pos)
            else:  # top of deck
                reserved, gold = board.reserve_from_deck(ft.tier)
            
            # Need to use discard so taht auto_take isn't called.
            # If player reserves with 10 gems, they must have a discard.
            # need to add this notice to the game.
            player.reserved_cards.append(reserved)
            if gold[5]:
                if move.discard:
                    player.gems -= move.discard
                    board.gems += move.discard
                player.gems += gold
                board.gems -= gold

    def apply_ai_move(self, move_idx: int) -> None:
        """Deeply sorry for the magic numbers approach."""
        player, board = self.active_player, self.board

        # Take gems moves
        if move_idx < player.take_dim:
            gems_to_take: np.ndarray = np.zeros(6)
            if move_idx < 40: # all_takes_3; 10 * 4discards
                gems_to_take = player.all_takes_3[move_idx // 4]
            elif move_idx < 55: # all_takes_2_same; 5 * 3discards
                gems_to_take = player.all_takes_2_same[(move_idx-40) // 3]
            elif move_idx < 85: # all_takes_2_diff; 10 * 3discards
                gems_to_take = player.all_takes_2_diff[(move_idx-55) // 3]
            elif move_idx < 95:  # all_takes_1; 5 * 2discards
                gems_to_take = player.all_takes_1[(move_idx-85) // 2]
            else:  # All else is illegal, discard
                legal_discards = np.where(player.gems > 0)[0]
                discard_idx = np.random.choice(legal_discards)
                player.gems[discard_idx] -= 1
                board.gems[discard_idx] += 1

            taken_gems, _ = player.auto_take(gems_to_take)
            board.take_gems(taken_gems)
            
            return

        # Buy card moves
        move_idx -= player.take_dim
        if move_idx < player.buy_dim:
            if move_idx < 24:  # 12 cards * w&w/o gold
                idx = move_idx // 2
                bought_card = board.take_card(idx//4, idx%4)  # Tier, card idx
            else:  # Buy reserved, 3 cards * w&w/o gold
                card_index = (move_idx-24) // 2
                bought_card = player.reserved_cards.pop(card_index)

            # Spend the tokens
            with_gold = move_idx % 2  # All odd indices are gold spends
            spent_gems = player.auto_spend(bought_card.cost, with_gold=with_gold)  # type: ignore

            board.return_gems(spent_gems)
            player.get_bought_card(bought_card)

            """Noble visit and end-of-game"""
            self._check_noble_visit(player)

            if player.points >= 15:
                self.victor = True
                player.victor = True

            return
        
        # Reserve card moves
        move_idx -= player.buy_dim
        if move_idx < player.reserve_dim:
            tier = move_idx // 5  # 4 cards + top of deck
            card_index = move_idx % 5

            if card_index < 4:  # Reserve from regular tier
                reserved_card, gold = board.reserve(tier, card_index)
            else:  # Reserve top of deck
                reserved_card, gold = board.reserve_from_deck(tier)

            player.reserved_cards.append(reserved_card)
            if gold[5]:
                discard_if_gt10, _ = player.auto_take(gold)
                board.take_gems(discard_if_gt10)

            return

    def _check_noble_visit(self, player) -> None:
        for index, noble in enumerate(self.board.nobles):
            if noble and np.all(player.cards >= noble.cost):
                self.board.nobles[index] = None
                player.points += 3
    
    def to_state(self) -> np.ndarray:
        cur_player = self.active_player
        enemy_player = self.players[(self.half_turns+1) % 2]

        board_vector = self.board.to_state(cur_player.effective_gems,         # 232
                                          enemy_player.effective_gems)
        hero_vector = self.active_player.to_state()                          # 47
        enemy_vector = enemy_player.to_state()                               # 47

        vector = np.concatenate((board_vector, hero_vector, enemy_vector))   # 326
        return vector.astype(np.float32)
