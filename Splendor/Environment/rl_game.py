# Splendor/Environment/rl_game.py

import numpy as np

from Environment import Board, Player
from RL import BasicRewardEngine, SparseRewardEngine  # type: ignore


class RLGame:
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
        self.move_idx: int = 0
        self.victor: bool = False
    
    @property
    def active_player(self):
        return self.players[self.half_turns % 2]
    
    @property
    def inactive_player(self):
        return self.players[(self.half_turns + 1) % 2]

    def turn(self):
        state = self.to_state()
        move_idx = self.active_player.choose_move(self.board, state)
        # assert isinstance(move_idx, int), f"move_idx is {type(move_idx)}"
        self.move_idx = move_idx  # type: ignore
        self.rewards._cache.clear()
        reward = self.apply_move(move_idx)  # type: ignore
        reward -= self.rewards.constant_penalty
        self.half_turns += 1

        # assert np.all(self.board.gems >= 0), "Board gems lt0"
        # assert np.all(self.board.gems[:5] <= 4), "Board gems gt4"
        # assert self.active_player.gems.sum() >= 0, "Player gems lt0"
        # assert self.active_player.gems.sum() <= 10, "Player gems gt10"

        # Remember
        next_state = self.to_state()
        legal_mask = self.active_player.get_legal_moves(self.board)
        sarsld = [state, move_idx, reward, 
                  next_state, legal_mask, 
                  self.victor]
        self.model.remember(sarsld)

    def apply_move(self, move_idx: int) -> float:
        """Deeply sorry for the magic numbers approach."""
        player, board = self.active_player, self.board

        # Take gems moves
        if move_idx < player.take_dim:
            if move_idx < 40: # all_takes_3; 10 * 4discards
                gems_to_take = player.all_takes_3[move_idx // 4]
            elif move_idx < 55: # all_takes_2_same; 5 * 3discards
                gems_to_take = player.all_takes_2_same[(move_idx-40) // 3]
            elif move_idx < 85: # all_takes_2_diff; 10 * 3discards
                gems_to_take = player.all_takes_2_diff[(move_idx-55) // 3]
            elif move_idx < 95:  # all_takes_1; 5 * 2discards
                gems_to_take = player.all_takes_1[(move_idx-85) // 2]
            else:  # All else is illegal, discard
                # I don't think this pathway is possible to activate.
                legal_discards = np.where(player.gems > 0)[0]
                discard_idx = np.random.choice(legal_discards)
                player.gems[discard_idx] -= 1
                board.gems[discard_idx] += 1
                return -0.2

            taken_gems, discards = player.auto_take(gems_to_take)
            board.take_gems(taken_gems)

            return self.rewards.gems(taken_gems, discards)

        # Buy card moves
        move_idx -= player.take_dim
        if move_idx < player.buy_dim:
            if move_idx < 24:  # 12 cards * w&w/o gold
                idx = move_idx // 2
                bought_card = board.take_card(idx//4, idx%4)  # Tier, card idx
            else:  # Buy reserved, 3 cards * w&w/o gold
                card_index = (move_idx-24) // 2
                bought_card = player.reserved_cards.pop(card_index)
            # assert bought_card is not None, "bought_card is none inside apply_move buy card moves"

            # Player spends the tokens
            with_gold = bool(move_idx % 2)  # All odd indices are gold spends
            spent_gems = player.auto_spend(bought_card.cost, with_gold=with_gold)  # type: ignore
            board.return_gems(spent_gems)
            
            player.get_bought_card(bought_card)

            """Noble visit and end-of-game"""
            # Base reward value
            reward = self.rewards.buy(bought_card)
            reward += self.rewards.noble(self._check_noble_visit())

            if player.points >= 15:
                self.victor = True
                player.victor = True
                reward += self.rewards.game(winner=True)
                self.model.memory[-1][2] += self.rewards.game(winner=False)  # Loser reward (we haven't appended memory yet so -1 is still the loser)
                self.model.memory[-1][5] = True  # Mark loser's memory as done
            
            return reward
        
        # Reserve card moves
        move_idx -= player.buy_dim
        if move_idx < player.reserve_dim:
            tier = move_idx // 5  # 4 cards + top of deck
            card_index = move_idx % 5

            if card_index < 4:  # Reserve from regular tier
                reserved_card, gold = board.reserve(tier, card_index)
            else:  # Reserve top
                reserved_card, gold = board.reserve_from_deck(tier)

            discards = np.zeros(6, dtype=int)
            player.reserved_cards.append(reserved_card)
            if gold[5]:
                discard_if_gt10, discards = player.auto_take(gold)
                board.take_gems(discard_if_gt10)

            return self.rewards.reserve(reserved_card, gold, discards)
        
        raise ValueError(f"Illegal move_idx {move_idx}")

    def _check_noble_visit(self):
        player = self.active_player
        visited = 0
        for index, noble in enumerate(self.board.nobles):
            if noble and np.all(player.cards >= noble.cost):
                self.board.nobles[index] = None
                player.points += 3
                visited += 1
        return visited
    
    def to_state(self):
        cur_player = self.active_player
        enemy_player = self.players[(self.half_turns+1) % 2]

        board_vector = self.board.to_state(cur_player.effective_gems,       # 232
                                           enemy_player.effective_gems)
        cur_vector = cur_player.to_state()                                  # 47
        enemy_vector = enemy_player.to_state()                              # 47

        # Tempo (can-buy-now) and points-if-bought, both perspectives (60)
        shop_cards = [c for tier in self.board.cards for c in tier]
        pad = [None] * 3
        cur_cards = shop_cards + (cur_player.reserved_cards + pad)[:3]
        enemy_cards = shop_cards + (enemy_player.reserved_cards + pad)[:3]

        vector = np.concatenate((board_vector, cur_vector, enemy_vector, cur_player.tempo_and_winrate(cur_cards), enemy_player.tempo_and_winrate(enemy_cards)))  # 386
        return vector.astype(np.float32)
