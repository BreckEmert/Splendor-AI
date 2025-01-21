# Splendor/Environment/Splendor_components/player.py

# import copy
import numpy as np
import itertools as it


class Player:
    def __init__(self, name, model):
        self.name: str = name
        self.model = model
        self.action_dim = 140
        self.reset()
        self._initialize_all_takes()
    
    def reset(self):
        self.gems: np.ndarray = np.zeros(6, dtype=int)  # Gold gem so 6
        self.cards: np.ndarray = np.zeros(5, dtype=int)  # No gold card so 5
        self.reserved_cards: list = []

        self.card_ids: list = [[[] for _ in range(5)] for _ in range(4)]
        self.points: float = 0.0
        self.victor: bool = False

    def _initialize_all_takes(self):
        """Preloads all possible take indices that can 
        be filtered and combined during gameplay.  Avoids 
        a lot of recalculation each turn.
        """
        # Take 3
        indices = list(it.combinations(range(5), 3))
        all_takes = np.zeros((10, 5), dtype=int)
        for index, combo in enumerate(indices):
            all_takes[index, combo] = 1
        self.all_takes_3 = all_takes

        # Take 2 (different)
        indices = list(it.combinations(range(5), 2))
        all_takes = np.zeros((10, 5), dtype=int)
        for index, combo in enumerate(indices):
            all_takes[index, combo] = 1
        self.all_takes_2_diff = all_takes
        
        # Take 2 (same)
        self.all_takes_2_same = np.eye(5, dtype=int)*2

        # Take 1
        self.all_takes_1 = np.eye(5, dtype=int)

    def get_bought_card(self, card):
        """Handles all buying on the player's endexcept for 
        the gems, which is handled by _auto_discard.
        """
        self.cards[card.gem] += 1
        self.points += card.points
        self.card_ids[card.tier][card.gem].append(card.id)

    def _auto_spend_gold(self, card_cost):
        """For now, random spend logic.  Modifies player gems 
        IN PLACE.  Also ENSURE that this and other methods 
        recieve .copy() objects, as this does modify card_cost.
        """
        spent_gems = np.zeros(6, dtype=np.int8)
        card_cost -= self.cards
        card_cost = np.maximum(card_cost, 0)
        
        for index, gem_count in enumerate(card_cost):
            while gem_count > 0:
                # If we have a gem of that color, spend it
                if self.gems[index] > 0:
                    self.gems[index] -= 1
                    spent_gems[index] += 1
                    card_cost[index] -= 1
                    continue
                
                # Otherwise pay with gold
                self.gems[5] -= 1
                spent_gems[5] += 1
                gem_count -= 1

        return spent_gems

    def _auto_discard(self, gems_to_take):
        """For now, random discard logic.  Modifies self.gems
        IN PLACE.  Good logic could be cosine similarity of 
        gross gems with all card costs!!
        """
        player_gems = self.gems[:5]
        n_discards = max(0, -7 + self.gems.sum() + gems_to_take.sum())

        discards = np.zeros(5, dtype=np.int8)
        while discards.sum() < n_discards:
            # Preferred discards that don't obstruct with what we took
            discard_prefs = player_gems * (1-gems_to_take)  # Or a bitwise inversion?
            discard_prefs_mask = np.where(discard_prefs > 0)[0]
            if discard_prefs_mask.size:
                random_choice = np.random.choice(discard_prefs_mask)
                player_gems[random_choice] -= 1
                discards[random_choice] += 1
                continue
            
            # Otherwise discard gems we took
            discard_mask = np.where(player_gems > 0)[0]
            random_choice = np.random.choice(discard_mask)

            player_gems[random_choice] -= 1
            discards[random_choice] += 1

        return discards

    def _get_legal_takes(self, board_gems):
        """For each possible take, there are ||take|| possible
        discards.  Because these are automatically discarded
        there is no combinatorics needed.
        """
        n_gems = self.gems.sum()
        board_gems = board_gems[:5]
        legal_take_mask = np.zeros(95, dtype=bool)

        """TAKE 3"""
        n_discards = max(0, -7+n_gems)
        for index, combo in enumerate(self.all_takes_3):
            if np.all(board_gems >= combo):
                legal_take_mask[4*index + n_discards] = True

        """TAKE 2 (DIFFERENT)"""
        for index, combo in enumerate(self.all_takes_2_diff):
            if np.all(board_gems >= combo):
                legal_take_mask[40 + 3*index + n_discards] = True

        """TAKE 2 (SAME)"""
        n_discards = max(0, n_discards-1)
        for gem_index in range(5):
            if board_gems[gem_index] >= 4:
                legal_take_mask[70 + 3*gem_index + n_discards]

        """TAKE 1"""
        n_discards = max(0, n_discards-1)
        for index, combo in enumerate(self.all_takes_1):
            if np.all(board_gems >= combo):
                legal_take_mask[85 + 2*index + n_discards] = True

        """Complete list of legal takes"""
        return legal_take_mask

    def _get_legal_buys(self, board_cards):
        # Treat purchased cards as gems
        effective_gems = self.gems.copy()
        effective_gems[:5] += self.cards

        # Returned object that we will append to
        legal_buy_mask = []

        # Buy card
        for tier in board_cards:
            for card in tier:
                if card:
                    can_afford = can_afford_with_gold = True
                    gold_needed = 0

                    for gem_index, amount in enumerate(card.cost):
                        if effective_gems[gem_index] < amount:
                            can_afford = False
                            gold_needed += amount - effective_gems[gem_index]
                            if gold_needed > effective_gems[5]:
                                can_afford_with_gold = False
                                break

                    if can_afford_with_gold:
                        legal_buy_mask.extend([True, True])
                    elif can_afford:
                        legal_buy_mask.extend([True, False])
                    else:
                        legal_buy_mask.extend([False, False])

        # Buy a reserved card
        for card_index, card in enumerate(self.reserved_cards):
            can_afford = can_afford_with_gold = True
            gold_needed = 0

            for gem_index, amount in enumerate(card.cost):
                if effective_gems[gem_index] < amount:
                    can_afford = False
                    gold_needed += amount - effective_gems[gem_index]
                    if gold_needed > effective_gems[5]:
                        can_afford_with_gold = False
                        break

            if can_afford_with_gold:
                legal_buy_mask.extend([True, True])
            elif can_afford:
                legal_buy_mask.extend([True, False])
            else:
                legal_buy_mask.extend([False, False])
        
        # Pad buys if less than 3 cards are reserved
        n_reserved = len(self.reserved_cards)
        if n_reserved < 3:
            legal_buy_mask.extend([False, False] * (3-n_reserved))

        length = len(legal_buy_mask)
        assert length == 30, f"legal_buy_mask is length {length}"
        return legal_buy_mask

    def _get_legal_reserves(self, board):
        """This almost exclusively irrelevant to search the board,
        but while the model learns it may actually buy out all of
        the deck, so we have to confirm there are no None.
        """
        if len(self.reserved_cards) < 3:
            legal_reserve_mask = []
            for tier_index, tier in enumerate(board.cards):
                for card in tier:
                    legal_reserve_mask.append(bool(card))
                remaining_deck = board.deck_mapping[tier_index].cards
                legal_reserve_mask.append(bool(remaining_deck))
        else:
            legal_reserve_mask = [False] * 15
        
        length = len(legal_reserve_mask)
        assert length == 15, f"legal_reserve_mask is length {length}"
        return legal_reserve_mask

    def _get_legal_moves(self, board):
        legal_take_mask = self._get_legal_takes(board.gems)
        legal_buy_mask = self._get_legal_buys(board.cards)
        legal_reserve_mask = self._get_legal_reserves(board)
        # print(len(legal_take_mask), len(legal_buy_mask), len(legal_reserve_mask))
        
        return np.concatenate(
            [legal_take_mask, legal_buy_mask, legal_reserve_mask]
        )

    def choose_move(self, board, state):
        legal_mask = self._get_legal_moves(board)
        rl_moves = self.model.get_predictions(state, legal_mask)
        return np.argmax(rl_moves)

    def to_state_vector(self):
        reserved_cards_vector = np.zeros(33)
        for i, card in enumerate(self.reserved_cards):
            reserved_cards_vector[i*11:(i+1)*11] = card.vector

        state_vector = np.concatenate((
            self.gems / 4,  # length 6 (5 gold but 5/4 is ratio with others)
            [sum(self.gems) / 10],  # length 1
            self.cards / 4,  # length 5
            reserved_cards_vector,  # length 11*3 = 33
            [self.points / 15]  # length 1
        ))

        # assert len(state_vector) == 46, "Player state is {len(state_vector)}"
        return state_vector  # length 46
