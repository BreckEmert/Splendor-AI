# Splendor/Environment/Splendor_components/player.py

import copy
import numpy as np
import itertools as it
import tensorflow as tf


class Player:
    def __init__(self, name, model):
        self.name: str = name
        self.model = model
        self.action_dim = 165
        self.reset()
        self._initialize_all_takes()
    
    def reset(self):
        self.gems: np.ndarray = np.zeros(6, dtype=int)  # Gold gem so 6
        self.cards: np.ndarray = np.zeros(5, dtype=int)  # No gold card so 5
        self.reserved_cards: list = []

        self.card_ids: list = [[[] for _ in range(5)] for _ in range(4)]
        self.victor: bool = False

    def _initialize_all_takes(self):
        """Preloads all possible take indices that can 
        be filtered and combined during gameplay.  Avoids 
        a lot of recalculation each turn.
        """
        # Take 3
        take_3 = list(it.combinations(range(5), 3))
        take_3 = tf.constant(take_3, dtype=tf.int32)
        take_3 = tf.one_hot(take_3, depth=5, axis=-1, dtype=tf.int32)
        take_3 = tf.reduce_sum(take_3, axis=1)
        self.all_takes_3 = take_3

        # Take 2
        take_2_diff = list(it.combinations(range(5), 2))
        take_2_diff = tf.constant(take_2_diff, dtype=tf.int32)
        take_2_diff = tf.one_hot(take_2_diff, depth=5, axis=-1, dtype=tf.int32)
        take_2_diff = tf.reduce_sum(take_2_diff, axis=1)
        self.all_takes_2_diff = take_2_diff

        self.all_takes_2_same = tf.eye(5, dtype=tf.int8)*2

        # Take 1
        self.all_takes_1 = tf.eye(5, dtype=tf.int8)

    def get_bought_card(self, card):
        self.cards[card.gem] += 1
        self.points += card.points
        self.card_ids[card.tier][card.gem].append(card.id)

    def _auto_spend(self, card_cost):
        """For now, random spend logic.  Modifies player gems 
        IN PLACE.  Also ENSURE that this and other methods 
        recieve .copy() objects, as this does modify card_cost.
        """
        spent_gems = np.zeros(6, dtype=np.int8)
        card_cost -= self.cards
        card_cost = np.max(card_cost, 0)
        
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
        """For now, random discard logic.  Good logic could be 
        cosine similarity of gross gems with all card costs!!
        """
        player_gems = self.gems[:5].copy()
        n_discards = max(0, 7 - self.gems.sum() - gems_to_take.sum())

        discards = np.zeros(5, dtype=np.int8)
        while discards.sum() < n_discards:
            # Preferred discards that don't obstruct with what we took
            discard_prefs = player_gems * (1-gems_to_take)  # Or a bitwise inversion?
            discard_prefs_mask = np.where(discard_prefs > 0)[0]
            if discard_prefs_mask.size > 0:
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

    def _scatter_legal_takes(self, legal_action_mask, board_mask, n_discards, offset):
        """Updates the legal action mask that will filter model 
        actions based on what's legal to take from the board
        """
        # Get the corresponding all_takes indices to the legal moves we found
        # (need a stable vector for the model)
        legal_indices = tf.where(board_mask)
        legal_indices = tf.reshape(legal_indices, [-1])  # Flatten

        # Now we can actually update the legal_action_mask
        action_indices = offset + legal_indices*n_discards + n_discards  # broadcast using stride=4
        action_indices = tf.expand_dims(action_indices, axis=1)  # expand back for scattering
        action_updates = tf.ones_like(action_indices, dtype=tf.bool)  # True where the indices are
        legal_action_mask = tf.tensor_scatter_nd_update(legal_action_mask, action_indices, action_updates)

        return legal_action_mask

    def _get_legal_takes(self, board_gems):
        n_gems = self.gems.sum()
        board_gems = tf.constant(board_gems[:5], dtype=tf.int8)
        legal_take_mask = tf.zeros([self.action_dim], dtype=tf.bool)
        offset = 0  # Offsets updates to legal_take_mask, increasing as we go

        """TAKE 3"""
        n_discards = max(0, -7+n_gems)
        # Filter self.all_takes_3 to where the board actually has gems
        board_gt0_mask = tf.cast(board_gems > 0, tf.int8)  # Board > 0 indicator
        takes_ltboard_mask = tf.reduce_all(self.all_takes_3 <= board_gt0_mask, axis=1)
        legal_take_mask = self._scatter_legal_takes(legal_take_mask, takes_ltboard_mask, n_discards, offset)
        offset += tf.shape(takes_ltboard_mask)[0]

        """TAKE 2"""
        n_discards = max(0, n_discards-1)
        """TAKE 2 - SAME"""
        # Filter self.all_takes_2_same to where the board has at least 4 gems of a color
        board_gt4_mask = tf.greater_equal(board_gems, 4)  # Board >= 4 indicator
        takes_gtboard_mask = tf.reduce_any(self.all_takes_2 <= board_gt4_mask, axis=1)  # WANT TO TEST WETHER ANY IS NEEDED....
        legal_take_mask = self._scatter_legal_takes(legal_take_mask, takes_gtboard_mask, n_discards, offset)
        offset += tf.shape(takes_gtboard_mask)[0]

        """TAKE 2 - DIFFERENT"""
        # Filter self.all_takes_2_diff to where the board has any gems
        takes_ltboard_mask = tf.reduce_all(self.all_takes_2_diff <= board_gt0_mask, axis=1)
        legal_take_mask = self._scatter_legal_takes(legal_take_mask, takes_ltboard_mask, n_discards, offset)
        offset += tf.shape(takes_ltboard_mask)[0]

        """TAKE 1"""
        n_discards = max(0, n_discards-1)
        # Filter self.all_takes_1 to where the board has any gems
        takes_ltboard_mask = tf.reduce_all(self.all_takes_1 <= board_gt0_mask, axis=1)
        legal_take_mask = self._scatter_legal_takes(legal_take_mask, takes_ltboard_mask, n_discards, offset)

        """Complete list of legal takes"""
        return legal_take_mask

    def _get_legal_buys(self, board_cards):
        # Treat purchased cards as gems
        effective_gems = self.gems.copy()
        effective_gems[:5] += self.cards

        # Returned object that we will append to
        legal_buy_mask = []

        # Buy card
        for tier_index, tier in enumerate(board_cards[:3]):
            for card_index, card in enumerate(tier):
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

        length = len(legal_buy_mask)
        assert length == 30, f"legal_buy_mask is length {length}"
        return legal_buy_mask

    def _get_legal_reserves(self, board):
        # Return object that we'll append to
        legal_reserve_mask = []

        if len(self.reserved_cards) < 3:
            for tier_index, tier in enumerate(board.cards[:3]):
                for card in tier:
                    legal_reserve_mask.append(bool(card))
                remaining_deck = board.deck_mapping[tier_index].cards
                legal_reserve_mask.append(bool(remaining_deck))
        
        length = len(legal_reserve_mask)
        assert length == 12, f"legal_reserve_mask is length {length}"
        return legal_reserve_mask

    def _get_legal_moves(self, board):
        # Take gems
        legal_take_mask = self._get_legal_takes(board.gems)
        legal_take_mask = tf.constant(legal_take_mask, dtype=tf.bool)

        # Buy card
        legal_buy_mask = self._get_legal_buys(board.cards)
        legal_buy_mask = tf.constant(legal_buy_mask, dtype=tf.bool)

        # Reserve card
        legal_reserve_mask = self._get_legal_reserves(board)
        legal_reserve_mask = tf.constant(legal_reserve_mask, dtype=tf.bool)
        
        legal_action_mask = tf.concat(
            [legal_take_mask, legal_buy_mask, legal_reserve_mask], 
            axis=0
        )
        print("Length of legal moves: ", len(legal_action_mask))
        return legal_action_mask

    def choose_move(self, board, state):
        legal_mask = self._get_legal_moves(board)
        rl_moves = self.model.get_predictions(state, legal_mask)
        return np.argmax(rl_moves)  # Things like this we need to confirm tf compatibility.  I'd like to eventually use only tf

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
