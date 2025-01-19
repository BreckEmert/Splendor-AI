# Splendor/Environment/Splendor_components/player.py

import copy
import numpy as np
import itertools as it
import tensorflow as tf


class Player:
    def __init__(self, name, model):
        self.name: str = name
        self.model = model
        self.action_dim = 400  # Guessing lol
        # self.state_offset: int = 150  # No longer needed?  action_dim-1 maybe?
        self.reset()
        self._initialize_all_takes()
    
    def reset(self):
        self.gems: np.ndarray = np.zeros(6, dtype=int)  # Gold gem so 6
        self.cards: np.ndarray = np.zeros(5, dtype=int)  # No gold card so 5
        self.reserved_cards: list = []
        self.points: int = 0

        self.card_ids: list = [[[], [], [], [], []], [[], [], [], [], []], 
                               [[], [], [], [], []], [[], [], [], [], []]]
        self.victor: bool = False
        self.move_index: int = 9999  # Set to impossible to avoid confusion

        self.discard_disincentive: float = -0.1

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

        take_2_same = tf.eye(5, dtype=tf.int8)*2

        self.all_takes_2 = tf.concat([take_2_diff, take_2_same], axis=0)

        # Take 1
        self.all_takes_1 = tf.eye(5, dtype=tf.int8)

    def take_or_spend_gems(self, gems_to_change):
        if len(gems_to_change) < 6:  # Pads gold dim if needed
            gems_to_change = np.pad(gems_to_change, (0, 6-len(gems_to_change)))
        self.gems += gems_to_change

        # Validate gem counts
        # assert np.all(self.gems >= 0), f"Illegal player gems: {self.gems}, {gems_to_change}"
        # assert sum(self.gems) <= 10, f"Illegal player gems: {self.gems}, {gems_to_change}"

    def get_bought_card(self, card):
        self.cards[card.gem] += 1
        self.points += card.points
        self.card_ids[card.tier][card.gem].append(card.id)

    def buy_with_gold_loop(self, next_state, move_index, card):
        starting_gems = self.gems.copy()
        chosen_gems = np.zeros(6, dtype=int)
        legal_mask = np.zeros(61, dtype=bool) # Action vector size
        cost = card.cost - self.cards
        cost = np.maximum(cost, 0)
        cost = np.append(cost, 0)
        state = next_state.copy()

        while sum(cost) > 0:
            gems = starting_gems + chosen_gems  # Update the player's gems to a local variable

            # Legal tokens to spend
            legal_mask[10:15] = (gems*cost != 0)[:5]  # Can only spend gems where card cost remains
            legal_mask[60] = True if gems[5] else False  # Enable spending gold as a legal move

            # Predict token to spend
            rl_moves = self.model.get_predictions(state, legal_mask)
            move_index = np.argmax(rl_moves)
            gem_index = move_index-10 if move_index != 60 else 5

            # Remember
            next_state = state.copy()
            next_state[gem_index+self.state_offset] -= 0.25
            memory = [state.copy(), move_index, 1/30, next_state.copy(), 1]
            self.model.remember(memory, legal_mask.copy())

            # Update player in game state
            state = next_state.copy()

            # Propagate move
            chosen_gems[gem_index] -= 1
            cost[gem_index] -= 1

        return chosen_gems

    def _auto_discard(self, legal_takes, n_discards):
        if n_discards:
            net_takes = []
            for take in legal_takes.numpy():
                net_take = self._auto_discard(self.gems[:5], take, n_discards)
                # 1) apply 
                net_takes.append(net_take)
            net_takes = tf.to_tensor(net_takes)  # Don't know how to handle this
        else:
            net_takes = legal_takes

        return net_takes

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

    def get_legal_takes(self, board_gems):
        n_gems = self.gems.sum()
        n_discards = 13 - n_gems
        board_gems = tf.constant(board_gems[:5], dtype=tf.int8)
        legal_action_mask = tf.zeros([self.action_dim], dtype=tf.bool)
        offset = 0  # Offsets updates to legal_action_mask, increasing as we go

        """TAKE 3"""
        # Filter self.all_takes_3 to where the board actually has gems
        board_gt0_mask = tf.cast(board_gems > 0, tf.int8)  # Board > 0 indicator
        takes_ltboard_mask = tf.reduce_all(self.all_takes_3 <= board_gt0_mask, axis=1)
        legal_action_mask = self._scatter_legal_takes(legal_action_mask, takes_ltboard_mask, n_discards, offset)
        offset += tf.shape(takes_ltboard_mask)[0]

        """TAKE 2 - SAME"""
        # Filter self.all_takes_2_same to where the board has at least 4 gems of a color
        board_gt4_mask = tf.greater_equal(board_gems, 4)  # Board >= 4 indicator
        takes_gtboard_mask = tf.reduce_any(self.all_takes_2 <= board_gt4_mask, axis=1)
        legal_action_mask = self._scatter_legal_takes(legal_action_mask, takes_gtboard_mask, n_discards)
        offset += tf.shape(takes_gtboard_mask)[0]

        """TAKE 2 - DIFFERENT"""
        # Filter self.all_takes_2_diff to where the board has any gems
        takes_ltboard_mask = tf.reduce_all(self.all_takes_2_diff <= board_gt0_mask, axis=1)
        legal_action_mask = self._scatter_legal_takes(legal_action_mask, takes_ltboard_mask, n_discards)
        offset += tf.shape(takes_ltboard_mask)[0]

        """TAKE 1"""
        # Filter self.all_takes_1 to where the board has any gems
        takes_ltboard_mask = tf.reduce_all(self.all_takes_1 <= board_gt0_mask, axis=1)
        legal_action_mask = self._scatter_legal_takes(legal_action_mask, takes_ltboard_mask, n_discards)

        """Complete list of legal takes"""
        return legal_action_mask

    def get_legal_moves(self, board):
        # Treat purchased cards as gems
        effective_gems = self.gems.copy()
        effective_gems[:5] += self.cards

        # Will append to this as we find legal moves
        legal_moves = []

        # Buy card
        for tier_index, tier in enumerate(board.cards[:3]):
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

                    if can_afford:
                        legal_moves.append(('buy', (tier_index, card_index)))
                    elif can_afford_with_gold:
                        legal_moves.append(('buy with gold', (tier_index, card_index)))

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

            if can_afford:
                legal_moves.append(('buy reserved', (None, card_index)))
            elif can_afford_with_gold:
                legal_moves.append(('buy reserved with gold', (None, card_index)))
        
        # Take gems
        legal_moves.append(self.get_legal_takes(board.gems))

        # Reserve a card
        if len(self.reserved_cards) < 3:
            for tier_index, tier in enumerate(board.cards[:3]):
                for card_index, card in enumerate(tier):
                    if card:
                        legal_moves.append(('reserve', (tier_index, card_index)))
                if board.deck_mapping[tier_index].cards:
                    legal_moves.append(('reserve top', (tier_index, None)))
        
        return legal_moves

    def legal_to_vector(self, legal_moves):
        legal_mask = np.zeros(61, dtype=bool)
        for move, details in legal_moves:
            tier, card_index = details
            match move:
                case 'take':
                    gem, amount = details # Overriding tier and card_index
                    if amount == 1:
                        legal_mask[gem] = True
                    elif amount == 2:
                        legal_mask[gem+5] = True
                    elif amount == -1:
                        legal_mask[gem+10] = True
                case 'buy':
                    legal_mask[15 + 4*tier + card_index] = True
                case 'buy reserved':
                    legal_mask[27 + card_index] = True
                case 'buy with gold':
                    legal_mask[30 + 4*tier + card_index] = True
                case 'buy reserved with gold':
                    legal_mask[42 + card_index] = True
                case 'reserve':
                    legal_mask[45 + 4*tier + card_index] = True
                case 'reserve top':
                    legal_mask[57 + tier] = True

        return legal_mask

    def vector_to_details(self, state, board, legal_mask, move_index):
        tier = move_index % 15 // 4
        card_index = move_index % 15 % 4

        if move_index < 15:  # Take (includes discarding a gem)
            if move_index < 5 or move_index >= 10: # Take 3
                chosen_gems = self.take_tokens_loop(state, board.gems[:5], move_index)
            else: # Take 2
                # Remember
                gem_index = move_index % 5
                next_state = state.copy()
                next_state[gem_index+self.state_offset] += 0.5
                memory = [state.copy(), move_index, 0, next_state.copy(), 1]
                self.model.remember(memory, legal_mask.copy())

                chosen_gems = np.zeros(6, dtype=int)
                chosen_gems[gem_index] = 2
            
            move = ('take', (chosen_gems, None))

        elif move_index < 45: # Buy
            # Remember
            # ~15/1.3 purchases in a game? y=\frac{2}{15}-\frac{2}{15}\cdot\frac{1.3}{15}x
            # reward = max(3/15-3/15*1.3/15*sum(self.gems), 0.0)
            reserved_card_index = move_index%15 - 12  # First 12 are shop cards, last 3 are reserved cards
            if tier < 3:  # Buy
                points = board.cards[tier][card_index].points
            else:  # Buy reserved
                points = self.reserved_cards[reserved_card_index].points
            reward = min(points, 15-self.points) / 15

            # Check noble visit and end of game
            if self.check_noble_visit(board):
                reward += min(3, 15-self.points) / 15

            if self.points+points >= 15:
                reward += 10
                memory = [state.copy(), move_index, reward, state.copy(), 0]
                self.model.remember(memory, legal_mask.copy())
                self.model.memory[-1].append(legal_mask.copy())
                self.victor = True
                return None

            next_state = state.copy()
            offset = 11 * (4*tier + card_index)
            next_state[offset:offset+11] = board.deck_mapping[tier].peek_vector()
            memory = [state.copy(), move_index, reward, next_state.copy(), 1]
            self.model.remember(memory, legal_mask.copy())
            
            # Buy moves
            if move_index < 27:
                move = ('buy', (tier, card_index))
            elif move_index < 30:
                move = ('buy reserved', (None, reserved_card_index))
            elif move_index < 42:
                card = board.cards[tier][card_index]
                spent_gems = self.buy_with_gold_loop(next_state, move_index, card)
                move = ('buy with gold', ((tier, card_index), spent_gems))
            else:
                card = self.reserved_cards[reserved_card_index]
                spent_gems = self.buy_with_gold_loop(next_state, move_index, card)
                move = ('buy reserved with gold', (card_index, spent_gems))

        else: # < 60 Reserve
            if move_index < 57:
                offset = 11 * (4*tier + card_index)
                move = ('reserve', (tier, card_index))
            else:
                offset = self.state_offset + len(self.reserved_cards)*11
                move = ('reserve top', (move_index-57, None))

            # Remember
            next_state = state.copy()
            next_state[offset:offset+11] = board.deck_mapping[tier].peek_vector()
            reward = 0.0 if sum(self.gems) < 10 else self.discard_disincentive
            memory = [state.copy(), move_index, reward, next_state.copy(), 1]
            self.model.remember(memory, legal_mask.copy())

        return move

    def choose_move(self, board, state):
        legal_moves = self.get_legal_moves(board)
        legal_mask = self.legal_to_vector(legal_moves)
        rl_moves = self.model.get_predictions(state, legal_mask)
        
        self.move_index = np.argmax(rl_moves)
        return self.vector_to_details(state, board, legal_mask, self.move_index)

    def check_noble_visit(self, board):
        for index, noble in enumerate(board.cards[3]):
            if noble and np.all(self.cards >= noble.cost):
                self.points += noble.points
                board.cards[3][index] = None
                return True  # No logic to tie-break, seems too insignificant for training
        return False

    def get_state(self):
        # chosen_move = int(self.move_index)
        # self.move_index = 9999
        return {
            'gems': self.gems.tolist(), 
            'cards': copy.deepcopy(self.card_ids), 
            'reserved_cards': [(card.tier, card.id) for card in self.reserved_cards], 
            'chosen_move': int(self.move_index), 
            'points': int(self.points)
        }

    def to_vector(self):
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
