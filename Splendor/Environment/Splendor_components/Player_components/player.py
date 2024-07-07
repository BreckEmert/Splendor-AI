# Splendor/Environment/Splendor_components/player.py

import numpy as np


class Player:
    def __init__(self, name, model):
        self.name: str = name
        self.model = model
        self.state_offset: int = 150
    
    def reset(self):
        self.gems: np.ndarray = np.zeros(6, dtype=int)
        self.cards: np.ndarray = np.zeros(5, dtype=int)
        self.reserved_cards: list = []
        self.points: int = 0

        self.card_ids: list = [[[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []]]
        self.victor: bool = False

    def take_or_spend_gems(self, gems_to_change):
        if len(gems_to_change) < 6:
            gems_to_change = np.pad(gems_to_change, (0, 6-len(gems_to_change)))
        self.gems += gems_to_change
        assert np.all(self.gems >= 0) and sum(self.gems) <= 10, f"Illegal player gems {self.gems}, {gems_to_change}"

    def get_bought_card(self, card):
        self.cards[card.gem] += 1
        self.points += card.points
        self.card_ids[card.tier][card.gem].append(card.id)

    def choose_discard(self, state, player_gems, progress=0, reward=0.0, move_index=None):
        # Set legal mask to only legal discards
        legal_mask = np.zeros(61, dtype=bool)
        legal_mask[10:15] = player_gems[:5] > 0

        if not move_index:
            # Call the model to choose a discard
            rl_moves = self.model.get_predictions(state, legal_mask)
            move_index = np.argmax(rl_moves)

        gem_index = move_index - 10
        discard = np.zeros(5, dtype=int)
        discard[gem_index] = -1

        # Remember
        next_state = state.copy()
        next_state[gem_index+self.state_offset] -= 0.25
        state[196] = 0.2*progress # 0.2 * (moves remaining+1), indicating progression through loop
        self.model.remember([state.copy(), move_index, reward, next_state.copy(), 1], legal_mask.copy())

        # Update player in the game state (not board anymore)
        state = next_state.copy() # Do we need to do .copy()

        return discard, state
    
    def choose_take(self, state, available_gems, progress, reward=0.0, take_index=None):
        # Set legal mask to only legal takes
        legal_mask = np.zeros(61, dtype=bool)
        legal_mask[:5] = available_gems > 0

        if not take_index:
            # Call the model to choose a take
            rl_moves = self.model.get_predictions(state, legal_mask)
            take_index = np.argmax(rl_moves)
        
        take = np.zeros(5, dtype=int)
        take[take_index] = 1

        # Remember
        next_state = state.copy()
        next_state[take_index+self.state_offset] += 0.25
        state[196] = progress # 0.2 * (moves remaining+1), indicating progression through loop
        self.model.remember([state.copy(), take_index, reward, next_state.copy(), 1], legal_mask.copy())

        return take, next_state

    def take_tokens_loop(self, state, board_gems, move_index=None):
        player_gems = self.gems[:5].copy()
        total_gems = sum(self.gems)
        board_gems = (board_gems>0).astype(int)

        state = state.copy()

        takes = min(3, sum(board_gems))
        discards = total_gems - 7
        discard_reward = -1/30*discards
        chosen_gems = np.zeros(5, dtype=int)

        # Perform the move that was initially chosen
        if move_index:
            if move_index < 5:
                chosen_gem, state = self.choose_take(state, board_gems, 0.6, 0.0, move_index)
                takes -= 1
            else:
                chosen_gem, state = self.choose_discard(state, self.gems, 0.6, discard_reward, move_index)
                discards -= 1
            chosen_gems += chosen_gem

        # Choose necessary discards
        while discards > 0:
            discard, state = self.choose_discard(state, player_gems+chosen_gems, progress=discards, reward=discard_reward)
            # discard_reward = 0.0
            chosen_gems += discard
            discards -= 1
        
        # Choose necessary takes
        while takes > 0:
            take, state = self.choose_take(state, board_gems-chosen_gems, progress=takes, reward=-1/30)
            chosen_gems += take
            takes -= 1

        return chosen_gems

    def buy_with_gold_loop(self, next_state, move_index, card):
        starting_gems = self.gems.copy()
        chosen_gems = np.zeros(6, dtype=int)
        legal_mask = np.zeros(61, dtype=bool) # Action vector size
        cost = card.cost - self.cards
        cost = np.maximum(cost, 0)
        cost = np.append(cost, 0)
        state = next_state.copy()

        while sum(cost) > 0:
            gems = starting_gems + chosen_gems # Update the player's gems to a local variable

            # Legal tokens to spend
            legal_mask[10:15] = (gems*cost != 0)[:5] # Can only spend gems where card cost remains
            legal_mask[60] = True if gems[5] else False # Enable spending gold as a legal move

            # Predict token to spend
            rl_moves = self.model.get_predictions(state, legal_mask)
            move_index = np.argmax(rl_moves)
            gem_index = move_index-10 if move_index != 60 else 5

            # Remember
            next_state = state.copy()
            next_state[gem_index+self.state_offset] -= 0.25
            self.model.remember([state.copy(), move_index, 1/30, next_state.copy(), 1], legal_mask.copy())

            # Update player in game state
            state = next_state.copy()

            # Propagate move
            chosen_gems[gem_index] -= 1
            cost[gem_index] -= 1

        return chosen_gems

    def get_legal_moves(self, board):
        effective_gems = self.gems.copy()
        effective_gems[:5] += self.cards
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

        # Buy reserved card
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

        # if len(legal_moves) > 0:
        #     return legal_moves
        
        # Take gems
        if sum(self.gems) < 10:
            for gem, amount in enumerate(board.gems[:5]):
                if amount:
                    legal_moves.append(('take', (gem, 1)))
                    if sum(self.gems) <= 8: # We actually can take 2 if more than 8 but will need discard
                        if amount >= 4:
                            legal_moves.append(('take', (gem, 2)))
        else: # Discard first, per take_tokens_loop logic
            for gem, amount in enumerate(self.gems[:5]):
                if amount:
                    legal_moves.append(('take', (gem, -1)))

        # Reserve card
        if len(self.reserved_cards) < 3:
            for tier_index, tier in enumerate(board.cards[:3]):
                for card_index, card in enumerate(tier):
                    if card:
                        legal_moves.append(('reserve', (tier_index, card_index)))
                if board.deck_mapping[tier_index].cards:
                    legal_moves.append(('reserve top', (tier_index, None))) # Setting card_index to None because it shouldn't be needed
        
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
                self.model.remember([state.copy(), move_index, -1/30, next_state.copy(), 1], legal_mask.copy())

                chosen_gems = np.zeros(6, dtype=int)
                chosen_gems[gem_index] = 2
            
            move = ('take', (chosen_gems, None))

        elif move_index < 45: # Buy
            # Remember
            # ~15/1.3 purchases in a game? y=\frac{2}{15}-\frac{2}{15}\cdot\frac{1.3}{15}x
            # reward = max(3/15-3/15*1.3/15*sum(self.gems), 0.0)
            reserved_card_index = move_index-27 if move_index<30 else move_index-42
            if tier < 3: # Buy
                points = board.cards[tier][card_index].points
            else: # Buy reserved
                points = self.reserved_cards[reserved_card_index].points
            reward = min(points, 15-self.points) / 30

            # Check noble visit and end of game
            if self.check_noble_visit(board):
                reward += min(3, 15-self.points) / 15

            if self.points+points >= 15:
                reward += 10
                self.model.remember([state.copy(), move_index, reward, state.copy(), 0], legal_mask.copy())
                self.model.memory[-1].append(legal_mask.copy())
                self.victor = True
                return None

            next_state = state.copy()
            offset = 11 * (4*tier + card_index)
            next_state[offset:offset+11] = board.deck_mapping[tier].peek_vector()
            self.model.remember([state.copy(), move_index, reward, next_state.copy(), 1], legal_mask.copy())
            
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
            reward = 0.0 if sum(self.gems) < 10 else 0.0
            self.model.remember([state.copy(), move_index, reward, next_state.copy(), 1], legal_mask.copy())
        
        return move
    
    def choose_move(self, board, state):
        legal_moves = self.get_legal_moves(board)
        legal_mask = self.legal_to_vector(legal_moves)
        rl_moves = self.model.get_predictions(state, legal_mask)
        
        self.move_index = np.argmax(rl_moves)
        self.chosen_move = self.vector_to_details(state, board, legal_mask, self.move_index)
        return self.chosen_move
    
    def check_noble_visit(self, board):
        for index, noble in enumerate(board.cards[3]):
            if noble and np.all(self.cards >= noble.cost):
                self.points += noble.points
                board.cards[3][index] = None
                return True # No logic to tie-break, seems too insignificant for training
        return False

    def get_state(self):
        return {
            'gems': self.gems.tolist(), 
            'cards': self.card_ids, 
            'reserved_cards': [(card.tier, card.id) for card in self.reserved_cards], 
            'points': self.points
        }

    def to_vector(self):
        reserved_cards_vector = np.zeros(33)
        for i, card in enumerate(self.reserved_cards):
            reserved_cards_vector[i*11:(i+1)*11] = card.vector

        state_vector = np.concatenate((
            self.gems/4, # length 6, there are actually 5 gold but 0 is all that matters
            [sum(self.gems)/10], # length 1
            self.cards/4, # length 5
            reserved_cards_vector, # length 11*3 = 33
            [self.points/15] # length 1
        ))

        return state_vector # length 46