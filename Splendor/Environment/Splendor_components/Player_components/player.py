# Splendor/Environment/Splendor_components/player.py

import numpy as np


class Player:
    def __init__(self, name, rl_model):
        self.name: str = name
        self.rl_model = rl_model
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

    def get_bought_card(self, card):
        self.cards[card.gem] += 1
        self.points += card.points
        self.card_ids[card.tier][card.gem].append(card.id)

    def choose_discard(self, game_state, move_index=None):
        if not move_index:
            # Set legal mask to only legal discards
            legal_mask = np.zeros(61, dtype=int)
            legal_mask[10:15] = game_state[self.state_offset:self.state_offset+5] > 0 # Based on player's gems

            # Call the model to choose a discard
            rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
            move_index = np.argmax(rl_moves)

        gem_index = move_index - 10
        discard = np.zeros(5, dtype=int)
        discard[gem_index] = 1

        # Remember
        next_state = game_state.copy()
        next_state[gem_index+self.state_offset] -= 0.25
        self.rl_model.remember(game_state, move_index, -1/15, next_state, False)

        # Update player in the game state (not board anymore)
        game_state = next_state.copy() # Do we need to do .copy()

        return discard, game_state

    def take_tokens_loop(self, board_gems, game_state, move_index=None):
        total_gems = sum(self.gems)
        chosen_gems = np.zeros(5, dtype=int)

        takes_remaining = 3
        legal_selection = board_gems.copy()
        state = game_state.copy()

        while takes_remaining and np.any(legal_selection):
            # Discard if required
            if total_gems == 10:
                if takes_remaining == 3 and move_index >= 10:
                    discard = np.zeros(5, dtype=int)
                    discard[move_index-10] = -1
                    state[move_index-10 + self.state_offset] -= 0.25
                else:
                    discard, state = self.choose_discard(state)

                # Implement move
                total_gems -= 1
                chosen_gems -= discard
                legal_selection += discard # Taking a second of the same gem is now legal
            
            if takes_remaining == 3 and move_index < 5:
                gem_index = move_index.copy()
            else:
                # Takes are only legal where gems exist and not previously taken
                legal_mask = np.zeros(61, dtype=int)
                legal_mask[:5] = (legal_selection > 0).astype(int)

                # Call the model to choose a take
                rl_moves = self.rl_model.get_predictions(state, legal_mask)
                gem_index = np.argmax(rl_moves)

            # Remember
            next_state = state.copy()
            next_state[gem_index+self.state_offset] += 0.25
            reward = -0.04 if takes_remaining == 3 else 0
            self.rl_model.remember(state, gem_index, reward, next_state, False)
            state = next_state.copy()

            # Implement move
            total_gems += 1
            takes_remaining -= 1
            chosen_gems[gem_index] += 1
            legal_selection[gem_index] *= 0 # Taking this gem again is now illegal unless previously discarded

        return chosen_gems

    def buy_with_gold_loop(self, game_state, move_index, card):
        chosen_gems = np.zeros(6, dtype=int)
        legal_mask = np.zeros(61, dtype=int) # Action vector size
        cost = card.cost - self.cards
        cost = np.maximum(cost, 0)
        cost = np.append(cost, 0)
        state = game_state.copy()

        while sum(cost) > 0:
            gems = self.gems + chosen_gems # Update the player's gems to a local variable

            # Legal tokens to spend
            legal_mask[10:15] = (gems*cost != 0).astype(int)[:5] # Can only spend gems where card cost remains
            legal_mask[60] = 1 if gems[5] else 0 # Enable spending gold as a legal move

            # Predict token to spend
            rl_moves = self.rl_model.get_predictions(state, legal_mask)
            move_index = np.argmax(rl_moves)
            gem_index = move_index-10 if move_index != 60 else 5

            # Remember
            next_state = state.copy()
            next_state[gem_index+self.state_offset] -= 0.25
            self.rl_model.remember(state, move_index, 0, next_state, False)

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

        # Take gems
        if sum(self.gems < 10):
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
        
        return legal_moves
    
    def legal_to_vector(self, legal_moves):
        legal_mask = np.zeros(61, dtype=int)
        for move, details in legal_moves:
            tier, card_index = details
            match move:
                case 'take':
                    gem, amount = details # Overriding tier and card_index
                    if amount == 1:
                        legal_mask[gem] = 1
                    elif amount == 2:
                        legal_mask[gem+5] = 1
                    elif amount == -1:
                        legal_mask[gem+10] = 1
                case 'buy':
                    legal_mask[15 + 4*tier + card_index] = 1
                case 'buy reserved':
                    legal_mask[27 + card_index] = 1
                case 'buy with gold':
                    legal_mask[30 + 4*tier + card_index] = 1
                case 'buy reserved with gold':
                    legal_mask[42 + card_index] = 1
                case 'reserve':
                    legal_mask[45 + 4*tier + card_index] = 1
                case 'reserve top':
                    legal_mask[57 + tier] = 1

        return legal_mask
    
    def vector_to_details(self, board, game_state, move_index):
        tier = move_index % 15 // 4
        card_index = move_index % 15 % 4

        if move_index < 15:  # Take (includes discarding a gem)
            if move_index < 5 or move_index >= 10: # Take 3
                chosen_gems = self.take_tokens_loop(board.gems[:5], game_state, move_index)
            else: # Take 2
                # Remember
                gem_index = move_index % 5
                next_state = game_state.copy()
                next_state[gem_index+self.state_offset] += 0.5
                self.rl_model.remember(game_state, move_index, -0.04, next_state, False)

                chosen_gems = np.zeros(6, dtype=int)
                chosen_gems[gem_index] = 2
            
            move = ('take', (chosen_gems, None))

        elif move_index < 45: # Buy
            # Remember
            # ~15/1.3 purchases in a game? y=\frac{2}{15}-\frac{2}{15}\cdot\frac{1.3}{15}x
            reward = max(2/15-2/15*1.3/15*sum(self.gems), 0)
            reserved_card_index = move_index-27 if move_index<30 else move_index-42
            if tier < 3: # Buy
                points = board.cards[tier][card_index].points
                reward += min(points, 15-self.points) / 15
            else: # Buy reserved
                points = self.reserved_cards[reserved_card_index].points
                reward += min(points, 15-self.points) / 15

            # Check noble visit and end of game
            if self.check_noble_visit(board):
                reward += min(3, 15-self.points) / 15

            if self.points+points >= 15:
                reward += 1
                self.rl_model.remember(game_state, move_index, reward, game_state, True)
                self.victor = True
                return None

            next_state = game_state.copy()
            offset = 11 * (4*tier + card_index)
            next_state[offset:offset+11] = board.deck_mapping[tier].peek_vector()
            self.rl_model.remember(game_state, move_index, reward, next_state, False)
            
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
            next_state = game_state.copy()
            next_state[offset:offset+11] = board.deck_mapping[tier].peek_vector()
            self.rl_model.remember(game_state, move_index, -0.04, next_state, False)
        
        return move
    
    def choose_move(self, board, game_state):
        legal_moves = self.get_legal_moves(board)
        legal_mask = self.legal_to_vector(legal_moves)
        rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
        
        self.move_index = np.argmax(rl_moves)
        self.chosen_move = self.vector_to_details(board, game_state, self.move_index)
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
            self.cards/4, # length 5
            reserved_cards_vector, # length 11*3 = 33
            [self.points/15] # length 1
        ))

        return state_vector