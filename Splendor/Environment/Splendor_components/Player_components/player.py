# Splendor/Environment/Splendor_components/player.py

from itertools import combinations
import numpy as np


class Player:
    def __init__(self, name, strategy, strategy_strength, rl_model, turn_order_index):
        self.name: str = name
        self.turn_order_index = turn_order_index # Unused
        self.state_offset = 156 + 45*turn_order_index

        self.gems: np.ndarray = np.zeros(6, dtype=int)
        self.cards: np.ndarray = np.zeros(5, dtype=int)
        self.reserved_cards: list = []
        self.points: int = 0

        self.card_ids = [[], [], [], []]
        self.rl_model = rl_model
        self.entered_loop = False
        self.victor = False
        #self.strategy: strategy = strategy
        #self.strategy_strength: int = strategy_strength
    
    def take_or_spend_gems(self, gems_to_change):
        self.gems += np.pad(gems_to_change, (0, 6-len(gems_to_change)))

    def get_bought_card(self, card):
        self.cards[card.gem] += 1
        self.points += card.points
        self.card_ids[card.tier].append(card.id)

    def take_tokens_loop(self, game_state):
        total_gems = sum(self.gems)
        chosen_gems = np.zeros(6, dtype=int)

        takes_remaining = 3
        required_discards = max(0, total_gems - 7)

        while takes_remaining:
            # Discard if required
            if total_gems == 10:
                # Set legal mask to only legal discards
                legal_mask = np.zeros(61, dtype=int)
                legal_mask[10:15] = game_state[self.state_offset:self.state_offset+5].astype(bool).astype(int) # Based on player's gems

                # Call the model to choose a discard
                rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
                move_index = np.argmax(rl_moves)
                gem_index = move_index - 10

                next_state = game_state.copy()
                next_state[gem_index] += 1  # Update board's gems
                next_state[gem_index + self.state_offset] -= 1  # Update player's gems
                reward = -1 if takes_remaining == 3 else 0
                self.rl_model.remember(game_state, move_index, reward, next_state, 0)

                # Implement move
                game_state = next_state
                required_discards -= 1
                chosen_gems[gem_index] -= 1 # Update for apply_move()?
            
            # Set legal mask to only legal takes
            legal_mask = [0] * 61
            legal_mask[:5] = game_state[:5].astype(bool).astype(int) # Based on board's gems

            # Call the model to choose a take
            rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
            move_index = np.argmax(rl_moves)

            next_state = game_state.copy()
            next_state[move_index] += 1 # Update board's gems
            next_state[move_index+self.state_offset] += 1 # Update player's gems
            reward = -1 if takes_remaining == 3 else 0
            self.rl_model.remember(game_state, move_index, reward, next_state, 0)
            
            # Implement move
            game_state = next_state
            takes_remaining -= 1
            chosen_gems[move_index] += 1

        return chosen_gems

    def buy_with_gold_loop(self, game_state, move_index, card):
        # Model remembers selecting the move and the reward first
        reward = card.points
        self.rl_model.remember(game_state, move_index, reward, game_state, 0) # Same state for now

        # Need to confirm if its ok to be modifying game_state locally - that we don't need it elsewhere.
        # For now, we'll just use the same five action slots as discarding, since neither are chosen moves but are just required for certain moves
        # This is nice because these nodes shouldn't have been punished and remain in a similar reward space
        chosen_gems = np.zeros(6, dtype=int)
        legal_mask = np.zeros(61, dtype=int) # Action vector size
        cost = np.append(card.cost, 0)
        while sum(cost) > 0:
            # Enable spending gold as a legal move
            legal_mask[10:15] = (self.gems >= cost).astype(int)[:5]
            legal_mask[60] = 1 if game_state[6+self.state_offset] else 0

            rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
            move_index = np.argmax(rl_moves)
            gem_index = 5 if move_index == 60 else move_index-10

            # Remember
            next_state = game_state.copy()
            next_state[gem_index] += 1 # Update board's gems
            next_state[gem_index+self.state_offset] -= 1 # Update player's gems
            self.rl_model.remember(game_state, move_index, 0, next_state, 0)

            # Propagate move
            chosen_gems[gem_index] -= 1
            cost[gem_index] -= 1
            game_state[gem_index] += 1 # Board gains the gem
            game_state[gem_index+self.state_offset] -= 1 # Player spends the gem

        return chosen_gems

    def get_legal_moves(self, board):
        legal_moves = []

        # Take gems
        for gem, amount in enumerate(self.gems[:5]):
            if amount > 0:
                legal_moves.append(('take', (gem, 1))) # This is also overriden
                if amount >= 4:
                    legal_moves.append(('take', (gem, 2)))
        if not legal_moves:
            legal_moves.append(('take', (0, 1))) # Random of the 5 moves because right now it gets overriden

        # Reserve card
        if len(self.reserved_cards) < 3:
            for tier_index, tier in enumerate(board.cards[:3]):
                for card_index, card in enumerate(tier):
                    if card:
                        legal_moves.append(('reserve', (tier_index, card_index)))
                if board.deck_mapping[tier_index]:
                    legal_moves.append(('reserve top', (tier_index, None))) # Setting card_index to None because it shouldn't be needed

        # Buy card
        for tier_index, tier in enumerate(board.cards[:3]):
            for card_index, card in enumerate(tier):
                if card:
                    can_afford = can_afford_with_gold = True
                    gold_needed = 0

                    for gem_index, amount in enumerate(card.cost):
                        if self.gems[gem_index] < amount:
                            can_afford = False
                            gold_needed += amount - self.gems[gem_index]
                            if gold_needed > self.gems[5]:
                                can_afford_with_gold = False
                                break

                    if can_afford_with_gold:
                        legal_moves.append(('buy with gold', (tier_index, card_index)))
                        if can_afford:
                            legal_moves.append(('buy', (tier_index, card_index)))

        # Buy reserved card
        for card_index, card in enumerate(self.reserved_cards):
            can_afford = can_afford_with_gold = True
            gold_needed = 0

            for gem_index, amount in enumerate(card.cost):
                if self.gems[gem_index] < amount:
                    can_afford = False
                    gold_needed += amount - self.gems[gem_index]
                    if gold_needed > self.gems[5]:
                        can_afford_with_gold = False
                        break

            if can_afford_with_gold:
                legal_moves.append(('buy reserved with gold', (None, card_index)))
                if can_afford:
                    legal_moves.append(('buy reserved', (None, card_index)))
        
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
                    if gem > 5:
                        print("take is off-index", gem)
                case 'buy':
                    if 15 + 4*tier + card_index >= 45:
                        print("buy is off-index", tier, card_index)
                    legal_mask[15 + 4*tier + card_index] = 1
                case 'buy reserved':
                    if 27 + card_index > 30:
                        print("buy reserved is off-index", card_index)
                    legal_mask[27 + card_index] = 1
                case 'buy with gold':
                    if 30 + 4*tier + card_index > 42:
                        print("buy with gold is off-index", tier, card_index)
                    legal_mask[30 + 4*tier + card_index] = 1
                case 'buy reserved with gold':
                    if 42 + card_index > 45:
                        print("buy reserved with gold is off-index", card_index)
                    legal_mask[42 + card_index] = 1
                case 'reserve':
                    if 45 + 4*tier + card_index > 57:
                        print("reserve is off-index", tier, card_index)
                    legal_mask[45 + 4*tier + card_index] = 1
                case 'reserve top':
                    if 57 + tier > 60:
                        print("reserve top is off-index", tier)
                    legal_mask[57 + tier] = 1

        return legal_mask
    
    def vector_to_details(self, move_index):
        tier = move_index % 15 // 5
        card_index = tier % 4

        if move_index < 15:  # Take (includes discarding a gem)
            gem_index = move_index % 5
            gems_to_take = np.zeros(6, dtype=int)

            if move_index < 5:
                gems_to_take[gem_index] = 1
                move = ('take', (gems_to_take, None))
            elif move_index < 10:
                gems_to_take[gem_index] = 2
                move = ('take', (gems_to_take, None))
            else:
                gems_to_take[gem_index] = -1
                move = ('take', (gems_to_take, None))
        elif move_index == 60:
            move = ('take', ([0, 0, 0, 0, 0, 1], None))

        elif move_index < 45: # Buy
            if move_index < 27:
                move = ('buy', (tier, card_index))
            elif move_index < 30:
                move = ('buy reserved', (None, move_index-27))
            elif move_index < 42:
                dummy_gems = 1 # dummy
                move = ('buy with gold', ((tier, card_index), dummy_gems))
            else:
                dummy_gems = 1 # dummy
                move = ('buy reserved with gold', ((None, move_index-42), dummy_gems))

        elif move_index < 60: # Reserve
            if move_index < 57:
                move = ('reserve', (tier, card_index))
            elif move_index < 60:
                move = ('reserve top', (move_index-57, None))
        
        return move
    
    def choose_move(self, board, game_state):
        legal_moves = self.get_legal_moves(board)
        legal_mask = self.legal_to_vector(legal_moves)
        rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
        #strategic_moves = self.strategy.strategize(game_state, rl_moves, self.strategy_strength)
        
        self.move_index = move_index = np.argmax(rl_moves)
        
        # If the move takes a single token call take_tokens_loop
        if move_index < 5:
            self.entered_loop = True
            chosen_move = ('take', (self.take_tokens_loop(game_state), None)) # Appending None for apply_move() format
        elif 30 <= move_index < 45:
            self.entered_loop = True # We'll remember() the move in the loop
            
            if move_index < 42:
                tier = move_index % 15 // 5
                card_index = tier % 4
                card = board.cards[tier][card_index]
                spent_gems = self.buy_with_gold_loop(game_state, move_index, card)
                chosen_move = ('buy with gold', ((tier, card_index), spent_gems))
            else:
                card_index = move_index - 42
                card = self.reserved_cards[card_index]
                spent_gems = self.buy_with_gold_loop(game_state, move_index, card)
                chosen_move = ('buy reserved with gold', (card_index, spent_gems))
        else:
            chosen_move = self.vector_to_details(move_index)

        return chosen_move
    
    def get_state(self):
        return {
            'gems': self.gems.tolist(), 
            'cards': self.card_ids, 
            'reserved_cards': [(card.tier, card.id) for card in self.reserved_cards], 
            'points': self.points
        }

    def to_vector(self):
        reserved_cards_vector = []
        for card in self.reserved_cards:
            reserved_cards_vector.extend(card.vector)
        reserved_cards_vector.extend([0] * 11*(3-len(self.reserved_cards)))

        state_vector = np.concatenate((
            self.gems, # length 6
            self.cards, # length 5
            reserved_cards_vector, # length 11*3 = 33
            [self.points] # length 1
        ))

        assert len(state_vector) == 45, f"Player state vector is not 45, but {len(state_vector)}"
        return state_vector