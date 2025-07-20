# Splendor/Environment/Splendor_components/player.py

import numpy as np
import itertools as it


class Player:
    def __init__(self, name, model):
        self.name: str = name
        self.model = model
        self.reset()
        self._initialize_all_takes()
        self._initialize_dimensions()
    
    def reset(self):
        self.gems: np.ndarray = np.zeros(6, dtype=int)  # Gold gem so 6
        self.cards: np.ndarray = np.zeros(6, dtype=int)  # 5th dim unused
        # self.cards = np.full(6, 1, dtype=int)  # DELETE THIS LATER, UNCOMMENT CHECK NOBLES
        # self.cards[5] = 0  # DELETE THIS LATER
        self.reserved_cards: list = []

        self.card_ids: list = [[] for _ in range(5)]
        self.points: float = 0.0
        self.victor: bool = False

    @property
    def effective_gems(self):
        return self.gems + self.cards

    def _initialize_all_takes(self):
        """Preloads all possible take indices that can 
        be filtered and combined during gameplay.  Avoids 
        a lot of recalculation each turn.
        """
        # Take 3
        indices = list(it.combinations(range(5), 3))
        all_takes = np.zeros((10, 6), dtype=int)
        for index, combo in enumerate(indices):
            all_takes[index, combo] = 1
        self.all_takes_3 = all_takes

        # Take 2 (different)
        indices = list(it.combinations(range(5), 2))
        all_takes = np.zeros((10, 6), dtype=int)
        for index, combo in enumerate(indices):
            all_takes[index, combo] = 1
        self.all_takes_2_diff = all_takes
        
        # Take 2 (same)
        self.all_takes_2_same = np.zeros((5, 6), dtype=int)
        self.all_takes_2_same[np.arange(5), np.arange(5)] = 2

        # Take 1
        self.all_takes_1 = np.zeros((5, 6), dtype=int)
        self.all_takes_1[np.arange(5), np.arange(5)] = 1

    def _initialize_dimensions(self):
        """Get indices used in other parts of 
        the code to avoid some hardcoding.
        """
        self.take_dim = (
            len(self.all_takes_3) * 4 +       # 10 * 4
            len(self.all_takes_2_same) * 3 +  # 5 * 3
            len(self.all_takes_2_diff) * 3 +  # 10 * 3
            len(self.all_takes_1) * 2 +       # 5 * 2
            1                                 # discard
        )

        self.buy_dim = (
            3 *  # 3 tiers
            4 *  # 4 cards per tier
            2 +  # Buy with and without gold
            3 *  # 3 reserve slots
            2    # Buy with and without gold
        )

        self.reserve_dim = (
            3 *  # 3 tiers
            5    # 4 cards per tier + top of deck
        )

        self.action_dim = self.take_dim + self.buy_dim + self.reserve_dim

    def get_bought_card(self, card):
        """Handles all buying on the player's end except
        for the gems, which is handled by _auto_discard.
        """
        self.cards[card.gem] += 1
        self.points += card.points
        self.card_ids[card.gem].append((card.tier, card.id))

    def auto_spend(self, raw_cost, with_gold):
        """For now, random spend logic.  Modifies player gems 
        IN PLACE.  Also ENSURE that this and other methods 
        recieve .copy() objects, as this does modify card_cost.
        raw_cost is guaranteed to be affordable, so doing
        spent_gems[5] = card_cost.sum() is completely fine.
        """
        # Discount the cost with our purchased cards
        card_cost = np.maximum(raw_cost - self.cards, 0)

        # Pay what we can with regular gems
        spent_gems = np.minimum(self.gems, card_cost)

        # Pay the rest with gold
        if with_gold:
            spent_gems[5] = card_cost.sum() - spent_gems.sum()

        # Return spent_gems so the board can update as well
        self.gems -= spent_gems
        return spent_gems

    def auto_take(self, gems_to_take):
        """Add gems_to_take to self.gems, while accounting for 
        discards by trying to discard gems that weren't taken.
        This avoids combinatorial discard space and does not 
        significantly limit gameplay.
        
        Ideally would have logic to competetively discard, 
        (sometimes I have wanted to discard gold so that the
        gems are frozen for the enemy player) but for now that 
        is out of the learning scope.
        """
        # Check if this is a reserve reward (gold only)
        gold_only = gems_to_take[5] > 0

        # Add gems to self.gems and handle reserve gold reward
        self.gems += gems_to_take  # Always add gems
        self_gems = self.gems[:5]
        
        # Now discard if required
        n_discards = max(0, self.gems.sum() - 10)
        discards = np.zeros(6, dtype=int)

        while discards.sum() < n_discards:
            # Try to prefer discarding gems we didn't take
            discard_prefs = np.where((self_gems > 0) & (gems_to_take[:5] == 0))[0]
            if discard_prefs.size > 0:
                color = np.random.choice(discard_prefs)
            else:
                discardable = np.where(self_gems > 0)[0]
                color = np.random.choice(discardable)

            # Discard 1 gem from that color
            self_gems[color] -= 1
            discards[color] += 1

        # Gems we were supposed to take minus what we had to disard
        net_take = gems_to_take - discards

        # Add back on the gold
        if gold_only:
            net_take[5] = 1

        return net_take, n_discards

    def _get_legal_takes(self, board_gems):
        """For each possible take, there are ||take|| possible
        discards.  Because these are automatically discarded
        there is no combinatorics needed.
        """
        legal_take_mask = np.zeros(96, dtype=bool)

        """TAKE 3"""
        n_discards = max(0, -7+self.gems.sum())
        for index, combo in enumerate(self.all_takes_3):
            if np.all(board_gems >= combo):
                legal_take_mask[4*index + n_discards] = True

        """TAKE 2 (SAME)"""
        n_discards = max(0, n_discards-1)
        for gem_index in range(5):
            if board_gems[gem_index] >= 4:
                legal_take_mask[40 + 3*gem_index + n_discards] = True

        """TAKE 2 (DIFFERENT)"""
        for index, combo in enumerate(self.all_takes_2_diff):
            if np.all(board_gems >= combo):
                legal_take_mask[55 + 3*index + n_discards] = True

        """TAKE 1"""
        n_discards = max(0, n_discards-1)
        for index, combo in enumerate(self.all_takes_1):
            if np.all(board_gems >= combo):
                legal_take_mask[85 + 2*index + n_discards] = True

        """Backup discard"""
        if self.gems.sum() == 10:
            legal_take_mask[-1] = True

        return legal_take_mask

    def _get_legal_buys(self, board_cards):
        # Treat purchased cards as gems
        effective_gems = self.gems.copy()
        effective_gems += self.cards  # Replace this with @effective_gems

        # Returned object that we will append to
        legal_buy_mask = []

        # Buy card
        for tier_index in range(3):
            for card_index in range(4):
                if board_cards[tier_index][card_index]:
                    card = board_cards[tier_index][card_index]

                    # Calculate costs (complicated because of gold)
                    gem_difference = card.cost - effective_gems
                    gold_needed = np.maximum(gem_difference, 0).sum()

                    # Check if we can afford it
                    can_afford = (gold_needed == 0)
                    can_afford_with_gold = (gold_needed <= effective_gems[5])
                    
                    # Append results to the mask
                    legal_buy_mask.extend([can_afford, can_afford_with_gold])
                else:
                    legal_buy_mask.extend([False, False])

        # Buy a reserved card
        for reserve_index in range(3):
            if reserve_index < len(self.reserved_cards):
                card = self.reserved_cards[reserve_index]

                # Calculate costs (complicated because of gold)
                gem_difference = card.cost - effective_gems
                gold_needed = np.maximum(gem_difference, 0).sum()

                # Check if we can afford it
                can_afford = (gold_needed == 0)
                can_afford_with_gold = (gold_needed <= effective_gems[5])
                
                # Append results to the mask
                legal_buy_mask.extend([can_afford, can_afford_with_gold])
            else:
                legal_buy_mask.extend([False, False])

        return legal_buy_mask

    def _get_legal_reserves(self, board):
        """This will almost never happen after a bit of training,
        but while the model learns it may actually buy out all of
        the deck, so we have to confirm there are no None.
        """
        if len(self.reserved_cards) < 3:
            legal_reserve_mask = []
            for tier_index, tier in enumerate(board.cards):
                for card in tier:
                    legal_reserve_mask.append(bool(card))
                remaining_deck = board.decks[tier_index].cards
                legal_reserve_mask.append(bool(remaining_deck))
        else:
            legal_reserve_mask = [False] * 15
        
        return legal_reserve_mask

    def get_legal_moves(self, board):
        legal_take_mask = self._get_legal_takes(board.gems)
        legal_buy_mask = self._get_legal_buys(board.cards)
        legal_reserve_mask = self._get_legal_reserves(board)
        
        legal_mask = np.concatenate(
            [legal_take_mask, legal_buy_mask, legal_reserve_mask]
        )

        return legal_mask

    def choose_move(self, board, state):
        legal_mask = self.get_legal_moves(board)
        rl_moves = self.model.get_predictions(state, legal_mask)
        return np.argmax(rl_moves)

    def to_state(self):
        """Some overwriting occurs because of the 6-dim vector
        standardization, so note that not all [5] have meaning.
        """
        state_vector = np.zeros(47, dtype=np.float32)

        # Gems (6+1 = 7)
        state_vector[:6] = self.gems / 4.0
        state_vector[5] /= 1.25  # Normalize to 5
        state_vector[6] = self.gems.sum() / 10.0

        # Cards (5+1 = 6)
        state_vector[7:13] = self.cards  # Note there are no gold cards
        state_vector[12] = self.cards.sum() / 10  # so we overwrite [12]

        # Reserved cards (11*3 = 33)
        start = 13
        for card in self.reserved_cards:
            state_vector[start:start+11] = card.to_vector(self.effective_gems)
            start += 11

        # Points
        state_vector[-1] = self.points / 15

        return state_vector  # length 47
