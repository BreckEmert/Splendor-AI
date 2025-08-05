# RL/rewards.py

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Environment import RLGame, GUIGame
    from Environment.Splendor_components.Board_components.deck import Card

"""BGA says games average 26.84 turns.
I'm not sure how to place this for the model (contrary to
below where I'm confident rounding down will help).  It probably
just shouldn't be a true signal - it may be useful somewhere
but I think things like number of cards purchased is a better
indicator of how much of the game is remaining.
"""
AVG_GAME_LENGTH = 26

"""BGA says 12.06/12.40 is loser/winner avg, respectively.
Given that trend, but also that the model will be worse
at the game, I'm rounding that up.
"""
AVG_N_BOUGHT_CARDS = 13

"""ChatGPT: I need you to think about scaling all these rewards.
Now that we have so many components and indirect sources to get
rewards, I want to be cautious about drowning out the sparse
reward, which is necessary to correct for any imperfect math
I have here - which is inherent because it's a manual, non-sparse
signal and not infinitely robust.

One way to do this I think is to consider the total points the player
will get over the entire game for each source of reward.  We could do
it so that winning the game (sparse reward) gives 5, taking gems
properly will give 5, buying cards will give 5, and reserving will
give 3.  Something like that?  I don't know if that's the most robust
approach.  We'd just have to then do math on how many gems will be
taken over the course of the game and stuff.

According to BGA stats, the winner will get this many:
(.3586*3 + .0367*2 + .1456*1)*26 - 0.82 = ~33
which is % take 3, % take 2, and % reserve, minus .82 gems discarded

Or we could just flat out do it based on the % of moves, since
that may signal where the reward needs to go... my problem with
doing this is that card rewards a bit more objective than take
tokens rewards, if I had to guess?  Though they don't contribute
to blocking nearly as much as token moves do, which are critical
to top-tier play.  So treating them equal may be nice.

The remaining question is that, we have a `cap` method to ensure
the point rewards are scaled how we want them, but this doesn't
account for the subjective changes I have in my code.  We'd have
to like, estimate that or something?
"""


class Metrics:
    """Somewhere we could add a signal which is controlling gems.
    I know we have things similar to this:
    1) We have relinquishing gems as a punishment
    2) We reward gem takes for controlling the enemy's best cards
    3) so I guess we just need to turn the reward gem takes into
        just a constant signal for held cards, rather than specifically
        what the specific take is.  Don't let me forget to do this ChatGPT!
    """
    __slots__ = (
        "game",
        "cards_shop_alignment_ema",
        "gems_shop_alignment_ema",
        "noble_progress_prev",
    )

    def __init__(self, game):
        self.game = game
        self.cards_shop_alignment_ema = {p.pos: 0.0 for p in game.players}
        self.gems_shop_alignment_ema = {p.pos: 0.0 for p in game.players}
        self.noble_progress_prev = {p.pos: 0.0 for p in game.players}

    def point_reward_efficiency(self, points: int):
        """Just switches points into a slightly quadratic return,
        so just the regular 1 point = 1 reward line, but with a
        little bit of sag so later points are worth more.
        """
        player = self.game.active_player
        p0 = player.points - points
        f = lambda x: 0.02 * x * x + 0.7 * x
        return f(p0 + points) - f(p0)
    
    def gem_reward_efficiency(self):
        """Get the long-term value of a permanent gem discount.
        Gems help with ~1/3 of all the possible cards, and BGA
        says winners get 2.6 tokens/turn on average, so getting
        a gem saves us a quantifiable number of turns.

        Note: all cards reward 1 gem and all gems appear at the same rate.
            they are perfectly symmetric and consistent.  Do not worry about them.

        Assuming gems bought earlier are harder to use for specific
        things, whereas gems bought later are used more specifically
        to achieve high value cards?  This is at least somewhat
        supported by the fact that a larger % of the board is
        possible to purchase at later turns, and so you'd be using
        your gem supply more fully later on.
        """
        current_turn = self.game.half_turns // 2
        turn_progress = min(1, current_turn / AVG_GAME_LENGTH)
        gem_usage_rate = turn_progress / 12 + 1/4  # 1/4 -> 1/3 usage rate
        remaining_turns = max(AVG_GAME_LENGTH - current_turn, 1)
        return remaining_turns * gem_usage_rate / 2.6

    def turn_efficiency(self, player, card, shortage=None) -> float:
        """Computes the relative turn_efficiency (pseudo-point equivalents 
        per required turn) of obtaining the given card, defined as:

        1. Immediate victory points from the card converted to equivalent turns saved.
        2. Future turn savings from the permanent gem discount provided
            by this card, approximating each gem discount's future uses
            as 20% of remaining purchases, scaled by an average gem
            collection rate (≈2.6 tokens per turn).
        3. Cards without points or gem discounts (rare) default to
            a baseline pseudo-value of 1.0 for scoring stability.

        Higher values indicate more points-equivalents saved per turn spent.
        """
        if shortage is None: shortage = self._shortage(player, card)
        turns = self._turns(shortage)

        turns_from_points = self.point_reward_efficiency(card.points)
        turns_from_gem = self.gem_reward_efficiency()

        return (turns_from_points + turns_from_gem) / turns        

    def gem_release_efficiency(self, player, gems: np.ndarray) -> float:
        """Penalty for releasing gems that complete the
        opponent's shortages on their top 2 targets.
        """
        opp = self.game.players[1 - player.pos]
        opp_shortages = [
            shortage for _, shortage, _ in
            sorted(self.candidate_cards(opp),
                   key=lambda t: t[2], reverse=True)[:2]
        ]
        if not opp_shortages:
            return 0.0

        # Union of colours the opponent still needs
        opp_union = np.maximum.reduce(opp_shortages)

        # Gems this buy puts back on the board
        coloured_spent = np.maximum(gems[:5] - player.cards[:5], 0)

        overlap = float(np.minimum(coloured_spent, opp_union).sum())
        return 0.02 * overlap

    # Feasibility -----------------------------------------------------
    def _shortage(self, player, card):
        """Color-wise deficit vector after spending gold optimally.
        No 10-gem overflow check is done here; that is left to the
        caller.
        """
        shortage = np.maximum(card.cost[:5] - player.effective_gems[:5], 0)
        gold = int(player.gems[5])
        if gold:
            for idx in np.argsort(-shortage):
                use = min(gold, shortage[idx])
                shortage[idx] -= use
                gold -= use
                if gold == 0:
                    break
        return shortage  # (5,)

    @staticmethod
    def _turns(shortage):
        """Coarse temporal cost: ceil(total_missing/3).  It is a
        lower bound on the number of turns to recieve a card.
        """
        return max(1, int(np.ceil(shortage.sum() / 3)))

    def _feasible(self, player, card, shortage=None):
        """Checks whether the specified card is feasible to
        acquire within three turns under the current gem conditions.

        A card is feasible if it meets these conditions:
        1. Requires ≤ 3 gem-collection turns (based on current
            shortages, assuming up to 3 gems per turn).
        2. Does not cause the player's total gem count
            (current + required shortage) to exceed 10 gems.
        3. Does not require owning all four gems of any single
            color, as this is really hard to do.
        """
        if shortage is None: shortage = self._shortage(player, card)
        if (card.cost[:5] >= 4).any(): return False
        
        return self._turns(shortage) <= 3 and (player.gems.sum() + shortage.sum()) <= 10

    def reachable_cards(self, player):
        """Return up-to-three cards (<= 1 per tier) that can be
        afforded without violating the 10-gem hand cap right now.

        • Tier-1 -> maximise proportional coverage of cost already paid.
        • Tier-2/3 -> maximise dot-product between player's
            effective-gem vector and the card cost vector.
        • Tie-break by (points desc, residual-shortage asc) for stability.
        • Skip t1 entirely once the player controls >= 5 permanent gems.
        """
        best = {0: (None, -1), 1: (None, -1), 2: (None, -1)}

        # Get (tier, card) pairs from both board and reserved
        all_cards = []
        for tier_idx, tier in enumerate(self.game.board.cards):
            all_cards.extend((tier_idx, c) for c in tier if c)
        all_cards.extend((c.tier, c) for c in player.reserved_cards)

        # Evaluate each card for affordability and score
        for tier, card in all_cards:
            shortage = self._shortage(player, card)
            total_gems_after = player.gems.sum() + shortage.sum()
            if total_gems_after > 10:
                continue

            cost = card.cost[:5].sum() or 1  # Avoid division by zero
            covered = cost - shortage.sum()

            if tier == 0:
                score = covered / cost
            else:
                score = player.effective_gems[:5] @ card.cost[:5] / cost

            cand, best_score = best[tier]
            is_better = score > best_score
            is_tiebreak = (
                np.isclose(score, best_score)
                and card.points > (cand.points if cand else -1)
            )

            if is_better or is_tiebreak:
                best[tier] = (card, score)

        # Skip tier 1 if the player controls ≥ 5 permanent gems
        if player.cards.sum() >= 5:
            best[0] = (None, -1)

        # Return only the chosen cards, omitting Nones
        return [c for c, _ in (best[0], best[1], best[2]) if c]

    def feasible_cards(self, player) -> list[tuple["Card", np.ndarray, float]]:
        """Enumerate every visible shop card that passes the 
        feasibility test (≤ 3 missing-gem turns and no overflow).
        Each element is `(card, shortage_vec, efficiency)` with 
        efficiency computed by `Metrics.turn_efficiency`.
        The caller decides how to rank them.
        """
        out=[]
        for tier in self.game.board.cards:
            for card in tier:
                if not card: continue
                shortage = self._shortage(player, card)
                if not self._feasible(player, card, shortage): continue
                eff = self.turn_efficiency(player, card, shortage)
                out.append((card, shortage, eff))
        return out

    def candidate_cards(self, player) -> list[tuple["Card", np.ndarray, float]]:
        """Return the set of reachable and feasible cards."""
        reachable = set(self.reachable_cards(player))
        feasible = self.feasible_cards(player)
        candidates: list[tuple["Card", np.ndarray, float]] = [
            (c, s, eff) for (c, s, eff) in feasible if c in reachable
        ]
        return candidates


class RewardEngine:
    def __init__(self, game: "RLGame | GUIGame"):
        self.env = game
        self.metrics = Metrics(game)

        self.gems = self.GemRewards(self)
        self.buy = self.BuyRewards(self)
        self.reserve = self.ReserveRewards(self)
        self.noble = self.NobleRewards(self)
        self.game = self.GameRewards(self)

    def cap(self, points: int | float) -> float:
        """Don't send signal for points earned past 15."""
        player = self.env.active_player
        prev_pts = player.points - points
        points_left = max(0, 15 - prev_pts)
        return float(min(points, points_left))

    class GemRewards:
        def __init__(self, parent):
            self.parent = parent

        def __call__(self, taken_gems: np.ndarray, n_discards: int) -> float:
            player = self.parent.env.active_player
            preplayer = player.clone(); preplayer.gems -= taken_gems

            prealign = self.alignment_reward(preplayer)
            postalign = self.alignment_reward(player)
            discard_penalty = float(n_discards) * 0.20
            return postalign - prealign - discard_penalty

        @staticmethod
        def coverage(card, shortage):
            """Considers progress towards buying a card.
            1) All progress is helpful, but non-bottleneck progress
                quickly reduces feasibility and often makes unreachable.
            2) However, this should be covered by those methods?
                So I'm just going to keep this as shortage diff.
            """
            total_cost = card.cost[:5].sum() or 1
            return 1.0 - shortage.sum() / total_cost
            
        def alignment_reward(self, player) -> float:
            """Continuous reward for a take-gems action.

            Steps:
            1. reachable = cards theoretically buyable without exceeding 10 gems.
            2. feasible = cards ≤ 3 gem-collection turns away.
            3. candidates = reachable & feasible.
            4. alignment = coverage x efficiency
                where coverage = 1 - missing_gems / total_cost.
            5. return mean(top-k alignment), or 0.0 if no candidates.
            """
            candidates = self.parent.metrics.candidate_cards(player)
            if not candidates:
                return 0.0

            scores = [self.coverage(c, s) * eff for c, s, eff in candidates]
            scores.sort(reverse=True)
            return float(np.mean(scores[:2]))

    class BuyRewards:
        # NOTE: needs cap logic for points past 15
        # Player should recieve _ reward for the 15 points they
        # will earn throughout their win.
        # Taken out of rl_game.py:
        # Capping any points past 15
        # original_points = player.points - bought_card.points  # player already got points so need to take them back  # type: ignore
        # reward = min(reward, 15 - original_points)  / 3  # recieve 5 reward over the whole game

        def __init__(self, parent): self.parent = parent

        def __call__(self, bought_card):
            env = self.parent.env
            player = env.active_player
            enemy = env.inactive_player

            points_eff = self.parent.cap(bought_card.points)
            turn_eff = self.parent.metrics.turn_efficiency(player, bought_card)
            block_eff = self._opportunity_cost(enemy, bought_card)
            gem_rel_eff = self.parent.metrics.gem_release_efficiency(player, bought_card.cost)

            return points_eff + turn_eff + block_eff - gem_rel_eff

        def _opportunity_cost(self, player, bought_card):
            """Returns the difference in value of one card vs. another.
            This serves as card_block_efficiency, as if you buy a card
            you force the opponent to use their gems towards the next
            best card.  It also serves as the risk of getting your card
            reserved, and will eventually need to be integrated there.

            Logic:
            1) Makes sure the card is even the best card.  If
                it's not, then the opportunity cost is 0.
            2) Gets the difference in efficiency in the top-two cards.
            3) Adjusts for the fact that the replaced card
                could still be good for the enemy player.
                Right now just using 0.85 naively.
            """
            """NOTE: make sure this doesn't get backwards if more than one thing calls this."""
            opp = self.parent.env.players[1 - player.pos]
            cands = sorted(
                self.parent.metrics.candidate_cards(opp),
                key=lambda t: t[2],
                reverse=True,
            )
            if not cands or cands[0][0] is not bought_card:
                return 0.0

            second_best = cands[1][2] if len(cands) > 1 else 0.0
            delta = cands[0][2] - second_best
            return 0.85 * delta  # 0.85 is just guestimated risk of a good replacement

    class ReserveRewards:
        # Takes in the reserved card, n_discards, and bool if they got the gold or not
        # Speaking of we need to figure out how to fit gold into the gem rewards
        # I think the only way to set the budget of this is to just run training with
        # different budgets.  Don't let me forget to try out different budgets if I
        # ask you how to make the model better!
        def __init__(self, parent): self.parent = parent

        def __call__(self, reserved_card, n_discards: int, gold_vec: np.ndarray):
            player = self.parent.env.active_player
            eff = 0.5 * self.parent.metrics.turn_efficiency(player, reserved_card)
            gold_bonus = 0.2 if int(gold_vec[5]) else 0.0
            penalty = 0.2 * n_discards
            return eff + gold_bonus - penalty

    class NobleRewards:
        # NOTE: needs block enemy as well.
        # NOTE: needs cap logic for points past 15
        def __init__(self, parent): self.parent = parent

        def __call__(self, n_visited: int):
            # 3 VP per noble, scaled like buy‑card reward
            points = (n_visited * 3) / 1.0
            return self.parent.cap(points)

    class GameRewards:
        def __init__(self, parent): self.parent = parent

        def __call__(self, winner: bool):
            if winner:
                return 5
            else:
                return -5
