# Splendor/RL/rewards.py

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Environment.Splendor_components.Board_components.deck import Card


Z6 = np.zeros(6, dtype=int)

"""BGA says games average 26.84 turns.
I'm not sure how to place this for the model (contrary to
below where I'm confident rounding down will help).  It probably
just shouldn't be a true signal - it may be useful somewhere
but I think things like number of cards purchased is a better
indicator of how much of the game is remaining.
"""
AVG_GAME_LENGTH = 26  # Note this is WHOLE turns

"""BGA says 12.06/12.40 is loser/winner avg, respectively.
Given that trend, but also that the model will be worse
at the game, I'm rounding that up.
"""
AVG_N_BOUGHT_CARDS = 13

"""Need to think about scaling all these rewards.
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

According to BGA stats, the winner will get this many gems:
(.3586*3 + .0367*2 + .1456*1)*26 - 0.82 = ~33
which is % take 3, % take 2, and % reserve, minus .82 gems discarded.
Directly from the averages it says 34.04

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
    def __init__(self, game):
        self.game = game

    def point_reward_efficiency(self, player, points: int) -> float:
        """Just switches points into a slightly quadratic return,
        so just the regular 1 point = 1 reward line, but with a
        little bit of sag so later points are worth more.
        """
        capped_points = player.cap(points)
        f = lambda x: 0.02 * x * x + 0.7 * x
        return f(player.points + capped_points) - f(player.points)
    
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

    def turn_efficiency(self, player, card) -> float:
        """Computes the relative pseudo-point equivalents per
        required turn of obtaining the given card, defined as:

        1. Immediate victory points from the card converted to equivalent turns saved.
        2. Local turn savings from the permanent gem discount provided
            by this card, looking at what's on the board.
        3. Future turn savings from the permanent gem discount provided
            by this card, approximating each gem discount's future uses
            as 20% of remaining purchases, scaled by an average gem
            collection rate (≈2.6 tokens per turn).
        4. Cards without points default to a baseline pseudo-value
            of 1.0 for scoring stability.

        Higher values indicate more points-equivalents saved per turn spent.
        """
        shortage = self.shortage(player, card)
        turns = self.turns(shortage)

        turns_from_points = self.point_reward_efficiency(player, card.points)
        turns_from_gem_local = self.reachability_gain(player, card.gem_one_hot)
        turns_from_gem_longterm = self.gem_reward_efficiency() * 0.5  # TUNABLE

        return (turns_from_points + turns_from_gem_local + turns_from_gem_longterm) / turns        

    # Feasibility -----------------------------------------------------
    def shortage(self, player, card, extra_gems: np.ndarray | None = None):
        """Color-wise deficit vector after spending gold optimally.
        No 10-gem overflow check is done here; that is left to the caller.
        """
        if extra_gems is None: extra_gems = Z6
        eff = player.effective_gems[:5].copy() + extra_gems[:5]
        shortage = np.maximum(card.cost[:5] - eff[:5], 0)
        gold = int(player.gems[5] + extra_gems[5])
        if gold:
            for idx in np.argsort(-shortage):
                use = min(gold, shortage[idx])
                shortage[idx] -= use
                gold -= use
                if gold == 0: break
        return shortage

    # AI GENERATED FUNCTION:
    def _net_take_for_card(self, player, delta, card):
        g0 = player.gems.astype(int)
        g  = g0.copy() + np.asarray(delta, dtype=int)
        # Greedy: drop tokens whose removal least increases shortage for this card
        while g.sum() > 10:
            best_i, best_up = None, None
            pre = self.shortage(player, card, g - g0).sum()
            for i in range(6):
                if g[i] <= 0: continue
                g[i] -= 1
                up = self.shortage(player, card, g - g0).sum() - pre
                g[i] += 1
                if best_up is None or up < best_up:
                    best_up, best_i = up, i
            g[best_i] -= 1
        return g - g0  # discard-aware net delta

    def coverage_gain(self, player, card, gems) -> float:
        """Returns progress towards buying a card given additional gems."""
        pre_short = self.shortage(player, card)
        net = self._net_take_for_card(player, gems, card)
        post_short = self.shortage(player, card, net)
        total_cost = float(card.cost[:5].sum() or 1)
        gain = (pre_short.sum() - post_short.sum()) / total_cost
        return max(0.0, float(gain))
        
    def turns(self, shortage):
        """Lower bound on turns required to purchase a card:
            - colored need at three per turn
            - reserving one gold at a time for bank shortfall
        """
        need = shortage[:5]
        bank = self.game.board.gems[:5]
        forced_gold = np.maximum(need - bank, 0).sum()
        coloured_need = need.sum() - forced_gold
        n_gem_moves = np.ceil(coloured_need / 3.0) + forced_gold
        return 1 if n_gem_moves <= 0 else int(n_gem_moves + 1)  # Have to spend a turn buying

    def _feasibility(self, player, card) -> float:
        """Downweight a card the more it requires all four gems."""
        need_after_gold = np.maximum(card.cost[:5] - player.cards[:5] - 1, 0)
        mx = need_after_gold.max()
        return 0.7 if mx >= 4 else (0.9 if mx == 3 else 1.0)

    def _reachable(self, player, card) -> bool:
        """Return if the card can be afforded, given enough gems,
        without violating the 10-gem hand cap.  Does allow for
        discarding irrelevant gems.
        """
        g = player.gems.astype(int)
        if g.sum() <= 10:
            # Don't use cap to gate reachability when we're at/under cap.
            # (Affordability and bank constraints are handled elsewhere.)
            return True

        # When over-cap: keep only what can contribute to this card.
        need = np.maximum(card.cost[:5] - player.cards[:5], 0)
        keep_colored = int(np.minimum(g[:5], need).sum())
        shortfall_after_keep = int(max(0, need.sum() - keep_colored))
        keep_gold = int(min(g[5], shortfall_after_keep))

        min_kept = keep_colored + keep_gold
        return min_kept <= 10

    def reachable_cards(self, player) -> list["Card"]:
        """Plural wrapper for _reachable."""
        out: list["Card"] = []
        for tier in self.game.board.cards:
            for card in tier:
                if card and self._reachable(player, card):
                    out.append(card)
        for card in player.reserved_cards:
            if card and self._reachable(player, card):
                out.append(card)
        return out

    def top_reachable(self, player, k: int = 2) -> list[tuple["Card", float]]:
        """Filters reachable_cards to just valuable cards."""
        cache = self.game.rewards._cache
        key = ("top", id(player),
                      player.gems.tobytes(),
                      player.cards.tobytes(),
                      k)
        if key in cache:
            return cache[key]

        cards = self.reachable_cards(player)
        if not cards:
            cache[key] = []
            return []
        
        scored = [
            (c, self.turn_efficiency(player, c) * self._feasibility(player, c))
            for c in cards
        ]
        scored.sort(key=lambda t: t[1], reverse=True)

        top = scored[:k]
        cache[key] = top
        return top

    def reachability_gain(self, player, gem_one_hot) -> float:
        """Measures how much this card's permanent gem improves access
        to cards that are currently out of reach or inefficient.
        """
        cache = self.game.rewards._cache
        key = ("reach", id(player),
                        player.gems.tobytes(),
                        player.cards.tobytes(),
                        gem_one_hot.tobytes())
        if key in cache:
            return cache[key]
        
        deltas = []
        for tier in self.game.board.cards:
            for card in tier:
                if card:
                    deltas.append(self.coverage_gain(player, card, gem_one_hot))

        gain = float(np.mean(deltas)) if deltas else 0.0
        cache[key] = gain
        return gain


class _PlayerView:
    __slots__ = ("_base","gems","effective_gems","cards","reserved_cards")
    def __init__(self, base, extra):
        self._base = base
        self.gems = base.gems + extra
        self.effective_gems = base.effective_gems + extra
        self.cards = base.cards
        self.reserved_cards = base.reserved_cards
    def __getattr__(self, name):
        return getattr(self._base, name)

def _player_with_extra(player, extra):
    return _PlayerView(player, np.asarray(extra, dtype=int))


class BasicRewardEngine:
    """Basic rewards; only really shapes discards and reserves:
       - constant step penalty
       - point value for bought cards (capped to 15)
       - small discard penalty on token takes / reserves
       - point value for nobles (3 VP each, capped)
       - game win/loss bonus
    """
    def __init__(self, game):
        self.env = game
        self.metrics = Metrics(game)
        self.constant_penalty = 0.2
        self._cache = {}
        self.gems = self.GemRewards(self)
        self.buy = self.BuyRewards(self)
        self.reserve = self.ReserveRewards(self)
        self.noble = self.NobleRewards(self)
        self.game = self.GameRewards(self)

    class GemRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, taken_gems: np.ndarray, discards: np.ndarray) -> float:
            """All returns are non-positive."""
            m = self.parent.metrics; p = self.parent.env.active_player
            bank = m.game.board.gems[:5]
            count = int(taken_gems.sum())
            base = 0.1 * float(discards.sum() + np.maximum(0, 3-count))  # effective discards
            if taken_gems[5]:
                return -.1 * discards.sum()

            bank_colors = int((bank > 0).sum())
            p_total = int(p.gems.sum())
            is_double = taken_gems.max() == 2
            could_take_three = (bank_colors >= 3 and p_total <= 7)

            if p_total <= 7:  # strongly prefer taking 3, except double-take
                if count < 3 and could_take_three and not is_double:
                    return -(0.6 + base)  # unacceptable move
                else:
                    return -base  # 3 available and taken or double-take
            else:
                return -base  # Best doable, but bad territory
        def debug_gem_breakdown(self, taken_gems: np.ndarray, discards: np.ndarray): pass

    class BuyRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, bought_card) -> float:
           # reward equals VP of the card, capped so we don't exceed 15
            player = self.parent.env.active_player
            # return player.cap(float(bought_card.points))
            return self.parent.metrics.point_reward_efficiency(player, bought_card.points) * .31
        def opportunity_cost(self, player, bought_card):
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
            m = self.parent.metrics

            # Opponent's highest two ranked reachable cards
            top = m.top_reachable(player, k=2)

            # 1) If the bought card wasn't their top option, opportunity cost is zero.
            if not top or top[0][0] is not bought_card:
                return 0.0

            # 2) Difference between top-1 and top-2 efficiencies (top-2 may not exist).
            second_best = top[1][1] if len(top) > 1 else 0.0
            delta = top[0][1] - second_best

            # 3) Damp because the replacement may still be strong.
            return 0.88 * delta  # 0.88 is just guestimated risk of a good replacement
        def debug_buy_breakdown(self, *a, **kw): pass

    class ReserveRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, reserved_card, gold_vec: np.ndarray, discards: np.ndarray) -> float:
            p = self.parent.env.active_player
            opp = self.parent.env.inactive_player
            block_eff = self.parent.buy.opportunity_cost(opp, reserved_card) * .7

            p_gold = _player_with_extra(p, gold_vec)  # include the reserved gold in our view
            anti_block = self.parent.buy.opportunity_cost(p_gold, reserved_card) * .7

            capped = len(self.parent.env.active_player.reserved_cards) == 3
            capped_penalty = float(capped) * -0.2
            discard_penalty = self.parent.gems(gold_vec, discards)

            prog = min(1.0, (self.parent.env.half_turns//2)/AVG_GAME_LENGTH)
            t1_pen = 0.3 if (reserved_card.tier==0 and prog>0.5) else 0.1

            tot = block_eff + anti_block + capped_penalty + discard_penalty - t1_pen
            return  max(tot - 0.1, -0.6)  # to match bad gem moves at worst case
        def debug_reserve_breakdown(self, reserved_card, gold_vec: np.ndarray, discards: np.ndarray):
            p = self.parent.env.active_player
            opp = self.parent.env.inactive_player
            block_eff = self.parent.buy.opportunity_cost(opp, reserved_card) * .6

            p_gold = _player_with_extra(p, gold_vec)  # include the reserved gold in our view
            anti_block = self.parent.buy.opportunity_cost(p_gold, reserved_card) * .6

            capped = len(self.parent.env.active_player.reserved_cards) == 3
            capped_penalty = float(capped) * -0.2
            discard_penalty = self.parent.gems(gold_vec, discards)
            prog = min(1.0, (self.parent.env.half_turns//2)/AVG_GAME_LENGTH)
            t1_pen = 0.3 if (reserved_card.tier==0 and prog>0.5) else 0.0
            total = block_eff + anti_block + capped_penalty + discard_penalty - t1_pen
            print(f"block_eff={block_eff:.3f} anti_block={anti_block:.3f} "
                  f"capped_penalty={capped_penalty:.3f} "
                  f"discard_penalty={discard_penalty:.3f} t1_pen={t1_pen} "
                  f"total={total:.3f}")

    class NobleRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, n_visited: int) -> float:
            # 3 VP per noble, apply same capping logic
            player = self.parent.env.active_player
            # return player.cap(3.0 * float(n_visited))
            # Making this smaller than regular point rewards on purpose:
            return self.parent.metrics.point_reward_efficiency(player, 3*n_visited) * .26

    class GameRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, winner: bool) -> float:
            return 10.0 if winner else -10.0


class SparseRewardEngine:
    """Only has game win/loss sparse rewards."""
    def __init__(self, game):
        self.env = game
        self.metrics = Metrics(game)
        self.constant_penalty = 0.0
        self._cache = {}
        self.gems = self.GemRewards(self)
        self.buy = self.BuyRewards(self)
        self.reserve = self.ReserveRewards(self)
        self.noble = self.NobleRewards(self)
        self.game = self.GameRewards(self)

    class GemRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, taken_gems, discards): return 0
        def debug_gem_breakdown(self, *a, **kw): return 0

    class BuyRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, bought_card): return 0
        def debug_buy_breakdown(self, *a, **kw): return 0

    class ReserveRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, reserved_card, gold_vec, discards): return 0
        def debug_reserve_breakdown(self, *a, **kw): return 0

    class NobleRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, n_visited): return 0

    class GameRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, winner: bool) -> float:
            return 10.0 if winner else -10.0


class SparseReserveRewardEngine:
    """Sparse plus trying to fix rewards.  I realize the problem now
    which is that the model can't explore reserving, because at all
    points it has three reserved cards.  It fills up and then can't
    learn further.

    Delaying this for now and trying a constant negative penalty of
    number of held reserved cards to try to dissociate rewards from
    having three reserved cards.
    """
    def __init__(self, game):
        self.env = game
        self.metrics = Metrics(game)
        self.constant_penalty = 0.0
        self._cache = {}
        self.gems = self.GemRewards(self)
        self.buy = self.BuyRewards(self)
        self.reserve = self.ReserveRewards(self)
        self.noble = self.NobleRewards(self)
        self.game = self.GameRewards(self)

    class GemRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, taken_gems, discards): pass
        def debug_gem_breakdown(self, *a, **kw): pass

    class BuyRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, bought_card): pass
        def debug_buy_breakdown(self, *a, **kw): pass

    class ReserveRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, reserved_card, gold_vec, discards): pass
        def debug_reserve_breakdown(self, *a, **kw): pass

    class NobleRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, n_visited): pass

    class GameRewards:
        def __init__(self, parent): self.parent = parent
        def __call__(self, winner: bool) -> float:
            return 10.0 if winner else -10.0
