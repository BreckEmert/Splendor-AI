# RL/old_rewards.py

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from Environment.rl_game import RLGame


class old_Metrics:
    """Read-only utilities for reward shaping (alignment, progress, PPG, etc.)."""
    # Noting: could have reward for player controlling things well
    # We may have some functions like that?  But also just a basic
    # reward for controlling all four of a gem-type.
    __slots__ = (
        "game",
        # EMA baselines -- stored per player‐position so we can emit Δ-alignment
        "cards_shop_alignment_ema",   # permanent-gems ↔ shop-demand
        "gems_shop_alignment_ema",    # temporary-gems ↔ shop-demand
        "noble_progress_prev",        # last-step noble progress per player
    )

    def __init__(self, game):
        self.game = game
        # Exponential-moving baseline and (optional) frozen snapshot per player-pos
        self.cards_shop_alignment_ema = {p.pos: 0.0 for p in game.players}
        self.gems_shop_alignment_ema  = {p.pos: 0.0 for p in game.players}
        self.noble_progress_prev      = {p.pos: 0.0 for p in game.players}

    def _raw_cards_shop_alignment(self, player) -> float:
        """cards_shop_alignment: permanent-gem profile ↔ current shop demand (0-1)."""
        profile = player.cards[:5].astype(np.int32)
        shop_cost = np.add.reduce(
            [card.cost[:5] for tier in self.game.board.cards for card in tier if card],
            axis=0, dtype=np.int32
        )
        denom = shop_cost.sum() or 1
        dot = float(profile @ shop_cost)
        return min(dot, denom) / denom  # ∈ [0, 1]

    def _raw_gems_shop_alignment(self, player) -> float:
        """temp-gem profile ↔ current shop demand (0-1)."""
        profile = player.gems[:5].astype(np.int32)
        if profile.sum() == 0:  # no colored gems held
            return 0.0
        shop_cost = np.add.reduce(
            [c.cost[:5] for tier in self.game.board.cards for c in tier if c],
            axis=0, dtype=np.int32,
        )
        denom = shop_cost.sum() or 1
        dot = float(profile @ shop_cost)
        return min(dot, denom) / denom  # ∈ [0, 1]

    # ── Feasibility primitives ────────────────────────────────────────────
    def _shortage(self, player, card) -> np.ndarray:
        """Color-wise shortage after cards+gems, then greedily spend gold."""
        shortage = np.maximum(card.cost[:5] - player.effective_gems[:5], 0).astype(int)
        gold = int(player.gems[5])
        if gold:  # greedily cancel largest deficits
            for idx in np.argsort(-shortage):
                use = min(gold, shortage[idx])
                shortage[idx] -= use
                gold -= use
                if gold == 0:
                    break
        return shortage  # shape (5,)

    @staticmethod
    def _turns(shortage: np.ndarray) -> int:
        """Minimum 'take-gems' turns needed (≤ 3 colored gems per turn)."""
        return int(np.ceil(shortage.sum() / 3.0))

    def _feasible(self, player, card, shortage=None) -> bool:
        """True if card is buyable within ≤ 3 turns and won't overflow hand."""
        if shortage is None:
            shortage = self._shortage(player, card)
        enough_turns = self._turns(shortage) <= 3
        room_ok = (player.gems.sum() + shortage.sum()) <= 10
        return enough_turns and room_ok

    def reachable_cards(self, player):
        """
        Return up to three ‘focus’ cards (one per tier, visible or reserved)
        that are reachable under the 10-gem limit.
        Tier-1: select by coverage ratio, only if player has <5 permanent gems.
        Tier-2/3: select by overlap with player's effective gems.
        """
        best_by_tier = {0: (None, -1.0), 1: (None, -1.0), 2: (None, -1.0)}

        # Unified stream of (tier, card): visible shop + player's reserved cards
        stream = [
            (tier_idx, card)
            for tier_idx, tier in enumerate(self.game.board.cards)
            for card in tier if card
        ] + [
            (card.tier, card)
            for card in player.reserved_cards
        ]

        for tier_idx, card in stream:
            shortage = self._shortage(player, card)
            if player.gems.sum() + shortage.sum() > 10:
                continue  # hard cap violated ⇒ not reachable

            cost_sum = card.cost[:5].sum() or 1
            covered = cost_sum - shortage.sum()

            if tier_idx == 0:
                score = covered / cost_sum  # coverage ratio for t1
            else:
                overlap = player.effective_gems[:5] @ card.cost[:5]
                score = overlap / cost_sum  # overlap ratio for t2/t3

            # Tie-break: higher points, then smaller shortage
            current_best, best_score = best_by_tier[tier_idx]
            if (
                score > best_score or
                (np.isclose(score, best_score) and current_best and
                (card.points, -shortage.sum()) > (current_best.points,
                                                -self._shortage(player, current_best).sum()))
            ):
                best_by_tier[tier_idx] = (card, score)

        # Drop tier-1 target if engine has ≥5 permanent gems
        if player.cards.sum() >= 5:
            best_by_tier[0] = (None, -1.0)

        # Return list without Nones, preserving tier order
        return [
            card for card, _ in (
                best_by_tier[0],
                best_by_tier[1],
                best_by_tier[2]
            ) if card
        ]

    # ── Card valuation & selection ────────────────────────────────────────
    def _card_value(self, card) -> float:
        """Heuristic quality: points-per-gem minus imbalance penalty."""
        return self.points_per_gem(card)  # re‑use existing helper

    def feasible_cards(self, player):
        """Return [(card, shortage, value)] for every
        **visible** shop card that passes `_feasible`.
        Order is arbitrary; caller decides ranking.
        """
        out = []
        for tier in self.game.board.cards:
            for card in tier:
                if not card:  # empty slot
                    continue
                shortage = self._shortage(player, card)
                if not self._feasible(player, card, shortage):
                    continue
                value = self._card_value(card)
                out.append((card, shortage, value))
        return out

    # def top_targets(self, player, k: int = 2):
    #     """
    #     Shortage vectors of the k highest‑value feasible cards
    #     (default = top‑two).  May return fewer than k if list shorter.
    #     """
    #     cards = self.feasible_cards(player)
    #     if not cards:
    #         return []
    #     # sort by value descending, then points for deterministic tie‑break
    #     cards.sort(key=lambda t: (-t[2], -t[0].points))
    #     return [t[1] for t in cards[:k]]

    def enemy_targets(self, player, k: int = 2):
        """Top-k feasible target shortages for *the* opponent of `player`.

        Returns a (possibly shorter) list of color-wise shortage vectors.
        Use this so reward terms can evaluate how much an action blocks
        what the enemy most wants right now.
        """
        opp = next(p for p in self.game.players if p is not player)
        return self.top_targets(opp, k=k)

    def current_targets(self, player, k: int = 2):
        """Public wrapper so Reward classes don't poke internals."""
        return self.top_targets(player, k=k)
    
    def feasibility_empty(self, player) -> bool:
        """True iff no card satisfies the feasibility test."""
        return not self.feasible_cards(player)

    def feasibility_score(self, card, player) -> float:
        """
        Continuous ∈ (0, 1] measure of how realistic it is for *player* to buy *card*:

        score = 1 / (turns_required + overflow_penalty + color_contention)

        • turns_required  = ceil(shortage.sum() / 3).
        • overflow_penalty = 1 if shortage would push hand >10 gems, else 0.
        • color_contention = 1 per color where opponent ALSO lacks ≥1 gem
        for their own top-k target (k=2).

        The inverse keeps the metric bounded in (0, 1]; higher ⇒ more feasible.

        I know this isn't used but this is an idea I want - so
        this should be incorporated somewhere?  Thoughts?
        """
        shortage = self._shortage(player, card)
        turns_req = self._turns(shortage)

        # overflow check
        overflow_penalty = int(player.gems.sum() + shortage.sum() > 10)

        # color contention with opponent’s current top‑2 targets
        opp_shortages = self.enemy_targets(player, k=2)
        if opp_shortages:
            opp_union = np.maximum.reduce(opp_shortages)
            contention = int(np.any((shortage > 0) & (opp_union > 0)))
        else:
            contention = 0

        denom = turns_req + overflow_penalty + contention
        return 1.0 / max(denom, 1)  # safe div‑by‑zero

    def card_alignment_delta(self, player, *, tier: int | None = None) -> float:
        """Potential-scaled Δ-alignment w/ EMA baseline and tier weight."""
        pos = player.pos
        align_now = self._raw_cards_shop_alignment(player)
        prev = self.cards_shop_alignment_ema[pos]
        gamma = 1.5  # try to keep delta-alignment
        base_delta = (align_now ** gamma) - (prev ** gamma)
        weight = 1.0 + 0.4 * (tier or 0)
        delta = weight * base_delta

        # update EMA baseline
        self.cards_shop_alignment_ema[pos] = 0.9 * prev + 0.1 * align_now
        return delta

    def gems_shop_alignment_delta(self, player) -> float:
        pos = player.pos
        now = self._raw_gems_shop_alignment(player)
        prev = self.gems_shop_alignment_ema[pos]
        delta = now - prev
        self.gems_shop_alignment_ema[pos] = 0.9 * prev + 0.1 * now
        return delta

    def block_potential(self, card, opponent) -> float:
        """
        How painful is it for *opponent* that this card is unavailable?
        1.0 = buyable right now
        0.66 = 1 gem short
        0.15 = 2-3 gems short
        0.0 = ≥4 gems short or wrong colors
        """
        shortage = np.maximum(card.cost - opponent.effective_gems, 0)[:5].sum()
        if shortage == 0:
           return 1.0
        elif shortage == 1:
            return 0.66
        elif shortage <= 3:
            return 0.15
        return 0.0

    def noble_progress(self, player) -> float:
        """Aggregate fractional progress toward every noble.
        Each noble contributes (1 - shortage/4) ∈ [0, 1],
        so the total lies in [0, 3].
        """
        progress = 0.0
        for noble in self.game.board.nobles:
            if noble is None:
                continue
            shortage = np.maximum(noble.cost - player.cards, 0)[:5].sum()
            progress += 1.0 - shortage / 4.0
        return progress

    def noble_progress_delta(self, player, kappa: float = 0.1) -> float:
        """+κ for every incremental step toward visiting 
        any noble (potential-based → policy-invariant).
        """
        pos = player.pos
        now = self.noble_progress(player)
        prev = self.noble_progress_prev[pos]
        self.noble_progress_prev[pos] = now
        return kappa * (now - prev)

    def points_per_gem(self, card) -> float:
        cost = card.cost[:5].sum()
        if cost == 0:  # safeguard for zero-cost bug
            return 0.
        ratio = card.points / cost
        # imbalance penalty
        costs = np.sort(card.cost[:5])[::-1]  # desc
        imbalance = (costs[0] - costs[1]) // 2
        return ratio - 0.1 * imbalance


class old_RewardEngine:
    """Hub for all reward signals.  Each method returns
    a scalar and has no side effects to the game.
    I'm not treating this with normal OOP rules
    because I want things to be as localized to
    their method as possible or things will get
    messy as experimentation goes.
    """
    # Let's think of some logic
    # 1) Blocking an enemy player from buying something is good
    # 2) Blocking an enemy player from taking gems that would let them buy something is good
    # 3) Blocking an enemy player from reserving something they want is good
    # 4) Bringing the gem supply to 0 is much better than bring it to 1
    # 5) Bringing the gem supply lower than the enemy player needs for a buy is good
    def __init__(self, game: "RLGame"):
        # It really seems like block logic is universal.  Like we have
        # gem blocking, buy blocking, and reserve blocking.  
        self.env = game
        self.metrics = Metrics(game)

        self.gems = self.GemRewards(self)
        self.buy = self.BuyRewards(self)
        self.reserve = self.ReserveRewards(self)
        self.noble = self.NobleRewards(self)
        self.game_rewards = self.GameRewards(self)

    class GemRewards:
        def __init__(self, parent): self.parent = parent

        def discard(self, n_discards: int = 1) -> float:
            """Quadratic discard penalty."""
            return -0.05 * (n_discards ** 2)

        def enemy_block_overlap(self, player, taken_gems, k=2) -> float:
            """Fraction of colored gems that hurt the opponent's top-k targets."""
            colored = taken_gems[:5]
            total = colored.sum()
            if total == 0:
                return 0.0
            opp_shortages = self.parent.metrics.enemy_targets(player, k=k)
            if not opp_shortages:
                return 0.0
            union = np.maximum.reduce(opp_shortages)
            overlap = float(np.minimum(colored, union).sum())
            return overlap / total
    
        def block_enemy(
            self,
            player,
            taken_gems: np.ndarray,
            k: int = 2,
        ) -> float:
            """Positive bonus when taken gems overlap the enemy’s targets."""
            bonus = 0.03 * self.enemy_block_overlap(player, taken_gems, k)
            return bonus

        # -------- main gem‑take shaping -----------------------------------------
        def take(
            self,
            player,
            taken_gems: np.ndarray,
            n_discards: int = 0,
        ) -> float:
            """Return discard penalty + ω · alignment with top-2 feasible cards."""
            base = self.discard(n_discards)
            ω = 0.04
            bonus = 0.0

            shortages = self.parent.metrics.current_targets(player, k=2)
            for shortage in shortages:  # loop no-op if list empty
                s_sum = shortage.sum()
                if s_sum:
                    bonus += ω * float(np.dot(taken_gems[:5], shortage) / s_sum)

            # Add in blocking enemy
            bonus += self.block_enemy(player, taken_gems)

            return base + bonus
        
        def opportunity_cost(self, player) -> float:
            """Small constant punishment when *no* card is currently feasible."""
            return -0.05 if self.parent.metrics.feasibility_empty(player) else 0.0

    class BuyRewards:
        def __init__(self, parent): self.parent = parent
        # Reward curve should be nonlinear:
        # \frac{y}{15}=\left(\frac{p+x}{15}\right)^{1.5}\ -\left(\frac{p}{15}\right)^{1.5}
            # Goal being points earned at lower points are worth less, 
            # because points earned at higher points can be purchased
            # for less effective turns.  I'm not 100% sure on this...
            # in the sense that it could be controlled for by other 
            # means.  But still a factor if not controlled for.

            # I guess a good question is if x being the current points
            # is a good baseline - I guess truly it should be the
            # number of gem cards the player has.  And even better
            # would be how well these align with the cards available
            # for purchase in the shop.

        def card(self, player, card) -> float:
            # base nonlinear/point reward still handled elsewhere
            m = self.parent.metrics
            β_card_align = 0.04  # cards↔shop weight (long-term)
            β_gem_align = 0.001  # gems↔shop weight (short-term)

            # --- nonlinear point shaping (late points worth more) ---
            φ = lambda p: (p / 15) ** 1.5
            nonlinear = 15 * (φ(player.points + card.points) - φ(player.points))

            # --- alignment bonuses ---
            tier = getattr(card, "tier", 0)
            align_bonus = (
                β_card_align * m.cards_shop_alignment_delta(player, tier=tier) +
                β_gem_align * m.gems_shop_alignment_delta(player)
            )

            # --- tiny points-per-gem heuristic nudger ---
            ppg_bonus = 0.05 * m.points_per_gem(card)  # tiny PPG heuristic
            engine_bonus = 0.05  # +η per permanent gem gained

            return nonlinear + align_bonus + ppg_bonus + engine_bonus

    class ReserveRewards:
        # 1) Under-reserving is bad
            # Blocking an enemy player from making noble progress is good
            # Blocking an enemy player from straight up making a buy is really good
                # Mostly for more expensive and specific buys...
                # Blocking early is still good but less efficient becuase they've allocated less moves
                # Blocking when they have an alternative is extremely less useful
                    # If their buy comes from gem supply, it is usless
                    # But blocking when it comes from card supply is quite good
            # 
        # 2) Over-reserving is bad
            # Reserving cards the enemy has no reason to block is useless
            # Reserving when the enemy has no reserves and cant afford it is useless
            # Reserving kind of is always a slower move, and should have some negative bias
        def __init__(self, parent): self.parent = parent

        def card(self, card, opponent, n_discards: int = 0) -> float:
            m = self.parent.metrics
            β = 0.08  # weight for block delta

            # Δ block-potential (potential-based shaping)
            prev = getattr(card, "_block_prev", 0.0)
            curr = m.block_potential(card, opponent)
            card._block_prev = curr  # mutate *card*, not env

            base  = -0.05 * (1 + n_discards)  # constant bias + discard penalty
            bonus = β * (curr - prev)
            return base + bonus

    class NobleRewards:
        def __init__(self, parent): self.parent = parent

        def progress(self, player) -> float:
            """Dense reward: incremental noble progress (κ · Δ)."""
            return self.parent.metrics.noble_progress_delta(player)

        def visit(self, n: int = 1) -> float:
            return 3.0 * n

    class GameRewards:
        def __init__(self, parent): self.parent = parent

        def win(self) -> float:
            return 5.0
