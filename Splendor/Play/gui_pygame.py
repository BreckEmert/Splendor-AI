# Splendor/Play/gui_pygame.py
"""Conducts the game and renderers through pygame."""

import sys
from io import BytesIO
from PIL import Image
from typing import TYPE_CHECKING

import numpy as np
import pygame

if TYPE_CHECKING:
    from Environment.Splendor_components.Board_components.deck import Card
from Play.common_types import GUIMove
from Play import ClickMap, FocusTarget
from Play.render import (
    BoardRenderer,
    OverlayRenderer,
    Coord
)


def pil_to_surface(pil_image):
    """Convert PIL image to pygame surface."""
    return pygame.image.fromstring(
        pil_image.tobytes(), pil_image.size, pil_image.mode
    ).convert()


class UILock:
    """Block all UI clicks while AI is thinking and
    after each human move to allow the player to see
    the consequences of their move.
    """
    def __init__(self, game, human):
        self._game = game
        self._human = human
        self.locked_until: int | None = None  # in ms

    @property
    def active(self) -> bool:
        now = pygame.time.get_ticks()
        awaiting_move = getattr(self._human, "awaiting_move", False)
        human_pause = self.locked_until is not None and now < self.locked_until
        return human_pause or not awaiting_move

    def arm_delay(self, ms: int) -> None:
        self.locked_until = pygame.time.get_ticks() + ms


class SplendorGUI:
    def __init__(self, game, human, preview_rewards: bool):
        self.game = game
        self.human = human
        self.overlay: OverlayRenderer
        self._renderer = BoardRenderer()
        self.rewards = game.rewards
        self.preview_rewards = preview_rewards
        self.window = None
        self.running = True

        # UI locks
        self._awaiting_ai = False
        self.delay_after_move: int = 3000
        self.lock = UILock(game, human)

        # Caches
        self._preview_state: tuple[FocusTarget | None, tuple[int, ...]] = (None, ())
        self._preview_lines: list[str] = []

        # State
        self._focus_target: FocusTarget | None = None
        self._take_picks: list[int] = []
        self._take_discards: list[int] = []
        self._ctx_rects = {}  # maps overlay button to a move_idx
        self._spend_state = None  # Context for when player manually spends

    def _reset_overlay_inputs(self):
        self._focus_target = None
        self._take_picks.clear()
        self._take_discards.clear()
        self._ctx_rects.clear()
        self._spend_state = None

    def _is_gem_click_allowed(self, color: int) -> bool:
        """Click is allowed if 4 Splendor rules pass."""
        supply = self.game.board.gems[color]
        picked = self._take_picks

        # 1. Max three picks total (two for taking two of the same).
        if len(picked) >= 3:
            return False
        elif len(picked) == 2 and picked[0] == picked[1]:
            return False
        
        # 2. There must be at least one token of that kind.
        if supply == 0:
            return False
        
        # 3. A second click of the same color is allowed when:
        if len(picked) == 1 and picked[0] == color and supply < 4:
            return False
        
        # 4. Once you've picked 2 diff you must keep picking diff
        if len(picked) >= 2 and color in picked:
            return False
        
        # 5. A player can have at most 10 gems.
        # These clicks are allowed, but the Confirm button
        # will not enable until self.discards_required is 0.
        
        return True
    
    def _is_reserve_legal(self) -> bool:
        return len(self.game.active_player.reserved_cards) < 3

    @property
    def discards_required(self) -> int:
        # Move this to the HumanAgent class?
        player = self.game.active_player
        n_picked = len(self._take_picks)
        n_discarded = len(self._take_discards)
        return max(0, player.gems.sum() + n_picked - n_discarded - 10)

    @staticmethod
    def compute_spend(card_cost, player) -> np.ndarray:
        card_cost = np.maximum(card_cost - player.cards, 0)
        spent_gems = np.minimum(player.gems, card_cost)
        remainder = card_cost.sum() - spent_gems.sum()
        spent_gems[5] = min(remainder, player.gems[5])
        return spent_gems
    
    def _handle_board_click(self, mouse_x, mouse_y, button: int) -> None:
        # Spend-selection mode; only accept player_gem clicks
        if self._spend_state:
            for rect, token in reversed(list(self.clickmap.items())):
                if rect.contains(mouse_x, mouse_y):
                    if token[0] == "player_gem":
                        self._handle_spend_click(token[1], button)
                    break
            return

        # Normal mode
        for rect, token in reversed(list(self.clickmap.items())):
            if rect.contains(mouse_x, mouse_y):
                # Right click unfocuses any card
                if button == 3:
                    self._focus_target = None
                    if not token[0].endswith("gem"):
                        break
                
                # Clicking gems unfocuses cards and vice-versa
                if button == 1 and token[0].endswith("gem"):
                    self._focus_target = None
                elif button == 1 and token[0].endswith("card"):
                    self._take_picks.clear()
                    self._take_discards.clear()

                # Now apply click logic
                kind = token[0]
                if kind == "board_card":
                    self._focus_target = FocusTarget.from_index(*token[1:])
                elif kind == "reserved_card":
                    self._focus_target = FocusTarget("reserved", reserve_idx=token[1])
                elif kind == "board_gem":
                    # Add/remove gem based on l/r click
                    color = token[1]
                    if button == 3 and color in self._take_picks:
                        self._take_picks.remove(color)
                    elif button == 1 and self._is_gem_click_allowed(color):
                        self._take_picks.append(color)
                elif kind == "player_gem":
                    if self._spend_state is None and self.discards_required == 0:
                        break

                    # Add/remove gem based on l/r click
                    color = token[1]
                    if button == 3 and color in self._take_discards:
                        self._take_discards.remove(color)
                    elif button == 1:
                        # Keep things linear and one-by-one:
                        if color in self._take_discards:
                            self._take_discards.remove(color)
                        else:
                            self._take_discards.append(color)

                break
    
    def _handle_context_menu_click(self, payload: tuple[str, GUIMove]) -> None:
        button_choice, move = payload

        if button_choice == "clear":
            ss = self._spend_state
            if ss is not None:
                if ss["spend"].sum():
                    ss["spend_picks"].clear()
                    ss["spend"][:] = 0
                else:
                    self._spend_state = None
            elif self._focus_target is not None:
                self._focus_target = None
                self._ctx_rects.clear()
            else:
                self._reset_overlay_inputs()
            return
        
        if button_choice == "confirm" and move is not None:
            if move.kind == "buy_choose":
                self._start_spend_mode(move.source, move.card)
                return
            
            # Take, buy, reserve, or buy with chosen spend
            self.human.feed_move(move)
            self.lock.arm_delay(self.delay_after_move)
            self._reset_overlay_inputs()
            return

    def _handle_mouse_event(self, event: pygame.event.Event) -> None:
        mouse_x, mouse_y = event.pos
        sx, sy = self.overlay.scale()
        mouse_x = int(mouse_x / sx)
        mouse_y = int(mouse_y / sy)

        for rect, payload in self._ctx_rects.items():
            # Context menu buttons
            if rect.contains(mouse_x, mouse_y):
                self._handle_context_menu_click(payload)
                break
        else:
            # Regular board click
            self._handle_board_click(mouse_x, mouse_y, event.button)

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            print("Exiting through _handle_event")
            self.running = False
            # pygame.quit()
            # sys.exit()
        elif (event.type == pygame.MOUSEBUTTONDOWN
              and not self.lock.active):
            self._handle_mouse_event(event)
        elif event.type == pygame.VIDEORESIZE:
            # Lock aspect ratio
            w, h = event.size
            ratio = self._renderer.geom.canvas[0] / self._renderer.geom.canvas[1]
            if w / h > ratio:
                w = int(h * ratio)
            else:
                h = int(w / ratio)
            self.window = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            self.overlay.update_window(self.window)
        elif event.type == pygame.USEREVENT and self._awaiting_ai:
            # Unlock UI after trigger we set goes off
            print("unarming lock")
            self._awaiting_ai = False

    def _card_menu_options(self, focus: FocusTarget) -> list[tuple[str, GUIMove]]:
        """Get legal (label, move) pairs for card clicks.
        We do this so we can draw the Confirm/Select button,
        and keep the legality checking to this one step.
        """

        p = self.game.active_player
        opts: list[tuple[str, GUIMove]] = []
        match focus.kind:
            case "shop":
                card = self.game.board.cards[focus.tier][focus.pos]
                afford_wo_gold, afford_with_gold = p.can_afford_card(card)
                if afford_wo_gold or afford_with_gold:
                    if afford_with_gold and p.gold_choice_exists(card):
                        # Allow for manually spending gems
                        move = GUIMove("buy_choose", card=card, source=focus)
                        opts.append(("Buy (mnl)", move))
                    
                    # Always offer an auto-spend option
                    spend = self.compute_spend(card.cost, p)
                    move = GUIMove("buy", card=card, source=focus, spend=spend)
                    opts.append(("Buy (auto)", move))

                if self._is_reserve_legal():
                    move = GUIMove("reserve", card=card, source=focus)
                    opts.append(("Reserve", move))

            case "deck":
                if self._is_reserve_legal():
                    move = GUIMove("reserve", card=None, source=focus)
                    opts.append(("Reserve", move))

            case "reserved":
                card = p.reserved_cards[focus.reserve_idx]
                afford_wo_gold, afford_with_gold = p.can_afford_card(card)
                if afford_wo_gold or afford_with_gold:
                    if afford_with_gold and p.gold_choice_exists(card):
                        # Allow for manually spending gems
                        move = GUIMove("buy_choose", card=card, source=focus)
                        opts.append(("Buy (mnl)", move))
                    
                    # Always offer an auto-spend option
                    spend = self.compute_spend(card.cost, p)
                    move = GUIMove("buy", card=card, source=focus, spend=spend)
                    opts.append(("Buy (auto)", move))

        return opts

    def _start_spend_mode(self, focus, card) -> None:
        """Engaged when player will start choosing gems to spend."""
        p = self.game.active_player
        cost = np.maximum(card.cost - p.cards, 0)
        self._spend_state = {
            "focus": focus,
            "card": card,
            "cost": cost,
            "colored_max": np.minimum(p.gems[:5], cost[:5]).astype(int),
            "gold_max": int(min(p.gems[5], cost[:5].sum())),
            "spend_picks": [],
            "spend": np.zeros(6, dtype=int),
        }
        self._take_picks.clear()
        self._take_discards.clear()
        self._focus_target = focus  # keep the yellow outline on the card

    def _handle_spend_click(self, color: int, button: int) -> None:
        ss = self._spend_state
        if not ss or button not in (1,3):
            return

        # --- one-by-one clickstream (mirror of take-picks) ---------------------
        picks = ss["spend_picks"]
        need = ss["cost"][:5].sum()  # colored need only
        used = np.bincount(picks, minlength=6)[:6]
        total = used[:5].sum() + used[5]
        cap = ss["colored_max"][color] if color < 5 else ss["gold_max"]

        if button == 1:
            # Add one if under the need
            if used[color] < cap and total < need:
                picks.append(color)
        else:
            # Remove one on right click
            for i in range(len(picks) - 1, -1, -1):
                if picks[i] == color:
                    picks.pop(i)
                    break

        ss["spend"][:] = np.bincount(picks, minlength=6)[:6]

    def _get_rewards(self) -> list[str]:
        """Return RewardEngine previews for current selection/focus."""
        state = (self._focus_target, tuple(self._take_picks))
        if state == self._preview_state:
            return self._preview_lines
        else:
            self._preview_state = state
        
        lines: list[str] = []
        rewards = self.rewards
        player = self.game.active_player

        # Gems
        if self._take_picks:
            taken_gems = np.zeros(6, dtype=int)
            for c in self._take_picks:
                taken_gems[c] += 1
            tmp = player.clone()
            _, discards = tmp.auto_take(taken_gems.copy())
            rewards.gems.debug_gem_breakdown(taken_gems, discards)
            lines.append(f"Take gems: {rewards.gems(taken_gems, discards):+.2f}")

        # Focused card (buy and reserve)
        if self._focus_target:
            ft = self._focus_target
            if ft.kind == "shop":
                card = self.game.board.cards[ft.tier][ft.pos]
                if card:
                    lines.append(f"Buy: {rewards.buy(card):+.2f}")
                    rewards.buy.debug_buy_breakdown(card)  # OPTIONAL DEBUG
                    gold = np.zeros(6, dtype=int)
                    gold[5] = int(self.game.board.gems[5] > 0)
                    discards = np.zeros(6, dtype=int)
                    if gold[5]:
                        tmp = player.clone()
                        _, discards = tmp.auto_take(gold.copy())
                    rewards.reserve.debug_reserve_breakdown(card, gold, discards)  # OPTIONAL DEBUG
                    lines.append(f"Reserve: {rewards.reserve(card, gold, discards):.2f}")
            elif ft.kind == "reserved":
                card = player.reserved_cards[ft.reserve_idx]
                rewards.buy.debug_buy_breakdown(card)  # OPTIONAL DEBUG
                lines.append(f"Buy reserved: {rewards.buy(card):+.2f}")

        self._preview_lines = lines
        return lines

    def run(self):
        """Side thread for GUI handling."""
        pygame.init()
        self.window = pygame.display.set_mode(
            self._renderer.geom.default_canvas_scale, 
            pygame.RESIZABLE
        )
        self.overlay = OverlayRenderer(self.window, self.preview_rewards)
        pygame.display.set_caption("Splendor RL - Human vs DDQN")

        while self.running and not self.game.victor:
            # Render frame and clickmap to buffer
            buf = BytesIO()
            self.clickmap: ClickMap = self._renderer.render(self.game, buf)
            buf.seek(0)
            frame = pil_to_surface(Image.open(buf))
            frame = pygame.transform.smoothscale(
                frame,
                self.window.get_size()
            )
            self.window.blit(frame, (0, 0))

            # pygame events
            for event in pygame.event.get():
                self._handle_event(event)

            # Don't continue if locked
            if self.lock.active:
                pygame.display.flip()
                continue
            
            # Draw UI
            self.overlay.draw_selection_highlights(
                self.clickmap, self._focus_target,
                self._take_picks, self._take_discards,
                self._spend_state["spend"] if self._spend_state else [0]*6
            )
            if self.discards_required > 0:
                self.overlay.draw_discard_notice()
            if self.preview_rewards:
                self.overlay.draw_reward_preview(self._get_rewards())

            if self._spend_state:
                # Draw confirm when selected gems equals the cost
                state = self._spend_state
                need = state["cost"][:5].sum()
                cur = state["spend"][:5].sum() + state["spend"][5]
                confirm_enabled = (cur == need)
                move = GUIMove(
                    kind="buy",
                    card=state["card"],
                    source=state["focus"],
                    spend=state["spend"].copy(),
                )
                self._ctx_rects = self.overlay.draw_move_confirm_button(
                    move, confirm_enabled, clear_enabled=True
                )
            elif self._focus_target:
                # Draw Submit/Clear button and context menu for clicked cards
                options = self._card_menu_options(self._focus_target)
                origin = None

                match self._focus_target.kind:
                    case "shop":
                        origin = next(
                            (Coord(r.x0, r.y0) for r, p in self.clickmap.items()
                             if p == ("board_card",
                                      self._focus_target.tier,
                                      self._focus_target.pos)),
                            None,
                        )
                    case "deck":
                        origin = next(
                            (Coord(r.x0, r.y0) for r, p in self.clickmap.items()
                             if p == ("board_card", self._focus_target.tier, 4)),
                            None,
                        )
                    case "reserved":
                        origin = next(
                            (Coord(r.x0, r.y0) for r, p in self.clickmap.items()
                             if p[0] == "reserved_card"
                             and p[1] == self._focus_target.reserve_idx),
                            None,
                        )

                if origin is not None and options:
                    self._ctx_rects = self.overlay.draw_card_context_menu(
                        origin, options,
                    )
            elif self._take_picks:
                # Draw Submit/Clear button for clicked tokens
                move = GUIMove(
                    kind="take",
                    take=np.bincount(self._take_picks, minlength=6)[:6],
                    discard=np.bincount(self._take_discards, minlength=6)[:6]
                )
                confirm_enabled = (self.discards_required == 0)
                clear_enabled = bool(self._take_picks)
                self._ctx_rects = self.overlay.draw_move_confirm_button(
                    move, confirm_enabled, clear_enabled
                )
            else:
                self._ctx_rects.clear()

            pygame.display.flip()
        
        else:
            if self.game.victor:
                font = pygame.font.SysFont(None, 72)
                txt  = font.render(
                    f"{self.game.active_player.name} wins!",
                    True, (255, 215, 0)
                )
                rect = txt.get_rect(center=self.window.get_rect().center)
                self.window.blit(txt, rect)
                pygame.display.flip()

                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type in (
                            pygame.QUIT,
                            pygame.KEYDOWN,
                            pygame.MOUSEBUTTONDOWN
                        ):
                            print("quitting because of ", event)
                            waiting = False

        pygame.quit()
        sys.exit()
