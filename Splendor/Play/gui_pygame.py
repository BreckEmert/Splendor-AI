# Splendor/Play/gui_pygame.py
"""Conducts the game and renderers through pygame."""

import sys
import pygame
from io import BytesIO
from PIL import Image

from Play import ClickMap, FocusTarget
from Play.render import (
    BoardRenderer,
    OverlayRenderer,
    Coord,
    take_3_indices, 
    take_2_diff_indices
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
        self._locked_until: int | None = None  # in ms

    @property
    def active(self) -> bool:
        now = pygame.time.get_ticks()
        awaiting_move = getattr(self._human, "awaiting_move", False)
        human_pause = self._locked_until is not None and now < self._locked_until
        return human_pause or not awaiting_move

    def arm_delay(self, ms: int) -> None:
        self._locked_until = pygame.time.get_ticks() + ms


class SplendorGUI:
    def __init__(self, game, human):
        self.game = game
        self.human = human
        self.overlay: OverlayRenderer
        self._renderer = BoardRenderer()
        self.window = None
        self.running = True

        # UI locks
        self.delay_after_move: int = 3000
        self.lock = UILock(game, human)

        # State
        self._focus_target: FocusTarget | None = None
        self._picked_tokens: list[int] = []
        self._ctx_rects = {}  # maps overlay button to a move_idx

    def _is_move_legal(self, move_idx: int | None) -> bool:
        return (move_idx is not None) and bool(self.human.legal_mask[move_idx])

    def _gem_click_allowed(self, color: int) -> bool:
        """Click is allowed if 4 Splendor rules pass."""
        supply = self.game.board.gems[color]
        picked = self._picked_tokens

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
        # THIS IS NOT CORRECT - NEED TO IMPLEMENT PLAYER GEMS AS VALID DISCARD CLICKS INSTEAD WHEN THIS FAILS
        if self.game.active_player.gems.sum() + len(picked) + 1 > 10:
            return False
        
        return True

    def _gems_to_move(self, picked: list[int]) -> int | None:
        """Map selected gems to the engine's move_idx.
        Returns None if the move is not yet legal.

        Goal with these two "to_move" methods:
            1) flashing red card on illegal click
            2) Confirm button goes green when a valid move is selected
        """
        player = self.game.active_player
        sel = sorted(picked)
        n = len(sel)
        discards = max(0, player.gems.sum() + n - 10)

        # Take 3 different, 0‑39
        if n == 3 and len(set(sel)) == 3:
            a, b, c = sel
            idx = take_3_indices.index((a, b, c))
            return idx*4 + discards

        # Take 2 same, 40‑54
        if n == 2 and sel[0] == sel[1]:
            return 40 + sel[0]*3 + discards

        # Take 2 different, 55‑84
        if n == 2:
            a, b = sel
            idx = take_2_diff_indices.index((a, b))
            return 55 + idx*3 + discards

        # Take 1, 85‑94
        if n == 1:
            return 85 + sel[0]*2 + discards

        # Here, we have to return None (different than _card_to_move)
        # because a selection can *eventually* end up legal.  If the
        # player discards gems after going over 10, it becomes legal.
        return None
    
    def _card_to_move(self, tier: int, pos: int, variant: str) -> int:
        player = self.game.active_player

        match variant:
            case "buy":
                return player.take_dim + 2*(tier*4 + pos)
            case "buy_with_gold":
                return player.take_dim + 2*(tier*4 + pos) + 1
            case "reserve":
                return player.take_dim + player.buy_dim + tier*5 + pos
            case "buy_reserved":
                return player.take_dim + 24 + pos*2  # indices may need validated?
            case "buy_reserved_with_gold":
                return player.take_dim + 24 + pos*2 + 1

        raise ValueError("Error: no legal card move_idx was found.")

    def _handle_board_click(self, mouse_x, mouse_y, button: int) -> None:
        for rect, token in reversed(list(self.clickmap.items())):
            if rect.contains(mouse_x, mouse_y):
                # Right click unfocuses any card
                if button == 3 and token[0] != "gem":
                    self._focus_target = None
                    break

                if token[0] == "board_card":
                    clicked_target = FocusTarget.from_index(*token[1:])
                    self._picked_tokens.clear()
                    self._focus_target = clicked_target
                elif token[0] == "reserved_card":
                    self._picked_tokens.clear()
                    reserve_idx = token[1]
                    self._focus_target = FocusTarget("reserved", reserve_idx=reserve_idx)
                elif token[0] == "gem":
                    # Add/remove token based on l/r click
                    color = token[1]
                    if button == 3 and color in self._picked_tokens:
                        self._picked_tokens.remove(color)
                    elif button == 1 and self._gem_click_allowed(color):
                        self._focus_target = None
                        self._picked_tokens.append(color)

                break
    
    def _handle_context_menu_click(self, payload) -> None:
        action, move_idx = payload

        if action == "clear":
            self._focus_target = None
        elif action == "confirm" and move_idx is not None:
            self.human.feed_move(move_idx)
            self.lock.arm_delay(self.delay_after_move)
        
        # Reset overlay
        self._focus_target = None
        self._picked_tokens.clear()
        self._ctx_rects.clear()

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

    def _card_menu_options(self, focus: FocusTarget) -> list[tuple[str, int]]:
        """Get legal (label, move_idx) pairs for card clicks."""

        legal, p = [], self.game.active_player
        match focus.kind:
            case "shop":
                tier, pos = focus.tier, focus.pos
                assert tier is not None, "FocusTarget.kind == 'shop' has tier of None"
                assert pos is not None, "FocusTarget.kind == 'shop' has pos of None"
                card = self.game.board.cards[tier][pos]
                for mode in ("buy", "buy_with_gold", "reserve"):
                    idx = self._card_to_move(tier, pos, mode)
                    if self._is_move_legal(idx):
                        legal.append((mode, idx))

            case "deck":
                assert focus.tier is not None, \
                    "FocusTarget.kind == 'deck' has tier of None"
                idx = self._card_to_move(focus.tier, 4, "reserve")
                if self._is_move_legal(idx):
                    legal.append(("reserve", idx))
                card = None  # unknown top of deck

            case "reserved":
                card = p.reserved_cards[focus.reserve_idx]
                for mode in ("buy_reserved", "buy_reserved_with_gold"):
                    assert focus.reserve_idx is not None, \
                        "FocusTarget.kind == 'reserved' has reserve_idx of None"
                    idx = self._card_to_move(0, focus.reserve_idx, mode)  # tier ignored
                    if self._is_move_legal(idx):
                        legal.append((mode, idx))

        opts = []
        for mode, idx in legal:
            match mode:
                case "buy" | "buy_reserved":
                    opts.append(("Buy 🔵⚪🟤", idx))
                case "buy_with_gold" | "buy_reserved_with_gold":
                    if card and p.gold_choice_exists(card.cost):
                        opts.append(("Buy 🪙", idx))
                    else:
                        opts.append(("Buy 🔵⚪🪙", idx))
                case "reserve":
                    opts.append(("Reserve", idx))

        return opts

    def run(self):
        """Side thread for GUI handling."""
        pygame.init()
        self.window = pygame.display.set_mode(
            self._renderer.geom.default_canvas_scale, 
            pygame.RESIZABLE
        )
        self.overlay = OverlayRenderer(self.window)
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

            # Don't continue if locked
            if self.lock.active:
                pygame.display.flip()
                continue

            # pygame events
            for event in pygame.event.get():
                self._handle_event(event)

            # Draw UI
            self.overlay.draw_selection_highlights(
                self.clickmap, self._focus_target, self._picked_tokens
            )

            if self._focus_target:
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
                             if p == ("board_card",
                                      self._focus_target.tier,
                                      4)),
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
            elif self._picked_tokens:
                # Draw Submit/Clear button for clicked tokens
                move_idx = self._gems_to_move(self._picked_tokens)
                confirm_enabled = self._is_move_legal(move_idx)
                clear_enabled = bool(self._picked_tokens)
                self._ctx_rects = self.overlay.draw_move_confirm_button(
                    move_idx, confirm_enabled, clear_enabled
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
