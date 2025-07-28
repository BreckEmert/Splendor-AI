# Splendor/Play/gui_pygame.py
"""Drive a Splendor game with one HumanAgent."""

import sys
import pygame
from io import BytesIO
from PIL import Image
from typing import Any, NamedTuple

from Play.render import (
    BoardRenderer,
    OverlayRenderer,
    Rect,
    take_3_indices, 
    take_2_diff_indices
)


def pil_to_surface(pil_image):  # Should this really be outside of class?
    """Convert PIL image to pygame surface."""
    return pygame.image.fromstring(
        pil_image.tobytes(), pil_image.size, pil_image.mode
    ).convert()


class CardIndex(NamedTuple):
    tier: int
    pos: int


class SplendorGUI:
    def __init__(self, game, human):
        self.game = game
        self.human = human
        self.window = None
        self.overlay: OverlayRenderer
        self.running = True
        self._renderer = BoardRenderer()

        # State
        self._focused_card: CardIndex | None = None
        self._picked_tokens: list[int] = []
        self._ctx_rects = {}  # maps overlay button to a move_index

    def _is_move_legal(self, move_index: int | None) -> bool:
        return (move_index is not None) and bool(self.human.legal_mask[move_index])

    def _gem_click_allowed(self, color: int) -> bool:
        """Click is allowed if 4 Splendor rules pass."""
        supply = self.game.board.gems[color]
        picked = self._picked_tokens

        # 1. Toggle-off always allowed.
        if color in picked:
            return True if picked.count(color) == 1 else False

        # 2. Max three picks total (two for taking two of the same).
        if len(picked) >= 3:
            return False
        elif len(picked) == 2 and picked[0] == picked[1]:
            return False
        
        # 3. There must be at least one token of that kind.
        if supply == 0:
            return False
        
        # 4. A second click of the same color is allowed when:
        if len(picked) == 1 and picked[0] == color and supply < 4:
            return False
        
        return True

    def _gems_to_move(self, picked: list[int]) -> int | None:
        """Map selected gems to the engine's move_index.
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

        raise ValueError("Error: no legal card move_index was found.")

    def _handle_board_click(self, mouse_x, mouse_y, event) -> None:
        for rect, token in self.clickmap.items():
            if rect.contains(mouse_x, mouse_y):
                # Card was clicked
                if token[0] == "card":
                    self._focused_card = CardIndex(*token[1:])

                # Gem was clicked
                elif token[0] == "gem":
                    color = token[1]

                    left_click = event.button == 1
                    right_click = event.button == 3

                    # Add/remove token from _picked_tokens based on l/r click
                    if right_click and color in self._picked_tokens:
                        self._picked_tokens.remove(color)
                    elif left_click and self._gem_click_allowed(color):
                        self._picked_tokens.append(color)
                
                # Reserved card was clicked
                elif token[0] == "move":
                    self.human.feed_move(token[1])

                break
    
    def _handle_context_menu_click(self, payload) -> None:
        action, move_index = payload

        if action == "clear":
            self._picked_tokens.clear()
        elif action == "confirm" and move_index is not None:
            self.human.feed_move(move_index)

        # Reset overlay
        self._focused_card = None
        self._ctx_rects.clear()

    def _handle_mouse_event(self, event) -> None:
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
            self._handle_board_click(mouse_x, mouse_y, event)

    def _handle_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            pygame.quit()
            sys.exit()
        elif (event.type == pygame.MOUSEBUTTONDOWN
              and self.game.active_player.agent is self.human):
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

    def _card_menu_options(self, tier: int, pos: int) -> list[tuple[str, Any]]:
        # All possible
        buy, buy_gold, reserve = (
            self._card_to_move(tier, pos, mode)
            for mode in ("buy", "buy_with_gold", "reserve")
        )

        # Legal ones
        buy_ok, gold_ok, reserve_ok = (
            self._is_move_legal(move)
            for move in (buy, buy_gold, reserve)
        )

        options = []
        card = self.game.board.cards[tier][pos] if pos < 4 else None

        if card:
            # Show only one option, prioritizing manual_spend
            if gold_ok and self.game.active_player.gold_choice_exists(card.cost):
                options.append(("Buy 🪙", buy_gold))
            elif buy_ok:
                options.append(("Buy 🔵⚪🪙", buy))
        if reserve_ok:
            options.append(("Reserve", reserve))

        return options

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
            self.clickmap: dict[Rect, tuple] = self._renderer.render(self.game, buf)
            buf.seek(0)
            frame = pil_to_surface(Image.open(buf))
            frame = pygame.transform.smoothscale(
                frame,
                self.window.get_size()
            )
            self.window.blit(frame, (0, 0))
            self.overlay.draw_selection_highlights(
                self.clickmap, self._focused_card, self._picked_tokens
            )
            pygame.display.flip()

            # pygame events
            for event in pygame.event.get():
                self._handle_event(event)

            # Draw UI
            if self._focused_card:
                tier, pos = self._focused_card
                opts = self._card_menu_options(tier, pos)
                self._ctx_rects = self.overlay.draw_card_context_menu(
                    tier, pos, opts
                )
            elif self._picked_tokens:
                move_index = self._gems_to_move(self._picked_tokens)
                confirm_enabled = self._is_move_legal(move_index)
                clear_enabled = bool(self._picked_tokens)
                self._ctx_rects = self.overlay.draw_move_confirm_button(
                    move_index, confirm_enabled, clear_enabled
                )
            else:
                self._ctx_rects.clear()
