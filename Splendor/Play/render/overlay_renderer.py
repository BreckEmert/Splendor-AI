# Splendor/Play/render/overlay_renderer.py
"""
Renders any object that is not part of the base game
"""

import pygame
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Play.common_types import GUIMove
from Play import FocusTarget, ClickMap
from Play.render import BoardGeometry, Coord, Rect


class OverlayRenderer:
    def __init__(self, window, preview_rewards: bool):
        """To implement:
        1) glow around most recently purchased card or taken gems
        """
        self.geom = BoardGeometry()
        self.window = window
        self.font = pygame.font.SysFont(None, 32)
        self.small_font = pygame.font.SysFont(None, 28)

        self.preview_rewards = preview_rewards

    def scale(self) -> tuple[float, float]:
        """Scale coordinates because the window can resize."""
        board_w, board_h = self.geom.canvas
        window_w, window_h = self.window.get_size()
        return window_w / board_w, window_h / board_h
    
    def to_window(self, rect: Rect) -> Rect:
        scaled_x, scaled_y = self.scale()
        return rect.scaled(scaled_x, scaled_y)

    def update_window(self, window: pygame.Surface) -> None:
        self.window = window

    def draw_selection_highlights(
            self, 
            clickmap: ClickMap, 
            focus_target: FocusTarget | None, 
            picked_gems: list[int],
            discard_gems: list[int]
        ) -> None:
        """Highlights selections that are queued for a move."""
        sx, sy = self.scale()

        def outline(rect, color):
            r_win = rect.scaled(sx, sy).to_pygame()
            pygame.draw.rect(self.window, color, r_win, 6)

        if focus_target:
            match focus_target.kind:
                case "shop":
                    key = ("board_card", focus_target.tier, focus_target.pos)
                    for r, payload in clickmap.items():
                        if payload == key:
                            outline(r, (255, 255, 0))
                            break
                case "deck":
                    key = ("board_card", focus_target.tier, 4)
                    for r, payload in clickmap.items():
                        if payload == key:
                            outline(r, (255, 255, 0))
                            break
                case "reserved":
                    for r, payload in clickmap.items():
                        if payload[0] == "reserved_card" and payload[1] == focus_target.reserve_idx:
                            outline(r, (255, 255, 0))
                            break
        else:  # gems
            pick_cnts = Counter(picked_gems)
            disc_cnts = Counter(discard_gems)
            for r, payload in clickmap.items():
                gem_type, gem_idx = payload[0], payload[1]
                if gem_type == "board_gem" and gem_idx in pick_cnts:
                    outline(r, (0, 255, 0))
                    c = pick_cnts[gem_idx]
                    if len(pick_cnts) == 1 and c == 2:  # x2 visual cue
                        r_win = r.scaled(sx, sy).to_pygame()
                        tag = self.small_font.render(f"x{c}", True, (255, 255, 0))
                        self.window.blit(
                            tag,
                            (r_win.right - tag.get_width() - 4,
                             r_win.bottom - tag.get_height() - 4)
                        )
                elif gem_type == "player_gem" and gem_idx in disc_cnts:
                    outline(r, (255, 0, 0))

    def _draw_button(self, rect: Rect, label: str, alpha: int) -> None:
        """Draws the move Submit/Clear button."""
        # Scale
        r_win = self.to_window(rect).to_pygame()
        x0, y0 = r_win.topleft
        w, h = r_win.size

        # Background
        surface = pygame.Surface((w, h), pygame.SRCALPHA)
        surface.fill((30, 30, 30, alpha))
        self.window.blit(surface, (x0, y0))

        # Border
        pygame.draw.rect(self.window, (255, 255, 255), (x0, y0, w, h), 2)

        # Label
        txt = self.small_font.render(label, True, (255, 255, 255))
        # txt = self.small_font.render(label, fgcolor=(255,255,255))
        tx = x0 + (w - txt.get_width()) // 2  # center horizontally
        ty = y0 + (h - txt.get_height()) // 2  # center vertically
        self.window.blit(txt, (tx, ty))

    def draw_card_context_menu(
            self, 
            origin: Coord,
            button_specs: list[tuple[str, "GUIMove"]],
        ) -> dict:
        """When the player clicks a card, this paints buttons at the 
        card's top-right corner and returns {button_rect: move_idx}.

        That button will then lock the move in as the current
        selected move until Clear or another card menu is hit.
        """
        # Layout
        g = self.geom
        card_x, card_y = origin
        menu_x, menu_y = card_x + g.card.x - g.button.x, card_y

        # Draw
        rects = {}
        for i, (label, move) in enumerate(button_specs):
            # Box
            menu_height = i * g.button.y
            rect = Rect.from_size(menu_x, menu_y+menu_height, *g.button)
            self._draw_button(rect, label, alpha=200)
            rects[rect] = ("confirm", move)
        
        return rects
    
    def draw_move_confirm_button(
            self, 
            move: "GUIMove | None", 
            confirm_enabled: bool, 
            clear_enabled: bool
        ) -> dict:
        """Draws the top-level Confirm/Clear buttons.
        Update every time self._picked is changed.
        """
        # Whether to have Confirm and Clear active
        confirm_opacity = 255 if confirm_enabled else 80
        clear_opacity = 255 if clear_enabled else 80

        button_specs = [
            ("Confirm", ("confirm", move), confirm_opacity),
            ("Clear", ("clear", None), clear_opacity)
        ]

        # Draw buttons
        g = self.geom
        buttons = {}
        button_x, cur_y = g.confirm_origin
        for label, payload, opacity in button_specs:
            rect = Rect.from_size(button_x, cur_y, *g.button)
            self._draw_button(rect, label, opacity)
            cur_y += g.button.y

            # Only has payload when clickable/full opacity
            if opacity == 255:
                buttons[rect] = payload

        return buttons

    def draw_discard_notice(self) -> None:
        """Small 'Discard required' prompt to the right of the shop gems."""
        sx, sy = self.scale()
        x = int((self.geom.gem_origin.x + self.geom.gem.x + 40) * sx)
        y = int((self.geom.gem_origin.y - 30) * sy)
        txt = self.small_font.render("Discard required", True, (255, 0, 0))
        self.window.blit(txt, (x, y))

    def draw_reward_preview(self, lines: list[str]) -> None:
        """Bottom-left HUD: one line per entry in *lines*."""
        if not lines:
            return
        sx, sy = self.scale()
        x, y = self.geom.reward_origin
        x, y = int(x * sx), int(y * sy)

        for txt in lines:
            surf = self.font.render(txt, True, (255, 255, 0))
            self.window.blit(surf, (x, y))
            y += surf.get_height() + 4
