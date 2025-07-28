# Splendor/Play/render/overlay_renderer.py
"""
Renders any object that is not part of the base game
"""

import pygame

from Play.render import BoardGeometry, Rect


class OverlayRenderer:
    def __init__(self, window):
        self.geom = BoardGeometry()
        self.window = window
        self.font = pygame.font.SysFont(None, 32)
        self.small_font = pygame.font.SysFont(None, 28)

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

    def draw_selection_highlights(self, clickmap, focused_card, picked_gems):
        """Highlights selections that are queued for a move."""
        sx, sy = self.scale()

        def outline(rect, color):
            r_win = rect.scaled(sx, sy).to_pygame()
            pygame.draw.rect(self.window, color, r_win, 6)

        # Focused board card
        if focused_card:
            for r, payload in clickmap.items():
                if payload[0] == "card" and payload[1:] == focused_card:
                    outline(r, (255, 255, 0))
        else:
            # Picked gems
            for r, payload in clickmap.items():
                if payload[0] == "gem" and payload[1] in picked_gems:
                    outline(r, (0, 255, 0))

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
        tx = x0 + (w - txt.get_width()) // 2  # center horizontally
        ty = y0 + (h - txt.get_height()) // 2  # center vertically
        self.window.blit(txt, (tx, ty))

    def draw_card_context_menu(
            self, 
            tier: int, # Noting that this is class CardIndex, but that would be circular...
            pos: int, 
            button_specs: list[tuple[str, int]]
        ) -> dict:
        """When the player clicks a card, this paints buttons at the 
        card's top-right corner and returns {button_rect: move_index}.

        That button will then lock the move in as the current
        selected move until Clear or another card menu is hit.
        """
        # Layout of the menu
        g = self.geom
        card_x = g.deck_origin.x + (1 + pos)*(g.card.x + g.card_offset.w)
        card_y = g.deck_origin.y + (2 - tier)*(g.card.y + g.card_offset.h)

        menu_x, menu_y = card_x + g.card.x - g.button.x, card_y

        # Draw buttons on the menu
        rects = {}
        for i, (label, move) in enumerate(button_specs):
            menu_height = i * g.button.y
            r = Rect.from_size(menu_x, menu_y + menu_height, *g.button)
            r_win = self.to_window(r).to_pygame()
            pygame.draw.rect(self.window, (30,30,30), r_win)
            pygame.draw.rect(self.window, (255,255,255), r_win, 2)

            # Render text in that window
            txt = self.small_font.render(label, True, (255,255,255))
            self.window.blit(txt, (r_win.x+8, r_win.y+8))
            rects[r] = ("confirm", move)
        
        return rects
    
    def draw_move_confirm_button(
            self, 
            move_idx: int | None, 
            confirm_enabled: bool, 
            clear_enabled: bool
        ) -> dict:
        """Draws the top-level Confirm/Clear buttons.
        Update every time self._picked is changed.
        """
        # Whether to have Confirm and Clear active
        if confirm_enabled:
            confirm_opacity = 255
        else:
            move_idx = None
            confirm_opacity = 80
        clear_opacity = 255 if clear_enabled else 80

        button_specs = [
            ("Confirm", ("confirm", move_idx), confirm_opacity),
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

            # Not clickable when low opacity
            if opacity == 255:
                buttons[rect] = payload

        return buttons
