# Play/render/overlay_renderer.py
"""
Renders any object that is not part of the base game
"""

import pygame

from Play.render.board_renderer import card_width, card_height


class OverlayRenderer:
    def __init__(self, window):
        self.window = window
        self.font = pygame.font.SysFont(None, 32)
        self.small_font = pygame.font.SysFont(None, 28)
        # Any other persistent resources (colors, pre-loaded surfaces) can go here

    def _draw_button(self, rect, label: str, opacity: int) -> None:
        """Draws all UI buttons."""
        # Background
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((30, 30, 30, opacity))
        self.window.blit(surf, rect[:2])
        
        # Border
        pygame.draw.rect(self.window, (255, 255, 255), rect, 2)

        # Label
        font = pygame.font.SysFont(None, 32)
        txt = font.render(label, True, (255, 255, 255))
        self.window.blit(txt, (rect[0] + 20, rect[1] + 20))

    def draw_card_context_menu(self, tier: int, pos: int, button_specs) -> dict:
        """Upon the player clicking on a card, this paints 
        up to two buttons at the card's top‑right corner 
        and returns {button_rect: move_index}.

        That button will then lock the move in as the current
        selected move until Clear or another card menu is hit.

        Consults _card_to_move to ......
        """
        # list[(label, move_idx)] is already legality‑filtered by caller
        legal_moves = button_specs

        # Layout of the menu
        button_width, button_height  = 140, 60  # size of each row
        card_x = 200 + card_width + 50 + pos*(card_width+10)
        card_y = 680 + (2 - tier)*(card_height + 50)

        menu_x, menu_y = card_x + card_width - button_width, card_y

        # Draw buttons on the menu
        rects = {}
        for i, (label, move) in enumerate(legal_moves):
            r = (menu_x,
                 menu_y + i*button_height,
                 menu_x + button_width,
                 menu_y + (i+1)*button_height)
            pygame.draw.rect(self.window, (30,30,30), r)
            pygame.draw.rect(self.window, (255,255,255), r, 2)

            # Render text in that window
            font = pygame.font.SysFont(None, 28)
            txt = font.render(label, True, (255,255,255))
            self.window.blit(txt, (r[0]+8, r[1]+8))
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
        button_width, button_height = 200, 80
        base_x, base_y = 1000, 2600  # Under gems row in 5000x3000

        # Whether to have Confirm active
        if confirm_enabled:
            confirm_opacity = 255
        else:
            move_idx = None
            confirm_opacity = 80

        # Whether to have Clear active
        if clear_enabled:
            clear_opacity = 255
        else:
            clear_opacity = 80

        # Always show Clear, show Confirm when available
        button_specs = [
            ("Confirm", ("confirm", move_idx), confirm_opacity),
            ("Clear", ("clear", None), clear_opacity)
        ]

        # Draw available buttons
        buttons = {}
        for i, (label, payload, opacity) in enumerate(button_specs):
            rect = (
                base_x,
                base_y + i * button_height,
                base_x + button_width,
                base_y + (i + 1) * button_height,
            )
            self._draw_button(rect, label, opacity)
            buttons[rect] = payload

        return buttons
