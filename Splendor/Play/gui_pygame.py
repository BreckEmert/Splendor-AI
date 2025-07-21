# Splendor/Play/gui_pygame.py
"""
Drive a Splendor game with one HumanAgent opponent.
"""

import os
import sys
import threading
import pygame
from PIL import Image

from Environment.game import Game
from RL.model import RLAgent
from Play.human_agent import HumanAgent
from Play.clickmap_renderer import render_game_state, card_width, card_height
from meta.generate_images import take_3_indices, take_2_diff_indices
from run import get_paths


def pil_to_surface(pil_image):
    """Convert PIL image to pygame surface."""
    return pygame.image.fromstring(
        pil_image.tobytes(), pil_image.size, pil_image.mode
    ).convert()

class SplendorGUI:
    def __init__(self, game, human):
        pygame.init()
        self.game = game
        self.human = human
        self.window = pygame.display.set_mode((1600, 960), pygame.RESIZABLE)
        pygame.display.set_caption("Splendor RL - Human vs DDQN")
        self.running = True
        self._focus = None  # (tier, pos) of clicked card
        self._ctx_rects = {}  # overlay button → move_index
        self._picked: list[int] = []  # colors clicked this turn

    def _is_selection_legal(self, selection, legal_mask):
        """Checks if the currently selected move is legal."""
        return legal_mask[selection]

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

    def _draw_card_context_menu(self, tier: int, pos: int, legal_mask):
        """Paints three small buttons at the card's top‑right
        corner and returns {button_rect: move_index}.
        Consults _card_to_move to ......
        """
        # This just has self.game.active_player... this means we never need to pass player around, no?
        player = self.game.active_player

        # Render only context options which are legal
        buy = self._card_to_move(tier, pos, "buy", player)
        reserve = self._card_to_move(tier, pos, "reserve", player)

        # This is going to have to have logic for buy with gold.
        # Speaking of, I'm not actually sure how this would integrate.
        # Because my game logic only has auto spend.  Maybe I can still
        # choose manually no problem?
        options = [("Buy 🔵⚪🪙", buy), ("Reserve", reserve)]
        legal_opts = [
            (label, move) for label, move in options 
            if legal_mask[move]
        ]
        buttons = [
            ()
        ]

        # Layout of the menu
        button_width, button_height  = 140, 60  # size of each row
        card_x = 200 + card_width + 50 + pos*(card_width+10)
        card_y = 680 + (2 - tier)*(card_height + 50)

        menu_x, menu_y = card_x + card_width - button_width, card_y

        # Draw buttons on the menu
        rects = {}
        for i, (label, move) in enumerate(legal_opts):
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
            rects[r] = move
        
        return rects
    
    @staticmethod
    def _gems_to_move(picked: list[int], player) -> int | None:
        """Map selected gems to the engine's move_index.
        Returns None if the move is not yet legal.

        Goal with these two "to_move" methods:
            1) flashing red card on illegal click
            2) Confirm button goes green when a valid move is selected
        """
        sel = sorted(picked)
        n = len(sel)
        discards = max(0, player.gems.sum() + n - 10)

        # Take 3 different
        if n == 3 and len(set(sel)) == 3:
            idx = take_3_indices.index(tuple(sel))  # 0‑9
            return idx*4 + discards                 # 0‑39

        # Take 2 same
        if n == 2 and sel[0] == sel[1]:
            return 40 + sel[0]*3 + discards  # 40‑54

        # Take 2 different
        if n == 2:
            idx = take_2_diff_indices.index(tuple(sel))  # 0‑9
            return 55 + idx*3 + discards                 # 55‑84

        # Take 1
        if n == 1:
            return 85 + sel[0]*2 + discards  # 85‑94

        # Here, we have to return None (different than _card_to_move)
        # because a selection can *eventually* end up legal.  If the
        # player discards gems after going over 10, it becomes legal.
        return None
    
    def _card_to_move(self, tier: int, pos: int, variant: str) -> int:
        """Map selected card + option to a move_index.
        Always returns an int because caller filters legality.
        """
        player = self.game.active_player
        take_dim = player.take_dim
        buy_dim = player.buy_dim

        match variant:
            case "buy":
                return take_dim + 2*(tier*4 + pos)
            case "buy_gold":
                return take_dim + 2*(tier*4 + pos) + 1
            case "reserve":
                return take_dim + buy_dim + tier*5 + pos

        raise ValueError("Error: no legal card move_index was found.")

    def _draw_move_confirm_button(self):
        """Draws the top-level Confirm/Clear buttons.
        Update every time self._picked is changed.
        """
        button_width, button_height = 200, 80
        base_x, base_y = 1000, 2600  # Under gems row in 5000x3000
        move_idx = self._gems_to_move(self._picked, self.game.active_player)

        # Always show Clear, show Confirm when available
        opacity = 255 if move_idx is not None else 80
        button_specs = [
            ("Confirm", ("confirm", None), 255),
            ("Clear", ("clear", None), opacity)
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

    def run(self):
        """Side thread for GUI handling."""
        while self.running and not self.game.victor:
            # Render
            temp_path = "/tmp/frame.jpg"
            clickmap = render_game_state(self.game, temp_path)

            frame = pil_to_surface(Image.open(temp_path))
            frame = pygame.transform.smoothscale(frame, self.window.get_size())
            self.window.blit(frame, (0, 0))
            pygame.display.flip()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()

                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and self.game.active_player is self.human
                ):
                    mouse_x, mouse_y = event.pos
                    scale_x, scale_y = (
                        frame.get_width(), frame.get_height()
                    )
                    mouse_x = int(mouse_x * 5000 / scale_x)
                    mouse_y = int(mouse_y * 3000 / scale_y)

                    # Check context-menu buttons
                    for rect, payload in self._ctx_rects.items():
                        x0, y0, x1, y1 = rect
                        if x0 <= mouse_x <= x1 and y0 <= mouse_y <= y1:
                            action, arg = payload
                            if action == "clear":
                                self._picked.clear()
                            elif action == "confirm" and arg is not None:
                                self.human.feed_move(arg)

                            self._focus = None
                            self._ctx_rects.clear()
                            break

                    else:
                        # Normal board clicks
                        for (x0, y0, x1, y1), token in clickmap.items():
                            if (x0 <= mouse_x <= x1 and
                                    y0 <= mouse_y <= y1):
                                if token and token[0] == "card":
                                    self._focus = token[1:]
                                elif token and token[0] == "gem":
                                    color = token[1]
                                    if color in self._picked:
                                        self._picked.remove(color)
                                    elif len(self._picked) < 3:
                                        self._picked.append(color)
                                break

            # Overlay
            if self._focus:
                self._ctx_rects = self._draw_card_context_menu(*self._focus)
            elif self._picked:
                self._ctx_rects = self._draw_move_confirm_button()
            else:
                self._ctx_rects.clear()


def play_one_game(model_path: str):
    # Agents
    paths = get_paths([512, 512, 256], model_path, None, 0)  # This layer sizes is hardcoded and should probably be dynamic
    rl_agent = RLAgent(paths)
    human_agent = HumanAgent()

    # Game
    players = [("Human", human_agent), ("DDQN", rl_agent)]
    game = Game(players, rl_agent)

    # GUI thread
    gui = SplendorGUI(game, human_agent)
    gui_thread = threading.Thread(target=gui.run, daemon=True)
    gui_thread.start()

    # Game loop
    while not game.victor and gui.running:
        game.turn()  # blocks if human's turn

    if gui.running:
        winner = game.players[0] if game.players[0].victor else game.players[1]
        print("Game over - winner:", winner.name)


if __name__ == "__main__":
    default_model = os.getenv("MODEL_PATH", None)
    if not default_model:
        msg = "Set MODEL_PATH to a .keras file or pass as CLI arg."
        raise SystemExit(msg)
    
    play_one_game(default_model)
