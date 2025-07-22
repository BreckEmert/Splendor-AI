# Splendor/Play/gui_pygame.py
"""
Drive a Splendor game with one HumanAgent opponent.
"""

import os
import sys
import threading
import pygame
from PIL import Image

from RL.model import RLAgent
from Play.gui_game import GUIGame
from Play.human_agent import HumanAgent
from Play.render.board_renderer import render_game_state
from Play.render.overlay_renderer import OverlayRenderer
from meta.generate_images import take_3_indices, take_2_diff_indices
from train import get_paths


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
        self.overlay = OverlayRenderer(self.window)
        pygame.display.set_caption("Splendor RL - Human vs DDQN")
        self.running = True
        self._focus = None  # (tier, pos) of clicked card
        self._ctx_rects = {}  # maps overlay button to a move_index
        self._picked: list[int] = []  # colors clicked this turn

    def _is_move_legal(self, move_index: int | None) -> bool:
        """Checks if the currently selected move is legal."""
        # Remember that legal_mask is fresh, from await_move()
        return (move_index is not None) and bool(self.human.legal_mask[move_index])

    def _gem_click_allowed(self, color: int) -> bool:
        """Returns whether the player is allowed to click this gem.

        Rules implemented:
        1. Toggle-off always allowed.
        2. Max three picks total (two for taking two of the same).
        3. There must be at least one token of that kind.
        4. A second click of the same color is allowed when:
            - it would be the 2nd pick overall
            - the stack has ≥4 tokens
        """
        supply = self.game.board.gems[color]
        picked = self._picked

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
        
        # All rules satisfied, return True
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

        # Take 3 different
        if n == 3 and len(set(sel)) == 3:
            a, b, c = sel  # sel is variable length so using 3 for the index would raise pylance... fix better later?
            idx = take_3_indices.index((a, b, c))  # 0‑9
            return idx*4 + discards                 # 0‑39

        # Take 2 same
        if n == 2 and sel[0] == sel[1]:
            return 40 + sel[0]*3 + discards  # 40‑54

        # Take 2 different
        if n == 2:
            a, b = sel
            idx = take_2_diff_indices.index((a, b))  # 0‑9
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

        match variant:
            case "buy":
                return player.take_dim + 2*(tier*4 + pos)
            case "buy_gold":
                return player.take_dim + 2*(tier*4 + pos) + 1
            case "reserve":
                return player.take_dim + player.buy_dim + tier*5 + pos

        raise ValueError("Error: no legal card move_index was found.")

    def _handle_board_click(self, mouse_x, mouse_y, event):
        """Click anywhere but on a card context menu"""
        for (x0, y0, x1, y1), token in self.clickmap.items():
            if x0 <= mouse_x <= x1 and y0 <= mouse_y <= y1:
                # Board card was clicked
                if token[0] == "card":
                    self._focus = token[1:]

                # Board gem was clicked
                elif token[0] == "gem":
                    color = token[1]

                    left_click = event.button == 1
                    right_click = event.button == 3

                    # Remove token from _picked upon right click
                    if right_click and color in self._picked:
                        self._picked.remove(color)

                    # Add to token to _picked if allowed
                    elif left_click and self._gem_click_allowed(color):
                        self._picked.append(color)
                
                # Reserved card was clicked
                elif token[0] == "move":
                    self.human.feed_move(token[1])

                break
    
    def _handle_context_menu_click(self, payload):
        """Click on a card context menu"""
        action, move_index = payload

        # Clear selection
        if action == "clear":
            self._picked.clear()

        # Confirm selected move
        elif action == "confirm" and move_index is not None:
            self.human.feed_move(move_index)

        # Reset overlay
        self._focus = None
        self._ctx_rects.clear()

    def _handle_mouse_event(self, event, frame):
        mouse_x, mouse_y = event.pos
        scale_x, scale_y = frame.get_width(), frame.get_height()
        mouse_x = int(mouse_x * 5000 / scale_x)
        mouse_y = int(mouse_y * 3000 / scale_y)

        # Context menu buttons
        for rect, payload in self._ctx_rects.items():
            x0, y0, x1, y1 = rect
            if x0 <= mouse_x <= x1 and y0 <= mouse_y <= y1:
                self._handle_context_menu_click(payload)
                break

        # Regular board click
        else:
            self._handle_board_click(mouse_x, mouse_y, event)

    def _handle_event(self, event, frame):
        if event.type == pygame.QUIT:
            self.running = False
            pygame.quit()
            sys.exit()
        elif (event.type == pygame.MOUSEBUTTONDOWN
              and self.game.active_player is self.human):
            self._handle_mouse_event(event, frame)

    def _card_menu_options(self, tier, pos):
        """Returns an ordered list of pairs of 
        legal moves for the card context menu.
        """
        player = self.game.active_player
        buy = self._card_to_move(tier, pos, "buy")
        reserve = self._card_to_move(tier, pos, "reserve")
        moves = [("Buy 🔵⚪🪙", buy), ("Reserve", reserve)]

        # Return only legal menu options to be rendered
        return [(label, move) for label, move in moves if self._is_move_legal(move)]

    def run(self):
        """Side thread for GUI handling."""
        while self.running and not self.game.victor:
            # Render frame and clickmap
            temp_path = "/tmp/frame.jpg"
            self.clickmap = render_game_state(self.game, temp_path)

            frame = pil_to_surface(Image.open(temp_path))
            frame = pygame.transform.smoothscale(
                frame,
                self.window.get_size()
            )
            self.window.blit(frame, (0, 0))
            pygame.display.flip()

            # Process pygame events
            for event in pygame.event.get():
                self._handle_event(event, frame)

            # Draw UI overlay (context menus or confirm buttons)
            if self._focus:
                tier, pos = self._focus
                opts = self._card_menu_options(tier, pos)
                self._ctx_rects = self.overlay.draw_card_context_menu(
                    tier, pos, opts
                )
            elif self._picked:
                move_index = self._gems_to_move(self._picked)
                confirm_enabled = self._is_move_legal(move_index)
                clear_enabled = bool(self._picked)
                self._ctx_rects = self.overlay.draw_move_confirm_button(
                    move_index, confirm_enabled, clear_enabled
                )
            else:
                self._ctx_rects.clear()


def play_one_game(model_path: str):
    # Agents
    paths = get_paths([1, 1, 1], model_path, None, 0)  # Layer sizes is stuck as an arg for now
    rl_agent = RLAgent(paths)
    human_agent = HumanAgent()

    # Game
    players = [("Human", human_agent), ("DDQN", rl_agent)]
    game = GUIGame(players, rl_agent)

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
