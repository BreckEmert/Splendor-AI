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
from Play.clickmap_renderer import render_game_state
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
        self.window = pygame.display.set_mode(
            (1600, 960), pygame.RESIZABLE
        )
        pygame.display.set_caption("Splendor RL - Human vs DDQN")
        self.running = True

    def run(self):
        """Side thread for GUI handling."""
        while self.running and not self.game.victor:

            # Render and clickmap
            temp_path = "/tmp/frame.jpg"
            clickmap = render_game_state(self.game, temp_path)

            frame = pil_to_surface(Image.open(temp_path))
            frame = pygame.transform.smoothscale(
                frame, self.window.get_size()
            )
            self.window.blit(frame, (0, 0))
            pygame.display.flip()

            # Events
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
                    scale_x, scale_y = frame.get_width(), frame.get_height()
                    mouse_x = int(mouse_x * 5000 / scale_x)
                    mouse_y = int(mouse_y * 3000 / scale_y)

                    for (x0, y0, x1, y1), move_index in clickmap.items():
                        if x0 <= mouse_x <= x1 and y0 <= mouse_y <= y1:
                            legal_move = self.is_legal(move_index)
                            if legal_move:
                                self.human.feed_move(move_index)
                            break

    def is_legal(self, move_index: int) -> bool:
        """Check legality of human move."""
        legal_moves = self.game.active_player.get_legal_moves(
            self.game.board
        )
        return legal_moves[move_index]


def play_one_game(model_path: str):
    # Agents
    paths = get_paths([512, 512, 256], model_path, None, 0)
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
        print("Game over - winner:", winner)


if __name__ == "__main__":
    default_model = os.getenv("MODEL_PATH", None)
    if not default_model:
        raise SystemExit(
            "Set MODEL_PATH to a .keras file or pass as CLI arg."
        )
    play_one_game(default_model)
