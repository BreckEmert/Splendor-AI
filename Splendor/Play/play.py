# Splendor/Play/play.py
"""
python -m Splendor.Play.play /path/to/trained_model.keras
Small shim that delegates to gui_pygame.play_one_game().
"""

import sys
from Play.gui_pygame import play_one_game


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m Splendor.Play.play <model>.keras")
    play_one_game(sys.argv[1])
