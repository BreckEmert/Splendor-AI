# Splendor/play_vs_ai.py
"""
Thin wrapper for gui_pygame.play_one_game().

Run from in the container:
export DISPLAY=host.docker.internal:0
export SDL_VIDEODRIVER=x11
python -m play

(on linux `export DISPLAY=:0`)

# Testing if something goes wrong:
echo $DISPLAY
ls -l /tmp/.X11-unix
glxinfo -B | head -4
glxgears -info | head -3
"""

import os
import sys
import threading
from pathlib import Path

from Environment.Splendor_components.Player_components import HumanAgent
from RL import InferenceAgent
from Play import GUIGame
from Play.gui_pygame import SplendorGUI


def _resolve_model_path(argv: list[str]) -> str:
    # Prioritize passed in model
    if len(argv) >= 2:
        return argv[1]

    env = os.getenv("MODEL_PATH")
    if env:
        return env
    
    # Default model
    trained_agent_dir = Path(__file__).with_name("RL") / "trained_agents"
    model = trained_agent_dir / "inference_model.keras"
    if model.exists():
        return str(model)
    
    raise SystemExit(
        "No model found or supplied.\n"
        "Pass a path (python -m Splendor.play /path/model.keras), "
        "set MODEL_PATH, or drop a *.keras file in RL/trained_agents/"
    )

def play_one_game(model_path: str):
    # Game
    rl_agent = InferenceAgent(model_path)
    human_agent = HumanAgent()
    players = [("DDQN", rl_agent, 0), ("Human", human_agent, 1)]
    game = GUIGame(players, rl_agent)

    # GUI thread
    gui = SplendorGUI(game, human_agent)
    gui_thread = threading.Thread(target=gui.run, daemon=True)
    gui_thread.start()

    # Game loop
    while not game.victor and gui.running:
        game.turn()

    if gui.running:
        winner = game.players[0] if game.players[0].victor else game.players[1]
        print("Game over - winner:", winner.name)


if __name__ == "__main__":
     model_path = _resolve_model_path(sys.argv)
     play_one_game(model_path)
