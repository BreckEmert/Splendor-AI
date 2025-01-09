# Splendor/run.py

import os
import sys
from datetime import datetime, timedelta

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from RL import ddqn_loop, debug_game, find_fastest_game  # type: ignore


def get_paths(layer_sizes):
    backup_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.getenv('WORKSPACE_DIR', backup_dir)
    agent_dir = os.path.join(base_dir, "RL", "trained_agents")
    nickname = "_".join(map(str, layer_sizes))

    model_save_path = os.path.join(agent_dir, nickname)
    log_dir = os.path.join(agent_dir, "game_logs")
    tensorboard_dir = os.path.join(agent_dir, "tensorboard_logs", nickname)

    return (model_save_path, log_dir, tensorboard_dir)

def main():
    layer_sizes = [64]
    model_save_path, log_dir, tensorboard_dir = get_paths(layer_sizes)

    # Function calls
    ddqn_loop(model_save_path = model_save_path, 
              preexisting_model_path = None, 
              layer_sizes = layer_sizes,  # model.py may need to be updated
              preexisting_memory = "random_memory.pkl",  # [None, random_memory.pkl, memory.pkl]
              log_dir = log_dir, 
              tensorboard_dir = tensorboard_dir)
    # debug_game(layer_sizes=layer_sizes, memory_path=None, log_path=log_path)
    # find_fastest_game(log_dir, append_to_prev_mem=True, base_dir)  # Ensure line 205 in player.py is uncommented


if __name__ == "__main__":
    # Need to go through and make sure .copy() and .deepcopy() are fully utilized
    # Just fixed two so there are probably more
    main()
