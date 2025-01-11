# Splendor/run.py

import os
import sys
from datetime import datetime, timedelta

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from RL import ddqn_loop, debug_game, find_fastest_game  # type: ignore


def get_paths(layer_sizes, model_from_name, memory_buffer_name):
    backup_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.getenv('WORKSPACE_DIR', backup_dir)
    rl_dir = os.path.join(base_dir, "RL")
    trained_agents_dir = os.path.join(rl_dir, "trained_agents")
    saved_files_dir = os.path.join(rl_dir, "saved_files")

    nickname = "_".join(map(str, layer_sizes))
    if model_from_name:
        model_from_path = os.path.join(trained_agents_dir, model_from_name)
        assert os.path.exists(model_from_path), f"{model_from_path} doesn't exist."
    else:
        model_from_path = None
    if memory_buffer_name:
        memory_buffer_path = os.path.join(saved_files_dir, memory_buffer_name)
    else:
        memory_buffer_path = None

    paths = {
        "base_dir": base_dir, 
        "layer_sizes": layer_sizes,
        "model_from_path": model_from_path, 
        "model_save_path": os.path.join(trained_agents_dir, nickname), 
        "memory_buffer_path": memory_buffer_path, 
        "rl_dir": rl_dir, 
        "saved_files_dir": saved_files_dir, 
        "log_dir": os.path.join(saved_files_dir, "game_logs"), 
        "states_log_dir": os.path.join(saved_files_dir, "game_states", nickname), 
        "tensorboard_dir": os.path.join(saved_files_dir, "tensorboard_logs", nickname)
    }

    for key, path in paths.items():
        if key.endswith("_dir") or key == "model_save_path":
            os.makedirs(path, exist_ok=True)

    return paths

def main():
    layer_sizes = [64]
    model_from_name = None  # "64_32.keras"
    memory_buffer = None  # "random_memory.pkl"
    paths = get_paths(layer_sizes, model_from_name, memory_buffer)

    # Function calls
    # ddqn_loop(paths, memory_buffer="random", log_rate=10)
    # debug_game(paths, memory_buffer=None)
    find_fastest_game(paths, n_games=2, log_states=True, append_to_prev_mem=False)  
        # !Uncomment line 205 in player.py!


if __name__ == "__main__":
    # Need to go through and make sure .copy() and .deepcopy() are fully utilized
    # Just fixed two so there are probably more
    main()
