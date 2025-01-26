# Splendor/run.py

import os
from datetime import datetime, timedelta

from RL import ddqn_loop, find_fastest_game


def get_unique_filename(layer_sizes):
    nickname = "-".join(map(str, layer_sizes))
    timestamp = datetime.now() - timedelta(hours=6)
    timestamp = timestamp.strftime("%m-%d-%H-%M")  # mm/dd/hh/mm
    return f"{timestamp}__{nickname}"

def get_paths(layer_sizes, model_from_name, memory_buffer_name):
    """I use '_dir' for folders and '_path' for things with extensions.
    Could be a dataclass if I get around to it.
    """
    backup_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.getenv('WORKSPACE_DIR', backup_dir)

    rl_dir = os.path.join(base_dir, "RL")
    trained_agents_dir = os.path.join(rl_dir, "trained_agents")
    saved_files_dir = os.path.join(rl_dir, "saved_files")

    nickname = get_unique_filename(layer_sizes)

    model_from_path = os.path.join(trained_agents_dir, model_from_name) if model_from_name else None
    memory_buffer_path = os.path.join(saved_files_dir, memory_buffer_name) if memory_buffer_name else None
    assert not model_from_path or os.path.exists(model_from_path), f"{model_from_path} doesn't exist."

    paths = {
        "base_dir": base_dir, 
        "layer_sizes": layer_sizes,
        "model_from_path": model_from_path, 
        "model_save_path": os.path.join(trained_agents_dir, f"{nickname}.keras"), 
        "memory_buffer_path": memory_buffer_path, 
        "rl_dir": rl_dir, 
        "saved_files_dir": saved_files_dir, 
        "images_dir": os.path.join(saved_files_dir, "rendered_games", nickname), 
        "tensorboard_dir": os.path.join(saved_files_dir, "tensorboard_logs", nickname)
    }

    # Make all of the directories
    for key, path in paths.items():
        if key.endswith("_dir") :
            os.makedirs(path, exist_ok=True)
        elif key.endswith("_path") and path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    return paths

def main():
    layer_sizes = [256, 256]
    model_from_name = None  # "64_01_14_04_51.keras"
    memory_buffer = 'memory.pkl'  # 'memory.pkl' 'random_memory.pkl'
    paths = get_paths(layer_sizes, model_from_name, memory_buffer)
    print(paths)

    # Function calls
    ddqn_loop(paths, log_rate=0)
    # debug_game(paths, memory_buffer=None)
    # find_fastest_game(paths, n_games=2, log_states=False)
        # !Uncomment line 205 in player.py!

 
if __name__ == "__main__":
    """If you're ever having issues make sure everything you pull from 
    the game is immutable - using .copy() and copy.deepcopy() where needed.
    """
    main()
       