# Splendor/train_ai.py

import os
from datetime import datetime, timedelta

from RL import ddqn_loop


def get_unique_filename(layer_sizes):
    nickname = "-".join(map(str, layer_sizes))
    timestamp = datetime.now() - timedelta(hours=6)
    timestamp = timestamp.strftime("%m-%d-%H-%M")  # mm/dd/hh/mm
    return f"{timestamp}__{nickname}"

def get_paths(layer_sizes, model_from_name, memory_buffer_name, log_rate):
    """I use '_dir' for folders and '_path'for things with extensions.
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

    images_dir = os.path.join(saved_files_dir, "rendered_games", nickname) if log_rate else None

    paths = {
        "base_dir": base_dir, 
        "layer_sizes": layer_sizes,
        "model_from_path": model_from_path, 
        "model_save_path": os.path.join(trained_agents_dir, f"{nickname}.keras"), 
        "memory_buffer_path": memory_buffer_path, 
        "rl_dir": rl_dir, 
        "saved_files_dir": saved_files_dir, 
        "images_dir": images_dir, 
        "tensorboard_dir": os.path.join(saved_files_dir, "tensorboard_logs", nickname)
    }

    # Make all of the directories
    for key, path in paths.items():
        if path:
            if key.endswith("_dir") :
                os.makedirs(path, exist_ok=True)
            elif key.endswith("_path"):
                os.makedirs(os.path.dirname(path), exist_ok=True)

    return paths

def main():
    layer_sizes = [512, 512, 512, 256]
    model_from_name = None  # "01-25-22-05__256-256.keras"
    memory_buffer = 'memory.pkl'  # None, 'memory.pkl' ?DOESNT EXIST ANHMORE?:'random_memory.pkl'
    log_rate = 25_000
    paths = get_paths(layer_sizes, model_from_name, memory_buffer, log_rate)
    print(paths)

    # Function calls
    ddqn_loop(paths, log_rate=log_rate)
    # debug_game(paths, memory_buffer=None)


if __name__ == "__main__":
    """If you're ever having issues make sure everything 
    you pull from the game is immutable - using .copy() 
    and copy.deepcopy() where needed.
    """
    main()
