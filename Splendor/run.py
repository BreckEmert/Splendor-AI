# Splendor/run.py

import os
from datetime import datetime

from RL import ddqn_loop, debug_game, find_fastest_game  # type: ignore


def get_unique_filename(layer_sizes):
    nickname = "_".join(map(str, layer_sizes))
    timestamp = datetime.now().strftime("%m_%d_%H_%M")  # mm/dd/hh/mm
    return f"{nickname}_{timestamp}"

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
        # "log_dir": os.path.join(saved_files_dir, "game_logs"), 
        "states_log_dir": os.path.join(saved_files_dir, "game_states", nickname), 
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
    layer_sizes = [64]
    model_from_name = None  # "64_32.keras"
    memory_buffer = "random_memory.pkl"  # "random_memory.pkl"
    paths = get_paths(layer_sizes, model_from_name, memory_buffer)
    print(paths)

    # Function calls
    ddqn_loop(paths, log_rate=10)
        #! Comment line 205 in player.py!
    # debug_game(paths, memory_buffer=None)
    # find_fastest_game(paths, n_games=1, log_states=True)  f
        # !Uncomment line 205 in player.py!

 
if __name__ == "__main__":
    """If you're ever having issues make sure everything you pull from 
    the game is immutable - using .copy() and copy.deepcopy() where needed.  
    Also note that the model will remember multiple times per turn, 
    because of the system I have to deal with the combinatoric space 
    required to take 3 tokens in a single pass.  This means states, game
    frames, and memories may not agree in length.
    """
    main()
