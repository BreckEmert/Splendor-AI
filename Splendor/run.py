# Splendor/run.py

import os
import sys
from datetime import datetime, timedelta

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from RL import ddqn_loop, debug_game, find_fastest_game  # type: ignore


def main():
    # Establish some paths
    backup_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.getenv('WORKSPACE_DIR', backup_dir)
    
    # Set the RL model layer sizes
    layer_sizes = [64]
    layer_sizes_str = "_".join(map(str, layer_sizes))
    nickname = layer_sizes_str
    
    # Model and memory paths
    agent_path = os.path.join(base_dir, "RL", "trained_agents")
    model_from_path = os.path.join(agent_path, "model.keras")  # (changeable)
    model_save_path = os.path.join(agent_path, nickname, layer_sizes_str)

    log_dir = os.path.join(agent_path, "game_logs")
    time = (datetime.now() - timedelta(hours=5)).strftime("%m%d-%H%M")
    tensorboard_dir = os.path.join(log_dir, "tensorboard_logs", time)

    # Print paths
    print(f"Base directory: {base_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Model save path: {model_save_path}")
    print(f"Preexisting model path: ", model_from_path)
    print(f"Tensorboard directory: {tensorboard_dir}")


    # Function calls
    # ddqn_loop(model_save_path=model_save_path, 
    #           model_from_path=None, 
    #           layer_sizes=layer_sizes, 
    #           preexisting_memory=None, 
    #           log_path=log_path, 
    #           tensorboard_dir=tensorboard_dir)
    # debug_game(layer_sizes=layer_sizes, memory_path=None, log_path=log_path)
    find_fastest_game(base_dir, append_to_prev_mem=True)  # Ensure line 196 in player.py is uncommented


if __name__ == "__main__":
    # Need to go through and make sure .copy() and .deepcopy() are fully utilized
    # Just fixed two so there are probably more
    main()