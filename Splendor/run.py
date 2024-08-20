# Splendor/run.py

import os
import sys

from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from RL import ddqn_loop, debug_game, find_fastest_game  # type: ignore


if __name__ == "__main__":
    log_path = "/workspace/RL/trained_agents/game_logs"
    time = (datetime.now() - timedelta(hours=5)).strftime("%m%d-%H%M")
    tensorboard_dir = os.path.join(log_path, "tensorboard_logs", time)
    nickname = "ffn_pos_sumgems"
    # layer_sizes = [256, 128, 256, 128]
    # Previous1 [256, 64, 256, 64]
    # layer_sizes = [128, 64, 128, 64]  # For regular training
    layer_sizes = [64]

    base_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))
    layer_sizes_str = "_".join(map(str, layer_sizes))
    model_path = os.path.join(base_dir, "RL", "trained_agents", nickname, layer_sizes_str)
    from_model_path = os.path.join(model_path, "model.keras")

    # memory_path = "/workspace/RL/random_memory.pkl"
    memory_path = "/workspace/RL/memory.pkl"
    
    ddqn_loop(model_path=model_path, 
              from_model_path = from_model_path,  # do not run without deleting line 194 player.py
              layer_sizes=layer_sizes, 
              memory_path=memory_path, 
              log_path=log_path, 
              tensorboard_dir=tensorboard_dir)
    # debug_game(layer_sizes=layer_sizes, memory_path=None, log_path=log_path)
    # find_fastest_game(memory_path=memory_path) f# uncomment line 194
