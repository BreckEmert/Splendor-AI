# Splendor/run.py

if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from RL import ddqn_loop, debug_game # type: ignore
    
    log_path = "/workspace/RL/trained_agents/game_logs"
    layer_sizes = [1024, 512, 512]

    base_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))
    layer_sizes_str = "_".join(map(str, layer_sizes))
    model_path = os.path.join(base_dir, "RL", "trained_agents", f"{layer_sizes_str}.keras")

    # train_agent(base_save_path, log_path, layer_sizes, none_path)
    ddqn_loop(layer_sizes, None)
    # debug_game(log_path, layer_sizes, None)