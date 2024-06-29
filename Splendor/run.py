# Splendor/run.py

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from RL import ddqn_loop, debug_game, find_fastest_game # type: ignore
    
    log_path = "/workspace/RL/trained_agents/game_logs"
    time = (datetime.now() - timedelta(hours=5)).strftime("%m%d-%H%M")
    tensorboard_dir = os.path.join(log_path, "tensorboard_logs", time)
    layer_sizes = [512, 256, 128]
    layer_sizes = [256, 128, 512, 128]

    base_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))
    layer_sizes_str = "_".join(map(str, layer_sizes))
    model_path = os.path.join(base_dir, "RL", "trained_agents", f"{layer_sizes_str}.keras")
    model_path = "/workspace/RL/trained_agents/model.keras"

    # ddqn_loop(model_path=model_path, 
    #           layer_sizes=layer_sizes, 
    #           memories=True, 
    #           log_path=log_path, 
    #           tensorboard_dir=tensorboard_dir)
    debug_game(layer_sizes=layer_sizes, memories=True, log_path=log_path)
    # find_fastest_game(model_path=None, layer_sizes=layer_sizes, append_to_previous=False)