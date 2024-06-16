# Splendor/run.py

if __name__ == "__main__":
    import sys
    import os

    # Add the parent directory to the Python path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

    from RL import priority_play, debug_game # type: ignore
    
    model_path = "C:/Users/Public/Documents/Python_Files/Splendor/RL/trained_agents/1024_512_512"
    none_path = None

    layer_sizes = [1024, 512, 512]
    # train_agent(base_save_path, log_path, layer_sizes, none_path)
    priority_play(layer_sizes, model_path)
    # debug_game(base_save_path, log_path, layer_sizes, none_path)