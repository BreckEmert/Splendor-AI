# Splendor/run.py

if __name__ == "__main__":
    import sys
    import os

    # Add the parent directory to the Python path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

    from RL import train_agent # type: ignore
    
    base_save_path = 'RL/trained_agents'
    log_path = "C:/Users/Public/Documents/Python_Files/Splendor/RL/trained_agents/game_logs"
    primary_model_paths = [
        "C:/Users/Public/Documents/Python_Files/Splendor/RL/trained_agents/Player1_48_48_48/model.keras",
        "C:/Users/Public/Documents/Python_Files/Splendor/RL/trained_agents/Player2_48_48_48/model.keras"
    ]
    token_model_paths = [
        None, 
        None
    ]

    primary_model_sizes = [48, 48, 48]
    token_model_sizes = [48, 48, 10]
    train_agent(base_save_path, log_path, primary_model_sizes, token_model_sizes, primary_model_paths, token_model_paths)
