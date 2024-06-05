# Splendor/run.py

if __name__ == "__main__":
    import sys
    import os

    # Add the parent directory to the Python path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

    from RL import train_agent

    model_save_path = 'RL/trained_agents/rl_agent_model.keras'
    log_path = "C:/Users/Public/Documents/Python_Files/Splendor/RL/trained_agents/game_logs"

    train_agent(model_save_path, log_path)
