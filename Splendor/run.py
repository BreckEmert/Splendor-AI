# Splendor/run.py

if __name__ == "__main__":
    import sys
    import os

    # Add the parent directory to the Python path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

    from RL import train_agent

    model_save_path = 'rl_agent_model.keras'
    train_agent(model_save_path)
