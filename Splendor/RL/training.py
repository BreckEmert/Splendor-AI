# Splendor/RL/training.py

import numpy as np

from Environment.game import Game # type: ignore
from Environment.Splendor_components.Player_components.strategy import ( # type: ignore
    BestStrategy, RandomStrategy, OffensiveStrategy, ResourceHog, ObliviousStrategy
)


def train_agent(model_save_path):
    # Define the players and their strategies
    players = [('Player1', BestStrategy(), 1), ('Player2', BestStrategy(), 1)]
    
    # Initialize the RL agent
    state_size = 263  # ADJUST LATER
    batch_size = 32

    # Training loop
    for episode in range(1):  # Number of episodes for training
        game = Game(players)  # Reset the game for each episode
        state = np.array(game.to_vector())
        state = np.reshape(state, [1, state_size])
        done = False

        while not done:
            game.turn()  # Perform a turn in the game
            next_state = np.array(game.to_vector())
            next_state = np.reshape(next_state, [1, state_size])
            reward = 0  # Initial reward

            if game.is_final_turn:
                reward = 1 if game.get_victor().name == 'Player1' else -1
                done = True

            # Agent remembers
            active_player = game.active_player
            active_player.rl_model.remember(
                state, 
                active_player.move_index,  # Chosen move
                reward, 
                next_state, 
                done
            )

            # Update state
            state = next_state

            if len(active_player.rl_model.memory) > batch_size:
                active_player.rl_model.replay()

        # Log the progress
        print(f"Episode {episode+1}/1000 - Reward: {reward}")

    # Save the trained model
    for player in game.players:
        player.rl_model.save_model(f"{model_save_path}_{player.name}")

if __name__ == '__main__':
    model_save_path = 'rl_agent_model.keras'
    train_agent(model_save_path)