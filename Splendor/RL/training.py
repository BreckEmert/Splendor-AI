# Splendor/RL/training.py

import numpy as np
import os

from Environment.Splendor_components.Player_components.strategy import ( # type: ignore
    BestStrategy, RandomStrategy, OffensiveStrategy, ResourceHog, ObliviousStrategy
)
from Environment.game import Game # type: ignore
from RL import RLAgent # type: ignore


def debug_game(base_save_path, log_path, layer_sizes, model_path):
     # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_path)
    player2_model = RLAgent(layer_sizes, model_path)
    
    players = [
        ('Player1', BestStrategy(), 1, player1_model, 0),
        ('Player2', BestStrategy(), 1, player2_model, 1)
    ]

    for episode in range(1000):
        # Enable logging for all games
        log_state = open(os.path.join(log_path, f"game_states_episode_{episode}.json"), 'w')
        log_move = open(os.path.join(log_path, f"moves_episode_{episode}.json"), 'w')

        simulate_game(players, True, log_state, log_move)
        # game = simulate_game(players, False, None, None)
        # print(f"Simulated game {episode}")

def simulate_game(players):
    game = Game(players)

    while not game.victor and game.half_turns < 350:
        game.turn()
    
    return game

def priority_play(layer_sizes, model_path):
    """searches tons of games and selects the fastest 10% for training"""
    # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_path)
    player2_model = RLAgent(layer_sizes, model_path)
    
    players = [
        ('Player1', BestStrategy(), 1, player1_model),
        ('Player2', BestStrategy(), 1, player2_model)
    ]

    num_wins = 1
    max_states = 400
    target_percentage = 0.1

    action_size = player1_model.action_size
    state_size = player1_model.state_size
    batch_size = 128
    
    while True:
        # Set player2 to be the most updated player 1 model
        player2_model.model.set_weights(player1_model.model.get_weights())

        # Preallocate space to avoid padding
        final_states = np.empty((num_wins, max_states, state_size), dtype=int) # Will need to be floats when we do calculations
        final_actions = np.zeros((num_wins, max_states, action_size), dtype=int)

        final_masks = np.zeros((num_wins, max_states), dtype=bool)
        final_lengths = np.empty((num_wins, ), dtype=int)
        
        i, current_wins = 0, 0
        while current_wins < num_wins:
            game = Game(players)
            game.turn() # So that active_player exists
            while not game.victor and game.active_player.rl_model.num_predicts < max_states-6: # Single turn makes up to 6 predictions
                game.turn()

            for player in game.players:
                if player.victor:
                    print("won")
                    num_moves = len(player.rl_model.action_memory)

                    # Place game into preallocated arrays
                    final_states[current_wins][ :num_moves] = player.rl_model.state_memory
                    final_actions[current_wins][np.arange(num_moves), player.rl_model.action_memory] = 1

                    final_masks[current_wins][ :num_moves] = True
                    final_lengths[current_wins] = player.rl_model.num_predicts
                    
                    current_wins += 1

                # Reset the game
                player.rl_model.state_memory = np.empty((0, state_size), dtype=float)
                player.rl_model.action_memory = np.empty((0, ), dtype=int)
                player.rl_model.num_predicts = 0

            i += 1
        
        # Update the rolling average target
        if current_wins/i > target_percentage:
            max_states = int(max_states * 0.98)
        print("Percentage of games won:", round(current_wins/i, 2), "max_states:", max_states)

        # Flatten the arrays to match the model input
        final_states = final_states.reshape(-1, state_size)
        final_actions = final_actions.reshape(-1, action_size)
        final_masks = final_masks.reshape(-1)

        # Shuffle the data
        indices = np.arange(final_states.shape[0])
        np.random.shuffle(indices)
        final_states = final_states[indices]
        final_actions = final_actions[indices]
        final_masks = final_masks[indices]

        # Train on memories in batches
        num_samples = final_states.shape[0]
        print("num_samples:", num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_states = final_states[start:end]
            batch_actions = final_actions[start:end]
            batch_masks = final_masks[start:end]

            player1_model.train_batch(batch_states, batch_actions, batch_masks)

        # Save model
        player1_model.save_model(model_path)