# Splendor/RL/training.py

import numpy as np
import os
from scipy.stats import norm

from Environment.Splendor_components.Player_components.strategy import ( # type: ignore
    BestStrategy, RandomStrategy, OffensiveStrategy, ResourceHog, ObliviousStrategy)
from Environment.game import Game # type: ignore
from RL import RLAgent # type: ignore


def debug_game(log_path, layer_sizes, model_path):
    import json

     # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_path)
    player2_model = RLAgent(layer_sizes, model_path)
    
    players = [
        ('Player1', BestStrategy(), 1, player1_model),
        ('Player2', BestStrategy(), 1, player2_model)
    ]

    for episode in range(5):
        # Enable logging for all games
        log_state = open(os.path.join(log_path, f"game_states_episode_{episode}.json"), 'w')
        log_move = open(os.path.join(log_path, f"moves_episode_{episode}.json"), 'w')

        game = Game(players)
        while not game.victor:
            game.turn()
            json.dump(game.get_state(), log_state)
            log_state.write('\n')
            log_move.write(str(game.active_player.chosen_move) + '\n')

        print(f"Simulated game {episode}")

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

    avg_half_turns = 500
    num_wins = 5
    max_half_turns = 140
    target_percentage = 0.1

    action_size = player1_model.action_size
    state_size = player1_model.state_size
    
    while avg_half_turns > 75:
        # Set player2 to be the most updated player 1 model
        player2_model.model.set_weights(player1_model.model.get_weights())
        
        i, current_wins = 0, 0
        game_lengths = []
        states_all, actions_all = [], []
        while current_wins < num_wins:
            game = Game(players)
            game.turn() # So that active_player exists
            while not game.victor and game.half_turns < max_half_turns:
                game.turn()

            game_lengths.append(game.half_turns)
            for player in game.players:
                if player.victor:
                    # print("won")
                    current_wins += 1
                    num_moves = len(player.rl_model.action_memory)
                    randoms = np.random.permutation(num_moves)

                    states = player.rl_model.state_memory[randoms]
                    actions = np.zeros((num_moves, action_size), dtype=int)
                    actions[np.arange(num_moves), player.rl_model.action_memory[randoms]] = 1

                    states_all.append(states)
                    actions_all.append(actions)

                # Reset the game
                player.rl_model.state_memory = np.empty((0, state_size), dtype=float)
                player.rl_model.action_memory = np.empty((0, ), dtype=int)

            i += 1
        
        # Batch train
        states_all = np.vstack(states_all)
        actions_all = np.vstack(actions_all)

        num_samples = states_all.shape[0]
        for start in range(0, num_samples, 128):
            end = min(start + 128, num_samples)
            player1_model.train_batch(states_all[start:end], actions_all[start:end])

        # Update hyperparameters
        avg_half_turns = np.mean(game_lengths)
        std_dev_half_turns = np.std(game_lengths)
        z_score_target = norm.ppf(1 - target_percentage)
        max_half_turns = int(avg_half_turns - z_score_target * std_dev_half_turns)

        epsilon = max(1.1 - np.exp(-0.02*(avg_half_turns-45)), 0.05) # y=1.1-e^{\left(-0.02\cdot\left(x-45\right)\right)}
        lr = min(0.000009 + 0.000008*np.exp(0.0406*(avg_half_turns-80)), 0.01) # y=0.000009+0.000008\cdot e^{0.0406\left(x-80\right)}

        player1_model.update_parameters(epsilon, lr)
        player2_model.update_parameters(epsilon, lr)
        
        print("Average half_turns:", avg_half_turns)
        print("New max_half_turns:", max_half_turns)
        print("New epsilon:", epsilon)
        print("New learning rate:", lr)

        # Save model
        player1_model.save_model(model_path)