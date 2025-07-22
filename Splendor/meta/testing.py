# Splendor/RL/training.py

import numpy as np
import os
import random

from Environment.Splendor_components.Player_components.strategy import ( # type: ignore
    BestStrategy, RandomStrategy, OffensiveStrategy, ResourceHog, ObliviousStrategy
)
from Environment.rl_game import Game # type: ignore
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
    """searches tons of games and selects the out 10%s for training"""
    # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_path)
    player2_model = RLAgent(layer_sizes, model_path)
    
    players = [
        ('Player1', BestStrategy(), 1, player1_model, 0),
        ('Player2', BestStrategy(), 1, player2_model, 1)
    ]

    n_128 = 1
    sims = 1
    victor_memory = []
    loser_memory = []
    for i in range(sims):
        # game = simulate_game(players, False, None, None)
        game = simulate_game(players)
        print(f"simulated game {i}")
        for player in game.players:
            if player.victor:
                victor_memory.extend(player.rl_model.memory)
            else:
                loser_memory.extend(player.rl_model.memory)
        
    # Sort and select memory
    victor_memory.sort(key=len, reverse=True)
    loser_memory.sort(key=len, reverse=True)

    shortest_victor = random.sample(victor_memory, 1) # [ :sims//5]   128*n_128
    longest_loser = random.sample(loser_memory, 1)
    print("Average lengths of 10 percents:", np.mean([len(mem) for mem in shortest_victor]), np.mean([len(mem) for mem in longest_loser]))

    # Train on memories
    print("Training on memories")
    player1_model.train_batch(shortest_victor, True)
    player1_model.train_batch(longest_loser, False)

    # Save model
    print("Saving model")
    player1_model.save_model(model_path)