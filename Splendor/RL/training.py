# Splendor/RL/training.py

import json
import os
import pickle
from copy import deepcopy

from Environment.game import Game
from RL import RLAgent, RandomAgent


def debug_game(model_path=None, layer_sizes=None, memory_path=None, log_path=None):
    # Players and strategies (BestStrategy for training perfectly)
    # ddqn_model = RLAgent(layer_sizes=layer_sizes)

    # Make logging directories
    states_log_dir = os.path.join(log_path, "game_states")
    os.makedirs(states_log_dir, exist_ok=True)
    
    players = [
        ('Player1', RLAgent(layer_sizes=layer_sizes, memory_path=memory_path)),
        ('Player2', RLAgent(layer_sizes=layer_sizes, memory_path=memory_path))
    ]

    game = Game(players)
    for episode in range(10_000):
        # Enable logging for all games
        state_log_path = os.path.join(
            states_log_dir, f"states_episode_{episode}.json")
        log_state = open(state_log_path, 'w')

        game.reset()
        while not game.victor:
            game.turn()

            json.dump(game.get_state(), log_state)
            log_state.write('\n')
        # show_game_rewards(game.players)

        # if episode == 100:
        #     print(len(game.active_player.model.memory))
        #     write_to_csv(list(game.active_player.model.memory)[-2000:])
        #     break

        # print(f"Simulated game {episode}, game length * 2: {game.half_turns}")

def write_to_csv(memory):
    print("-------Writing to CSV------")
    import pandas as pd
    import numpy as np

    # Extract the components
    states = np.array([mem[0] for mem in memory]).reshape(states.shape[0], -1)
    actions = np.array([mem[1] for mem in memory]).reshape(-1, 1)
    # rewards = np.array([mem[2] for mem in memory])
    next_states = np.array([mem[3] for mem in memory]).reshape(next_states.shape[0], -1)
    # dones = np.array([mem[4] for mem in memory])

    # Create DataFrames
    states = np.round(states, 1)
    next_states = np.round(next_states, 1)

    df_states = pd.DataFrame(np.hstack((actions, states)))
    df_next_states = pd.DataFrame(np.hstack((actions, next_states)))

    # To CSV
    df_states.to_csv('states.csv', index=False)
    df_next_states.to_csv('next_states.csv', index=False)
    print("-------Wrote to CSV------")

def ddqn_loop(model_save_path=None, preexisting_model_path=None, layer_sizes=None, 
              preexisting_memory_path=None, log_dir=None, tensorboard_dir=None):
    """Add docstring
    """
    # Initialize players, their models, and a game (these get reset)
    ddqn_model = RLAgent(model_save_path, preexisting_model_path, layer_sizes, 
                         preexisting_memory_path, tensorboard_dir)
    players = [('Player1', ddqn_model), ('Player2', ddqn_model)]
    game = Game(players)
    game_lengths = []

    # Set a directory to log the game states
    game_states_dir = os.path.join(log_dir, "game_states")
    os.makedirs(game_states_dir, exist_ok=True)

    # Loop through games - can be stopped at any time
    for episode in range(5000):
        game.reset()

        # Enable logging
        if log_path: # and episode%10 == 0
            log_path = os.path.join(game_states_dir, f"states_episode_{episode}.json")
            log_state_file = open(log_path, 'w')
            logging = True
        else:
            logging = False

        # Play a game
        while not game.victor:
            game.turn()

            if logging:
                json.dump(game.get_state(), log_state_file)
                log_state_file.write('\n')

        game_lengths.append(game.half_turns)

        # Run replay after each game
        ddqn_model.replay()

        # Save every 10 episodes
        if episode % 10 == 0:
            ddqn_model.update_target_model()
            if episode % 100 == 0:
                ddqn_model.save_model(model_save_path)
                ddqn_model.write_memory()

            avg = sum(game_lengths)/len(game_lengths)/2
            print(f"Episode: {episode}")
            print(f"Average turns for last 10 games: {avg}")
            game_lengths = []
    
    # Save memory
    with open("/workspace/RL/real_memory.pkl", 'wb') as f:
        pickle.dump(list(ddqn_model.memory), f)

def find_fastest_game(log_dir, append_to_prev_mem=False):
    """"Simulates tons of games in a slightly intelligent way
    putting only ones below a move length into memory
    which thereby are better games for initial memory
    """
    fastest_memory = []
    while len(fastest_memory) < 2:
        # Initialize a game
        players = [('Player1', RandomAgent()), ('Player2', RandomAgent())]
        game = Game(players)
        checkpoint = deepcopy(game)
        original_checkpoint = deepcopy(game)

        last_buy_turn = 1
        buys_since_checkpoint = 0
        
        # Enable logging if requested, for generate_images.py
        filename = f"random_states_episode_{len(fastest_memory)}.json"
        log_state_path = os.path.join(log_dir, filename)
        log_state_file = open(log_state_path, 'w')

        # Play a game
        found = False
        while not found:
            game.turn()
            if log_dir: WE CANT DO THIS BECAUSE THE GAMES ARE AWFUL WOOPS
                json.dump(game.get_state(), log_state_file)
                log_state_file.write('\n')
            
            # Only buy moves can make progress towards a win (index 15-44)
            if 15 <= game.active_player.move_index < 44:
                # print("Buying")
                buys_since_checkpoint += 1
                if buys_since_checkpoint == 2:
                    if game.half_turns - last_buy_turn <= 16:
                        last_buy_turn = game.half_turns
                        # print("Setting last_buy_turn to ", last_buy_turn)
                        checkpoint = deepcopy(game)
                    else:
                        game = deepcopy(checkpoint)
                        # print("Loading old game at turn ", game.half_turns)
                    buys_since_checkpoint = 0
            
            # Just because someone won doesn't mean it was a short enough win
            if game.victor:
                if game.half_turns < 55:
                    print(game.half_turns)
                    for player in game.players:
                        if player.victor:
                            fastest_memory.append(list(player.model.memory.copy())[1:])
                    found = True
                else:
                    checkpoint = deepcopy(original_checkpoint)
                    game = deepcopy(original_checkpoint)
                    buys_since_checkpoint = 0
                    last_buy_turn = 1
            else:
                game.turn()
                if log_dir:
                    json.dump(game.get_state(), log_state_file)
                    log_state_file.write('\n')

    # Write out the memories of the winner of all the short games
    flattened_memory = [item for sublist in fastest_memory for item in sublist]
    game.active_player.model.write_memory(flattened_memory, base_dir, append_to_prev_mem)

def show_game_rewards(players):
    for num, player in enumerate(players):
        print(num)
        total_neg = total_pos = n_neg = n_pos = 0
        for mem in player.model.memory:
            reward = mem[2]
            if reward < 0:
                total_neg += reward
                n_neg += 1
            elif reward > 0:
                total_pos += reward
                n_pos += 1

        print(f"\nPlayer {num} with {player.points} points:")

        if total_neg:
            average_neg = total_neg / n_neg
            print("Negative Rewards:", total_neg, average_neg)
        if total_pos:
            average_pos = total_pos / n_pos
            print("Positive Rewards:", total_pos, average_pos, "\n")
        else:
            print("No positive rewards")
        
        if player.victor:
            winner_points = player.points
            winner_neg = total_neg
            winner_pos = total_pos
        else:
            loser_points = player.points
            loser_neg = total_neg
            loser_pos = total_pos

        player.model.memory.clear()
        player.model.memory.append([0, 0, 0, 0, 0])
        
    assert winner_points >= loser_points, f"Loser has {loser_points} points but winner only has {winner_points}"
    assert winner_pos > loser_pos, f"Loser has {loser_pos} rewards but winner only has {winner_pos}"
