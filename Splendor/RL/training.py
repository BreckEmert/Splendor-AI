# Splendor/RL/training.py

import json
import os
import shutil
from copy import deepcopy

from Environment.game import Game
from RL import RLAgent, RandomAgent


def debug_game(paths):
    """DEPRECATED, FOR NOW"""
    # Make logging directories
    states_log_dir = os.path.dirname(paths['states_log_dir'])
    states_log_dir = os.path.join(states_log_dir, "debug")
    os.makedirs(states_log_dir, exist_ok=True)
    
    ddqn_model = RLAgent(paths)
    players = [('Player1', ddqn_model), ('Player2', ddqn_model)]
    game = Game(players)

    for episode in range(10_000):
        # Enable logging for all games
        json_path = f"states_episode_{episode}.json"
        state_log_path = os.path.join(states_log_dir, json_path)
        log_state = open(state_log_path, 'w')

        game.reset()
        while not game.victor:
            game.turn()

            json.dump(game.to_state_vector(), log_state)
            log_state.write('\n')
        # show_game_rewards(game.players)

        # if episode == 100:
        #     print(len(game.active_player.model.memory))
        #     write_to_csv(list(game.active_player.model.memory)[-2000:])
        #     break

        # print(f"Simulated game {episode}, game length * 2: {game.half_turns}")

def ddqn_loop(paths, log_rate=0):
    """Add docstring
    """
    # Initialize players, their models, and a game (these get reset)
    ddqn_model = RLAgent(paths)
    players = [('Player1', ddqn_model), ('Player2', ddqn_model)]
    game = Game(players)
    game_lengths = []

    # Loop through games - can be stopped at any time
    for episode in range(750):  # Crashing after 750 now?
        game.reset()

        # Enable logging
        if log_rate and episode%log_rate == 0:
            log_path = os.path.join(paths['states_log_dir'], f"states_episode_{episode}.json")
            log_state_file = open(log_path, 'w')
            logging = True
        else:
            logging = False

        # Play a game
        while not game.victor:
            game.turn()

            if logging:
                json.dump(game.to_state_vector(), log_state_file)
                log_state_file.write('\n')
        else:
            game.active_player.model.memory[-2][2] -= 10  # Loser reward

        game_lengths.append(game.half_turns)

        # Run replay after each game
        ddqn_model.replay()

        # Save every 10 episodes
        if episode % 10 == 0:
            ddqn_model.update_target_model()
            if episode % 100 == 0:
                ddqn_model.save_model()
                ddqn_model.write_memory()

            avg = sum(game_lengths)/len(game_lengths)/2
            print(f"Episode: {episode}")
            print(f"Average turns for last 10 games: {avg}")
            game_lengths = []
    
    # Save memory
    ddqn_model.write_memory()

def find_fastest_game(paths, n_games, log_states=False):
    """"Simulates tons of games, putting only ones below a move length 
    into memory which thereby are better games for initial memory.
    This doesn't need fancy logic other than uncommenting line 205 in
    player.py as that stops the legal moves logic after it gets the
    legal buy moves.  So it buys whenever it can just from that.
    """
    # Make the log folder for states, for visualization later
    states_log_dir = os.path.dirname(paths['states_log_dir'])
    states_log_dir = os.path.join(states_log_dir, "random")
    shutil.rmtree(states_log_dir, ignore_errors=True)
    os.makedirs(states_log_dir, exist_ok=True)

    players = [('Player1', RandomAgent(paths)), ('Player2', RandomAgent(paths))]
    game = Game(players)
    
    completed_games = []
    while len(completed_games) < n_games:
        # Initialize a game
        game.reset()
        game.state_history = []
        for player in game.players:
            player.model.reset()
        original_checkpoint = deepcopy(game)  # Avoid completion bias by retrying even hard games
        
        # Enable logging if requested, for generate_images.py
        if log_states:
            filename = f"random_states_episode_{len(completed_games)}.json"
            log_state_path = os.path.join(states_log_dir, filename)

        # Play a game
        completed_quickly = False
        while not completed_quickly:
            for _ in range(2):
                game.turn()
                game.state_history.append(game.to_state_vector())
            
            # Just because someone won doesn't mean it was a short enough win
            if game.victor:
                if game.half_turns <= 60:
                    print(f"Victory in {game.half_turns}, {len(completed_games)+1} completed games.")
                    for player in game.players:
                        if player.victor:
                            completed_quickly = True
                            completed_games.append(deepcopy(player.model.memory))
                            # Each player will have different memory lengths so this number must be less than the game length
                            print(f"Appended {len(player.model.memory)} frames to completed_games")
                            if log_states:
                                with open(log_state_path, 'w') as f:
                                    for state in game.state_history:
                                        json.dump(state, f)
                                        f.write('\n')
                else:
                    # Hard reset if the game wasn't fast enough
                    print("Resetting, game was too long.", game.half_turns)
                    game = deepcopy(original_checkpoint)
                    for player in game.players:
                        player.model.reset()
                
                continue

    # Write out the memories of the winner of all the short games
    flattened_memory = [state for game in completed_games for state in game]
    print(f"flattened_memory has {len(flattened_memory)} length")
    game.active_player.model.write_memory(flattened_memory[1:])

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
