# Splendor/RL/training.py

import json
import os
import shutil
from copy import deepcopy

from Environment.game import Game
from meta.generate_images import draw_game_state
from RL import RLAgent, RandomAgent


def ddqn_loop(paths, log_rate=0):
    # Initialize players, their models, and a game (these get reset)
    model = RLAgent(paths)
    players = [('Player1', model), ('Player2', model)]
    game = Game(players, model)
    game_lengths = [30]  # Just to reduce variance
    step_counter = 0

    replay_freq = 120
    n = model.batch_size / replay_freq
    print(f"Replays are sampled {n} times on average.")
    """Need to consider whether because of the self-play correlation
    of states if my sampling should be reduced.  I definitely think 
    that I should target something lower than normal because of this.
    """

    # Loop through games - safe to stop at any time
    for episode in range(100_001):
        game.reset()
      
        # Enable logging
        logging = (log_rate and episode%log_rate == 0)

        # Play a game
        while not game.victor:
            game.turn()
            step_counter += 1
            
            # Run replay (roughly 512 batch size / 4 samples = 120 rate)
            if step_counter%replay_freq == 0:
                model.replay()

            # Draw the game state
            if logging:
                draw_game_state(episode, game)
        
        # End-of-game logging and saving
        game_lengths.append(game.half_turns)
        if episode % 100 == 0:
            avg = sum(game_lengths)/len(game_lengths)/2
            model.log_game_lengths(avg)
            game_lengths = []

            # Save the model and memory
            if episode % 10_000 == 0:
                model.save_model()
                model.write_memory()
    
    # Save memory
    model.write_memory()

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
                game.state_history.append(game.to_state())
            
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
    # game.active_player.model.write_memory(flattened_memory[1:])  # Doesn't work?  memories are split between players, no?

def write_to_csv(memory):
    print("-------Writing to CSV------")
    import pandas as pd
    import numpy as np

    # Extract the components
    states = np.array([mem[0] for mem in memory]).reshape(states.shape[0], -1)
    actions = np.array([mem[1] for mem in memory]).reshape(-1, 1)
    # rewards = np.array([mem[2] for mem in memory])
    next_states = np.array([mem[3] for mem in memory]).reshape(next_states.shape[0], -1)  # if this breaks later try len(next_states) instead of .shape[0]?
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
