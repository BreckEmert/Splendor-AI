# Splendor/RL/training.py

import json
import os
import shutil
from copy import deepcopy

from Environment.game import Game
from meta.generate_images import draw_game_state
from RL import RLAgent, RandomAgent


def ddqn_loop(paths, log_rate=0):
    """Add docstring
    """


    # Initialize players, their models, and a game (these get reset)
    model = RLAgent(paths)
    players = [('Player1', model), ('Player2', model)]
    game = Game(players, model)
    game_lengths = []

    # Loop through games - safe to stop at any time
    for episode in range(80000):
        game.reset()
      
        # Enable logging
        logging = (log_rate and episode%log_rate == 0)

        # Play a game
        while not game.victor:
            game.turn()

            if logging:
                draw_game_state(game)

            if logging:
                draw_game_state(episode, game)
        game_lengths.append(game.half_turns)

        # Run replay after each game
        model.replay()

        # Update target model every 10 episodes
        if episode % 10 == 0:
            model.update_target_model()

            # Printout avg game length every 100
            if episode % 100 == 0:
                avg = sum(game_lengths)/len(game_lengths)/2
                print(f"Episode: {episode}")
                print(f"Average turns for last 100 games: {avg}")
                game_lengths = []

                # And save every 500 games
                # if episode % 500 == 0:
                #     model.save_model()
                #     model.write_memory()
    
    # Save memory
    # model.write_memory()

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
