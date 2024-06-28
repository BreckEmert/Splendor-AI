# Splendor/RL/training.py

import os

from Environment.game import Game # type: ignore
from RL import RLAgent # type: ignore


def debug_game(log_path, layer_sizes, model_path):
    import json

     # Players and strategies (BestStrategy for training perfectly)
    ddqn_model = RLAgent(layer_sizes, model_path)
    
    players = [
        ('Player1', ddqn_model),
        ('Player2', ddqn_model)
    ]

    game = Game(players)
    for episode in range(1000):
        # Enable logging for all games
        log_state = open(os.path.join(log_path, "game_states", f"states_episode_{episode}.json"), 'w')
        log_move = open(os.path.join(log_path, "moves", f"moves_episode_{episode}.json"), 'w')

        game.reset()
        while not game.victor:
            game.turn()
            json.dump(game.get_state(), log_state)
            log_state.write('\n')
            log_move.write(str(game.active_player.chosen_move) + '\n') # Disabled for ddqn

        # write_to_csv(game.active_player.ddqn_model.memory[-500:])
        # break

        print(f"Simulated game {episode}")

def write_to_csv(memory):
        print("-------Writing to CSV------")
        import pandas as pd
        import numpy as np

        # Extract the components
        states = np.array([mem[0] for mem in memory])
        actions = np.array([mem[1] for mem in memory])
        # rewards = np.array([mem[2] for mem in memory])
        next_states = np.array([mem[3] for mem in memory])
        # dones = np.array([mem[4] for mem in memory])

        # Create DataFrames
        states = np.round(states, 1)
        next_states = np.round(next_states, 1)

        df_states = pd.DataFrame(np.hstack((actions.reshape(-1, 1), states.reshape(states.shape[0], -1))))
        df_next_states = pd.DataFrame(np.hstack((actions.reshape(-1, 1), next_states.reshape(next_states.shape[0], -1))))

        # To CSV
        df_states.to_csv('states.csv', index=False)
        df_next_states.to_csv('next_states.csv', index=False)
        print("-------Wrote to CSV------")

def ddqn_loop(layer_sizes, model_path, tensorboard_dir):
    # Players and strategies (BestStrategy for training perfectly)
    ddqn_model = RLAgent(layer_sizes, model_path, tensorboard_dir)
    
    players = [
        ('Player1', ddqn_model),
        ('Player2', ddqn_model)
    ]

    game = Game(players)
    game_lengths = []

    for episode in range(10_000):
        game.reset()

        while not game.victor:
            game.turn()
        game_lengths.append(game.half_turns)

        ddqn_model.train(game.half_turns*2) # Not enough length.  More memories than half turns.  Estimating *2
        ddqn_model.replay()

        if episode % 10 == 0:
            avg = sum(game_lengths)/len(game_lengths)/2
            ddqn_model.update_target_model()
            ddqn_model.update_learning_rate(avg)
            ddqn_model.save_model(model_path)

            print(f"Episode: {episode}, Epsilon: {ddqn_model.epsilon}, Average turns for last 10 games: {avg}, lr: {ddqn_model.lr}")
            length_memory = []