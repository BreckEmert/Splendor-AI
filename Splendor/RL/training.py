# Splendor/RL/training.py

import json
import os
import pickle

from Environment.game import Game
from RL import RLAgent, RandomAgent


def debug_game(model_path=None, layer_sizes=None, memory_path=None, log_path=None):
    import json

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
        log_state = open(os.path.join(states_log_dir, f"states_episode_{episode}.json"), 'w')

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

    # Load existing memory
    
def write_memory(memory, random=False, append_to_previous=False):
    if random:
        memory_path = "/workspace/RL/random_memory.pkl"
    else:
        memory_path = "/workspace/RL/memory.pkl"

    if append_to_previous and os.path.exists(memory_path):
        with open(memory_path, 'rb') as f:
            existing_memory = pickle.load(f)
        print(f"Loaded {len(existing_memory)} existing memories.")
        memory.extend(existing_memory)

    print(f"Wrote {len(memory)} memories")
    with open(memory_path, 'wb') as f:
        pickle.dump(memory, f)

def ddqn_loop(model_path=None, from_model_path=None, layer_sizes=None, memory_path=None, log_path=None, tensorboard_dir=None):
    # Players and strategies (BestStrategy for training perfectly)
    ddqn_model = RLAgent(model_path, from_model_path, layer_sizes, memory_path, tensorboard_dir)
    
    players = [
        ('Player1', ddqn_model),
        ('Player2', ddqn_model)
    ]

    game = Game(players)
    game_lengths = []

    for episode in range(5000):
        game.reset()

        # Enable logging
        if log_path: # and episode%10 == 0
            log_state = open(os.path.join(log_path, "game_states", f"states_episode_{episode}.json"), 'w')
            logging = True
        else:
            logging = False

        while not game.victor:
            game.turn()

            if logging:
                json.dump(game.get_state(), log_state)
                log_state.write('\n')

        game_lengths.append(game.half_turns)

        ddqn_model.replay()

        if episode % 10 == 0:
            ddqn_model.update_target_model()
            if episode % 100 == 0:
                ddqn_model.save_model(model_path)
                write_memory(ddqn_model.memory)

            avg = sum(game_lengths)/len(game_lengths)/2
            print(f"Episode: {episode}, Average turns for last 10 games: {avg}")
            game_lengths = []
    
    # Save memory
    with open("/workspace/RL/real_memory.pkl", 'wb') as f:
        pickle.dump(list(ddqn_model.memory), f)

def find_fastest_game(memory_path):
    from copy import deepcopy
    fastest_memory = []

    while len(fastest_memory) < 150:
        # Players
        if not len(fastest_memory):
            players = [
                ('Player1', RandomAgent(memory_path)),
                ('Player2', RandomAgent(memory_path))
            ]
        else:
            players = [
                ('Player1', RandomAgent()),
                ('Player2', RandomAgent())
            ]

        game = Game(players)

        checkpoint = deepcopy(game)
        original_checkpoint = deepcopy(game)
        last_buy_turn = 1
        buys_since_checkpoint = 0
        
        found = False
        while not found:
            game.turn()
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
            
            if game.victor:
                if game.half_turns < 54:
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

    flattened_memory = [item for sublist in fastest_memory for item in sublist]

    write_memory(flattened_memory, random=True)

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
