# Splendor/RL/training.py

import json
import os
import pickle

from Environment.game import Game # type: ignore
from RL import RLAgent, RandomAgent # type: ignore


def debug_game(model_path=None, layer_sizes=None, memories=False, log_path=None):
    import json

     # Players and strategies (BestStrategy for training perfectly)
    # ddqn_model = RLAgent(layer_sizes=layer_sizes)
    
    players = [
        ('Player1', RLAgent(layer_sizes=layer_sizes, memories=memories)),
        ('Player2', RLAgent(layer_sizes=layer_sizes, memories=memories))
    ]

    game = Game(players)
    for episode in range(10_000):
        # Enable logging for all games
        # log_state = open(os.path.join(log_path, "game_states", f"states_episode_{episode}.json"), 'w')
        # log_move = open(os.path.join(log_path, "moves", f"moves_episode_{episode}.json"), 'w')

        game.reset()
        while not game.victor:
            game.turn()
            # json.dump(game.get_state(), log_state)
            # log_state.write('\n')
            # log_move.write(str(game.active_player.chosen_move) + '\n') # Disabled for ddqn

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

def ddqn_loop(model_path=None, from_model_path=None, layer_sizes=None, memories_path=None, log_path=None, tensorboard_dir=None):
    # Players and strategies (BestStrategy for training perfectly)
    ddqn_model = RLAgent(model_path, from_model_path, layer_sizes, memories_path, tensorboard_dir)
    
    players = [
        ('Player1', ddqn_model),
        ('Player2', ddqn_model)
    ]

    game = Game(players)
    game_lengths = []

    for episode in range(1501):
        game.reset()

        # Enable logging
        if log_path: # and episode%10 == 0
            log_state = open(os.path.join(log_path, "game_states", f"states_episode_{episode}.json"), 'w')
            log_move = open(os.path.join(log_path, "moves", f"moves_episode_{episode}.json"), 'w')
            logging = True
        else:
            logging = False

        while not game.victor:
            game.turn()

            if logging:
                json.dump(game.get_state(), log_state)
                log_state.write('\n')
                log_move.write(str(game.active_player.chosen_move) + '\n') # Disabled for ddqn

        game_lengths.append(game.half_turns)

        ddqn_model.train(ddqn_model.game_length)
        ddqn_model.lr = max(ddqn_model.lr*0.985, 0.0001) if ddqn_model.step < 51 else ddqn_model.lr
        ddqn_model.replay()
        ddqn_model.lr = max(ddqn_model.lr*0.985, 0.0001)
        ddqn_model.replay()

        if episode % 10 == 0:
            avg = sum(game_lengths)/len(game_lengths)/2
            ddqn_model.update_target_model()
            ddqn_model.save_model(model_path)

            print(f"Episode: {episode}, Epsilon: {ddqn_model.epsilon}, Average turns for last 10 games: {avg}, lr: {ddqn_model.lr}")
            game_lengths = []
    
    # Save memories
    with open("/workspace/RL/real_memories.pkl", 'wb') as f:
        pickle.dump(list(ddqn_model.memory), f)

def find_fastest_game(memories_path, append_to_previous):
    from copy import deepcopy
    fastest_memories = []

    while len(fastest_memories) < 90:
        # Players
        players = [
            ('Player1', RandomAgent(memories_path)),
            ('Player2', RandomAgent(memories_path))
        ]

        game = Game(players)

        # Initialize a fake memory for remember() logic purposes
        for player in game.players:
            player.model.memory.append([None, None, None, None, None, None])

        checkpoint = deepcopy(game)
        original_checkpoint = deepcopy(game)
        last_buy_turn = 1
        buys_since_checkpoint = 0
        
        found = False
        while not found:
            game.turn()
            # print("\nTaking turn ", game.half_turns)
            if 15 <= game.active_player.move_index < 44:
                # print("Buying")
                buys_since_checkpoint += 1
                if buys_since_checkpoint == 2:
                    if game.half_turns - last_buy_turn < 16:
                        last_buy_turn = game.half_turns
                        # print("Setting last_buy_turn to ", last_buy_turn)
                        checkpoint = deepcopy(game)
                    else:
                        game = deepcopy(checkpoint)
                        # print("Loading old game at turn ", game.half_turns)
                    buys_since_checkpoint = 0
            
            if game.victor:
                # print(game.half_turns)
                if game.half_turns < 55:
                    print(game.half_turns)
                    for player in game.players:
                        if player.victor:
                            fastest_memories.append(list(player.model.memory.copy())[1:])
                    found = True
                else:
                    checkpoint = deepcopy(original_checkpoint)
                    game = deepcopy(original_checkpoint)
                    buys_since_checkpoint = 0
                    last_buy_turn = 1
            else:
                game.turn()

    flattened_memories = [item for sublist in fastest_memories for item in sublist]

    # Load existing memory
    memory_path = "/workspace/RL/random_memories.pkl"
    if append_to_previous and os.path.exists(memory_path):
        with open(memory_path, 'rb') as f:
            existing_memories = pickle.load(f)
        print(f"Loaded {len(existing_memories)} existing memories.")
        flattened_memories.extend(existing_memories)

    print("Number of memories:", len(flattened_memories))
    with open(memory_path, 'wb') as f:
        pickle.dump(flattened_memories, f)