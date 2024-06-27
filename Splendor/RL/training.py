# Splendor/RL/training.py

import os

from Environment.game import Game # type: ignore
from RL import RLAgent # type: ignore


def debug_game(log_path, layer_sizes, model_path):
    import json

     # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_path)
    player2_model = RLAgent(layer_sizes, model_path)
    
    players = [
        ('Player1', player1_model),
        ('Player2', player2_model)
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

        print(f"Simulated game {episode}")

def ddqn_loop(layer_sizes, model_path):
    # Players and strategies (BestStrategy for training perfectly)
    ddqn_model = RLAgent(layer_sizes, model_path)
    
    players = [
        ('Player1', ddqn_model),
        ('Player2', ddqn_model)
    ]

    game = Game(players)
    length_memory = []

    for episode in range(100):
        game.reset()

        while not game.victor:
            game.turn()
        length_memory.append(game.half_turns)

        ddqn_model.train(game.half_turns) # Not enough length.  More memories than half turns.
        ddqn_model.replay()

        if episode % 10 == 0:
            avg = sum(length_memory)/len(length_memory)/2
            ddqn_model.update_target_model()
            ddqn_model.update_learning_rate(avg)
            ddqn_model.save_model(model_path)

            print(f"Episode: {episode}, Epsilon: {ddqn_model.epsilon}, Average turns for last 10 games: {avg}, lr: {ddqn_model.lr}")
            length_memory = []