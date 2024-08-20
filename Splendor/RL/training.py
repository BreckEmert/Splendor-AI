# Splendor/RL/training.py

import json
import numpy as np
import os

from Environment.Splendor_components.Player_components.strategy import (
    BestStrategy, RandomStrategy, OffensiveStrategy, ResourceHog, ObliviousStrategy
)
from RL import RLAgent


def train_agent(base_save_path, log_path, layer_sizes, model_paths=None):
    from Environment.game import Game
    
    # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_paths[0])
    player2_model = RLAgent(layer_sizes, model_paths[1])
    
    players = [
        ('Player1', BestStrategy(), 1, player1_model, 0),
        ('Player2', BestStrategy(), 1, player2_model, 1)
    ]

    # Training loop
    for episode in range(5):  # Number of games
        logging = False
        game = Game(players)
        state = np.array(game.to_vector())

        # Log every 10 games
        if episode % 10 == 0:
            log_state = open(os.path.join(log_path, f"game_states_episode_{episode}.json"), 'w')
            log_move = open(os.path.join(log_path, f"moves_episode_{episode}.json"), 'w')
            logging = True

        while not game.victor:
            game.turn()  # Take a turn
            next_state = np.array(game.to_vector())

            if logging:
                json.dump(game.get_state(), log_state)
                log_state.write('\n')
                log_move.write(str(game.active_player.chosen_move) + '\n')

            # Agent remembers
            active_player = game.active_player
            if not active_player.entered_loop:
                active_player.rl_model.remember(
                    state, 
                    active_player.move_index, 
                    game.reward, 
                    next_state, 
                    1 if game.victor else 0
                )
            active_player.entered_loop = False

            # Update state
            state = next_state
        
        print(f"Episode {episode+1} game complete and took {game.half_turns} turns. Replaying.")
        # Final rewards
        for player in game.players:
            player.rl_model.replay()
            final_reward = 5 if player == game.victor else -5
            for state, action, reward, next_state, done in reversed(player.rl_model.memory):
                player.rl_model.train(state, action, final_reward, next_state, done)

            # Decay epsilon
            if player.rl_model.epsilon > player.rl_model.epsilon_min:
                player.rl_model.epsilon *= player.rl_model.epsilon_decay
        
        # Close logs
        if logging:
            log_state.close()
            log_move.close()

        # Log the progress
        print(f"Episode {episode+1} replayed")

    # Save models
    for player in game.players:
        player.rl_model.save_model(base_save_path, player.name)