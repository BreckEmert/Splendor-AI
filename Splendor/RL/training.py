# Splendor/RL/training.py

import json
import numpy as np
import os

from Environment.Splendor_components.Player_components.strategy import ( # type: ignore
    BestStrategy, RandomStrategy, OffensiveStrategy, ResourceHog, ObliviousStrategy
)


def train_agent(model_save_path, log_path):
    from Environment.game import Game # type: ignore
    # Players and strategies (BestStrategy for training perfectly)
    players = [('Player1', BestStrategy(), 1), ('Player2', BestStrategy(), 1)]
    
    state_size = 263  # ADJUST LATER
    batch_size = 32

    # Training loop
    for episode in range(1000):  # Number of games
        game = Game(players)
        state = np.array(game.to_vector())
        state = np.reshape(state, [1, state_size])

        # Log every 10 games
        if episode % 10 == 0:
            log_file = os.path.join(log_path, f"game_state_episode_{episode}.json")
            log = open(log_file, 'w')
            logging = True

        while not game.victor:
            game.turn()  # Take a turn
            next_state = np.array(game.to_vector())
            next_state = np.reshape(next_state, [1, state_size])

            if logging:
                json.dump(game.get_state(), log)
                log.write('\n')

            # Agent remembers
            active_player = game.active_player
            active_player.rl_model.remember(
                state, 
                active_player.move_index,  # Chosen move
                game.reward, 
                next_state, 
                1 if game.victor else 0
            )

            # Update state
            state = next_state

            if len(active_player.rl_model.memory) > batch_size:
                active_player.rl_model.replay()
        
        # Final rewards
        for player in game.players:
            final_reward = 10 if player == game.victor else -10
            for state, action, reward, next_state, done in reversed(player.rl_model.memory):
                player.rl_model.train(state, action, final_reward, next_state, done)

        # Log the progress
        print(f"Episode {episode+1}/1000")

    # Save models
    for player in game.players:
        player.rl_model.save_model(f"{model_save_path}_{player.name}")

if __name__ == '__main__':
    model_save_path = 'rl_agent_model.keras'
    train_agent(model_save_path)