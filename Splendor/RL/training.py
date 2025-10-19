# Splendor/RL/training.py

from Environment.rl_game import RLGame
from RL import RLAgent, RandomAgent
from Play.render import draw_game_state


def ddqn_loop(paths, log_rate=0):
    # Initialize players, their models, and a game (these get reset)
    model = RLAgent(paths)
    players = [('Player1', model, 0), ('Player2', model, 1)]
    game = RLGame(players, model)
    game_lengths = []
    step_counter = 0

    replay_freq = 256
    n = model.batch_size / replay_freq
    print(f"Replays are sampled {n} times on average.")
    """Need to consider whether because of the self-play correlation
    of states if my sampling should be reduced.  I definitely think 
    that I should target something lower than normal because of this.
    """

    # Loop through games - safe to stop at any time
    for episode in range(200_001):
        # print(f"Starting a game, episode {episode}")
        game.reset()
      
        # Enable logging
        logging = (log_rate and episode%log_rate == 0)

        # Play a game
        while not game.victor:
            game.turn()
            step_counter += 1
            
            # Run replay (roughly 512 batch size / 4 samples = 120 rate)
            if step_counter%replay_freq == 0:
                # model.replay()
                model.replay_two_ply()

            # Draw the game state (very rare at my default of 25k)
            if logging:
                draw_game_state(episode, game)
        
        # End-of-game logging and saving
        game_lengths.append(game.half_turns)
        if episode % 1_000 == 0:
            avg = sum(game_lengths)/len(game_lengths)/2
            model.log_game_lengths(avg)
            game_lengths = []

            # Save the model and memory
            if episode % 10_000 == 0:
                model.save_model()
                model.write_memory()
    
    # Save memory
    model.write_memory()
