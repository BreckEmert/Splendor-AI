# Splendor/train_azero.py
"""
Entry point for the AlphaZero-style loop (RL/azero.py): search-generated
self-play data -> train actor on visit distributions + critic on search-backed
values -> stronger nets -> stronger search. Warm-starts from the tournament
champion; checkpoints and TensorBoard land in the usual shared directories.

Knobs (env): AZ_GENS, AZ_GAMES_PER_GEN, AZ_SIMS, AZ_EVAL_BATCH, AZ_TEMP_MOVES,
AZ_WINDOW, AZ_EPOCHS, AZ_BATCH, AZ_LR, AZ_EVAL_GAMES, plus AZ_WARM_ACTOR /
AZ_WARM_CRITIC to start from a different checkpoint. Example:
    AZ_GENS=12 AZ_GAMES_PER_GEN=150 python train_azero.py
"""

import os
from datetime import datetime, timedelta

from RL.azero import AZeroLoop

CHAMP_ACTOR = ("05-31-21-54__vrpo_512-512-256__kl0.06_alr3e-4_clr1.5e-4_"
               "cbuf4_best_actor.keras")


def get_paths():
    backup_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.getenv('WORKSPACE_DIR', backup_dir)
    rl_dir = os.path.join(base_dir, "RL")
    agents_dir = os.path.join(rl_dir, "trained_agents")
    tb_root = os.path.join(rl_dir, "saved_files", "tensorboard_logs")

    stamp = (datetime.now() - timedelta(hours=6)).strftime("%m-%d-%H-%M")
    nickname = f"{stamp}__azero"

    warm_actor = os.getenv('AZ_WARM_ACTOR',
                           os.path.join(agents_dir, CHAMP_ACTOR))
    warm_critic = os.getenv('AZ_WARM_CRITIC',
                            warm_actor.replace('_actor.keras', '_critic.keras'))

    paths = {
        'warm_actor': warm_actor,
        'warm_critic': warm_critic,
        'gen_actor': os.path.join(agents_dir, nickname + "_gen{gen}_actor.keras"),
        'gen_critic': os.path.join(agents_dir, nickname + "_gen{gen}_critic.keras"),
        'best_actor': os.path.join(agents_dir, nickname + "_best_actor.keras"),
        'best_critic': os.path.join(agents_dir, nickname + "_best_critic.keras"),
        'tensorboard_dir': os.path.join(tb_root, nickname),
    }
    os.makedirs(paths['tensorboard_dir'], exist_ok=True)
    assert os.path.exists(warm_actor), f"warm-start actor missing: {warm_actor}"
    assert os.path.exists(warm_critic), f"warm-start critic missing: {warm_critic}"
    return paths


if __name__ == "__main__":
    AZeroLoop(get_paths()).run()
