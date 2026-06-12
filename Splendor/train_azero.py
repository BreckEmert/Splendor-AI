# Splendor/train_azero.py
"""
Entry point for the AlphaZero-style loop, v2 (RL/azero.py): one two-headed net
(policy + scalar tanh value), warm-started by copying the champion actor's
weights into the trunk + policy head. Gen 0 self-play uses the legacy
actor+critic search while the fresh value head trains; later gens use the
two-headed net (half the NN calls per leaf).

Knobs (env): AZ_GENS, AZ_GAMES_PER_GEN, AZ_SIMS, AZ_EVAL_BATCH, AZ_TEMP_MOVES,
AZ_WINDOW, AZ_EPOCHS, AZ_BATCH, AZ_LR, AZ_VLOSS, AZ_EVAL_GAMES, AZ_LAYERS
(e.g. "1024-1024-512": trunk size; warm start requires it to match the
champion's 512-512-256, otherwise the net trains from scratch), and
AZ_WARM_AZNET to RESUME from a saved v2 generation checkpoint. Examples:
    python train_azero.py
    AZ_WARM_AZNET=RL/trained_agents/<stamp>__aznet_gen7.keras python train_azero.py
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

    layers = [int(x) for x in
              os.getenv('AZ_LAYERS', '512-512-256').replace(',', '-').split('-')]
    stamp = (datetime.now() - timedelta(hours=6)).strftime("%m-%d-%H-%M")
    nickname = f"{stamp}__aznet"

    warm_actor = os.path.join(agents_dir, CHAMP_ACTOR)
    warm_critic = warm_actor.replace('_actor.keras', '_critic.keras')
    baseline = os.getenv('AZ_BASELINE', warm_actor)
    warm_aznet = os.getenv('AZ_WARM_AZNET')          # resume (optional)

    paths = {
        'layers': layers,
        'warm_actor': warm_actor,
        'warm_critic': warm_critic,
        'warm_aznet': warm_aznet,
        'baseline': baseline,
        'gen_net': os.path.join(agents_dir, nickname + "_gen{gen}.keras"),
        'best_net': os.path.join(agents_dir, nickname + "_best.keras"),
        'tensorboard_dir': os.path.join(tb_root, nickname),
    }
    os.makedirs(paths['tensorboard_dir'], exist_ok=True)
    for key in ('warm_actor', 'warm_critic', 'baseline'):
        assert os.path.exists(paths[key]), f"{key} missing: {paths[key]}"
    if warm_aznet:
        assert os.path.exists(warm_aznet), f"resume net missing: {warm_aznet}"
    return paths


if __name__ == "__main__":
    AZeroLoop(get_paths()).run()
