# Splendor/train_vrpo.py

"""
Entry point for VRPO training. Mirrors train_ai.py and shares the DQN's
directories (RL/trained_agents for checkpoints, RL/saved_files/tensorboard_logs
for logs) so the README's TensorBoard command shows VRPO runs alongside DQN
ones. Filenames carry a 'vrpo_' prefix (+ _actor/_critic) so nothing collides.

Hyperparameters are env-overridable for grid search, e.g.:
    VRPO_KL_COEF=0.05 VRPO_ACTOR_LR=1e-4 python train_vrpo.py
Key knobs are encoded into the run name so grid runs stay distinct in TensorBoard.
"""

import os
from datetime import datetime, timedelta

from RL.vrpo_loop import vrpo_loop

# Knobs worth distinguishing runs by (env var -> short tag). Only non-default
# overrides appear in the run name, keeping the common case tidy.
_NAME_KNOBS = [
    ('VRPO_KL_COEF', 'kl'), ('VRPO_ACTOR_LR', 'alr'), ('VRPO_CRITIC_LR', 'clr'),
    ('VRPO_LAMBDA', 'lam'), ('VRPO_CLIP', 'clip'), ('VRPO_ROLLOUT', 'roll'),
    ('VRPO_CRITIC_BUFFER', 'cbuf'),
]


def _config_suffix():
    parts = []
    for env, tag in _NAME_KNOBS:
        val = os.getenv(env)
        if val is not None:
            parts.append(f"{tag}{val}")
    # Critic layers: sanitize dashes to 'x' so the tag stays one token in the
    # run name (e.g. crit1024x1024x512), distinct from the actor's a-b-c block.
    cl = os.getenv('VRPO_CRITIC_LAYERS')
    if cl is not None:
        parts.append("crit" + cl.replace(',', '-').replace('-', 'x'))
    return ("__" + "_".join(parts)) if parts else ""


def get_unique_filename(layer_sizes):
    nickname = "-".join(map(str, layer_sizes))
    timestamp = datetime.now() - timedelta(hours=6)
    timestamp = timestamp.strftime("%m-%d-%H-%M")
    return f"{timestamp}__vrpo_{nickname}{_config_suffix()}"


def get_paths(layer_sizes):
    """'_dir' for folders, '_path' for files with extensions."""
    backup_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.getenv('WORKSPACE_DIR', backup_dir)

    rl_dir = os.path.join(base_dir, "RL")
    # Share the DQN's directories so the README's command
    #   tensorboard --logdir=/workspace/RL/saved_files/tensorboard_logs
    # picks up VRPO runs too (alongside DQN runs, for direct comparison).
    agents_dir = os.path.join(rl_dir, "trained_agents")
    saved_files_dir = os.path.join(rl_dir, "saved_files")

    nickname = get_unique_filename(layer_sizes)

    # Strength-eval opponent: the existing (superhuman) DQN inference model.
    dqn_eval_path = os.path.join(agents_dir, "inference_model.keras")

    paths = {
        "base_dir": base_dir,
        "layer_sizes": layer_sizes,
        "rl_dir": rl_dir,
        "saved_files_dir": saved_files_dir,
        "actor_save_path": os.path.join(agents_dir, f"{nickname}_actor.keras"),
        "critic_save_path": os.path.join(agents_dir, f"{nickname}_critic.keras"),
        "actor_best_path": os.path.join(agents_dir, f"{nickname}_best_actor.keras"),
        "critic_best_path": os.path.join(agents_dir, f"{nickname}_best_critic.keras"),
        "tensorboard_dir": os.path.join(saved_files_dir, "tensorboard_logs", nickname),
        "dqn_eval_path": dqn_eval_path if os.path.exists(dqn_eval_path) else None,
    }

    for key, path in paths.items():
        if isinstance(path, str):
            if key.endswith("_dir"):
                os.makedirs(path, exist_ok=True)
            elif key.endswith("_path") and not key.startswith("dqn"):
                os.makedirs(os.path.dirname(path), exist_ok=True)

    return paths


def main():
    layer_sizes = [512, 512, 256]
    paths = get_paths(layer_sizes)
    print(paths)

    iterations = int(os.getenv('VRPO_ITERS', 5000))
    vrpo_loop(paths, iterations=iterations)


if __name__ == "__main__":
    main()
