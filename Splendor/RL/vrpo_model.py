# Splendor/RL/vrpo_model.py
"""
Actor and critic networks for VRPO (Variance-Reduced Policy Optimization),
the algorithm from Fan & Farina, "GAE Falls Short in Imperfect-Information
Self-Play Reinforcement Learning" (arXiv:2605.19235).

Two SEPARATE networks (not a shared trunk) so the actor's policy-gradient
signal and the critic's regression signal don't fight over a shared body.
Both reuse the same MLP trunk shape as the existing DQN (Dense + LeakyReLU)
- this is the deliberate "pragmatic" choice: Splendor's state is a small,
fixed 251-vector with no natural sequence structure, so the paper's
transformer/Llama encoder buys little here. Swapping in attention or a
shared trunk later is isolated to this file.
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Stop NUMA/info spam

import tensorflow as tf
from keras.layers import Input, Dense, LeakyReLU
from keras.initializers import HeNormal


NEG_INF = -1e9  # masking sentinel; exp(NEG_INF) underflows cleanly to 0.0


def _trunk(state_input, layer_sizes, prefix):
    x = state_input
    for i, n in enumerate(layer_sizes):
        x = Dense(n, kernel_initializer=HeNormal(), name=f'{prefix}_dense{i+1}')(x)
        x = LeakyReLU(negative_slope=0.3)(x)
    return x


def build_actor(state_dim, action_dim, layer_sizes):
    """State -> raw policy logits over all actions (masking applied later)."""
    s = Input(shape=(state_dim,))
    x = _trunk(s, layer_sizes, 'actor')
    logits = Dense(action_dim, kernel_initializer=HeNormal(), name='policy_logits')(x)
    return tf.keras.Model(inputs=s, outputs=logits, name='vrpo_actor')


def build_critic(state_dim, action_dim, layer_sizes):
    """State -> Q(s, a) for every action. This is the centralized
    action-value critic that Q-boosting needs.
    """
    s = Input(shape=(state_dim,))
    x = _trunk(s, layer_sizes, 'critic')
    q = Dense(action_dim, kernel_initializer=HeNormal(), name='q_values')(x)
    return tf.keras.Model(inputs=s, outputs=q, name='vrpo_critic')


def masked_log_softmax(logits, mask):
    """log pi(.|s) over legal actions only. `mask` is a bool tensor,
    True where the action is legal. Illegal actions get NEG_INF, so their
    probability underflows to exactly 0.
    """
    neg = tf.fill(tf.shape(logits), NEG_INF)
    masked = tf.where(mask, logits, neg)
    return masked - tf.reduce_logsumexp(masked, axis=-1, keepdims=True)


def masked_softmax(logits, mask):
    return tf.exp(masked_log_softmax(logits, mask))


def policy_value(q_values, pi):
    """V^pi(s) = sum_a pi(a|s) Q(s, a).

    This is the heart of Q-boosting: the state value is the policy
    EXPECTATION over action-values, computed by enumerating actions,
    rather than the single sampled action GAE would use. Illegal actions
    contribute 0 because pi is 0 there.
    """
    return tf.reduce_sum(pi * q_values, axis=-1)
