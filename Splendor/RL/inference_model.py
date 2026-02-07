# Splendor/RL/inference_model.py

import tensorflow as tf
# from keras.config import enable_unsafe_deserialization
from keras.models import load_model


class InferenceAgent:
    def __init__(self, model_path: str):
        # enable_unsafe_deserialization()

        # Dimensions
        self.state_dim = 386
        self.action_dim = 141
        self.batch_size = 512

        # Model
        self.model: tf.keras.Model = load_model(model_path)
        self.step = 0

    def get_predictions(self, state, legal_mask):
        """Returns q-values where legal."""
        qs = self.model(state[None, :], training=False)[0]
        return tf.where(legal_mask, qs, tf.fill(qs.shape, -tf.float32.max))

    def remember(self, *_): pass
    def replay(self): pass
    def save_model(self): pass
    def write_memory(self): pass
