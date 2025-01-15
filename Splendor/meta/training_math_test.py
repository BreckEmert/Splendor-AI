import numpy as np
import tensorflow as tf

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(3,)), 
        tf.keras.layers.Dense(5, activation='relu'), 
        tf.keras.layers.Dense(2, activation=None)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

model = create_model()
target_model = create_model()

batch = [
    [[15, 16, 17], 0, 0, [15, 16, 17], 1, [True, True]], 
    [[15, 16, 17], 1, 1, [18, 19, 20], 0, [False, True]]
]

states = tf.convert_to_tensor([mem[0] for mem in batch], dtype=tf.float32)
actions = tf.convert_to_tensor([mem[1] for mem in batch], dtype=tf.int32)
rewards = tf.convert_to_tensor([mem[2] for mem in batch], dtype=tf.float32)
next_states = tf.convert_to_tensor([mem[3] for mem in batch], dtype=tf.float32)
dones = tf.convert_to_tensor([mem[4] for mem in batch], dtype=tf.float32)
legal_masks = tf.convert_to_tensor([mem[5] for mem in batch], dtype=tf.bool)

print("States: ")
print(states, "\n")
print("Actions: ")
print(actions, "\n\n")

# Calculate this turn's qs with primary model
qs = model(states, training=False)

# Predict next turn's actions with primary model
next_actions = model(next_states, training=False)
next_actions = tf.where(legal_masks, next_actions, tf.fill(next_actions.shape, -np.inf))
next_actions = tf.argmax(next_actions, axis=1, output_type=tf.int32)

# Calculate next turn's qs with target model
next_qs = target_model(next_states, training=False)
selected_next_qs = tf.gather_nd(next_qs, tf.stack([tf.range(len(next_actions)), next_actions], axis=1))

# Ground qs with reward and value trajectory
targets = rewards + dones * 0.9 * selected_next_qs
actions_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
target_qs = tf.tensor_scatter_nd_update(qs, actions_indices, targets)

# Fit
from collections import deque
dense_history = deque(maxlen=10)
action_history = deque(maxlen=10)

for step in range(1000):
    with tf.GradientTape() as tape:
        predictions = model(states, training=True)
        loss = tf.keras.losses.mean_squared_error(target_qs, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    dense_weights = model.get_layer('dense').get_weights()[0]
    action_weights = model.get_layer('dense_1').get_weights()[0]

    dense_history.append(dense_weights)
    action_history.append(action_weights)

    if step%50 == 0:
        dense_mean = tf.reduce_mean(tf.stack(dense_history), axis=0)
        action_mean = tf.reduce_mean(tf.stack(action_history), axis=0)
        print(dense_mean.numpy().round(2))
        print(action_mean.numpy().round(2))
        print()
