from collections import deque
from pathlib import Path

import gymnasium as gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()


def epsilon_greedy_policy(model, state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(0, 2)
    else:
        q_values = model(state[np.newaxis], training=False)[0]
        return np.argmax(q_values)


def sample_experiences(batch_size, replay_buffer):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(6)
    ]


def play_one_step(env, model, state, epsilon, replay_buffer):
    action = epsilon_greedy_policy(model, state, epsilon)
    next_state, reward, done, truncated, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done, truncated))
    return next_state, reward, done, truncated, info


def training_step(
    model, batch_size, discount_factor, optimizer, loss_fn, replay_buffer
):
    experiences = sample_experiences(batch_size, replay_buffer)
    states, actions, rewards, next_states, dones, truncateds = experiences

    next_q_values = model(next_states, training=False).numpy()
    max_next_q_values = next_q_values.max(axis=1)

    runs = 1.0 - (dones | truncateds)
    target_q_values = rewards + runs * discount_factor * max_next_q_values
    target_q_values = target_q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, 2)

    with tf.GradientTape() as tape:
        all_q_values = model(states)
        q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_q_values, q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train(
    model,
    n_episodes,
    max_steps,
    batch_size,
    discount_factor,
    lr,
    total_decay_episodes=100,
):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    replay_buffer = deque(maxlen=20000)
    optimizer = keras.optimizers.Nadam(learning_rate=lr)
    loss_fn = keras.losses.mean_squared_error

    episode_rewards = []

    for episode in range(n_episodes):
        curr_reward = 0
        obs, info = env.reset()

        epsilon = max(1 - episode / total_decay_episodes, 0.05)

        for step in range(max_steps):
            obs, reward, done, truncated, info = play_one_step(
                env, model, obs, epsilon, replay_buffer
            )
            curr_reward += reward
            if done or truncated:
                break

        episode_rewards.append(curr_reward)

        if len(replay_buffer) > batch_size:
            for _ in range(5):
                training_step(
                    model,
                    batch_size,
                    discount_factor,
                    optimizer,
                    loss_fn,
                    replay_buffer,
                )

    env.close()
    return model, episode_rewards


def save_plot(rewards, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Suma nagród", color="lightblue", alpha=0.7)

    window = 10
    if len(rewards) >= window:
        sma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(rewards)), sma, label=f"SMA ({window})", color="blue"
        )

    plt.title(title)
    plt.xlabel("Epizod")
    plt.ylabel("Nagroda")
    plt.grid(True, alpha=0.3)
    plt.legend()

    filepath = SCRIPT_DIR / filename
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Zapisano wykres: {filepath.name}")


if __name__ == "__main__":
    discount_factors = [0.9, 0.95, 0.99]
    hidden_activations = ["relu", "mish"]

    param_rewards = {}
    best_score = -float("inf")

    for df in discount_factors:
        for activ in hidden_activations:
            print(f"\nValidation Gamma: {df}, Activation: {activ}")

            model = keras.Sequential(
                [
                    keras.layers.Dense(24, activation=activ, input_shape=[4]),
                    keras.layers.Dense(24, activation=activ),
                    keras.layers.Dense(2, activation="linear"),
                ]
            )
            trained_model, episode_rewards = train(
                model,
                n_episodes=100,
                max_steps=200,
                batch_size=32,
                discount_factor=df,
                lr=0.005,
                total_decay_episodes=80,
            )
            avg_score = sum(episode_rewards[-10:]) / 10
            param_rewards[(df, activ)] = avg_score
            print(f"Mean reward: {avg_score:.2f}")

            save_plot(
                episode_rewards,
                f"DQN CartPole (Gamma: {df}, Activ: {activ})",
                f"dqn_eval_{df}_{activ}.png",
            )

            if avg_score > best_score:
                best_score = avg_score
                best_params = (df, activ)

    print("\n" + "=" * 50)
    print(f"Best params: Gamma={best_params[0]}, Activation='{best_params[1]}'")
    print("=" * 50)

    final_model = keras.Sequential(
        [
            keras.layers.Dense(24, activation=best_params[1], input_shape=[4]),
            keras.layers.Dense(24, activation=best_params[1]),
            keras.layers.Dense(2, activation="linear"),
        ]
    )

    final_model, final_rewards = train(
        final_model,
        n_episodes=500,
        max_steps=500,
        batch_size=32,
        discount_factor=best_params[0],
        lr=0.005,
        total_decay_episodes=300,
    )

    save_plot(
        final_rewards,
        f"Finalny Model DQN (Gamma: {best_params[0]}, Activ: {best_params[1]})",
        "dqn_final_best_model.png",
    )
