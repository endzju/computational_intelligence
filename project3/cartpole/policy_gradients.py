import time
from pathlib import Path

import gymnasium as gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = tf.random.uniform([1, 1]) > left_proba
        y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
        grads = tape.gradient(loss, model.trainable_variables)
        obs, reward, done, truncated, info = env.step(int(action[0, 0]))

        return obs, reward, done, truncated, grads


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []

    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs, info = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, truncated, grads = play_one_step(
                env, obs, model, loss_fn
            )
            current_rewards.append(reward)
            current_grads.append(grads)

            if done or truncated:
                break

        all_rewards.append(current_rewards)
        all_grads.append(current_grads)

    return all_rewards, all_grads


def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor

    return discounted


def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [
        discount_rewards(rewards, discount_factor) for rewards in all_rewards
    ]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    return [
        (discounted_rewards - reward_mean) / reward_std
        for discounted_rewards in all_discounted_rewards
    ]


def train_model(
    model,
    n_iterations,
    n_episodes_per_update,
    n_max_steps,
    discount_factor,
    lr,
):
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    optimizer = keras.optimizers.Nadam(learning_rate=lr)
    loss_fn = keras.losses.binary_crossentropy

    history_mean_returns = []

    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(
            env, n_episodes_per_update, n_max_steps, model, loss_fn
        )
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

        episode_returns = [sum(episode_rewards) for episode_rewards in all_rewards]
        mean_episode_return = np.mean(episode_returns)

        history_mean_returns.append(mean_episode_return)

        print(
            f"Training iteration {iteration} | Mean episode return: {mean_episode_return}"
        )

        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [
                    final_reward * grad[var_index]
                    for final_rewards, grads in zip(all_final_rewards, all_grads)
                    for final_reward, grad in zip(final_rewards, grads)
                ],
                axis=0,
            )
            all_mean_grads.append(mean_grads)

        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    env.close()
    return model, history_mean_returns


def test_model(model, n_episodes=5, render=True):
    render_mode = "human" if render else None
    test_env = gym.make("CartPole-v1", render_mode=render_mode)

    all_test_rewards = []

    for episode in range(n_episodes):
        obs, info = test_env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            left_proba = model(obs[np.newaxis], training=False)

            action = 0 if left_proba.numpy()[0][0] > 0.5 else 1

            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += float(reward)

            if render:
                time.sleep(0.02)

        all_test_rewards.append(total_reward)
        print(f"Test Episode: {episode + 1} | Total Reward: {total_reward}")

    test_env.close()

    mean_reward = np.mean(all_test_rewards)
    print(f"\nAverage Reward over {n_episodes} test episodes: {mean_reward}")

    return mean_reward


def save_training_plot(rewards_history, discount_factor, lr):
    plt.figure(figsize=(10, 6))
    plt.plot(
        rewards_history,
        label=f"Średnia nagroda ($\gamma$={discount_factor}, lr={lr})",
        color="blue",
    )
    plt.title(
        f"Krzywa uczenia REINFORCE - CartPole-v1\nDiscount Factor: {discount_factor}",
        fontsize=14,
    )
    plt.xlabel("Iteracje (Aktualizacje wag)", fontsize=12)
    plt.ylabel("Średnia suma nagród na epizod", fontsize=12)
    plt.axhline(y=200, color="r", linestyle="--", alpha=0.5, label="Cel CartPole (200)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filepath = (
        Path(__file__).parent / f"training_curve_gamma_{discount_factor}_lr_{lr}.png"
    )

    plt.savefig(filepath, dpi=300)
    plt.close()


if __name__ == "__main__":
    n_iterations = 150
    n_episodes_per_update = 10
    n_max_steps = 500

    lr = 0.01
    discount_factors = (0.9, 0.95, 0.99)

    for d in discount_factors:
        model = keras.Sequential(
            [
                keras.layers.Dense(5, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile()
        model, training_rewards = train_model(
            model, n_iterations, n_episodes_per_update, n_max_steps, d, lr
        )

        save_training_plot(training_rewards, d, lr)

        mean_reward = test_model(model)
        print(f"Discount factor: {d} | LR: {lr} | Mean test reward: {mean_reward}\n")
