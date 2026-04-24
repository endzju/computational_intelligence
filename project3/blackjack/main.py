from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


def exploration_policy(state, epsilon, q_table):
    r = np.random.random()

    if r <= epsilon:
        action = np.random.choice(len(q_table[state]))
    else:
        action = np.argmax(q_table[state])

    return action


def train_q_learning(
    episodes: int,
    alpha: float,
    alpha_decay: float,
    alpha_min: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    epsilon_min: float,
):
    env = gym.make("Blackjack-v1")

    action_space: spaces.Discrete = env.action_space  # ty:ignore[invalid-assignment]
    n_actions: int = action_space.n
    q_table = defaultdict(lambda: np.zeros(n_actions))

    episode_rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = exploration_policy(state, epsilon, q_table)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = float(reward)

            if terminated:
                target = float(reward)
            else:
                target = reward + gamma * np.max(q_table[next_state])

            q_table[state][action] = (1 - alpha) * q_table[state][
                action
            ] + alpha * target

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(alpha_min, alpha * alpha_decay)

        episode_rewards.append(total_reward)

        if (episode + 1) % 10000 == 0:
            print(
                f"Episode: {episode + 1} | Total reward: {total_reward} | Epsilon: {epsilon:.4f} | Alpha: {alpha:.4f}"
            )

    env.close()
    return q_table, episode_rewards


def plot_learning_curve(rewards, window_size=1000):
    weights = np.ones(window_size) / window_size
    smoothed_rewards = np.convolve(rewards, weights, mode="valid")

    plt.figure(figsize=(10, 6))
    plt.plot(
        smoothed_rewards, color="blue", label=f"Średnia nagroda (okno={window_size})"
    )

    plt.title("Krzywa uczenia - Q-Learning w Blackjack-v1", fontsize=14)
    plt.xlabel("Epizody", fontsize=12)
    plt.ylabel("Średnia suma nagród", fontsize=12)
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Linia remisu (0.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    final_q_table, rewards = train_q_learning(
        episodes=500_000,
        alpha=0.5,
        alpha_min=0.01,
        alpha_decay=0.9999,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9999,
    )
    print("Training finished.")

    plot_learning_curve(rewards, window_size=1000)
