import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from helpers import ALPHA, GAMMA, print_policy, print_values, new_policy, max_dict
from sklearn.kernel_approximation import RBFSampler


def epsilon_greedy(model, state, eps=0.1):
  if np.random.random() < (1 - eps):
    values = model.predict_actions(state)
    return np.argmax(values)
  else:
    return model.env.action_space.sample()

def gather_samples(env, episodes=10000):
  samples = []
  for _ in range(episodes):
    state, info = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
      action = env.action_space.sample()
      state_action = np.concatenate((state, [action]))
      samples.append(state_action)

      state, reward, done, truncated, info = env.step(action)
  return samples

class Model:
  def __init__(self, env):
    self.env = env

    samples = gather_samples(env)
    self.featurizer = RBFSampler()
    self.featurizer.fit(samples)

    dims = self.featurizer.n_components
    self.weight = np.zeros(dims)

  def predict(self, state, action):
    state_action = np.concatenate((state, [action]))
    x = self.featurizer.transform([state_action])[0]
    return x @ self.weight
  
  def predict_actions(self, state):
    return [self.predict(state, action) for action in range(self.env.action_space.n)]
  
  def gradient(self, state, action):
    print(f"Model Gradient")
    print(f"State: {state}")
    print(f"Action: {action}")
    state_action = np.concatenate((state, [action]))
    print(f"State action: {state_action}")
    x = self.featurizer.transform([state_action])[0]
    print(f"X: {x}")
    return x

def test_agent(model, env, episodes=20):
  reward_per_episode = np.zeros(episodes)
  for i in range(episodes):
    done = False
    truncated = False
    episode_reward = 0
    state, info = env.reset()
    while not (done or truncated):
      action = epsilon_greedy(model, state, eps=0)
      state, reward, done, truncated, info = env.step(action)
      episode_reward += reward
    reward_per_episode[i] = episode_reward
  return np.mean(reward_per_episode)

def watch_agent(model, env, eps):
  done = False
  truncated = False
  episode_reward = 0
  state, info = env.reset()
  while not (done or truncated):
    action = epsilon_greedy(model, state, eps)
    state, reward, done, truncated, info = env.step(action)
    episode_reward += reward
  print(f"\nEpsiode reward: {episode_reward}")

def run():
  env = gym.make("CartPole-v1", render_mode="rgb_array")

  model = Model(env)
  reward_per_episode = []
  watch_agent(model, env, 0)

  episodes = 2
  for i in range(episodes):
    print(f"\n\nIteration: {i}")
    state, info = env.reset()
    print(f"State: {state}")
    print(f"Info: {info}")
    episode_reward = 0
    done = False
    truncated = False

    while not (done or truncated):
      action = epsilon_greedy(model, state)
      print(f"Action: {action}")
      next_state, reward, done, truncated, info = env.step(action)

      print(f"Next state: {next_state}")
      print(f"Reward: {reward}")
      print(f"Done: {done}")
      print(f"Truncated: {truncated}")
      print(f"Info: {info}")
      if done:
        target = reward
        print(f"Target: {target}")
      else:
        values = model.predict_actions(next_state)
        print(f"Values: {values}")
        target = reward + GAMMA * np.max(values)
        print(f"Target: {target}")

      gradient = model.gradient(state, action)
      print(f"Gradient: {gradient}")
      error = target - model.predict(state, action)
      print(f"Error: {error}")
      model.weight += ALPHA * error * gradient

      episode_reward += reward
      print(f"Episode reward: {episode_reward}")
      
      state = next_state
      print(f"State: {state}")

    if (i + 1) % 50 == 0:
      print(f"Episode: {i + 1}, Reward: {episode_reward}")

    if i > 20 and np.mean(reward_per_episode[-20:]) == 500:
      print("Early exit")
      break
    
    reward_per_episode.append(episode_reward)
    print(f"Reward per epsiode: {reward_per_episode}")
  
  test_reward = test_agent(model, env)
  print(f"Average test reward: {test_reward}")

  plt.plot(reward_per_episode)
  plt.title("Reward Per Episode")
  plt.show()

  env = gym.make("CartPole-v1", render_mode="human")
  watch_agent(model, env, eps=0)

if __name__ == '__main__':
  run()