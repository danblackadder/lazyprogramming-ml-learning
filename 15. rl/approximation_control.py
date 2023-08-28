import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid import create_gridworld, create_negative_gridworld
from helpers import ALPHA, ACTION_SPACE, GAMMA, print_policy, print_values, new_policy, max_dict
from sklearn.kernel_approximation import RBFSampler

ACTION2INT = {a: i for i, a in enumerate(ACTION_SPACE)}
INT2ONEHOT = np.eye(len(ACTION_SPACE))

def epsilon_greedy(model, state, eps=0.1):
  if np.random.random() < (1 - eps):
    values = model.predict_actions(state)
    return ACTION_SPACE[np.argmax(values)]
  else:
    return np.random.choice(ACTION_SPACE)
  
def one_hot(k):
  return INT2ONEHOT[k]

def merge_state_action(s, a):
  ai = one_hot(ACTION2INT[a])
  return np.concatenate((s, ai))

def gather_samples(grid, episodes=10000):
  samples = []
  for _ in range(episodes):
    state = grid.reset()
    while not grid.game_over():
      action = np.random.choice(ACTION_SPACE)
      state_action = merge_state_action(state, action)
      samples.append(state_action)

      grid.move(action)
      state = grid.get_current_state()
  return samples

class Model:
  def __init__(self, grid):
    samples = gather_samples(grid)
    self.featurizer = RBFSampler()
    self.featurizer.fit(samples)

    dims = self.featurizer.n_components
    self.weight = np.zeros(dims)

  def predict(self, state, action):
    state_action = merge_state_action(state, action)
    x = self.featurizer.transform([state_action])[0]
    return x @ self.weight
  
  def predict_actions(self, state):
    return [self.predict(state, action) for action in ACTION_SPACE]
  
  def gradient(self, state, action):
    state_action = merge_state_action(state, action)
    x = self.featurizer.transform([state_action])[0]
    return x

def run():
  grid = create_negative_gridworld(step_cost=-0.1)

  print("\nRewards:")
  print_values(grid.rewards, grid)

  model = Model(grid)
  reward_per_episode = []
  state_visit_count = {}

  episodes = 10000
  for _ in range(episodes):
    state = grid.reset()
    state_visit_count[state] = state_visit_count.get(state, 0) + 1
    episode_reward = 0

    while not grid.game_over():
      action = epsilon_greedy(model, state)
      reward = grid.move(action)
      next_state = grid.get_current_state()
      state_visit_count[next_state] = state_visit_count.get(next_state, 0) + 1

      if grid.game_over():
        target = reward
      else:
        values = model.predict_actions(next_state)
        target = reward + GAMMA * np.max(values)

      gradient = model.gradient(state, action)
      error = target - model.predict(state, action)
      model.weight += ALPHA * error * gradient

      episode_reward += reward
      
      state = next_state

    reward_per_episode.append(episode_reward)
  
  plt.plot(reward_per_episode)
  plt.title("Reward Per Episode")
  plt.show()

  V = {}
  greedy_policy = {}
  states = grid.get_all_states()
  for s in states:
    if s in grid.actions:
      values = model.predict_actions(s)
      V[s] = np.max(values)
      greedy_policy[s] = ACTION_SPACE[np.argmax(values)]
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0
  
  print("\nValues:")
  print_values(V, grid)
  print("\nPolicy:")
  print_policy(greedy_policy, grid)

  print("State Visit Count:")
  state_sample_count_arr = np.zeros((grid.rows, grid.cols))
  for x in range(grid.rows):
    for y in range(grid.cols):
      if (x, y) in state_visit_count:
        state_sample_count_arr[x,y] = state_visit_count[(x,y)]
  df = pd.DataFrame(state_sample_count_arr)
  print(df)


if __name__ == '__main__':
  run()