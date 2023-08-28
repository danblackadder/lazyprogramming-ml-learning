import numpy as np
import matplotlib.pyplot as plt
from grid import create_gridworld
from helpers import ALPHA, ACTION_SPACE, GAMMA, print_policy, print_values, new_policy, max_dict, epsilon_greedy
from sklearn.kernel_approximation import RBFSampler

def gather_samples(grid, episodes=10000):
  samples = []
  for _ in range(episodes):
    state = grid.reset()
    samples.append(state)
    while not grid.game_over():
      action = np.random.choice(ACTION_SPACE)
      grid.move(action)
      next_state = grid.get_current_state()
      samples.append(next_state)
  return samples

class Model:
  def __init__(self, grid):
    samples = gather_samples(grid)
    self.featurizer = RBFSampler()
    self.featurizer.fit(samples)

    dims = self.featurizer.n_components
    self.weight = np.zeros(dims)

  def predict(self, state):
    x = self.featurizer.transform([state])[0]
    return x @ self.weight
  
  def gradient(self, state):
    x = self.featurizer.transform([state])[0]
    return x

def run():
  grid = create_gridworld()

  print("\nRewards:")
  print_values(grid.rewards, grid)

  greedy_policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }

  model = Model(grid)
  episode_mean_square_error = []

  episodes = 10000
  for _ in range(episodes):
    state = grid.reset()
    values = model.predict(state)

    steps = 0
    episode_error = 0
    while not grid.game_over():
      action = epsilon_greedy(greedy_policy, state)
      reward = grid.move(action)
      next_state = grid.get_current_state()

      if grid.is_terminal(next_state):
        target = reward
      else:
        next_values = model.predict(next_state)
        target = reward + GAMMA * next_values

      gradient = model.gradient(state)
      error = target - values
      model.weight += ALPHA * error * gradient

      steps += 1
      episode_error += error*error

      state = next_state
      values = next_values

    mean_square_error = episode_error / steps
    episode_mean_square_error.append(mean_square_error)
  
  plt.plot(episode_mean_square_error)
  plt.title("MSE per episode")
  plt.show()

  V = {}
  states = grid.get_all_states()
  for state in states:
    if state in grid.actions:
      V[state] = model.predict(state)
    else:
      # terminal state or state we can't otherwise get to
      V[state] = 0
  
  print("\nValues:")
  print_values(V, grid)
  print("\nPolicy:")
  print_policy(greedy_policy, grid)

if __name__ == '__main__':
  run()