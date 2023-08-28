import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grid import create_gridworld
from helpers import ALPHA, ACTION_SPACE, GAMMA, print_policy, print_values, new_policy, max_dict, epsilon_greedy

def run():
  grid = create_gridworld()

  print("\nRewards:")
  print_values(grid.rewards, grid)

  policy = new_policy(grid)
  # policy = {
  #   (2, 0): 'U',
  #   (1, 0): 'U',
  #   (0, 0): 'R',
  #   (0, 1): 'R',
  #   (0, 2): 'R',
  #   (1, 2): 'R',
  #   (2, 1): 'R',
  #   (2, 2): 'R',
  #   (2, 3): 'U',
  # }

  values = {}
  state_sample_count = {}
  states = grid.get_all_states()
  for state in states:
    values[state] = 0
    state_sample_count[state] = 0

  deltas = []

  episodes = 10000
  for i in range(episodes):
    state = grid.reset()
  
    delta = 0
    while not grid.game_over():
      action = epsilon_greedy(policy, state)
      reward = grid.move(action)
      next_state = grid.get_current_state()

      current_value = values[state]
      values[state] = values[state] + ALPHA * (reward + GAMMA * values[next_state] - values[state])
      delta = max(delta, np.abs(values[state] - current_value))
      state_sample_count[state] += 1

      state = next_state
    
    deltas.append(delta)
  
  plt.plot(deltas)
  plt.show()

  print("\nFinal Values:")
  print_values(values, grid)

  print("\nFinal Policy:")
  print_policy(policy, grid)

  print("State Sample Count:")
  state_sample_count_arr = np.zeros((grid.rows, grid.cols))
  for x in range(grid.rows):
    for y in range(grid.cols):
      if (x, y) in state_sample_count:
        state_sample_count_arr[x,y] = state_sample_count[(x, y)]
  df = pd.DataFrame(state_sample_count_arr)
  print(df)

if __name__ == '__main__':
  run()