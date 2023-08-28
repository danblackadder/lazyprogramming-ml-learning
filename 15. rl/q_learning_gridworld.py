import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grid import create_windy_gridworld
from helpers import ALPHA, ACTION_SPACE, GAMMA, print_policy, print_values, new_policy, max_dict, q_epsilon_greedy

def run():
  grid = create_windy_gridworld(step_cost=-0.1)

  print("\nRewards:")
  print_values(grid.rewards, grid)

  Q = {}
  states = grid.get_all_states()
  for state in states:
    Q[state] = {}
    for action in ACTION_SPACE:
      Q[state][action] = 0

  update_count = {}
  rewards_per_episode = []

  episodes = 10000
  for i in range(episodes):
    state = grid.reset()
    episode_reward = 0

    while not grid.game_over():
      action = q_epsilon_greedy(Q, state, eps=0.1)
      reward = grid.move(action)
      next_state = grid.get_current_state()

      episode_reward += reward

      max_q = max_dict(Q[next_state])[1]
      Q[state][action] = Q[state][action] + ALPHA * (reward + GAMMA * max_q - Q[state][action])

      update_count[state] = update_count.get(state, 0) + 1

      state = next_state
    
    rewards_per_episode.append(episode_reward)
  
  plt.plot(rewards_per_episode)
  plt.show()

  policy = {}
  values = {}

  for state in grid.get_actions().keys():
    argmax, max_q = max_dict(Q[state])
    policy[state] = argmax
    values[state] = max_q

  print("\nFinal Values:")
  print_values(values, grid)

  print("\nFinal Policy:")
  print_policy(policy, grid)

  print("State Sample Count:")
  total = np.sum(list(update_count.values()))
  for key, value in update_count.items():
    update_count[key] = float(value) / total
  print_values(update_count, grid)

if __name__ == '__main__':
  run()