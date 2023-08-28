import numpy as np

from helpers import ACTION_SPACE, GAMMA, THRESHOLD, print_policy, print_values, new_policy
from grid import create_gridworld

def get_probabilities_and_rewards(grid):
  probabilities = {}
  rewards = {}
  
  for x in range(grid.rows):
    for y in range(grid.cols):
      state = (x, y)
      if not grid.is_terminal(state):
        for action in ACTION_SPACE:
          next_state  = grid.get_next_state(state, action)
          probabilities[(state, action, next_state)] = 1
          if next_state in grid.rewards:
            rewards[(state, action, next_state)] = grid.rewards[next_state]

  return probabilities, rewards

def policy_evaluation(grid, policy, rewards, probabilities, initial_value=None):
  if initial_value is None:
    value = {}
    for state in grid.get_all_states():
      value[state] = 0
  else:
    value = initial_value

  i = 0
  
  while True:
    biggest_change = 0
    for state in grid.get_all_states():
      if not grid.is_terminal(state):
        current_v = value[state]
        new_value = 0

        for action in ACTION_SPACE:
          for next_state in grid.get_all_states():
            action_prob = 1 if policy.get(state) == action else 0
            new_reward = rewards.get((state, action, next_state), 0)
            new_value += action_prob * probabilities.get((state, action, next_state), 0) * (new_reward + GAMMA * value[next_state])
        
        value[state] = new_value
        biggest_change = max(biggest_change, np.abs(current_v - value[state]))
    
    print(f"Iteration: {i} - Biggest Change: {biggest_change}")
    print_values(value, grid)

    i += 1

    if  biggest_change < THRESHOLD:
      break
  
  return value

def policy_improvement(grid, policy, rewards, probabilities, value):
  optimal_policy = True
  for state in grid.actions.keys():
    previous_action = policy[state]
    new_action = None
    best_value = float("-inf")

    for action in ACTION_SPACE:
      new_value = 0
      for new_state in grid.get_all_states():
        new_reward = rewards.get((state, action, new_state), 0)
        new_value += probabilities.get((state, action, new_state), 0) * (new_reward + GAMMA * value[new_state])

      if new_value > best_value:
        best_value = new_value
        new_action = action

    policy[state] = new_action
    if new_action != previous_action:
      optimal_policy = False
    
  print("\nPolicy Improved:")
  print_policy(policy, grid)

  return optimal_policy, policy

def run():
  grid = create_gridworld()

  probabilities, rewards = get_probabilities_and_rewards(grid)
  print(rewards)
  return
  
  print("\nRewards:")
  print_values(rewards, grid)

  policy = new_policy(grid)

  print("\nPolicy:")
  print_policy(policy, grid)

  value = None

  while True:
    value = policy_evaluation(grid, policy, rewards, probabilities, initial_value=value)
    optimal_policy, policy = policy_improvement(grid, policy, rewards, probabilities, value)

    if optimal_policy:
      break
  
  print("\nValues:")
  print_values(value, grid)
  print("\nPolicy:")
  print_policy(policy, grid)
  print("\n\n")

if __name__ == "__main__":
  run()