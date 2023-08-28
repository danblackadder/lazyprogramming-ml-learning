import numpy as np

from helpers import ACTION_SPACE, GAMMA, THRESHOLD, print_policy, print_values
from grid import create_windy_gridworld

def get_probabilities_and_rewards(grid):
  probabilities = {}
  rewards = {}
  
  for (state, action), value in grid.probabilities.items():
    for next_state, probability in value.items():
      probabilities[(state, action, next_state)] = probability
      rewards[(state, action, next_state)] = grid.rewards.get(next_state, 0)

  return probabilities, rewards

def policy_evaluation(grid, rewards, probabilities):
  value = {}
  for state in grid.get_all_states():
    value[state] = 0

  i = 0
  
  while True:
    biggest_change = 0
    for state in grid.get_all_states():
      if not grid.is_terminal(state):
        current_v = value[state]
        new_value = float('-inf')

        for action in ACTION_SPACE:
          temp_value = 0
          for next_state in grid.get_all_states():
            new_reward = rewards.get((state, action, next_state), 0)
            temp_value += probabilities.get((state, action, next_state), 0) * (new_reward + GAMMA * value[next_state])
          
          if temp_value > new_value:
            new_value = temp_value

        value[state] = new_value
        biggest_change = max(biggest_change, np.abs(current_v - value[state]))
    
    print(f"Iteration: {i} - Biggest Change: {biggest_change}")
    print_values(value, grid)

    i += 1

    if  biggest_change < THRESHOLD:
      break
  
  return value

def policy_improvement(grid, rewards, probabilities, value):
  policy = {}
  for state in grid.actions.keys():
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
    
  print("\nPolicy Improved:")
  print_policy(policy, grid)

  return policy

def run():
  grid = create_windy_gridworld(step_cost=-0.2)

  probabilities, rewards = get_probabilities_and_rewards(grid)
  
  print("\nRewards:")
  print_values(rewards, grid)

  value = policy_evaluation(grid, rewards, probabilities)
  policy = policy_improvement(grid, rewards, probabilities, value)
  
  print("\nFinal Values:")
  print_values(value, grid)
  print("\nFinal Policy:")
  print_policy(policy, grid)
  print("\n\n")

if __name__ == "__main__":
  run()