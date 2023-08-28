import numpy as np

ACTION_SPACE = ('U', 'D', 'L', 'R')
DEFAULT_STATE = [(2, 0)]
GAMMA = 0.9
THRESHOLD = 1e-3
ALPHA = 0.1


def new_policy(grid):
  policy = {}
  for state in grid.actions.keys():
    policy[state] = np.random.choice(ACTION_SPACE)
  return policy


def print_policy(policy, grid):
  for x in range(grid.rows):
    print("------------------------")
    for y in range(grid.cols):
      action = policy.get((x, y), ' ')
      print(f"  {action}  |", end="")
    print("")
  print("------------------------\n")


def print_values(v, grid):
  for x in range(grid.rows):
    print("------------------------")
    for y in range(grid.cols):
      value = v.get((x, y), 0)
      if value >= 0:
        print(f" {format(round(value, 2), '.2f')}|", end="")
      else:
        print(f"{format(round(value, 2), '.2f')}|", end="")
    print("")
  print("------------------------\n")


def max_dict(d):
  max_val = max(d.values())
  max_keys = [key for key, val in d.items() if val == max_val]

  return np.random.choice(max_keys), max_val


def epsilon_greedy(policy, state, eps=0.1):
  if np.random.random() < (1 - eps):
    return policy[state]
  else:
    return np.random.choice(ACTION_SPACE)


def q_epsilon_greedy(Q, s, eps=0.1):
  if np.random.random() < eps:
    return np.random.choice(ACTION_SPACE)
  else:
    a_opt = max_dict(Q[s])[0]
    return a_opt
