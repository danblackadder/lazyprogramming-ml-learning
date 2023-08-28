import numpy as np
from helpers import DEFAULT_STATE

class Gridworld:
  def __init__(self, rows, cols, state = DEFAULT_STATE.copy()):
    self.rows = rows
    self.cols = cols
    self.states = state
    self.rewards = {}
    self.actions = {}

  def set_rewards(self, rewards):
    self.rewards = rewards
  
  def set_actions(self, actions):
    self.actions = actions
  
  def get_actions(self):
    return self.actions

  def get_current_state(self):
    return self.states[-1]
  
  def set_current_state(self, state):
    self.states.append(state)

  def is_terminal(self, state):
    return state not in self.actions
  
  def reset(self):#
    self.states = DEFAULT_STATE.copy()
    return self.get_current_state()

  def get_next_state(self, state, action):
    next_state = None
    if action in self.actions[state]:
      if action == "U":
        next_state = (state[0] - 1, state[1])
      elif action == "D":
        next_state = (state[0] + 1, state[1])
      elif action == "L":
        next_state = (state[0], state[1] - 1)
      elif action == "R":
        next_state = (state[0], state[1] + 1)
    return next_state

  def move(self, action):
    if action in self.actions[(self.get_current_state())]:
      if action == "U":
        current_state = self.get_current_state()
        next_state = (current_state[0] - 1, current_state[1])
        self.set_current_state(next_state)
      elif action == "D":
        current_state = self.get_current_state()
        next_state = (current_state[0] + 1, current_state[1])
        self.set_current_state(next_state)
      elif action == "L":
        current_state = self.get_current_state()
        next_state = (current_state[0], current_state[1] - 1)
        self.set_current_state(next_state)
      elif action == "R":
        current_state = self.get_current_state()
        next_state = (current_state[0], current_state[1] + 1)
        self.set_current_state(next_state)
    return self.rewards.get(self.get_current_state(), 0)

  def game_over(self):
    current_state = self.get_current_state()
    return current_state not in self.actions
  
  def get_state_history(self):
    return self.states
  
  def get_all_states(self):
    return set(self.actions.keys()) | set(self.rewards.keys())
  
def create_gridworld(rows = 3, cols = 4):
  grid = Gridworld(rows, cols)
  rewards = {(0,3): 1, (1,3): -1}
  actions = {
    (0, 0): ("D", "R"),
    (0, 1): ("L", "R"),
    (0, 2): ("L", "D", "R"),
    (1, 0): ("U", "D"),
    (1, 2): ("U", "D", "R"),
    (2, 0): ("U", "R"),
    (2, 1): ("L", "R"),
    (2, 2): ("L", "R", "U"),
    (2, 3): ("L", "U")
  }
  
  grid.set_rewards(rewards)
  grid.set_actions(actions)
  return grid

def create_negative_gridworld(rows = 3, cols = 4, step_cost = 0):
  grid = Gridworld(rows, cols)
  rewards = {    
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
    (0,3): 1, 
    (1,3): -1
  }
  actions = {
    (0, 0): ("D", "R"),
    (0, 1): ("L", "R"),
    (0, 2): ("L", "D", "R"),
    (1, 0): ("U", "D"),
    (1, 2): ("U", "D", "R"),
    (2, 0): ("U", "R"),
    (2, 1): ("L", "R"),
    (2, 2): ("L", "R", "U"),
    (2, 3): ("L", "U")
  }
  
  grid.set_rewards(rewards)
  grid.set_actions(actions)
  return grid


class WindyGridworld:
  def __init__(self, rows, cols, state = DEFAULT_STATE.copy()):
    self.rows = rows
    self.cols = cols
    self.states = state
    self.rewards = {}
    self.actions = {}
    self.probabilities = {}

  def set_rewards(self, rewards):
    self.rewards = rewards
  
  def set_actions(self, actions):
    self.actions = actions
    
  def get_actions(self):
    return self.actions

  def set_probabilities(self, probabilities):
    self.probabilities = probabilities

  def get_current_state(self):
    return self.states[-1]
  
  def set_current_state(self, state):
    self.states.append(state)

  def is_terminal(self, state):
    return state not in self.actions
  
  def reset(self):
    self.states = DEFAULT_STATE.copy()
    return self.get_current_state()

  def get_next_state(self, state, action):
    next_state = None
    if action in self.actions[state]:
      if action == "U":
        next_state = (state[0] - 1, state[1])
      elif action == "D":
        next_state = (state[0] + 1, state[1])
      elif action == "L":
        next_state = (state[0], state[1] - 1)
      elif action == "R":
        next_state = (state[0], state[1] + 1)
    return next_state

  def move(self, action):
    next_state_probabilities = self.probabilities[(self.get_current_state(), action)]
    next_states = list(next_state_probabilities.keys())
    next_probabilities = list(next_state_probabilities.values())
    next_state_idx = np.random.choice(len(next_states), p=next_probabilities)
    next_state = next_states[next_state_idx]

    self.set_current_state((next_state))

    return self.rewards.get(self.get_current_state(), 0)
    
  def game_over(self):
    current_state = self.get_current_state()
    return current_state not in self.actions
  
  def get_state_history(self):
    return self.states
  
  def get_all_states(self):
    return set(self.actions.keys()) | set(self.rewards.keys())
  
def create_windy_gridworld(rows = 3, cols = 4, step_cost = 0):
  grid = WindyGridworld(rows, cols)
  rewards = {    
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
    (0,3): 1, 
    (1,3): -1
  }
  actions = {
    (0, 0): ("D", "R"),
    (0, 1): ("L", "R"),
    (0, 2): ("L", "D", "R"),
    (1, 0): ("U", "D"),
    (1, 2): ("U", "D", "R"),
    (2, 0): ("U", "R"),
    (2, 1): ("L", "R"),
    (2, 2): ("L", "R", "U"),
    (2, 3): ("L", "U")
  }

  probabilities = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
  }
  
  grid.set_rewards(rewards)
  grid.set_actions(actions)
  grid.set_probabilities(probabilities)
  return grid