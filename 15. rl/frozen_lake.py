import random
import math
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_POS = (0, 0)
ACTION_SPACE = ("U", "D", "L", "R")
GAMMA = 0.9
ALPHA = 0.1


def print_values(v, grid):
    for x in range(grid.rows):
        print("------" * grid.cols)
        for y in range(grid.cols):
            value = v.get((x, y), 0)
            if value >= 0:
                print(f" {format(round(value, 2), '.2f')}|", end="")
            else:
                print(f"{format(round(value, 2), '.2f')}|", end="")
        print("")
    print("------" * grid.cols)
    print("\n")


def print_rewards(rewards, grid):
    cols = grid.rows + 1
    rows = grid.cols + 1
    for x in range(rows):
        print("------" * cols)
        for y in range(cols):
            value = rewards.get((x, y), 0)
            if value >= 0:
                print(f" {format(round(value, 2), '.2f')}|", end="")
            else:
                print(f"{format(round(value, 2), '.2f')}|", end="")
        print("")
    print("------" * cols)
    print("\n")


def print_policy(policy, grid):
    for x in range(grid.rows):
        print("------" * grid.cols)
        for y in range(grid.cols):
            action = policy.get((x, y), " ")
            print(f"  {action}  |", end="")
        print("")
    print("------" * grid.cols)
    print("\n")


def max_dict(d):
    max_val = max(d.values())
    max_keys = [key for key, val in d.items() if val == max_val]

    return np.random.choice(max_keys), max_val


def q_epsilon_greedy(Q, s, eps=0.1):
    if np.random.random() < eps:
        return np.random.choice(ACTION_SPACE)
    else:
        a_opt = max_dict(Q[s])[0]
        return a_opt


class FrozenLake:
    def __init__(self, rows=4, cols=4, starting_pos=DEFAULT_POS):
        self.rows = rows
        self.cols = cols
        self.starting_pos = starting_pos
        self.states = [starting_pos]
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

    def set_current_state(self, value):
        self.states.append(value)

    def get_all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

    def reset(self):
        self.states = [self.starting_pos]
        return self.get_current_state()

    def move(self, action):
        current_state = self.get_current_state()
        x = current_state[0]
        y = current_state[1]
        reward = 0

        if action in self.actions[(current_state)]:
            if action == "U":
                if x > 0:
                    x -= 1
                else:
                    x = 0
                    reward -= 1
            elif action == "D":
                if x < self.rows - 1:
                    x += 1
                else:
                    x = self.rows - 1
                    reward -= 1
            elif action == "L":
                if y > 0:
                    y -= 1
                else:
                    y = 0
                    reward -= 1
            elif action == "R":
                if y < self.cols - 1:
                    y += 1
                else:
                    y = self.cols - 1
                    reward -= 1

        next_state = (x, y)
        self.set_current_state(next_state)

        if reward == 0:
            reward = self.rewards.get(next_state, 0)

        return reward

    def game_over(self):
        current_state = self.get_current_state()
        return current_state not in self.actions


def is_valid(grid, rewards):
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= grid.rows or c_new < 0 or c_new >= grid.cols:
                    continue
                if rewards[r_new, c_new] == 1:
                    return True
                if rewards[r_new, c_new] != -1:
                    frontier.append((r_new, c_new))
    return False


def create_frozen_lake(rows, cols, step_cost=0):
    valid = False
    while not valid:
        grid = FrozenLake(rows, cols, DEFAULT_POS)

        rewards = {}
        actions = {}

        for x in range(rows):
            for y in range(cols):
                if x == rows - 1 and y == cols - 1:
                    rewards[(x, y)] = 1
                elif x == 0 and y == 0:
                    rewards[(x, y)] = step_cost
                    actions[(x, y)] = ACTION_SPACE
                else:
                    rewards[(x, y)] = step_cost
                    actions[(x, y)] = ACTION_SPACE

        lake_count = math.floor(rows * cols / 4)
        for _ in range(lake_count):
            all_rewards = actions.keys()
            lake = random.choice(list(all_rewards))
            if lake != (0, 0):
                rewards[lake] = -1
                del actions[lake]

        valid = is_valid(grid, rewards)

    grid.set_rewards(rewards)
    grid.set_actions(actions)

    return grid


def main():
    grid = create_frozen_lake(rows=10, cols=20, step_cost=-0.1)

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
            Q[state][action] = Q[state][action] + ALPHA * (
                reward + GAMMA * max_q - Q[state][action]
            )
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

    print("\nRewards:")
    print(grid.rewards)
    print_values(grid.rewards, grid)

    print("\nFinal Values:")
    print_values(values, grid)

    print("\nFinal Policy:")
    print_policy(policy, grid)

    print("State Sample Count:")
    total = np.sum(list(update_count.values()))
    for key, value in update_count.items():
        update_count[key] = float(value) / total
    print_values(update_count, grid)


if __name__ == "__main__":
    main()
