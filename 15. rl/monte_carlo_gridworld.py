import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import ACTION_SPACE, GAMMA, print_policy, print_values, new_policy, max_dict, epsilon_greedy
from grid import create_gridworld


def play_game(grid, policy, max_steps=20):
    state = grid.reset()
    action = epsilon_greedy(policy, state)

    states = [state]
    actions = [action]
    rewards = [0]

    for _ in range(max_steps):
        reward = grid.move(action)
        next_state = grid.get_current_state()

        rewards.append(reward)
        states.append(next_state)

        if grid.game_over():
            break
        else:
            action = epsilon_greedy(policy, next_state)
            actions.append(action)

    return states, actions, rewards


def run():
    grid = create_gridworld()

    print("\nRewards:")
    print_values(grid.rewards, grid)

    policy = new_policy(grid)

    Q = {}
    sample_counts = {}
    state_sample_count = {}
    states = grid.get_all_states()
    for state in states:
        if state in grid.actions:
            Q[state] = {}
            sample_counts[state] = {}
            state_sample_count[state] = 0
            for action in ACTION_SPACE:
                Q[state][action] = 0
                sample_counts[state][action] = 0
        else:
            pass

    deltas = []
    for i in range(10000):

        biggest_change = 0
        states, actions, rewards = play_game(grid, policy)

        states_actions = list(zip(states, actions))

        G = 0
        total = len(states)
        for i in range(total - 2, -1, -1):
            state = states[i]
            action = actions[i]

            G = rewards[i+1] + GAMMA * G

            if (state, action) not in states_actions[:i]:

                current_q = Q[state][action]
                sample_counts[state][action] += 1
                learning_rate = 1 / sample_counts[state][action]
                Q[state][action] = current_q + learning_rate * (G - current_q)

                policy[state] = max_dict(Q[state])[0]

                state_sample_count[state] += 1

                biggest_change = max(biggest_change, np.abs(current_q - Q[state][action]))
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    values = {}
    for state, _ in Q.items():
        values[state] = max_dict(Q[state])[1]

    print("\nFinal Values:")
    print_values(values, grid)

    print("\nFinal Policy:")
    print_policy(policy, grid)

    print("State Sample Count:")
    state_sample_count_arr = np.zeros((grid.rows, grid.cols))
    for x in range(grid.rows):
        for y in range(grid.cols):
            if (x, y) in state_sample_count:
                state_sample_count_arr[x, y] = state_sample_count[(x, y)]
    df = pd.DataFrame(state_sample_count_arr)
    print(df)


if __name__ == '__main__':
    run()
