
import numpy as np
import random

ROWS = 5
COLS = 5
GAMMA = 0.95
THETA = 1e-8
MAX_ITERATIONS = 10000

ACTIONS = ["U", "D", "R", "L"]

ACTION_DELTAS = {
    "U": (-1, 0),
    "D": (1, 0),
    "R": (0, 1),
    "L": (0, -1)
}

ACTION_SYMBOLS = {
    "U": "↑",
    "D": "↓",
    "R": "→",
    "L": "←"
}

REWARD_CASES = [
    (100, 110),
    (10, 100),
    (1, 10),
    (10, 15)
]


def create_reward_grid(R1, R2):
    return np.array([
        [R1, 1, 0, -1, R2],
        [2,  1, 0, -1, -2],
        [2,  1, 0, -1, -2],
        [2,  1, 0, -1, -2],
        [2,  1, 0, -1, -2]
    ], dtype=float)


def move(state, action):
    row, col = state
    dr, dc = ACTION_DELTAS[action]

    new_row = row + dr
    new_col = col + dc

    if new_row < 0 or new_row >= ROWS or new_col < 0 or new_col >= COLS:
        return state

    return (new_row, new_col)


def get_transition_outcomes(state, intended_action):
    outcomes = []

    for actual_action in ACTIONS:
        if actual_action == intended_action:
            probability = 0.7
        else:
            probability = 0.1

        next_state = move(state, actual_action)
        outcomes.append((probability, next_state))

    return outcomes



def calculate_action_value(state, action, values, rewards):
    row, col = state

    total = 0.0

    for probability, next_state in get_transition_outcomes(state, action):
        next_row, next_col = next_state

        total += probability * values[next_row, next_col]

    return rewards[row, col] + GAMMA * total


def value_iteration(rewards):
    values = np.zeros((ROWS, COLS))
    policy = np.full((ROWS, COLS), "", dtype=object)

    for iteration in range(MAX_ITERATIONS):
        new_values = np.zeros((ROWS, COLS))
        delta = 0.0

        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)

                action_values = []

                for action in ACTIONS:
                    q_value = calculate_action_value(state, action, values, rewards)
                    action_values.append(q_value)

                best_value = max(action_values)

                new_values[row, col] = best_value

                delta = max(delta, abs(new_values[row, col] - values[row, col]))

        values = new_values

        if delta < THETA:
            break

    for row in range(ROWS):
        for col in range(COLS):
            state = (row, col)

            best_action = None
            best_action_value = -float("inf")

            for action in ACTIONS:
                q_value = calculate_action_value(state, action, values, rewards)

                if q_value > best_action_value:
                    best_action_value = q_value
                    best_action = action

            policy[row, col] = ACTION_SYMBOLS[best_action]

    return values, policy, iteration + 1

