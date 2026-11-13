
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
