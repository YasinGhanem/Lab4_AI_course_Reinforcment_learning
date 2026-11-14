
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



def random_policy():
    policy = np.full((ROWS, COLS), "", dtype=object)

    for row in range(ROWS):
        for col in range(COLS):
            policy[row, col] = random.choice(ACTIONS)

    return policy


def policy_evaluation(policy, rewards):
    values = np.zeros((ROWS, COLS))

    for iteration in range(MAX_ITERATIONS):
        new_values = np.zeros((ROWS, COLS))
        delta = 0.0

        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)
                action = policy[row, col]

                new_values[row, col] = calculate_action_value(state, action, values, rewards)

                delta = max(delta, abs(new_values[row, col] - values[row, col]))

        values = new_values

        if delta < THETA:
            break

    return values


def policy_improvement(values, rewards, old_policy):
    new_policy = np.full((ROWS, COLS), "", dtype=object)
    policy_stable = True

    for row in range(ROWS):
        for col in range(COLS):
            state = (row, col)
            old_action = old_policy[row, col]

            best_action = None
            best_action_value = -float("inf")

            for action in ACTIONS:
                q_value = calculate_action_value(state, action, values, rewards)

                if q_value > best_action_value:
                    best_action_value = q_value
                    best_action = action

            new_policy[row, col] = best_action

            if best_action != old_action:
                policy_stable = False

    return new_policy, policy_stable


def policy_iteration(rewards):
    policy = random_policy()

    for iteration in range(MAX_ITERATIONS):
        values = policy_evaluation(policy, rewards)
        policy, stable = policy_improvement(values, rewards, policy)

        if stable:
            break

    symbol_policy = np.full((ROWS, COLS), "", dtype=object)

    for row in range(ROWS):
        for col in range(COLS):
            symbol_policy[row, col] = ACTION_SYMBOLS[policy[row, col]]

    return values, symbol_policy, iteration + 1



def print_grid(grid, title):
    print(title)
    print("-" * len(title))

    for row in range(ROWS):
        for col in range(COLS):
            print(f"{grid[row, col]:>10.2f}", end=" ")
        print()

    print()


def print_policy(policy, title):
    print(title)
    print("-" * len(title))

    for row in range(ROWS):
        for col in range(COLS):
            print(f"{policy[row, col]:>3}", end=" ")
        print()

    print()


def main():
    random.seed(42)
    np.random.seed(42)

    for case_number, (R1, R2) in enumerate(REWARD_CASES, start=1):
        print("=" * 70)
        print(f"Case {case_number}: R1 = {R1}, R2 = {R2}")
        print("=" * 70)

        rewards = create_reward_grid(R1, R2)

        print_grid(rewards, "Reward Grid")

        vi_values, vi_policy, vi_iterations = value_iteration(rewards)

        print(f"Value Iteration converged in {vi_iterations} iterations.\n")
        print_grid(vi_values, "Value Iteration Values")
        print_policy(vi_policy, "Value Iteration Policy")

        pi_values, pi_policy, pi_iterations = policy_iteration(rewards)

        print(f"Policy Iteration converged in {pi_iterations} iterations.\n")
        print_grid(pi_values, "Policy Iteration Values")
        print_policy(pi_policy, "Policy Iteration Policy")


if __name__ == "__main__":
    main()
