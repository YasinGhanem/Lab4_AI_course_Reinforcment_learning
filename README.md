# Stochastic GridWorld: MDP Planning Algorithms

## About the Project

This project uses Value Iteration and Policy Iteration to find the best actions for an agent moving in a 5×5 GridWorld.

The agent's movements are not always predictable, and the goal is to find the optimal policy based on the rewards assigned to each state.

The project also compares how different reward values affect the agent's decisions.

## Environment

- **Grid Size:** 5×5
- **Actions:** Up, Down, Left, Right
- **Movement Probability:** 70% for the chosen action and 10% for each other direction
- **Walls:** The agent stays in the same position if it hits a wall
- **Discount Factor:** 0.95

## Algorithms

**Value Iteration:** Uses Bellman updates to calculate the value of each state and find the optimal policy.

**Policy Iteration:** Starts with a random policy and repeatedly evaluates and improves it until the policy becomes stable.

## Reward Configurations

The algorithms are tested with four different reward combinations:

| Case | R1 | R2 |
|---|---|---|
| 1 | 100 | 110 |
| 2 | 10 | 100 |
| 3 | 1 | 10 |
| 4 | 10 | 15 |

## Technologies Used

Python, NumPy, Reinforcement Learning, Markov Decision Processes (MDPs), Dynamic Programming.

## How to Run

First, install NumPy:

```bash
pip install numpy
```

Then run the program:

```bash
python main.py
```

## Results

For each reward combination, the program shows the calculated state values, optimal policy, and number of iterations for both algorithms.

By comparing the results, we can see how changing rewards affects the agent's decisions and how Value Iteration and Policy Iteration approach the same problem.
