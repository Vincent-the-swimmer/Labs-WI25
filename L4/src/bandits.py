import numpy as np
import pandas as pd
from typing import List


def update(q: float, r: float, k: int) -> float:
    """
    Update the Q-value using the given reward and number of times the action has been taken.

    Parameters:
    q (float): The current Q-value.
    r (float): The reward received for the action.
    k (int): The number of times the action has been taken before.

    Returns:
    float: The updated Q-value.
    """
    # Note: since k is the number of times the action has been taken before this update, we need to add 1 to k before using it in the formula.
    # TODO
    k += 1
    new_Q = q + 1/(k)*(r-q)
    return new_Q


def greedy(q_estimate: np.ndarray) -> int:
    """
    Selects the action with the highest Q-value.

    Parameters:
    q_estimate (numpy.ndarray): 1-D Array of Q-values for each action.

    Returns:
    int: The index of the action with the highest Q-value.
    """
    
    # TODO
    return np.argmax(q_estimate)


def egreedy(q_estimate: np.ndarray, epsilon: float) -> int:
    """
    Implements the epsilon-greedy exploration strategy for multi-armed bandits.

    Parameters:
    q_estimate (numpy.ndarray): 1-D Array of estimated action values.
    epsilon (float): Exploration rate, determines the probability of selecting a random action.
    n_arms (int): Number of arms in the bandit. default is 10.

    Returns:
    int: The index of the selected action.
    """
    # TODO
    n_arms = len(q_estimate)
    if np.random.random() < (1-epsilon):
        return np.argmax(q_estimate)
    return np.random.randint(0,n_arms)


def empirical_egreedy(epsilon: float, n_trials: int, n_arms: int, n_plays: int) -> List[List[float]]:
    """
    Run the epsilon-greedy algorithm on a multi-armed bandit problem. For each play,
    the algorithm selects an action based on the epsilon-greedy strategy and updates
    the Q-value of the selected action. For each trial, the algorithm returns
    the rewards for each play, a total of n_plays.

    Args:
        epsilon (float): epsilon value for the epsilon-greedy algorithm.
        n_trials (int): number of trials to run the algorithm
        n_arms (int): number of arms in the bandit
        n_plays (int): number of plays in each trial

    Returns:
        List[List[float]]: A list of rewards for each play in each trial.
    """
    rewards = []  # stores the rewards for each trial
    # TODO
    rewards = [[None] * n_trials for _ in range(n_plays)]
    random_number = np.random.default_rng()
    for i in range(n_trials):
        mean = np.random.normal(0.0, 1.0, n_arms)
        q_val = np.zeros(n_arms)
        num_pulled = np.zeros(n_arms)
        for j in range(n_plays):
            action = egreedy(q_val, epsilon)
            action_reward = np.mean(np.random.normal(mean[action], 1))
            ##rewards[action].append(action_reward)
            num_pulled[action]+=1
            new_q = update(q_val[action], action_reward, num_pulled[action])
            print(action)
            q_val[int(action)] += new_q
        return rewards
