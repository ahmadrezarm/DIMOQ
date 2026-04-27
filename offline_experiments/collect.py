"""
Dataset collection utilities for the offline DIMOQ experiments.

Policy factories all return callables of the form: state (flat int) -> action (int).

Score tables are pre-computed once from a trained DIMOQ so that data collection
is just table lookups — calling get_q_dists_lst at every step would be too slow
for large datasets.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


# --------------------------------------------------------------------------- #
# Core collection loop
# --------------------------------------------------------------------------- #

def collect_dataset(env, num_transitions, policy_fn, flatten_fn=None):
    """Roll out policy_fn in env and log (s, a, r, s') until num_transitions reached.

    Args:
        env:              Gym-compatible environment.
        num_transitions:  Total number of transitions to collect.
        policy_fn:        Callable: flat_state (int) -> action (int).
        flatten_fn:       Callable: raw obs -> flat int.  None = identity.

    Returns:
        List of (s, a, r, s') tuples where s/s' are flat ints and r is ndarray.
    """
    _flatten = flatten_fn if flatten_fn is not None else (lambda x: x)
    dataset = []

    state, _ = env.reset()
    state = _flatten(state)

    while len(dataset) < num_transitions:
        action = policy_fn(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = _flatten(next_obs)
        dataset.append((int(state), int(action), reward.copy(), int(next_state)))

        if terminated or truncated:
            state, _ = env.reset()
            state = _flatten(state)
        else:
            state = next_state

    return dataset


# --------------------------------------------------------------------------- #
# Policy factories
# --------------------------------------------------------------------------- #

def random_policy(num_actions, rng):
    """Uniform random action selection."""
    return lambda s: int(rng.integers(num_actions))


def make_score_table(dimoq, score_fn_name='linear'):
    """Pre-compute per-state action scores from a trained DIMOQ.

    Args:
        dimoq:          Trained DIMOQ instance.
        score_fn_name:  'linear' | 'hypervolume' | 'distance'.

    Returns:
        dict: state -> np.ndarray of shape (num_actions,).
    """
    if score_fn_name == 'linear':
        score_fn = dimoq.score_linear_utility
    elif score_fn_name == 'hypervolume':
        score_fn = dimoq.score_hypervolume
    elif score_fn_name == 'distance':
        score_fn = dimoq.score_inter_distance
    else:
        raise ValueError(f'Unknown score_fn_name: {score_fn_name}')

    table = {}
    for s in range(dimoq.num_states):
        raw = score_fn(s, pool=None)
        table[s] = np.array(raw, dtype=float)
    return table


def make_scalarized_score_table(dimoq, alpha):
    """Pre-compute α·EV₀ + (1-α)·EV₁ per action from DIMOQ's Q-distributions.

    Args:
        dimoq:  Trained DIMOQ instance.
        alpha:  Weight on objective 0.  alpha=1 → pure obj-0 greedy.

    Returns:
        dict: state -> np.ndarray of shape (num_actions,).
    """
    table = {}
    for s in range(dimoq.num_states):
        q_dists_lst = dimoq.get_q_dists_lst(s, pool=None)
        scores = []
        for q_dists in q_dists_lst:
            if not q_dists:
                scores.append(0.0)
                continue
            evs = np.array([d.get_expected_value() for d in q_dists])
            mean_ev = evs.mean(axis=0)
            scores.append(alpha * mean_ev[0] + (1.0 - alpha) * mean_ev[1])
        table[s] = np.array(scores)
    return table


def greedy_policy(score_table, rng):
    """Greedy w.r.t. a pre-computed score table (breaks ties randomly)."""
    def policy(s):
        scores = score_table.get(s, np.zeros(1))
        return int(rng.choice(np.flatnonzero(scores == scores.max())))
    return policy


def epsilon_greedy_policy(score_table, num_actions, epsilon, rng):
    """ε-greedy over a pre-computed score table."""
    def policy(s):
        if rng.random() < epsilon:
            return int(rng.integers(num_actions))
        scores = score_table.get(s, np.zeros(num_actions))
        return int(rng.choice(np.flatnonzero(scores == scores.max())))
    return policy
