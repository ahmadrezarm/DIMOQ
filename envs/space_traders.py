import gymnasium as gym
import numpy as np

from gym import spaces


class SpaceTraders(gym.Env):
    """A class for the space traders environment."""

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.actions = {'Indirect': 0, 'Direct': 1, 'Teleport': 2}
        self.states = {'A': 0, 'B': 1, 'Goal': 2, 'Terminal': 3}
        self.description = {'A': {'Indirect': {'Reward': {'B': [0, -12], 'Terminal': [0, 0]},
                                               'Probability': {'B': 1., 'Terminal': 0}},
                                  'Direct': {'Reward': {'B': [0, -6], 'Terminal': [0, -1]},
                                             'Probability': {'B': 0.9, 'Terminal': 0.1}},
                                  'Teleport': {'Reward': {'B': [0, 0], 'Terminal': [0, 0]},
                                               'Probability': {'B': 0.85, 'Terminal': 0.15}}},
                            'B': {'Indirect': {'Reward': {'Goal': [1, -10], 'Terminal': [0, 0]},
                                               'Probability': {'Goal': 1., 'Terminal': 0}},
                                  'Direct': {'Reward': {'Goal': [1, -8], 'Terminal': [0, -7]},
                                             'Probability': {'Goal': 0.9, 'Terminal': 0.1}},
                                  'Teleport': {'Reward': {'Goal': [1, 0], 'Terminal': [0, 0]},
                                               'Probability': {'Goal': 0.85, 'Terminal': 0.15}}},
                            'Goal': {'Indirect': {'Reward': {'Goal': [0, 0]},
                                                  'Probability': {'Goal': 1.}},
                                     'Direct': {'Reward': {'Goal': [0, 0]},
                                                'Probability': {'Goal': 1.}},
                                     'Teleport': {'Reward': {'Goal': [0, 0]},
                                                  'Probability': {'Goal': 1.}}},
                            'Terminal': {'Indirect': {'Reward': {'Terminal': [0, 0]},
                                                      'Probability': {'Terminal': 1.}},
                                         'Direct': {'Reward': {'Terminal': [0, 0]},
                                                    'Probability': {'Terminal': 1.}},
                                         'Teleport': {'Reward': {'Terminal': [0, 0]},
                                                      'Probability': {'Terminal': 1.}}}}

        self.finite_horizon = True
        self.max_timesteps = 2
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        self.num_objectives = 2
        self._terminal_states = [self.states['Goal'], self.states['Terminal']]

        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)
        self.reward_space = spaces.Box(low=np.array([0., -12.]), high=np.array([1., 0.]))

        self._reward_function, self._transition_function = self._init_functions()

        self.start_state = 0
        self._state = self.start_state
        self._timestep = 0

    def _init_functions(self):
        """Initialize the reward and transition functions."""
        rewards = np.zeros((self.num_states, self.num_actions, self.num_states, self.num_objectives))
        transitions = np.zeros((self.num_states, self.num_actions, self.num_states))

        for state in self.description.keys():
            for action in self.description[state].keys():
                for next_state, reward in self.description[state][action]['Reward'].items():
                    rewards[self.states[state], self.actions[action], self.states[next_state]] = reward

                for next_state, prob in self.description[state][action]['Probability'].items():
                    transitions[self.states[state], self.actions[action], self.states[next_state]] = prob

        return rewards, transitions

    def get_config(self):
        """Get the configuration of the environment."""
        return {
            "seed": self.seed
        }

    def reset(self, seed=None, options=None):
        self._state = self.start_state
        self._timestep = 0

        return self._state, {}

    def step(self, action):
        """Take a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            int, ndarray, bool, bool, dict: The next state, the reward, whether the episode is done, whether the episode
                is truncated and no info.
        """
        next_state = self.rng.choice(self.num_states, p=self._transition_function[self._state, action])
        rewards = self._reward_function[self._state, action, next_state]

        done = next_state in self._terminal_states
        self._state = next_state
        self._timestep += 1

        return self._state, rewards, done, self._timestep == self.max_timesteps, {}
