#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py.
"""

from __future__ import print_function

import numpy as np

from .agent import BaseAgent
from .tilecode import Tilecoder


class Agent(BaseAgent):
    """Implements the SARSA learning algorithm."""

    def __init__(self):
        self.actions = None
        self.action_index = None
        self.action_feature = None

        self.q_values = None
        self.tilecoder = None

        self.last_obs = None
        self.last_action = None
        self.last_features = None

        self.alpha = None
        self.gamma = None
        self.epsilon = None

    def agent_init(self, agent_init_info={}):
        """Setup for the agent called when the experiment first starts."""

        self.actions = np.asarray(agent_init_info.get("actions", np.zeros(0)))
        self.action_index = {i: np.where(i == self.actions)[0][0]
                             for i in self.actions}
        self.action_feature = agent_init_info['action_in_features']

        self.tilecoder = Tilecoder(**agent_init_info)
        self.q_values = np.zeros((self.actions.size,
                                  self.tilecoder.num_features))

        self.gamma = float(agent_init_info.get('gamma', 1.0))
        self.alpha = float(agent_init_info.get('alpha', 0.1))
        self.epsilon = float(agent_init_info.get('epsilon', 0))

    def agent_start(self, observation, agent_start_info={}):
        """The first method called when the experiment starts, called after
        the environment starts.

        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.

        Returns:
            The first action the agent takes.
        """

        self.last_obs = observation
        self.last_features = self.tilecoder.get_features(observation)

        self.last_action = np.random.choice(self.actions)

        return self.last_action

    def choose_action(self, features):
        if np.random.uniform() > self.epsilon:
            action_values = np.einsum("ij,j->i",
                                      self.q_values,
                                      features)
            return self.actions[np.argmax(action_values)]
        else:
            return np.random.choice(self.actions)

    def action_value(self, features, action):
        return np.einsum("i,i->",  # vector dot product
                         self.q_values[self.action_index[action]],
                         features)

    def agent_step(self, reward, observation):
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step

        Returns:
            The action the agent is taking.
        """
        features = self.tilecoder.get_features(observation)

        action = self.choose_action(features)

        if self.action_feature:
            features[-self.actions.size:] = self.actions == action

        td_error = (reward
                    + self.gamma * self.action_value(features, action)
                    - self.action_value(self.last_features, self.last_action))

        self.q_values[self.last_action] += self.alpha * td_error

        self.last_action = action
        self.last_obs = observation
        self.last_features = features

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        td_error = reward - self.action_value(self.last_features,
                                              self.last_action)
        self.q_values[self.last_action] += self.alpha * td_error

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        self.last_action = None
        self.last_obs = None
        self.last_features = None

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.

        Args:
            message: The message passed to the agent.

        Returns:
            The response (or answer) to the message.
        """
        pass
