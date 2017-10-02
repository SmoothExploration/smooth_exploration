#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py.
"""

from __future__ import print_function

import numpy as np

from agent import Agent


class SARSA(Agent):
    """Implements the SARSA learning algorithm."""

    def __init__(self):
        self.actions = None
        self.q_values = None

        self.last_obs = None
        self.last_action = None

        self.alpha = None
        self.gamma = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

        self.actions = np.asarray(agent_info.get("actions", np.zeros(0)))

        states = np.asarray(agent_info.get("state_array", np.zeros(0)))
        self.q_values = np.zeros((states.size, self.actions.size))

        self.gamma = float(agent_info.get('gamma', 1.0))
        self.alpha = float(agent_info.get('alpha', 0.1))

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.

        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.

        Returns:
            The first action the agent takes.
        """

        self.last_obs = observation

        self.last_action = np.random.choice(self.actions)

        return self.last_action

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

        action = self.actions[np.argmax(self.q_values[observation])]

        td_error = (reward
                    + self.gamma * self.q_values[observation, action]
                    - self.q_values[self.last_obs, self.last_action])
        self.q_values[self.last_obs, self.last_action] += self.alpha * td_error

        self.last_action = action
        self.last_obs = observation

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        td_error = reward - self.q_values[self.last_obs, self.last_action]
        self.q_values[self.last_obs, self.last_action] += self.alpha * td_error

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        self.actions = None
        self.q_values = None
        self.last_action = None
        self.last_obs = None
        self.gamma = None

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.

        Args:
            message: The message passed to the agent.

        Returns:
            The response (or answer) to the message.
        """
        pass
