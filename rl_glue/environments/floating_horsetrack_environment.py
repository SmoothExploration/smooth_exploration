#!/usr/bin/env python

"""env *ignores* actions: rewards are all random
"""

from __future__ import print_function

import numpy as np

from .environment import BaseEnvironment


class Environment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    actions = [-1, 1]

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.current_state = 0
        self.reward_obs_term = (reward, observation, termination)
        self.a_bar = 0

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        reward = 0  # reward is 0 at each time step
        self.current_state = np.array([0.])

        return reward, self.current_state, False

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = np.array([0.])  # Agent starts at state 0

        return self.current_state

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        self.current_state += action - 0.3 * self.a_bar
        self.current_state %= 100

        self.a_bar *= 0.8
        self.a_bar += 0.2 * action

        reward = -1
        terminal = False

        # Terminal state is 50, reward is 1.
        if abs(self.current_state - 50) <= 0.5:
            terminal = True

        return reward, self.current_state, terminal

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        self.a_bar = 0

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"
