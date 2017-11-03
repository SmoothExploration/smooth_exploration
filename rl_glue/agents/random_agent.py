#!/usr/bin/env python

"""An example of a random agent.
"""

from .agent import BaseAgent
import random


class Agent(BaseAgent):
    """agent does *no* learning, selects action 0 always"""

    def __init__(self):
        self.actions = None
        self.q_values = None

    def agent_init(self, agent_init_info={}):
        """Setup for the agent called when the experiment first starts."""

        if "actions" in agent_init_info:
            self.actions = agent_init_info["actions"]

        if "state_array" in agent_init_info:
            self.q_values = agent_init_info["state_array"]

    def agent_start(self, observation, agent_start_info={}):
        """The first method called when the experiment starts, called after
        the environment starts.

        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.

        Returns:
            The first action the agent takes.
        """
        return random.choice(self.actions)

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
        return random.choice(self.actions)

    def agent_end(self, reward):
        """Run when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.

        Args:
            message: The message passed to the agent.

        Returns:
            The response (or answer) to the message.
        """
        pass
