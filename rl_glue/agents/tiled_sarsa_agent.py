#!/usr/bin/env python

from .agent import BaseAgent
import numpy as np
import tilecoder



class Agent(BaseAgent):
    """agent does *no* learning, selects action 0 always"""
    def __init__(self):
        self.last_action = None
        self.q_values = None
        self.actions = None
        self.iht = tilecoder.IHT(1024)
        self.epsilon = 0.1
        self.alpha = 0.1
        self.lam = 0.1 # lambda

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

        if "actions" in agent_info:
            self.actions = agent_info["actions"]

        if "state_array" in agent_info:
            self.q_values = agent_info["state_array"]

        if "epsilon" in agent_info:
            self.epsilon = agent_info["epsilon"]

        if "alpha" in agent_info:
            self.alpha = agent_info["alpha"]

        if "lambda" in agent_info:
            self.lam = agent_info["lambda"]

        self.last_action = 0

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.last_action = 0  # set first action to 0
        self.last_state = np.copy(observation)

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
        local_action = 0  # choose the action here

        # might do some learning here

        self.last_action = local_action
        self.last_state = np.copy(observation)

        return self.last_action

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