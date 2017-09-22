#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
 
  env *ignores* actions: rewards are all random
"""

from __future__ import print_function
from utils import rand_norm, rand_in_range, rand_un
import numpy as np

class Environment:
     """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required methods
    """

    def __init__(self):
        self.this_reward_observation = (None, None, None) # this_reward_observation: (floating point, NumPy array, Boolean)

    def env_init(self):
        """Setup for the environment called when the experiment first starts
        
        Note:
            Initialize a tuple with the reward, first state observation, boolean indicating if it's terminal
        """
        local_observation = np.zeros(0) # An empty NumPy array

        self.this_reward_observation = (0.0, local_observation, False)


    def env_start(self):
        """The first method called when the experiment starts, called before the agent starts
        
        Returns:
            Numpy array: the first state observation from the environment
        """
        return self.this_reward_observation[1]

    def env_step(self, this_action):
        """A step taken by the environment
        
        Args:
            this_action (Numpy array): the action taken by the agent

        Returns:
            (float, Numpy array, Boolean): a tuple of the reward, state observation, and boolean indicating if it's terminal
        """
        the_reward = rand_norm(0.0, 1.0) # rewards drawn from (0, 1) Gaussian

        self.this_reward_observation = (the_reward, self.this_reward_observation[1], False)

        return self.this_reward_observation

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        return

    def env_message(self, inMessage):
        """A message asking the environment for information
        
        Args:
            inMessage (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if inMessage == "what is your name?":
            return "my name is skeleton_environment!"
      
        # else
        return "I don't know how to respond to your message"
