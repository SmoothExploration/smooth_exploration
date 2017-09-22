#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions
 
"""

from __future__ import print_function
from utils import rand_in_range
import numpy as np

class Agent:
    """Implements the agent for an RLGlue environment

    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and agent_message are required methods
    """

    def __init__(self):
        self.last_action = None # last_action: NumPy array
        self.num_actions = 1

    def agent_init(self):
        """Setup for the agent called when the experiment first starts"""

        self.last_action = np.zeros(1) # generates a NumPy array with size 1 equal to zero

    def agent_start(self, this_observation):
        """The first method called when the experiment starts, called after the environment starts
        
        Args:
            this_observation (Numpy array): the state observation from the environment's evn_start function

        Returns:
            Numpy array: the first action the agent takes
        """

        self.last_action[0] = rand_in_range(self.num_actions)

        local_action = np.zeros(1)
        local_action[0] = rand_in_range(self.num_actions)

        return local_action[0]


    def agent_step(self, reward, this_observation): # returns NumPy array, reward: floating point, this_observation: NumPy array
        """A step taken by the agent
        
        Args:
            reward (float): the reward received for taking the last action taken
            this_observation (Numpy array): the state observation from the environment's step based, where the agent ended up
                after the last step

        Returns:
            Numpy array: the action the agent is taking
        """

        local_action = np.zeros(1)
        local_action[0] = rand_in_range(self.num_actions)

        # might do some learning here
        self.last_action = local_action

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward (float): the reward the agent received for entering the terminal state
        """
        return

    def agent_cleanup(self):
        """Cleanup done after the agent ends"""
        return

    def agent_message(self, inMessage):
        """A message asking the agent for information
        
        Args:
            inMessage (string): the message passed to the agent

        Returns:
            string: the response (or answer) to the message
        """

        if inMessage == "what is your name?":
            return "my name is skeleton_agent!"
      
        # else
        return "I don't know how to respond to your message"
