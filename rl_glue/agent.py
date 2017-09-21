#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions
 
"""

from __future__ import print_function
from utils import rand_in_range
import numpy as np

class Agent:

    def __init__(self):

        self.last_action = None # last_action: NumPy array
        self.num_actions = 1

    def agent_init(self):

        self.last_action = np.zeros(1) # generates a NumPy array with size 1 equal to zero

    def agent_start(self, this_observation): # returns NumPy array, this_observation: NumPy array

        self.last_action[0] = rand_in_range(self.num_actions)

        local_action = np.zeros(1)
        local_action[0] = rand_in_range(self.num_actions)

        return local_action[0]


    def agent_step(self, reward, this_observation): # returns NumPy array, reward: floating point, this_observation: NumPy array

        local_action = np.zeros(1)
        local_action[0] = rand_in_range(self.num_actions)

        # might do some learning here
        self.last_action = local_action

        return self.last_action

    def agent_end(self, reward): # reward: floating point
        # final learning update at end of episode
        return

    def agent_cleanup(self):
        # clean up
        return

    def agent_message(self, inMessage): # returns string, inMessage: string
        # might be useful to get information from the agent

        if inMessage == "what is your name?":
            return "my name is skeleton_agent!"
      
        # else
        return "I don't know how to respond to your message"
