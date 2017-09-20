#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
 
  env *ignores* actions: rewards are all random
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

class Environment:

    def __init__(self):
        self.this_reward_observation = (None, None, None) # this_reward_observation: (floating point, NumPy array, Boolean)

    def env_init(self):
        local_observation = np.zeros(0) # An empty NumPy array

        self.this_reward_observation = (0.0, local_observation, False)


    def env_start(self): # returns NumPy array
        return self.this_reward_observation[1]

    def env_step(self, this_action): # returns (floating point, NumPy array, Boolean), this_action: NumPy array
        the_reward = rand_norm(0.0, 1.0) # rewards drawn from (0, 1) Gaussian

        self.this_reward_observation = (the_reward, self.this_reward_observation[1], False)

        return self.this_reward_observation

    def env_cleanup(self):
        #
        return

    def env_message(self, inMessage): # returns string, inMessage: string
        if inMessage == "what is your name?":
            return "my name is skeleton_environment!"
      
        # else
        return "I don't know how to respond to your message"
