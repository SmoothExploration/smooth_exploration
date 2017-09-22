#!/usr/bin/env python

"""
 Copyright (C) 2017, Adam White, Mohammad M. Ajallooeian


"""

from __future__ import print_function
from importlib import import_module

class RLGlue:
    """RLGlue class

    args:
        env_name (string): the name of the module where the Environment class can be found
        agent_name (string): the name of the module where the Agent class can be found
    """

    def __init__(self, env_name, agent_name):
        self.environment = import_module(env_name).Environment()
        self.agent = import_module(agent_name).Agent()

        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

    def rl_init(self):
        """Initial method called when RLGlue experiment is created"""
        self.environment.env_init()
        self.agent.agent_init()

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self):
        """Starts RLGlue experiment

        Returns:
            tuple: (Numpy array, Numpy array)
        """
        total_reward = 0.0;
        num_steps = 1;

        last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def rl_agent_start(self, observation):
        """Starts the agent

        Args:
            observation (Numpy array): the first observation from the environment

        Returns:
            Numpy array: the action taken by the agent
        """
        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation):
        """Step taken by the agent
        
        Args:
            reward (float): the last reward the agent recieved for taking the last action
            observation (Numpy array): the state observation the agent recieves from the environment

        Returns:
            Numpy array: the action taken by the agent
        """
        return self.agent.agent_step(reward, observation)

    def rl_agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward (float): the reward the agent received when terminating
        """
        self.agent.agent_end(reward)

    def rl_env_start(self):
        """Starts RLGlue environement
        
        Returns:
            (float, Numpy array, Boolean): reward, state observation, boolean indicating termination
        """
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def rl_env_step(self, action):
        """Step taken by the environment based on action from agent

        Args:
            action (Numpy array): action taken by agent
        
        Returns:
            (float, Numpy array, Boolean): reward, state observation, boolean indicating termination
        """
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal == True:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def rl_step(self):
        """Step taken by RLGlue, takes environement step and either step or end by agent

        Returns:
            (float, Numpy array, Numpy array, Boolean): reward, last state observation, last action, boolean indicating termination
        """
        (this_reward, last_state, terminal) = self.environment.env_step(self.last_action)

        self.total_reward += this_reward;

        if terminal == True:
            self.num_episodes += 1
            self.agent.agent_end(this_reward)
            roa = (this_reward, last_state, None, terminal)
        else:
            self.num_steps += 1
            self.last_action = self.agent.agent_step(this_reward, last_state)
            roa = (this_reward, last_state, self.last_action, terminal)

        return roa

    def rl_cleanup(self):
        """Cleanup done at end of experiment"""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message):
        """Message passed to communicate with agent during experiment
        
        Args:
            message (string): the message (or question) to send to the agent

        Returns:
            string: the message back (or answer) from the agent

        """
        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_agent_response = self.agent.agent_message(message_to_send)
        if the_agent_response is None:
            return ""

        return the_agent_response

    def rl_env_message(self, message):
        """Message passed to communicate with environment during experiment
        
        Args:
            message (string): the message (or question) to send to the environment

        Returns:
            string: the message back (or answer) from the environment

        """

        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_env_response = self.environment.env_message(message_to_send)
        if the_env_response is None:
            return ""

        return the_env_response

    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode
        
        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode
        
        Returns:
            Boolean: if the episode should terminate
        """
        is_terminal = False

        self.rl_start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[3]

        return is_terminal


    def rl_return(self):
        """The total reward

        Returns:
            float: the total reward
        """
        return self.total_reward

    def rl_num_steps(self):
        """The total number of steps taken

        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes
        
        Returns
            Int: the total number of episodes

        """
        return self.num_episodes
