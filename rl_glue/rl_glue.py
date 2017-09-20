#!/usr/bin/env python

"""
 Copyright (C) 2017, Adam White, Mohammad M. Ajallooeian


"""

from environment import Environment
from agent import Agent

class RLGlue:

    def __init__(self, env_name, agent_name):
        self.environment = Environment()
        self.agent = Agent()

        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

    def rl_init(self):
        self.environment.env_init()
        self.agent.agent_init()

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self): # returns (NumPy array, NumPy array)
        total_reward = 0.0;
        num_steps = 1;

        last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def rl_agent_start(self, observation): # returns NumPy array, observation: NumPy array
        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation): # returns NumPy array, reward: floating point, observation: NumPy array
        return self.agent.agent_step(reward, observation)

    def rl_agent_end(self, reward): # reward: floating point
        self.agent.agent_end(reward)

    def rl_env_start(self):
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def rl_env_step(self, action): # returns (floating point, NumPy array, Boolean), action: NumPy array
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal == True:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def rl_step(self): # returns (floating point, NumPy array, NumPy array, Boolean)
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
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message): # returns string, message: string
        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_agent_response = self.agent.agent_message(message_to_send)
        if the_agent_response is None:
            return ""

        return the_agent_response

    def rl_env_message(self, message):  # returns string, message: string
        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_env_response = self.environment.env_message(message_to_send)
        if the_env_response is None:
            return ""

        return the_env_response

    def rl_episode(self, max_steps_this_episode): # returns Boolean, # max_steps_this_episode: integer
        is_terminal = False

        self.rl_start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[3]

        return is_terminal


    def rl_return(self): # returns floating point
        return self.total_reward

    def rl_num_steps(self): # returns integer
        return self.num_steps

    def rl_num_episodes(self): # returns integer
        return self.num_episodes
