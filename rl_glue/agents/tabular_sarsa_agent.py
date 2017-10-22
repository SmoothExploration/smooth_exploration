#!/usr/bin/env python

from .agent import BaseAgent
import numpy as np

class Agent(BaseAgent):
    """agent does *no* learning, selects action 0 always"""
    def __init__(self):
        self.last_action = None
        self.q_values = None
        self.world_size = None
        self.actions = None
        self.epsilon = 0.1
        self.alpha = 0.1
        self.lam = 0.1 # lambda
        self.gamma = 1.0

    def agent_init(self, agent_init_info={}):
        """Setup for the agent called when the experiment first starts."""

        if "actions" in agent_init_info:
            self.actions = agent_init_info["actions"]

        if "world_size" in agent_init_info:
            self.world_size = agent_init_info["world_size"]

        self.q_values = np.zeros(self.world_size * len(self.actions)).reshape(self.world_size, len(self.actions))

        # if "q_values" in agent_init_info:
        #     self.q_values = agent_init_info["q_values"]
        # else:
        #     self.q_values = np.zeros(self.world_size * len(self.actions)).reshape(self.world_size, len(self.actions))

        if "epsilon" in agent_init_info:
            self.epsilon = agent_init_info["epsilon"]

        if "alpha" in agent_init_info:
            self.alpha = agent_init_info["alpha"]

        # if "lambda" in agent_init_info:
        #     self.lam = agent_init_info["lambda"]

        self.last_action = 0

    def agent_start(self, observation, agent_start_info={}):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        # print(observation)
        # if "q_values" in agent_start_info:
        #     self.q_values = agent_start_info["q_values"]
        # if "epsilon" in agent_start_info:
        #     self.epsilon = agent_start_info["epsilon"]

        # if "alpha" in agent_start_info:
        #     self.alpha = agent_start_info["alpha"]

        # if "lambda" in agent_start_info:
        #     self.lam = agent_start_info["lambda"]

        if np.random.random() < self.epsilon:
            action = np.random.randint(0, len(self.actions))
        else:
            action = np.argmax(self.q_values[observation])

        self.last_state = np.copy(observation)
        self.last_action = np.copy(action)

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
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, len(self.actions))
        else:
            action = np.argmax(self.q_values[observation])

        # might do some learning here
        td_target = reward + self.gamma * self.q_values[observation[0]][action]
        td_error = td_target - self.q_values[self.last_state[0]][self.last_action]
        self.q_values[self.last_state[0]][self.last_action] += self.alpha * td_error

        self.last_state = np.copy(observation)
        self.last_action = np.copy(action)

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        td_target = reward
        td_error = td_target - self.q_values[self.last_state[0]][self.last_action]
        self.q_values[self.last_state[0]][self.last_action] += self.alpha * td_error
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