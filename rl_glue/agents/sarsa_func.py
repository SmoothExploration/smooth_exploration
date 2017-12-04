#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py.
"""

from __future__ import print_function

import numpy as np

from .agent import BaseAgent
from .tilecode import Tilecoder
from .state_aggregator import StateAggregator


class Agent(BaseAgent):
    """Implements the SARSA learning algorithm."""

    def __init__(self):
        self.actions = None
        self.action_index = None
        self.action_feature = None

        self.q_values = None
        self.feature_generator = None
        self.num_features = None
        self.feature_counts = None

        self.last_obs = None
        self.last_action = None
        self.last_features = None

        self.alpha = None
        self.gamma = None
        self.epsilon = None
        self.kappa = None
        self.time = None

        # for saving results
        self.track_actions = None
        self.track_q_values = None
        self.track_states = None
        self.track_reward = None
        self.track_feature_counts = None
        self.track_features = None
        self.filename = None


    def agent_init(self, agent_init_info={}):
        """Setup for the agent called when the experiment first starts."""

        self.actions = np.asarray(agent_init_info.get("actions", np.zeros(0)))
        self.action_index = {i: np.where(i == self.actions)[0][0]
                             for i in self.actions}
        self.action_feature = agent_init_info['action_in_features']

        self.feature_generator = StateAggregator(**agent_init_info)
        self.q_values = (np.ones((self.actions.size,
                                  self.feature_generator.num_features)) *
                         agent_init_info['initialization_values'])

        self.gamma = float(agent_init_info.get('gamma', 1.0))
        alpha0 = float(agent_init_info.get('alpha', 0.1))
        self.alpha = alpha0 / self.feature_generator.num_active_features
        self.epsilon = float(agent_init_info.get('epsilon', 0))
        self.kappa = float(agent_init_info.get('kappa', 0))
        self.time = 1

        if self.kappa:
            num_features = self.feature_generator.num_features
            if self.action_feature:
                num_features -= self.actions.size
            self.feature_counts = np.ones((2, num_features))
            self.feature_counts *= 0.5

        self.filename = agent_init_info.get('filename', None)

    def agent_start(self, observation, agent_start_info={}):
        """The first method called when the experiment starts, called after
        the environment starts.

        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
            agent_start_info (dict): parameters

        Returns:
            The first action the agent takes.
        """
        self.track_actions = []
        self.track_q_values = []
        self.track_states = []
        self.track_reward = []
        self.track_feature_counts = []
        self.track_features = []

        self.last_obs = observation
        self.last_features = self.feature_generator.get_features(observation)
        self.last_action = np.random.choice(self.actions)

        if self.action_feature:
            self.last_features[-self.actions.size:] = self.actions == self.last_action

        if self.kappa:
            ind = -self.actions.size if self.action_feature else None
            self.intrinsic_reward(self.last_features[:ind])

        self.time += 1

        return self.last_action

    def choose_action(self, features):
        if np.random.uniform() > self.epsilon:
            # find value of taking each action
            shuf = np.random.permutation(self.actions.shape[0])
            action_values = np.einsum("ij,j->i",
                                      self.q_values,
                                      features)
            return self.actions[shuf[np.argmax(action_values[shuf])]]
        else:
            return np.random.choice(self.actions)

    def action_value(self, features, action):
        return np.einsum("i,i->",  # vector dot product to find action value
                         self.q_values[self.action_index[action]],
                         features)

    def intrinsic_reward(self, features):
        rho0 = (self.feature_counts[0][~features]/self.time).prod()
        rho1 = (self.feature_counts[1][features]/self.time).prod()
        rho = rho0 * rho1
        rho_prime_i0 = (self.feature_counts[0][~features] + 1) / (self.time + 1)
        rho_prime_i1 = (self.feature_counts[1][features] + 1) / (self.time + 1)
        rho_prime = rho_prime_i0.prod() * rho_prime_i1.prod()

        self.feature_counts[0][~features] += 1
        self.feature_counts[1][features] += 1

        return rho * (1 - rho_prime) / (rho_prime - rho)

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
        features = self.feature_generator.get_features(observation)

        copy_actions = np.copy(self.last_action)
        self.track_actions.append(copy_actions)

        copy_q = np.copy(self.q_values)
        self.track_q_values.append(copy_q)

        copy_states = np.copy(self.last_obs[0])
        self.track_states.append(copy_states)

        copy_rewards = np.copy(reward)
        self.track_reward.append(copy_rewards)

        copy_features = np.copy(self.last_features)
        self.track_feature_counts.append(copy_features)

        action = self.choose_action(features)

        if self.action_feature:
            features[-self.actions.size:] = self.actions == action

        int_reward = 0
        if self.kappa:
            ind = -self.actions.size if self.action_feature else None
            pseudocount = self.intrinsic_reward(features[:ind])
            int_reward = self.kappa / np.sqrt(pseudocount)

        td_error = (reward
                    + int_reward
                    + self.gamma * self.action_value(features, action)
                    - self.action_value(self.last_features,
                                        self.last_action))
        td_error *= self.last_features

        self.q_values[self.last_action] += self.alpha * td_error

        self.last_action = action
        self.last_obs = observation
        self.last_features = features
        self.time += 1

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        copy_actions = np.copy(self.last_action)
        self.track_actions.append(copy_actions)

        copy_q = np.copy(self.q_values)
        self.track_q_values.append(copy_q)

        copy_states = np.copy(self.last_obs[0])
        self.track_states.append(copy_states)

        copy_rewards = np.copy(reward)
        self.track_reward.append(copy_rewards)

        copy_features = np.copy(self.last_features)
        self.track_feature_counts.append(copy_features)


        int_reward = 0
        if self.kappa:
            ind = -self.actions.size if self.action_feature else None
            pseudocount = self.intrinsic_reward(self.last_features[:ind])
            int_reward = self.kappa / np.sqrt(pseudocount)

        td_error = (reward
                    + int_reward
                    - self.action_value(self.last_features, self.last_action))
        td_error *= self.last_features
        self.q_values[self.last_action] += self.alpha * td_error

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

        copy_actions = np.copy(self.last_action)
        self.track_actions.append(copy_actions)

        copy_q = np.copy(self.q_values)
        self.track_q_values.append(copy_q)

        copy_states = np.copy(self.last_obs[0])
        self.track_states.append(copy_states)

        copy_features = np.copy(self.last_features)
        self.track_feature_counts.append(copy_features)

        self.save_results()

        self.last_action = None
        self.last_obs = None
        self.last_features = None

        self.track_actions = None
        self.track_q_values = None
        self.track_states = None
        self.track_reward = None
        self.track_feature_counts = None


    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.

        Args:
            message: The message passed to the agent.

        Returns:
            The response (or answer) to the message.
        """
        pass

    def save_results(self):
        for name, result in [("actions", self.track_actions),
                             ("q_values", self.track_q_values),
                             ("states", self.track_states),
                             ("rewards", self.track_reward),
                             ("feature_counts", self.track_feature_counts)]:
            filename = "data/save_runs/{}/{}".format(name, self.filename)
            np.save(filename, result)
