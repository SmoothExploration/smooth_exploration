#!/usr/bin/env python

from .agent import BaseAgent
import numpy as np

class Agent(BaseAgent):
    """Implement Tabular version of SARSA learning algorithm"""
    
    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.q_values = None
        self.world_size = None
        self.actions = None
        self.epsilon = None
        self.alpha = None
        self.lam = None # lambda
        self.gamma = None
        self.feature_counts = None
        self.time = None

        # for saving results
        self.track_actions = None
        self.track_q_values = None
        self.track_states = None 
        self.track_reward = None
        self.track_feature_counts = None
        self.filename = None

    def agent_init(self, agent_init_info={}):
        """Setup for the agent called when the experiment first starts."""

        self.actions = np.asarray(agent_init_info.get("actions", np.zeros(0)))
        self.min_obs = agent_init_info.get("min_obs", -50)
        self.max_obs = agent_init_info.get("max_obs", 50)
        self.world_size = self.max_obs - self.min_obs #agent_init_info.get("world_size", 100)

        self.q_values = np.ones(self.world_size * len(self.actions))\
                          .reshape(self.world_size, len(self.actions))\
                          * agent_init_info.get('initialization_values', 0)

        self.gamma = float(agent_init_info.get('gamma', 1.0))
        self.lam = float(agent_init_info.get('lambda', 0.0))
        self.alpha = float(agent_init_info.get('alpha', 0.1))
        self.epsilon = float(agent_init_info.get('epsilon', 0))
        self.kappa = float(agent_init_info.get('kappa', 0))
        self.time = 0        

        if self.kappa:
            self.feature_counts = np.zeros(self.world_size + 1)
            # self.feature_counts *= 0.5


        self.filename = agent_init_info.get('filename', None)

    def normalize_observation(self, observation):
        return observation - self.min_obs

    def agent_start(self, observation, agent_start_info={}):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        observation = self.normalize_observation(observation)

        self.track_actions = []
        self.track_q_values = []
        self.track_states = []
        self.track_reward = []
        self.track_feature_counts = []

        if np.random.random() < self.epsilon:
            action_choice = np.random.randint(2)
            action = self.actions[action_choice]
        else:
            action_choice = np.random.choice(np.flatnonzero(self.q_values[int(observation[0])] == np.max(self.q_values[int(observation[0])])))
            action = self.actions[action_choice]
        
        if self.kappa:
            self.feature_counts[int(observation[0])] += 1

        self.last_state = np.copy(observation)
        self.last_action = np.copy(action_choice)
        self.time += 1

        return action

    # def intrinsic_reward(self, features):
    #     rho0 = (self.feature_counts[0][~features]/self.time).prod()
    #     rho1 = (self.feature_counts[1][features]/self.time).prod()
    #     rho = rho0 * rho1
    #     rho_prime_i0 = (self.feature_counts[0][~features] + 1) / (self.time + 1)
    #     rho_prime_i1 = (self.feature_counts[1][features] + 1) / (self.time + 1)
    #     rho_prime = rho_prime_i0.prod() * rho_prime_i1.prod()

    #     return rho * (1 - rho_prime) / (rho_prime - rho)

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
        observation = self.normalize_observation(observation)

        self.track_actions.append(self.last_action)
        q_copy = np.copy(self.q_values)
        self.track_q_values.append(q_copy)
        self.track_states.append(self.last_state[0])
        self.track_reward.append(reward)
        feature_copy = np.copy(self.feature_counts)
        self.track_feature_counts.append(feature_copy)

        features = observation

        if np.random.random() < self.epsilon:
            action_choice = np.random.randint(2)
            action = self.actions[action_choice]
        else:
            action_choice = np.random.choice(np.flatnonzero(self.q_values[int(observation[0])] == np.max(self.q_values[int(observation[0])])))
            action = self.actions[action_choice]

        count_reward = 0

        if self.kappa:
            self.feature_counts[int(observation[0])] += 1
            count_reward = self.kappa / np.sqrt(self.feature_counts[int(observation[0])])

        td_target = reward + count_reward + self.gamma * self.q_values[int(observation[0])][action_choice]
        td_error = td_target - self.q_values[int(self.last_state[0])][self.last_action]
        
        self.q_values[int(self.last_state[0])][self.last_action] += self.alpha * td_error

        # self.q_values += count_reward

        self.last_state = np.copy(observation)
        self.last_action = np.copy(action_choice)
        self.last_features = features
        self.time += 1

        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        # """

        self.track_actions.append(self.last_action)
        q_copy = np.copy(self.q_values)
        self.track_q_values.append(q_copy)
        self.track_states.append(self.last_state[0])
        self.track_reward.append(reward)
        feature_copy = np.copy(self.feature_counts)
        self.track_feature_counts.append(feature_copy)

        count_reward = 0

        if self.kappa:
            self.feature_counts[-1] += 1
            count_reward = self.kappa / np.sqrt(self.feature_counts[-1])

        td_target = reward + count_reward
        td_error = td_target - self.q_values[int(self.last_state[0])][self.last_action]
        
        self.q_values[int(self.last_state[0])][self.last_action] += self.alpha * td_error

        # self.q_values += count_reward

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        self.track_actions.append(self.last_action)
        q_copy = np.copy(self.q_values)
        self.track_q_values.append(q_copy)
        self.track_states.append(self.last_state[0])
        feature_copy = np.copy(self.feature_counts)
        self.track_feature_counts.append(feature_copy)

        # self.save_results()

        self.track_actions = None
        self.track_q_values = None
        self.track_states = None
        self.track_reward = None
        self.track_feature_counts = None

        self.last_action = None
        self.last_obs = None
        self.last_features = None

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
            # with open(filename, "a+") as data_file:
            #     for i in range(len(result) - 1):
            #         data_file.write("{}, ".format(result[i]))

            #     if result:
            #         data_file.write("{}\n".format(result[-1]))





