from __future__ import print_function
import datetime
import time
import sys

import numpy as np

from environments import horsetrack_environment, floating_horsetrack_environment
from rl_glue import RLGlue  # Required for RL-Glue
from agents import random_agent, sarsa_tabular, sarsa_func

def save_results(data, data_size, filename):
    # data: floating point, data_size: integer, filename: string
    with open(filename, "a+") as data_file:
        for i in range(data_size - 1):
            data_file.write("{}, ".format(data[i]))

        if data:
            data_file.write("{}\n".format(data[-1]))


def main(agent_info, agent_class, steps, filename):
    env_class = floating_horsetrack_environment.Environment
    rl_glue = RLGlue(env_class, agent_class)

    max_steps = steps
    step = 0
    episode_end = []
    cum_reward = 0
    state_array = []
    action_array = []

    agent_info.update({"actions": env_class.actions})
    rl_glue.rl_init(agent_info)

    state = rl_glue.rl_start()
    state_array.append(state[0][0])

    is_terminal = False
    while not is_terminal:
        reward, state, action, is_terminal = rl_glue.rl_step()
        # cum_reward += reward
        state_array.append(state[0])
        # action_array.append(action)

    rl_glue.rl_cleanup()

    save_results(state_array, len(state_array), "data/first_episode_states_{}".format(filename))
    # save_results(action_array, len(action_array), "data/first_episode_actions_{}".format(filename))

if __name__ == '__main__':

    agent_names = {1: "random",
                  2: "sarsa_tabular",
                  3: "sarsa_func"}
    
    agents = {1: random_agent.Agent,
              2: sarsa_tabular.Agent,
              3: sarsa_func.Agent}
    
    agent_class_ = agents[int(sys.argv[1])]

    agent_info_ = {"epsilon": float(sys.argv[2]),
                   "alpha": float(sys.argv[3]),
                   "lambda": float(sys.argv[4]),
                   "kappa": float(sys.argv[5]),
                   "action_in_features": sys.argv[6].lower() == 'true',
                   "initialization_values": float(sys.argv[7]),
                   "gamma": 0.99,
                   "num_tilings": [16],
                   "num_tiles": [2],
                   "wrap_widths": [100],
                   }
    agent_info_['scale'] = [np.array(agent_info_['num_tiles']) /
                            np.array(agent_info_["wrap_widths"])]


    steps_ = int(sys.argv[-1])
    filename_ = "{}__{}__{epsilon}__{alpha}__{lambda}__{kappa}__{action_in_features}__{initialization_values}.dat"
    filename_ = filename_.format(agent_names[int(sys.argv[1])],
                                 "horsetrack",
                                 int(agent_info_['action_in_features']),
                                 **agent_info_)

    main(agent_info_, agent_class_, steps_, filename_)
