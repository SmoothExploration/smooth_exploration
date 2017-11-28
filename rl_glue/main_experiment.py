#!/usr/bin/env python
"""
Args:
    epsilon (float)
    alpha (float)
    lambda (float)
    action in state rep (True/False)
    feature pseudocounts (True/False)
    agent_index (int)
        1. random_agent
        2. SARSA
    num steps (int)
"""

from __future__ import print_function
import datetime
import time
import sys
import fcntl
from io import BytesIO

import numpy as np

from environments import horsetrack_environment, horsetrack_environment_m1, floating_horsetrack_environment, floating_horsetrack_environment_m1
from rl_glue import RLGlue  # Required for RL-Glue
from agents import random_agent
from agents import sarsa_tabular, sarsa_func


def save_results(data, param_info):
    # data: floating point, data_size: integer, filename: string
    # with open(filename, "a+") as data_file:
    #     for i in range(data_size - 1):
    #         data_file.write("{}, ".format(data[i]))

    #     if data:
    #         data_file.write("{}\n".format(data[-1]))
    with open("data/results", "a+") as data_file:
        fcntl.flock(data_file, fcntl.LOCK_EX)
        data_file.write('{}, '.format(param_info))

        s = BytesIO()
        np.savetxt(s, data, newline=",")
        data_file.write("{}\n".format(s.getvalue().decode()[:-1]))
        fcntl.flock(data_file, fcntl.LOCK_UN)


def main(agent_info, agent_class, env_info, env_class, steps, param_info):
    # env_class = horsetrack_environment.Environment
    rl_glue = RLGlue(env_class, agent_class)

    max_steps = steps
    max_episodes = 5
    step = 0
    episodes = 0
    episode_end = np.ones(max_episodes) * max_steps
    cum_reward = 0

    # max_steps = 20000

    agent_info.update({"actions": env_class.actions})
    rl_glue.rl_init(agent_info, env_info)

    while step < max_steps and episodes < max_episodes:
        rl_glue.rl_start()

        is_terminal = False

        while not is_terminal and step < max_steps:
            reward, state, action, is_terminal = rl_glue.rl_step()
            cum_reward += reward

            step += 1

        if is_terminal:
            episode_end[episodes] = step
            episodes += 1
        rl_glue.rl_cleanup()

    save_results(episode_end, "{}".format(param_info))


if __name__ == "__main__":
    start = time.time()
    agent_names = {1: "random",
                   2: "sarsa_tabular",
                   3: "sarsa_func"}
    agents = {1: random_agent.Agent,
              2: sarsa_tabular.Agent,
              3: sarsa_func.Agent}
    agent_class_ = agents[int(sys.argv[1])]

    environment_names = {1: "horsetrack",
                         2: "horsetrack_minus",
                         3: "floating_horsetrack",
                         4: "floating_horestrack_minus",}
    environments = {1: horsetrack_environment.Environment,
                    2: horsetrack_environment_m1.Environment,
                    3: floating_horsetrack_environment.Environment,
                    4: floating_horsetrack_environment_m1.Environment}
    env_class_ = environments[int(sys.argv[2])]

    agent_info_ = {"gamma": float(sys.argv[3]),
                   "epsilon": float(sys.argv[4]),
                   "alpha": float(sys.argv[5]),
                   "lambda": float(sys.argv[6]),
                   "kappa": float(sys.argv[7]),
                   "action_in_features": sys.argv[8].lower() == 'true',
                   "initialization_values": float(sys.argv[9]),
                   # "gamma": 0.99,
                   # "num_tilings": [16],
                   # "num_tiles": [2],
                   # "wrap_widths": [100],
                   "num_bins": int(sys.argv[-1]),
                   "min_obs": -50,
                   "max_obs": 50,
                   }

    env_info_ = {"min_obs": -50,
                 "max_obs": 50,
                }

    # agent_info_['scale'] = [np.array(agent_info_['num_tiles']) /
    #                         np.array(agent_info_["wrap_widths"])]
    steps_ = int(sys.argv[-2])

    param_info = "{},{},{gamma},{epsilon},{alpha},{lambda},{kappa},{action_in_features},{initialization_values},{num_bins}"
    # timestamp = int((datetime.datetime.now() -
    #                  datetime.datetime.utcfromtimestamp(0)).total_seconds())
    param_info = param_info.format(agent_names[int(sys.argv[1])],
                                 environment_names[int(sys.argv[2])],
                                 int(agent_info_['action_in_features']),
                                 **agent_info_)
    agent_info_["filename"] = param_info

    main(agent_info_, agent_class_, env_info_, env_class_, steps_, param_info)

    print("Done in {}s".format(time.time() - start))
