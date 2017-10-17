#!/usr/bin/env python

"""An abstract class that specifies the Experiment API for RL-Glue-py.

Usage: python horsetrack_experiment.py args

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

import numpy as np

from environments import floating_horsetrack_environment
from rl_glue import RLGlue  # Required for RL-Glue
from agents import random_agent
from agents import sarsa_func


def save_results(data, data_size, filename):
    # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))


def main(agent_info, agent_class, steps, filename):
    env_class = floating_horsetrack_environment.Environment
    rl_glue = RLGlue(env_class, agent_class)

    max_steps = steps
    step = 0
    episode_end = []
    cum_reward = 0

    agent_info.update({"actions": env_class.actions})
    rl_glue.rl_init(agent_info)

    while step < max_steps:
        rl_glue.rl_start()

        is_terminal = False

        while not is_terminal and step < max_steps:
            reward, state, action, is_terminal = rl_glue.rl_step()
            cum_reward += reward

            step += 1

        if is_terminal:
            episode_end.append(step)
        rl_glue.rl_cleanup()

    save_results(episode_end, len(episode_end), "data/{}".format(filename))


if __name__ == "__main__":
    start = time.time()
    agent_names = {1: "random",
                   2: "sarsa"}
    agents = {1: random_agent.Agent,
              2: sarsa_func.Agent}
    agent_class_ = agents[int(sys.argv[1])]

    agent_info_ = {"epsilon": float(sys.argv[2]),
                   "alpha": float(sys.argv[3]),
                   "lambda": float(sys.argv[4]),
                   "beta": float(sys.argv[5]),
                   "action_in_features": sys.argv[6].lower() == 'true',
                   "gamma": 1.0,
                   "num_tilings": [16],
                   "num_tiles": [2],
                   "wrap_widths": [100],
                   }
    agent_info_['scale'] = [np.array(agent_info_['num_tiles']) /
                            np.array(agent_info_["wrap_widths"])]
    steps_ = int(sys.argv[-1])

    filename_ = "{}__{}__{epsilon}__{alpha}__{lambda}__{beta}__{}.dat"
    # timestamp = int((datetime.datetime.now() -
    #                  datetime.datetime.utcfromtimestamp(0)).total_seconds())
    filename_ = filename_.format(agent_names[int(sys.argv[1])],
                                 "floating-horsetrack",
                                 int(agent_info_['action_in_features']),
                                 **agent_info_)

    main(agent_info_, agent_class_, steps_, filename_)

    print("Done in {}s".format(time.time() - start))