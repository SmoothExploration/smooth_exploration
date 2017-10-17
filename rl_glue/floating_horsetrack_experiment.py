#!/usr/bin/env python

"""An abstract class that specifies the Experiment API for RL-Glue-py.

Usage: python horsetrack_experiment.py args

Args:
    alpha (float)
    lambda (float)
    action in state rep (True/False)
    agent_index (int)
        1. random_agent
        2. SARSA
    num steps (int)
"""

from __future__ import print_function
import sys

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
    agent_info_ = {"epsilon": float(sys.argv[1]),
                   "alpha": float(sys.argv[2]),
                   "lambda": float(sys.argv[3]),
                   "action_in_features": sys.argv[4].lower() == 'true',
                   "gamma": 1.0,
                   "num_tilings": [16],
                   "num_tiles": [2],
                   "wrap_widths": [100],
                   "scale": [.02]}
    filename_ = "help.dat"

    agents = {1: random_agent.Agent,
              2: sarsa_func.Agent}
    agent_class_ = agents[int(sys.argv[5])]

    steps_ = int(sys.argv[6])

    main(agent_info_, agent_class_, steps_, filename_)
