#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Andrew Jacobsen, Victor Silva, Mohammad M. Ajallooeian
  Last Modified on: 16/9/2017

    An abstract class that specifies the Experiment API for RL-Glue-py.
"""

from __future__ import print_function
import sys

from rl_glue import RLGlue  # Required for RL-Glue
from agents import random_agent
from environments import horsetrack_environment
import datetime


def save_results(data, data_size, filename):
    # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))

def run_episode(rl_glue_instance=None, max_steps=100, agent_data={}):
    agent_info = {
        "actions" : rl_glue_instance.environment.actions,
    }

    if "q_values" in agent_data:
        agent_info["q_values"] = agent_data["q_values"]

    rl_glue_instance.rl_init(agent_info=agent_info)
    rl_glue_instance.rl_start()

    is_terminal = False
    # while rl_glue_instance.num_steps < max_steps - 1 and not is_terminal:
    while not is_terminal:
        reward, state, action, is_terminal = rl_glue_instance.rl_step()
        # optimal_action[rl_glue_instance.num_steps] += 1 if "action is optimal" else 0

    rl_glue_instance.rl_cleanup()
    # print(".", end='')
    sys.stdout.flush()

    agent_data["q_values"] = rl_glue_instance.agent.q_values

    return agent_data


def main(data_output_location="data"):
    
    env_class = horsetrack_environment.Environment
    agent_class = random_agent.Agent

    agent_name = agent_class.__module__[agent_class.__module__.find(".") + 1:]
    environment_name = env_class.__module__[env_class.__module__.find(".") + 1:]

    rl_glue = RLGlue(env_class, agent_class)

    num_episodes = 2000
    max_steps = 1000
    max_total_steps = 10000000

    print("Running Agent: {} on Environment: {}.".format(agent_name, environment_name))
    # print("\tPrinting one dot for every episode",
    #       end=' ')

    agent_data = {"epsilon": 0.1,
                  "alpha": 0.1,
                  }
    total_steps = 0
    termination_times = []

    while True:
        agent_data = run_episode(rl_glue_instance=rl_glue, 
                                 max_steps=max_steps,
                                 agent_data=agent_data)
        print("AGENT DATA")
        print(agent_data)
        total_steps += rl_glue.num_steps
        termination_times.append(total_steps)

        if total_steps >= max_total_steps:
            break

    epoch_datetime = int((datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds())
    
    save_results(termination_times, len(termination_times), 
                 "{}/{}_{}__{}.dat".format(data_output_location, 
                                          epoch_datetime,
                                          agent_name,
                                          environment_name))
    
    print("\nDone")


if __name__ == "__main__":
    main()
