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
from importlib import import_module
from rl_glue import RLGlue  # Required for RL-Glue

# Import agent and environment if you are not setting 
# them from the command line
from agents import example_agent
from environments import example_environment


def save_results(data, data_size, filename):
    # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))

def run_episode(rl_glue_instance=None, max_steps=100, optimal_action=[], agent_data={}):
    agent_info = {
        "actions" : rl_glue_instance.environment.actions,
    }

    if "q_values" in agent_data:
        agent_info["q_values"] = agent_data["q_values"]

    rl_glue_instance.rl_init(agent_info=agent_info)
    rl_glue_instance.rl_start()

    is_terminal = False
    while rl_glue_instance.num_steps < max_steps - 1 and not is_terminal:
        reward, state, action, is_terminal = rl_glue_instance.rl_step()
        optimal_action[rl_glue_instance.num_steps] += 1 if "action is optimal" else 0

    rl_glue_instance.rl_cleanup()
    print(".", end='')
    sys.stdout.flush()

    agent_data["q_values"] = rl_glue_instance.agent.q_values

    return agent_data


def main(data_output_location="data"):
    
    env_class = example_environment.Environment
    agent_class = example_agent.Agent

    rl_glue = RLGlue(env_class, agent_class)

    num_episodes = 2000
    max_steps = 1000

    print("\tPrinting one dot for every run: {}".format(num_episodes),
          end=' ')
    print("total runs to complete.")

    optimal_action = [0 for _ in range(max_steps)]

    agent_data = {}
    for i in range(num_episodes):
        agent_data = run_episode(rl_glue_instance=rl_glue, 
                                 max_steps=max_steps,
                                 optimal_action=optimal_action, 
                                 agent_data=agent_data)

    prop_optimal = [num_optimal / num_episodes for num_optimal in
                    optimal_action]
    save_results(prop_optimal, max_steps, 
                 "{}/RL_EXP_OUT.dat".format(data_output_location))
    print("\nDone")


if __name__ == "__main__":
    main()
