#!/usr/bin/env python

from __future__ import print_function
import sys

from rl_glue import RLGlue  # Required for RL-Glue
from agents import random_agent, tabular_sarsa_agent
from environments import horsetrack_environment
import datetime


def save_results(data, data_size, filename):
    # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))


def main(data_output_location="new_data"):
    
    env_class = horsetrack_environment.Environment
    agent_class = random_agent.Agent

    agent_name = agent_class.__module__[agent_class.__module__.find(".") + 1:]
    environment_name = env_class.__module__[env_class.__module__.find(".") + 1:]

    rl_glue = RLGlue(env_class, agent_class)

    # num_episodes = 2000
    # max_steps = 1000
    max_total_steps = 100_000

    for epsilon in [0.0, 0.1]:
        for alpha in [2, 1, 0.5, 0.25, 0.125, 0.0625]:
            print("Running Agent: {} on Environment: {}.".format(agent_name, environment_name))
            agent_init_info = {"actions" : [-1, 1],
                               "world_size": 100,
                               "epsilon": epsilon,
                               "alpha": alpha}
            termination_times = []

            rl_glue.rl_init(agent_init_info=agent_init_info)

            step_counter = 0

            while step_counter < max_total_steps:
                rl_glue.rl_start()
                is_terminal = False

                while step_counter < max_total_steps and not is_terminal:
                    reward, state, action, is_terminal = rl_glue.rl_step()
                    step_counter += 1

                rl_glue.rl_cleanup()
                # print(".", end='')
                sys.stdout.flush()

                if is_terminal:
                    termination_times.append(step_counter)

            epoch_datetime = int((datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds())
            
            save_results(termination_times, len(termination_times), 
                         "{}/{}_{}__{}__epsilon{}__alpha{}.dat".format(data_output_location, 
                                                  epoch_datetime,
                                                  agent_name,
                                                  environment_name,
                                                  epsilon,
                                                  alpha))
    
    print("\nDone")


if __name__ == "__main__":
    main()
