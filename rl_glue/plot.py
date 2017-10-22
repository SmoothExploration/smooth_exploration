#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
# import unicode

# if __name__ == "__main__":
#     V = np.loadtxt('data/1508191280_tabular_sarsa_agent__horsetrack_environment__epsilon0.1__alpha0.0625.dat')
#     V2 = np.loadtxt('data/1508193353_random_agent__horsetrack_environment__epsilon0.0__alpha2.dat')
#     plt.show()
#     y, x = zip(*[(i, x) for i, x in enumerate(V)])
#     # # print(x)
#     plt.plot(x, y)
#     y, x = zip(*[(i, x) for i, x in enumerate(V2)])
#     # # print(x)
#     plt.plot(x, y)
#     plt.xlim([0, 1_000])
#     plt.xticks([x for x in range(0, 1_100_000, 100_000)])
#     plt.show()
    # plt.show()
    # plt.show()
    # print V.shape
    # # for i, episode_num in enumerate([100, 1000, 8000]):
    # plt.plot(V, label='episode : ' + str(episode_num))
    # plt.xlim([0,100])
    # plt.xticks([1,25,50,75,99])
    # plt.xlabel('Capital')
    # plt.ylabel('Value estimates')
    # plt.legend()
    # plt.show()

# for _, _, files in os.walk("niko_data/data"):
#     for file in files:
#         agent_data = np.loadtxt('niko_data/data/{}'.format(file))[:1_000]
#         perfect_agent_data = np.loadtxt('new_data/1508197215_random_agent__horsetrack_environment__epsilon0.0__alpha1.dat')[:1_000]
#         y, x = zip(*[(i, x) for i, x in enumerate(agent_data)])
#         plt.plot(x, y)
#         y, x = zip(*[(i, x) for i, x in enumerate(perfect_agent_data)])
#         plt.plot(x, y)
#         plt.xlim([0, 1_000])
#         plt.xticks([x for x in range(0, 1100, 100)])
#         name = file[file.find("__")+2 : file.find(".dat")]
#         plt.suptitle('{}'.format(name), fontsize=10, fontweight='light')
#         plt.savefig('niko_data/figures_small/{}.png'.format(file))
#         plt.clf()

for _, _, files in os.walk("niko_data/data"):
    for file in files:
        agent_data = np.loadtxt('niko_data/data/{}'.format(file))[:20_000]
        perfect_agent_data = np.loadtxt('new_data/1508197215_random_agent__horsetrack_environment__epsilon0.0__alpha1.dat')[:20_000]
        y, x = zip(*[(i, x) for i, x in enumerate(agent_data)])
        plt.plot(x, y)
        y, x = zip(*[(i, x) for i, x in enumerate(perfect_agent_data)])
        plt.plot(x, y)
        plt.xlim([0, 1_000])
        plt.xticks([x for x in range(0, 21_000, 1_000)])
        name = file[file.find("__")+2 : file.find(".dat")]
        plt.suptitle('{}'.format(name), fontsize=10, fontweight='light')
        plt.savefig('niko_data/figures_med/{}.png'.format(file))
        plt.clf()