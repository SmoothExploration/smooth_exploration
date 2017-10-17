#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    V = np.load('data/1508177601_tabular_sarsa_agent__horsetrack_environment.dat')
    print(V)
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