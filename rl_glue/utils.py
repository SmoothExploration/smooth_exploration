#!/usr/bin/env python

"""
"""

import numpy.random as rnd


def rand_in_range(upper):
    """Returns a random integer in [0, upper).
    """
    return rnd.randint(upper)


def rand_un():
    """Returns a random float in [0, 1).
    """
    return rnd.uniform()


def rand_norm(mu, sigma):
    """Returns a random float drawn from a normal distribution.

    Args:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
    """
    return rnd.normal(mu, sigma)


randInRange = rand_in_range
randn = rand_norm
