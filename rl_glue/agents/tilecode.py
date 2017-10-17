import collections
from functools import partial
import numpy as np

from .tiles3 import IHT, tileswrap, tiles


class Tilecoder:
    """ Tilecodes an observation.

    Args:
        num_tilings (list of int): Number of tilings for each element of the
            observation.
        num_tiles (list of int): Number of tiles for each element of the
            observation.
        scale (list of float): Scaling factor for each element of the
            observation. Should be equal to num_tiles / (high - low), where
            the range of observations is [low, high].
        wrap_widths (list of int): Wrap widths for each element of the
            observation. None if no wrapping.

    """
    def __init__(self,
                 num_tilings,
                 num_tiles,
                 scale,
                 wrap_widths,
                 action_in_features,
                 actions,
                 **kwargs):
        num_tilings = np.asarray(num_tilings)
        num_tiles = np.asarray(num_tiles)
        iht_sizes = (num_tiles + 1) * num_tilings
        self.ihts = [IHT(sizeval=iht_size) for iht_size in iht_sizes]
        self.scale = scale

        def tilecode(i):
            return partial(tileswrap if wrap_widths[i] else tiles,
                           numtilings=num_tilings[i],
                           wrapwidths=[wrap_widths[i]])
        self.tilecoders = [tilecode(i) for i in range(len(wrap_widths))]

        self.num_features = iht_sizes.sum()
        if action_in_features:
            self.num_features += len(actions)

        self.num_active_features = num_tilings.sum()

    def get_features(self, observation):
        def tilecode(i, floats):
            if isinstance(floats, collections.Iterable):
                floats = [flt * self.scale[i] for flt in floats]
            else:
                floats = [floats * self.scale[i]]
            return self.tilecoders[i](ihtORsize=self.ihts[i], floats=floats)

        features = [tilecode(i, floats) for i, floats in enumerate(observation)]

        phi = np.zeros(self.num_features, dtype=np.bool)
        phi[np.concatenate([np.asarray(f) for f in features])] += True

        return phi
