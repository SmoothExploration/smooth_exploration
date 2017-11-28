import numpy as np


class StateAggregator:
    """ 1-D state aggregation
    Args:
        num_bins (int): Number of bins to aggregate state.
        min_obs (float): Minimum state value.
        max_obs (float): Maximum state value.
        action_in_features (bool): Whether the action is part of the features.
        actions (obj with len): Used to determine number of actions.
    """
    def __init__(self,
                 num_bins,
                 min_obs,
                 max_obs,
                 action_in_features,
                 actions,
                 **kwargs):

        self.min = min_obs
        self.max = max_obs
        self.num_bins = num_bins
        self.num_features = num_bins
        self.num_active_features = 1
        self.num_actions = 0

        if action_in_features:
            self.num_actions = len(actions)
            self.num_features += self.num_actions
            self.num_active_features += 1

    def get_features(self, observation):
        agg = np.asarray(np.histogram(observation,
                                      self.num_bins,
                                      (self.min, self.max))[0],
                         dtype=np.bool)

        if self.num_actions:
            return np.concatenate((agg, np.zeros(self.num_actions,
                                                 dtype=np.bool)))
        else:
            return agg