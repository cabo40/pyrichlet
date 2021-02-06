"""Base class for weighting structure models."""
from abc import ABCMeta, abstractmethod
import numpy as np


class BaseWeights(metaclass=ABCMeta):
    """Base class for weighting structure models.

    This abstract class specifies an interface for all weighting structure
    classes and provides basic common methods for weighting models.
    """
    def __init__(self, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.w = np.array([], dtype=np.float64)
        self.d = np.array([], dtype=np.int64)

    def fit(self, d):
        """Fit the weighting structure to a vector of assignments

        This method fits the parameters of the weighting model given the
        internal truncated weighting structure `self.w`.
        """
        self.d = np.array(d)

    @abstractmethod
    def random(self, size=None):
        """Do a random draw of the truncated weighting structure up to `n` obs.

        This method updates internal truncated weighting structure `self.w`
        as a step of the underlying Gibbs sampler.
        """
        pass

    @abstractmethod
    def tail(self, x):
        """Return an array of weights such that the sum is greater than `x`"""
        pass

    @abstractmethod
    def complete(self, size):
        """Return an array of weights with at least `n` elements"""
        pass

    @abstractmethod
    def structure_log_likelihood(self):
        """Return the log-likelihood of the complete weighting structure"""
        pass

    @abstractmethod
    def weight_log_likelihood(self, w=None):
        """Return the log-likelihood of the weights conditional to params"""
        pass

    def assign_log_likelihood(self, d=None):
        """Returns the log-likelihood of an assignment `d` given the weights"""
        if d is None:
            d = self.d
        self.complete(max(d))
        return np.sum(np.log(self.w[d]))

    def reset(self):
        """Resets the conditional vector `d` to None"""
        self.d = None

    def get_weights(self):
        """Returns the last weighting structure drawn"""
        return self.w

    def get_normalized_weights(self):
        """Returns the last weighting stricture normalized"""
        return self.w/np.sum(self.w)

    def get_normalized_cumulative_weights(self):
        """Returns the normalized cumulative weights"""
        return np.cumsum(self.get_normalized_weights())

    def get_size(self):
        """Returns the size of the truncated weighting structure"""
        return len(self.w)

    def random_assignment(self, size=None):
        """Returns a sample draw of the categorical assignment from the current
        state normalized weighting structure"""
        u = self.rng.uniform(size=size)
        self.tail(1-np.min(u))
        inverse_sampling = np.greater.outer(
            u, self.get_normalized_cumulative_weights()
        )
        return np.sum(inverse_sampling, axis=1)
