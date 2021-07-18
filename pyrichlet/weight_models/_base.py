"""Base class for weighting structure models."""
from abc import ABCMeta, abstractmethod
import numpy as np


class BaseWeight(metaclass=ABCMeta):
    """Base class for weighting structure models.

    This abstract class specifies an interface for all weighting structure
    classes and provides basic common methods for weighting models.
    """
    def __init__(self, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
        elif type(rng) is int:
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng

        self.w = np.array([], dtype=np.float64)
        self.d = np.array([], dtype=np.int64)
        self.variational_params = None
        self.variational_d = None
        self.variational_k = None

    def fit(self, d):
        """Fit the weighting structure to a vector of assignments

        This method fits the parameters of the weighting model given the
        internal truncated weighting structure `self.w`. Calls to any of the
        methods: `random`, `tail`, `complete`; after calling this method
        results in random draws from the posterior distribution.
        """
        self.d = np.array(d)

    @abstractmethod
    def random(self, size=None):
        """Do a random draw of the truncated weighting structure up to `n` obs.

        This method does a random draw from the posterior weighting
        distribution (or from the prior distribution if nothing has been
        fitted) and updates the internal truncated weighting structure
        `self.w`.
        """
        pass

    @abstractmethod
    def tail(self, x):
        """Return an array of weights such that the sum is greater than `x`

        This method appends weights to the truncated weighting structure
        `self.w` until the sum of its elements is greater than the input `x`
        and then returns `self.w`.
        """
        pass

    @abstractmethod
    def complete(self, size):
        """Return an array of weights with at least `n` elements

        This method appends weights to the truncated weighting structure
        `self.w` until reaching a length of `size` and then returns `self.w`.
        Note: This method is a constrain on the minimum number of elements in
        the truncated weighting structure. No truncation is induced in case
        `size` is less than `len(self.w)` and the full length of `self.w` is
        returned.
        """
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
        inverse_sampling = np.greater.outer(
            u, self.get_normalized_cumulative_weights()
        )
        return np.sum(inverse_sampling, axis=1)

    def fit_variational(self, variational_d):
        """Fits the variational distribution q

        This method fits the variational distribution q that minimizes the
        Kullback-Leiber divergence from q(w) to p(w|d) ($D_{KL}(q||p)$) where
        d has a discrete finite random distribution given by
        q(d_i = j) = variational_d[j, i] and q is truncated up to
        `k=len(variational_d)` so that q(w_k=1) = 1.
        """
        raise NotImplementedError

    def variational_mean_log_w_j(self, j):
        """Returns the mean of the logarithm of w_j

        This method returns the expected value of the logarithm of w_j under
        the variational distribution q.
        """
        raise NotImplementedError

    def variational_mean_log_p_d__w(self, variational_d=None):
        """Returns the mean of log p(d|w)

        This method returns the expected value of the logarithm of the
         probability of assignation d given w under the variational
         distribution q.
        """
        raise NotImplementedError

    def variational_mean_log_p_w(self):
        """Returns the mean of log p(d|w)

        This method returns the expected value of the logarithm of the
         probability of assignation d given w under the variational
         distribution q.
        """
        raise NotImplementedError

    def variational_mean_log_q_w(self):
        """Returns the mean of log p(d|w)

        This method returns the expected value of the logarithm of the
         probability of assignation d given w under the variational
         distribution q.
        """
        raise NotImplementedError
