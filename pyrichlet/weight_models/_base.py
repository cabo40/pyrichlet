"""Base class for weighting structure models."""
from abc import ABC, abstractmethod
import numpy as np


class BaseWeight(ABC):
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

    @abstractmethod
    def random(self, size=None):
        """Do a random draw of the truncated weighting structure up to `n` obs.

        This method does a random draw from the posterior weighting
        distribution (or from the prior distribution if nothing has been
        fitted) and updates the internal truncated weighting structure
        `self.w`.

        Parameters
        ----------
        size : int
            The desired size of the returned vector of weights

        Returns
        -------
        np.array
            Array of the weighted structure.
        """
        pass

    @abstractmethod
    def complete(self, size):
        """Return an array of weights with at least `n` elements

        This method appends weights to the truncated weighting structure
        `self.w` until reaching a length of `size` and then returns `self.w`.
        Note: This method sets a constraint on the minimum number of elements
        in the truncated weighting structure. No truncation is induced in case
        `size` is less than `len(self.w)` and the full length of `self.w` is
        returned.

        Parameters
        ----------
        size : int
            The desired size of the returned vector of weights

        Returns
        -------
        np.array
            Array of the weighted structure.
        """
        if size is not None and type(size) not in (int, np.int64):
            raise TypeError("size parameter must be integer or None")

    @abstractmethod
    def weighting_log_likelihood(self):
        """Return the given structure log-likelihood

        This method returns log f(w) for the underlying weighting model.

        Returns
        -------
        float
            The log-likelihood value
        """
        pass

    def fit(self, d):
        """Fit the weighting structure to a vector of assignments

        This method fits the parameters of the weighting model given the
        internal truncated weighting structure `self.w`. Any call to the
        methods `random`, `tail` or `complete` after calling this method
        results in a random draw from the posterior distribution.

        Parameters
        ----------
        d : array[int], np.array
            An array of integers representing the assigned group
        """
        self.d = np.array(d)

    def tail(self, x):
        """Return an array of weights such that the sum is greater than `x`

        This method appends weights to the truncated weighting structure
        `self.w` until the sum of its elements is greater than the input `x`
        and then returns `self.w`.

        Parameters
        ----------
        x : float
            A float in the range $[0,1)$ for which the sum of weights must be
            greater.

        Returns
        -------
        np.array
            Array of the completed weighted structure
        """
        if x >= 1 or x < 0:
            raise ValueError("Tail parameter not in range [0,1)")
        if len(self.w) == 0:
            self.random(1)
        while self.w.sum() < x:
            self.complete(len(self.w) + 1)
        return self.w

    def assignation_log_likelihood(self, d=None):
        """Returns the log-likelihood of an assignment `d` given the weights"""
        if d is None:
            d = self.d
        self.complete(max(d) + 1)
        return np.sum(np.log(self.w[d]))

    def reset(self):
        """Resets the conditional vector `d` to None"""
        self.d = np.array([], dtype=np.int64)

    def get_weights(self):
        """Returns the last weighting structure drawn"""
        return self.w

    def get_normalized_weights(self):
        """Returns the last weighting stricture normalized"""
        return self.w / np.sum(self.w)

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

    def variational_mean_w(self, j):
        """Returns the mean of w_j

        This method returns the expected value of the j-th weighting factor
        under the variational distribution q.
        """
        raise NotImplementedError

    def variational_mode_w(self, j):
        """Returns the mean of w_j

        This method returns the expected value of the j-th weighting factor
        under the variational distribution q.
        """
        raise NotImplementedError
