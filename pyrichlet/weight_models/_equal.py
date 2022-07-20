from ._base import BaseWeight
from ..exceptions import NotFittedError

import numpy as np


class EqualWeighting(BaseWeight):
    def __init__(self, n=1, rng=None):
        super().__init__(rng=rng)
        self.n = n

    def weighting_log_likelihood(self):
        return 0

    def random(self, size=None):
        self.w = np.repeat(1 / self.n, self.n)
        return self.w

    def complete(self, size):
        return self.random(size)

    def fit_variational(self, variational_d: np.ndarray):
        self.variational_d = variational_d
        self.n = self.variational_d.shape[0]
        self.variational_k = self.n

    def variational_mean_log_w_j(self, j):
        return np.log(1 / self.n) * (j < self.n)

    def variational_mean_log_p_d__w(self, variational_d=None):
        _variational_d = variational_d
        if _variational_d is None:
            _variational_d = self.variational_d
            if _variational_d is None:
                raise NotFittedError
        return _variational_d.shape[1] * np.log(self.n)

    def variational_mean_log_p_w(self):
        return 0

    def variational_mean_log_q_w(self):
        return 0

    def variational_mean_w(self, j):
        return 1 / self.n * (j < self.n)

    def variational_mode_w(self, j):
        return 1 / self.n * (j < self.n)
