from ._base import BaseWeight
from ..exceptions import NotFittedError

import numpy as np


class EqualWeighting(BaseWeight):
    def __init__(self, n=1, rng=None):
        super().__init__(rng=rng)
        self.n = n

    def random(self, size=None):
        self.w = np.repeat(1 / self.n, self.n)
        return self.w

    def complete(self, size):
        return self.random(size)

    def fit_variational(self, variational_d: np.ndarray):
        self.variational_d = variational_d
        self.n = self.variational_d.shape[1]

    def variational_mean_log_w_j(self, j):
        if self.variational_d is None:
            raise NotFittedError
        return 0

    def variational_mean_log_p_d__w(self, variational_d=None):
        if variational_d is None:
            _variational_d = self.variational_d
            if _variational_d is None:
                raise NotFittedError
        return 0

    def variational_mean_log_p_w(self):
        if self.variational_d is None:
            raise NotFittedError
        return 0

    def variational_mean_log_q_w(self):
        if self.variational_d is None:
            raise NotFittedError
        return 0

    def variational_mean_w(self, j):
        if j >= self.variational_k:
            return 0
        return 1 / self.n

    def variational_mode_w(self, j):
        if j >= self.variational_k:
            return 0
        return 1 / self.n
