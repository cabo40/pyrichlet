from ._base import BaseWeight
from ..exceptions import NotFittedError

import numpy as np


class FrequencyWeighting(BaseWeight):
    def __init__(self, n=1, rng=None):
        super().__init__(rng=rng)
        self.n = n

    def random(self, size=None):
        if len(self.d) == 0:
            self.w = np.repeat(1 / self.n, self.n)
        else:
            self.w = self.d / self.d.sum()

    def complete(self, size):
        return self.random(size)

    def fit_variational(self, variational_d: np.ndarray):
        self.variational_d = variational_d
        self.variational_k = variational_d.shape[1]

    def variational_mean_log_w_j(self, j):
        if self.variational_d is None:
            raise NotFittedError
        if j >= self.variational_k:
            return -np.inf
        return self.variational_d.sum(1)[j] / self.variational_d.sum()

    def variational_mean_log_p_d__w(self, variational_d=None):
        if variational_d is None:
            if self.variational_d is None:
                raise NotFittedError
            variational_d = self.variational_d
        else:
            self.variational_d = variational_d
        return np.sum(variational_d.sum(1) * np.log(self.variational_d.sum(1) /
                                                    self.variational_d.sum()))

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
        return self.variational_d.sum(1)[j] / self.variational_d.sum()

    def variational_mode_w(self, j):
        if j > self.variational_k:
            return 0
        return self.variational_d.sum(1)[j] / self.variational_d.sum()
