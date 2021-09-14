from ._base import BaseWeight
from ..exceptions import NotFittedError

import numpy as np


class EqualWeighting(BaseWeight):
    def __init__(self, n=1):
        super().__init__()
        self.n = n

    def structure_log_likelihood(self):
        return 0

    def weight_log_likelihood(self, w=None):
        return 0

    def fit(self, d):
        super(EqualWeighting, self).fit(d)
        self.n = max(self.d) + 1

    def random(self, size=None):
        if size is None:
            return np.repeat(1 / self.n, self.n)
        else:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
            if size <= self.n:
                return np.repeat(1 / self.n, size)
            else:
                return np.concatenate((np.repeat(1 / self.n, self.n),
                                       np.repeat(0, size - self.n)))

    def complete(self, size):
        return self.random(size)

    def tail(self, x):
        if x >= 1 or x < 0:
            raise ValueError("Tail parameter not in range [0,1)")
        size = np.ceil(x * self.n)
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
