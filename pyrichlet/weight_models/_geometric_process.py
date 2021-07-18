from ._base import BaseWeight
from ..exceptions import NotFittedError
from ..utils.functions import mean_log_beta

import numpy as np
from scipy.stats import beta
from scipy.special import loggamma


class GeometricProcess(BaseWeight):
    def __init__(self, a=1, b=1, rng=None):
        super().__init__(rng=rng)
        self.a = a
        self.b = b
        self.p = self.rng.beta(a=self.a, b=self.b)

        self.v = np.array([], dtype=np.float64)

    def structure_log_likelihood(self, v=None, p=None, a=None, b=None):
        if v is None:
            v = self.v
        if p is None:
            p = self.p
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        log_likelihood = self.weight_log_likelihood(v=v)
        log_likelihood += self.p_log_likelihood(p=p, a=a, b=b)
        return log_likelihood

    def weight_log_likelihood(self, v=None):
        if v is None:
            v = self.v
        if len(v) == 0:
            return 0
        if np.all(v == v[0]):
            return 0
        else:
            return -np.inf

    def p_log_likelihood(self, p=None, a=None, b=None):
        if p is None:
            p = self.p
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return beta.logpdf(p, a=a, b=b)

    def random(self, size=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is None:
            size = max(self.d) + 1
        else:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
        self.p = self.rng.beta(self.a + len(self.d), self.b + self.d.sum())
        if size <= 0:
            size = 1
        self.v = np.repeat(self.p, size)
        self.w = self.v * np.cumprod(np.concatenate(([1],
                                                     1 - self.v[:-1])))
        return self.w

    def complete(self, size):
        if type(size) is not int:
            raise TypeError("size parameter must be integer or None")
        if len(self.w) < size:
            self.v = np.repeat(self.p, size)
            self.w = self.v * np.cumprod(
                np.concatenate(([1], 1 - self.v[:-1])))
        return self.w

    def tail(self, x):
        if x >= 1 or x < 0:
            raise ValueError("Tail parameter not in range [0,1)")
        size = int(np.log(1 - x) / np.log(1 - self.p))
        if size <= 0:
            size = 1
        self.v = np.repeat(self.p, size)
        self.w = self.v * np.cumprod(np.concatenate(([1], 1 - self.v[:-1])))
        return self.w

    def get_p(self):
        return self.p

    def fit_variational(self, variational_d):
        self.variational_d = variational_d
        self.variational_k = len(self.variational_d)
        self.variational_params = np.empty(2, dtype=np.float64)
        self.variational_params[0] = self.a + len(self.variational_d[0]) - 1
        self.variational_params[1] = self.b + (
                self.variational_d[1:].T * range(1, 3)).sum()

    def variational_mean_log_w_j(self, j):
        if self.variational_d is None:
            raise NotFittedError
        res = mean_log_beta(self.variational_params[0],
                            self.variational_params[1]
                            )
        if j > 0:
            res += mean_log_beta(self.variational_params[1],
                                 self.variational_params[0]
                                 ) * j
        return res

    def variational_mean_log_p_d__w(self, variational_d=None):
        if variational_d is None:
            _variational_d = self.variational_d
            if _variational_d is None:
                raise NotFittedError
        else:
            _variational_d = variational_d
        res = mean_log_beta(self.variational_params[0],
                            self.variational_params[1]
                            ) * len(_variational_d[0])
        e_log_v_bar = mean_log_beta(self.variational_params[1],
                                    self.variational_params[0]
                                    )
        for j, nj in enumerate(np.sum(_variational_d, 1)):
            res += nj * e_log_v_bar
        return res

    def variational_mean_log_p_w(self):
        if self.variational_d is None:
            raise NotFittedError
        params = self.variational_params
        res = mean_log_beta(params[0], params[1]) * (self.a - 1)
        res += mean_log_beta(params[1], params[0]) * (self.b - 1)
        res += loggamma(self.a + self.b)
        res -= loggamma(self.a) + loggamma(self.b)
        return res

    def variational_mean_log_q_w(self):
        if self.variational_d is None:
            raise NotFittedError
        params = self.variational_params
        res = mean_log_beta(params[0], params[1]) * (params[0] - 1)
        res += mean_log_beta(params[1], params[0]) * (params[1] - 1)
        res += loggamma(params[0] + params[1])
        res -= loggamma(params[0]) + loggamma(params[1])
        return res
