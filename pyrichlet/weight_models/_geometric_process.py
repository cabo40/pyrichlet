from ._base import BaseWeight
from ..exceptions import NotFittedError
from ..utils.functions import mean_log_beta, log_likelihood_beta

import numpy as np
from scipy.special import loggamma


class GeometricProcess(BaseWeight):
    def __init__(self, a=1, b=1, rng=None):
        super().__init__(rng=rng)
        self.a = a
        self.b = b
        self.p = self.rng.beta(a=self.a, b=self.b)

        self.v = np.array([], dtype=np.float64)

    def weighting_log_likelihood(self):
        ret = log_likelihood_beta(self.p, self.a, self.b)
        return ret

    def random(self, size=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is None:
            size = max(self.d) + 1
        self.v = self.v[:0]
        self.p = self.rng.beta(self.a + len(self.d), self.b + self.d.sum())
        self.complete(size)
        return self.w

    def complete(self, size):
        super().complete(size)
        if len(self.v) < size:
            self.v = np.repeat(self.p, size)
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        return self.w

    def tail(self, x):
        if x >= 1 or x < 0:
            raise ValueError("Tail parameter not in range [0,1)")
        size = int(np.log(1 - x) / np.log(1 - self.p))
        self.complete(size)
        return self.w

    def fit_variational(self, variational_d):
        self.variational_d = variational_d
        self.variational_k = len(self.variational_d)
        self.variational_params = np.empty(2, dtype=np.float64)
        self.variational_params[0] = self.a + max(len(self.variational_d[0]),
                                                  1) - 1
        self.variational_params[1] = (
                self.b + (self.variational_d[1:].T *
                          range(1, self.variational_k)).sum()
        )

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

    def variational_mean_w(self, j):
        if j >= self.variational_k:
            return 1
        p = self.variational_params[0] / self.variational_params.sum()
        return p * (1 - p) ** j

    def variational_mode_w(self, j):
        if j > self.variational_k:
            return
        if self.variational_params[0] <= 1:
            if self.variational_params[1] <= 1:
                raise ValueError('multimodal distribution')
            else:
                return 0
        elif self.variational_params[1] <= 1:
            return 1 * (j == 0)
        p = ((self.variational_params[0] - 1) /
             (self.variational_params.sum() - 2))
        res = (1 - p) ** j * p
        return res
