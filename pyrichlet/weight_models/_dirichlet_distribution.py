from ._base import BaseWeight
from ..exceptions import NotFittedError
from ..utils.functions import mean_log_beta

import numpy as np
from scipy.stats import dirichlet
from scipy.special import loggamma


class DirichletDistribution(BaseWeight):
    def __init__(self, n=1, alpha=1, rng=None):
        super().__init__(rng=rng)
        assert type(n) == int, "parameter n must be of type int"
        self.n = n
        if type(alpha) in (list, np.ndarray):
            self.n = n
            self.alpha = np.array(alpha, dtype=np.float64)
        elif type(alpha) in (int, float):
            self.alpha = np.array([alpha] * self.n, dtype=np.float64)
        self.v = np.array([], dtype=np.float64)

    def structure_log_likelihood(self, v=None, alpha=None):
        if v is None:
            v = self.v
        if alpha is None:
            alpha = self.alpha
        return self.weight_log_likelihood(v=v, theta=alpha)

    def weight_log_likelihood(self, v=None, theta=None):
        if v is None:
            v = self.v
        if theta is None:
            theta = self.alpha
        if len(v) == 0:
            return 0
        return np.sum(dirichlet.logpdf(v, theta))

    def random(self, size=None):
        if len(self.d) > 0:
            if max(self.d) >= len(self.alpha):
                raise ValueError(
                    'fitted structure is incompatible with this model'
                )
            else:
                a_c = np.bincount(self.d)
                a_c.resize(len(self.alpha), refcheck=False)
                self.v = self.rng.dirichlet(self.alpha + a_c)
        else:
            self.v = self.rng.dirichlet(self.alpha)
        self.w = self.v * np.cumprod(np.concatenate(([1],
                                                     1 - self.v[:-1])))
        return self.w

    def complete(self, size=None):
        if len(self.w) == 0:
            self.random(None)
        return self.w

    def tail(self, x=None):
        if len(self.w) == 0:
            self.random(None)
        return self.w

    def fit_variational(self, variational_d):
        assert len(variational_d) == self.n, "variational distribution must" \
                                             "have the same length as the" \
                                             "Dirichlet distribution's" \
                                             "dimension"
        self.variational_k = self.n
        self.variational_d = variational_d
        self.variational_params = self.alpha + np.sum(self.variational_d, 1)

    def variational_mean_log_w_j(self, j):
        if self.variational_d is None:
            raise NotFittedError
        return mean_log_beta(self.variational_params[j],
                             self.variational_params.sum())

    def variational_mean_log_p_d__w(self, variational_d=None):
        if variational_d is None:
            _variational_d = self.variational_d
            if _variational_d is None:
                raise NotFittedError
        else:
            _variational_d = variational_d
        res = 0
        for j, nj in enumerate(np.sum(_variational_d, 1)):
            res += nj * self.variational_mean_log_w_j(j)
        return res

    def variational_mean_log_p_w(self):
        if self.variational_d is None:
            raise NotFittedError
        log_sum_w_j = 0
        for j in range(self.variational_k):
            log_sum_w_j += self.variational_mean_log_w_j(j)
        log_sum_w_j *= self.alpha.sum() - 1
        res = self._log_normalization_constant(self.alpha)
        res += log_sum_w_j
        return res

    def variational_mean_log_q_w(self):
        if self.variational_d is None:
            raise NotFittedError
        res = 0
        for j in range(self.variational_k):
            res += ((self.variational_params[j] - 1) *
                    self.variational_mean_log_w_j(j))
        res += self._log_normalization_constant(
            self.variational_params
        )
        return res

    def _log_normalization_constant(self, alpha=None):
        if alpha is None:
            _alpha = self.alpha
        else:
            _alpha = alpha
        log_sum = loggamma(np.sum(_alpha))
        for a_i in _alpha:
            log_sum -= loggamma(a_i)
        return log_sum
