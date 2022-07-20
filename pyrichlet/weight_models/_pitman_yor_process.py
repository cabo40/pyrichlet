from ._base import BaseWeight
from ..exceptions import NotFittedError
from ..utils.functions import mean_log_beta, log_likelihood_beta

import numpy as np
from scipy.special import loggamma


class PitmanYorProcess(BaseWeight):
    def __init__(self, pyd=0, alpha=1, truncation_length=-1, rng=None):
        super().__init__(rng=rng)
        assert -pyd < alpha, "alpha param must be greater than -pyd"
        self.pyd = pyd
        self.alpha = alpha
        self.v = np.array([], dtype=np.float64)
        self.truncation_length = truncation_length

    def weighting_log_likelihood(self):
        v = self.w[0]
        ret = log_likelihood_beta(v, 1 - self.pyd, self.alpha)
        prod_v = 1 - v
        for j in range(1, len(self.w)):
            v = self.w[j] / prod_v
            ret += log_likelihood_beta(v, 1 - self.pyd,
                                       self.alpha + j * self.pyd)
            prod_v *= (1 - v)
        return ret

    def random(self, size=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if len(self.d) == 0:
            pitman_yor_bias = np.arange(size)
            self.v = self.rng.beta(a=1 - self.pyd,
                                   b=self.alpha + pitman_yor_bias * self.pyd,
                                   size=size)
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        else:
            a_c = np.bincount(self.d)
            b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))

            if size is not None and size < len(a_c):
                a_c = a_c[:size]
                b_c = b_c[:size]

            pitman_yor_bias = np.arange(len(a_c))
            self.v = self.rng.beta(
                a=1 - self.pyd + a_c,
                b=self.alpha + pitman_yor_bias * self.pyd + b_c
            )
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
            if size is not None:
                self.complete(size)
        return self.w

    def complete(self, size):
        super().complete(size)
        if self.get_size() < size:
            pitman_yor_bias = np.arange(self.get_size(), size)
            self.v = np.concatenate(
                (
                    self.v,
                    self.rng.beta(a=1 - self.pyd,
                                  b=self.alpha + pitman_yor_bias * self.pyd)
                )
            )
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        return self.w

    def fit_variational(self, variational_d):
        self.variational_d = variational_d
        self.variational_k = len(self.variational_d)
        self.variational_params = np.empty((self.variational_k, 2),
                                           dtype=np.float64)
        a_c = np.sum(self.variational_d, 1)
        b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))
        self.variational_params[:, 0] = 1 - self.pyd + a_c
        self.variational_params[:, 1] = self.alpha + (
                1 + np.arange(self.variational_params.shape[0])
        ) * self.pyd + b_c

    def variational_mean_log_w_j(self, j):
        if self.variational_d is None:
            raise NotFittedError
        res = 0
        for jj in range(j):
            res += mean_log_beta(self.variational_params[jj][1],
                                 self.variational_params[jj][0])
        res += mean_log_beta(self.variational_params[j, 0],
                             self.variational_params[j, 1]
                             )
        return res

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
        res = 0
        for j, params in enumerate(self.variational_params):
            res += mean_log_beta(params[0], params[1]) * -self.pyd
            res += mean_log_beta(params[1], params[0]) * (
                    self.alpha + (j + 1) * self.pyd - 1
            )
            res += loggamma(self.alpha + j * self.pyd + 1)
            res -= loggamma(self.alpha + (j + 1) * self.pyd + 1)
            res -= loggamma(1 - self.pyd)
        return res

    def variational_mean_log_q_w(self):
        if self.variational_d is None:
            raise NotFittedError
        res = 0
        for params in self.variational_params:
            res += (params[0] - 1) * mean_log_beta(params[0], params[1])
            res += (params[1] - 1) * mean_log_beta(params[1], params[0])
            res += loggamma(params[0] + params[1])
            res -= loggamma(params[0]) + loggamma(params[1])
        return res

    def variational_mean_w(self, j):
        if j > self.variational_k:
            return 0
        res = 1
        for jj in range(j):
            res *= (self.variational_params[jj][1] /
                    self.variational_params[jj].sum())
        res *= self.variational_params[j, 0] / self.variational_params[j].sum()
        return res

    def variational_mode_w(self, j):
        if j > self.variational_k:
            return 0
        res = 1
        for jj in range(j):
            if self.variational_params[jj, 1] <= 1:
                if self.variational_params[jj, 0] <= 1:
                    raise ValueError('multimodal distribution')
                else:
                    return 0
            elif self.variational_params[jj, 0] <= 1:
                continue
            res *= ((self.variational_params[jj, 1] - 1) /
                    (self.variational_params[jj].sum() - 2))

        if self.variational_params[j, 0] <= 1:
            if self.variational_params[j, 1] <= 1:
                raise ValueError('multimodal distribution')
            else:
                return 0
        elif self.variational_params[j, 1] <= 1:
            return res
        res *= ((self.variational_params[j, 0] - 1) /
                (self.variational_params[j].sum() - 2))
        return res
