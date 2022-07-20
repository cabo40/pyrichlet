from ._base import BaseWeight
import numpy as np

from ..utils.functions import log_likelihood_beta, log_likelihood_beta_binom, \
    log_likelihood_binom


class BetaBinomial(BaseWeight):
    def __init__(self, n=0, alpha=1, rng=None):
        super().__init__(rng=rng)
        self.n = n
        self.alpha = alpha

        self.v = np.array([], dtype=np.float64)
        self.binomials = np.array([], dtype=int)

    def weighting_log_likelihood(self):
        b = self.binomials[0]
        res = log_likelihood_beta_binom(b, self.n, 1, self.alpha)
        for j in range(1, len(self.v) - 1):
            v = self.v[j]
            res += log_likelihood_beta(v, 1 + b,
                                       self.alpha + self.n - b)
            b = self.binomials[j]
            res += log_likelihood_binom(b, self.n, v)

        res += log_likelihood_beta(self.v[-1], 1 + b,
                                   self.alpha + self.n - b)
        return res

    def random(self, size=None):
        if size is None:
            if len(self.d) == 0:
                raise ValueError(
                    "Weight structure not fitted and `n` not passed.")
            size = 1
        self.v = self.v[:0]
        if len(self.d) == 0:
            self.complete(size)
        else:
            self._random_binomials()
            a_c = np.bincount(self.d)
            b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))
            beta_phased = self.binomials[:-1] + self.binomials[1:]
            a = 1 + a_c + beta_phased
            b = self.alpha + b_c + 2 * self.n - beta_phased
            self.v = self.rng.beta(a=a, b=b)
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
            if size is not None:
                self.complete(size)
        return self.w

    def complete(self, size):
        super().complete(size)
        if len(self.v) == 0:
            v0 = self.rng.beta(1, self.alpha)
            self.binomials = self.rng.binomial(self.n, v0, size=1)
            self.v = self.rng.beta(1 + self.binomials[-1],
                                   self.alpha + self.n - self.binomials[-1],
                                   size=1)
        while len(self.v) < size:
            self.binomials = np.append(self.binomials,
                                       self.rng.binomial(self.n, self.v[-1]))
            self.v = np.append(
                self.v, self.rng.beta(
                    1 + self.binomials[-1],
                    self.alpha + self.n - self.binomials[-1])
            )
        self.w = self.v * np.cumprod(np.concatenate(([1],
                                                     1 - self.v[:-1])))
        return self.w

    def _random_binomials(self):
        a_c = np.bincount(self.d)
        b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))
        a_c = np.append(0, a_c)
        b_c = np.append(0, b_c)
        beta_rv = self.rng.beta(1 + a_c, self.alpha + b_c)
        self.binomials = self.rng.binomial(self.n, beta_rv)
        return self.binomials
