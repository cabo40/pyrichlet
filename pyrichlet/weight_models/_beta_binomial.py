from ._base import BaseWeight
import numpy as np
from scipy.stats import beta, binom


class BetaBinomial(BaseWeight):
    def __init__(self, n=0, alpha=1, rng=None):
        super().__init__(rng=rng)
        self.n = n
        self.alpha = alpha

        self.v = np.array([], dtype=np.float64)
        self.binomials = np.array([], dtype=np.int)

    def random(self, size=None):
        if size is None:
            if len(self.d) == 0:
                raise ValueError(
                    "Weight structure not fitted and `n` not passed.")
            size = 1
        else:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
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
        if type(size) is not int:
            raise TypeError("size parameter must be integer or None")
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
