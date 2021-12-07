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

    def tail(self, x):
        if x >= 1 or x < 0:
            raise ValueError("Tail parameter not in range [0,1)")
        if len(self.w) == 0:
            self.random(1)

        w_sum = sum(self.w)
        while w_sum < x:
            bool_new_val = self.rng.binomial(n=1, p=1 - self.p)
            self.binomials = np.concatenate((self.binomials, [bool_new_val]))
            if bool_new_val:
                v_to_append = self.rng.beta(a=1, b=self.alpha, size=1)
                self.v = np.concatenate((self.v, v_to_append))
            else:
                self.v = np.concatenate((self.v, [self.v[-1]]))
            self.w = np.concatenate((self.w, [(1 - sum(self.w)) * self.v[-1]]))
            w_sum += self.w[-1]
        return self.w

    def structure_log_likelihood(self, v=None, binomials=None, theta=None):
        if v is None:
            v = self.v
        if binomials is None:
            binomials = self.binomials
        if theta is None:
            theta = self.alpha
        log_likelihood = self.weight_log_likelihood(v=v, theta=theta,
                                                    binomials=binomials)
        return log_likelihood

    def weight_log_likelihood(self, v=None, theta=None, binomials=None):
        raise NotImplementedError

    def _random_binomials(self):
        a_c = np.bincount(self.d)
        b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))
        a_c = np.append(0, a_c)
        b_c = np.append(0, b_c)
        beta_rv = self.rng.beta(1 + a_c, self.alpha + b_c)
        self.binomials = self.rng.binomial(self.n, beta_rv)
        return self.binomials

    def fit_variational(self, variational_d):
        raise NotImplementedError

    def variational_mean_log_w_j(self, j):
        raise NotImplementedError

    def variational_mean_log_p_d__w(self, variational_d=None):
        raise NotImplementedError

    def variational_mean_log_p_w(self):
        raise NotImplementedError

    def variational_mean_log_q_w(self):
        raise NotImplementedError

    def variational_mean_w(self, j):
        raise NotImplementedError

    def variational_mode_w(self, j):
        raise NotImplementedError
