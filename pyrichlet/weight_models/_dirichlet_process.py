from ._base import BaseWeights
import numpy as np
from scipy.stats import beta


class DirichletProcess(BaseWeights):
    def __init__(self, alpha=1, rng=None):
        super().__init__(rng=rng)
        self.alpha = alpha
        self.v = np.array([], dtype=np.float64)

    def structure_log_likelihood(self, v=None, alpha=None):
        if v is None:
            v = self.v
        if alpha is None:
            alpha = self.alpha
        return self.weight_log_likelihood(v=v, alpha=alpha)

    def weight_log_likelihood(self, v=None, alpha=None):
        if v is None:
            v = self.v
        if alpha is None:
            alpha = self.alpha
        if len(v) == 0:
            return 0
        return np.sum(beta.logpdf(v, a=1, b=alpha))

    def random(self, size=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is not None:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
        if len(self.d) == 0:
            self.v = self.rng.beta(a=1, b=self.alpha, size=size)
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        else:
            a_c = np.bincount(self.d)
            b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))

            if size is not None and size < len(a_c):
                a_c = a_c[:size]
                b_c = b_c[:size]

            self.v = self.rng.beta(a=1 + a_c, b=self.alpha + b_c)
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
            if size is not None:
                self.complete(size)
        return self.w

    def complete(self, size):
        if type(size) is not int:
            raise TypeError("size parameter must be integer or None")
        if len(self.v) < size:
            self.v = np.concatenate(
                (self.v,
                 self.rng.beta(a=1, b=self.alpha, size=size - len(self.v))))
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
            v_to_append = self.rng.beta(a=1, b=self.alpha, size=1)
            self.v = np.concatenate((self.v, v_to_append))
            self.w = np.concatenate((self.w, [(1 - sum(self.w)) * self.v[-1]]))
            w_sum += self.w[-1]
        return self.w
