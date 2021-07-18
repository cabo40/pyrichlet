from ._base import BaseWeight
import numpy as np
from scipy.stats import beta


class PitmanYorProcess(BaseWeight):
    def __init__(self, pyd=0, alpha=1, rng=None):
        super().__init__(rng=rng)
        assert -pyd < alpha, "alpha param must be greater than -pyd"
        self.pyd = pyd
        self.alpha = alpha
        self.v = np.array([], dtype=np.float64)

    def structure_log_likelihood(self, v=None, pyd=None, alpha=None):
        if v is None:
            v = self.v
        if pyd is None:
            pyd = self.pyd
        if alpha is None:
            alpha = self.alpha
        return self.weight_log_likelihood(v=v, pyd=pyd, alpha=alpha)

    def weight_log_likelihood(self, v=None, pyd=None, alpha=None):
        if v is None:
            v = self.v
        if pyd is None:
            pyd = self.pyd
        if alpha is None:
            alpha = self.alpha
        n = len(v)
        if n == 0:
            return 0
        pitman_yor_bias = np.arange(n)
        return np.sum(beta.logpdf(v,
                                  a=1 - pyd,
                                  b=alpha + pitman_yor_bias * pyd))

    def random(self, size=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is not None:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
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
        if type(size) is not int:
            raise TypeError("size parameter must be integer or None")
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

    def tail(self, x):
        if x >= 1 or x < 0:
            raise ValueError("Tail parameter not in range [0,1)")
        if len(self.w) == 0:
            self.random(1)

        w_sum = sum(self.w)
        while w_sum < x:
            v_to_append = self.rng.beta(
                a=1 - self.pyd,
                b=self.alpha + self.get_size() * self.pyd,
                size=1)
            self.v = np.concatenate((self.v, v_to_append))
            self.w = np.concatenate((self.w, [(1 - sum(self.w)) * self.v[-1]]))
            w_sum += self.w[-1]
        return self.w
