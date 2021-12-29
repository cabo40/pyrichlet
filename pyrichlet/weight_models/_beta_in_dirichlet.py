import numpy as np
from collections import defaultdict

from ._base import BaseWeight


class BetaInDirichlet(BaseWeight):
    def __init__(self, alpha=1, a=0, rng=None):
        super().__init__(rng=rng)
        self.a = a
        self.alpha = alpha
        self.v = np.array([], dtype=np.float64)

    def random(self, size=None, u=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is not None:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
        if len(self.d) == 0:
            self.complete(size)
        else:
            max_d = self.d.max()
            c = defaultdict(lambda: 0)
            c_prime = defaultdict(lambda: 1)
            while len(self.v) < max_d:
                p = np.array([1] * len(self.v) + [self.a], dtype=np.float64)
                p /= p.sum()
                j = self.rng.choice(range(len(self.v) + 1), p=p)
                if j <= len(self.v):
                    self.v = np.append(self.v, self.v[j])
                else:
                    self.v = np.append(self.v, self.rng.beta(1, self.alpha))
            if u is None:
                self.w = self.v * np.cumprod(np.concatenate(([1],
                                                             1 - self.v[:-1])))
                u = self.rng.uniform(0, self.w[self.d])
            for k, dk in enumerate(self.d):
                c_j = u[k]
                if dk > 0:
                    c_j /= np.prod(1 - self.v[:dk])
                c[dk] = max(c[dk], c_j)
                for j in range(dk):
                    c_j_prime = self.v[dk] * np.prod(1 - self.v[:dk])
                    c_j_prime = 1 - u[k] * (1 - self.v[j]) / c_j_prime
                    c_prime[j] = min(c_prime[j], c_j_prime)
            for j in range(len(self.v)):
                mask = np.full_like(self.v, True, dtype=np.bool)
                mask[j] = False
                mask = mask & (self.v > c[j]) & (self.v < c_prime[j])
                temp_v = self.v[mask]
                len_temp_v = len(temp_v)
                if len_temp_v == 0 and self.a == 0:
                    k = 0
                else:
                    p = np.array([1] * len_temp_v + [self.a],
                                 dtype=np.float64)
                    p /= p.sum()
                    k = self.rng.choice(range(len_temp_v + 1), p=p)
                if k < len_temp_v:
                    self.v[j] = temp_v[k]
                else:
                    trunc_beta = self.rng.uniform(
                        1 - np.power(1 - c[j], self.alpha),
                        1 - np.power(1 - c_prime[j], self.alpha)
                    )
                    trunc_beta = 1 - np.power(1 - trunc_beta, 1 / self.alpha)
                    self.v[j] = trunc_beta
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        return self.w

    def complete(self, size):
        if len(self.v) == 0:
            self.v = self.rng.beta(1, self.alpha, size=1)
        while len(self.v) < size:
            p = np.array([1] * len(self.v) + [self.a], dtype=np.float64)
            p /= p.sum()
            j = self.rng.choice(range(len(self.v) + 1), p=p)
            if j <= len(self.v):
                self.v = np.append(self.v, self.v[j])
            else:
                self.v = np.append(self.v, self.rng.beta(1, self.alpha))
        self.w = self.v * np.cumprod(np.concatenate(([1],
                                                     1 - self.v[:-1])))
