import numpy as np
from collections import defaultdict

from ._base import BaseWeight
from ..utils.functions import log_likelihood_beta, dirichlet_eppf


class BetaInDirichlet(BaseWeight):
    def __init__(self, alpha=1, a=0, rng=None):
        super().__init__(rng=rng)
        self.a = a
        self.alpha = alpha
        self.v = np.array([], dtype=np.float64)
        self._v_base = np.array([], dtype=np.float64)
        self._d_base = []

    def weighting_log_likelihood(self):
        v = [self.w[0]]
        prod_v = 1 - v[-1]
        for wj in self.w[1:]:
            v.append(wj / prod_v)
            prod_v *= (1 - v[-1])
        v_unique, v_counts = np.unique(v, return_counts=True)
        ret = 0
        for vj in v_unique:
            ret += log_likelihood_beta(vj, 1, self.alpha)
        ret += dirichlet_eppf(self.a, v_counts)
        return ret

    def random(self, size=None, u=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is not None:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
        self.v = self.v[:0]
        self._v_base = self._v_base[:0]
        self._d_base = self._d_base[:0]
        if len(self.d) == 0:
            self.complete(size)
        else:
            max_d = self.d.max()
            c = defaultdict(lambda: 0)
            c_prime = defaultdict(lambda: 1)
            self.complete(max_d + 1)
            v_conj_prod = np.concatenate([[1], np.cumprod(1 - self.v[:-1])])
            if u is None:
                u = self.rng.uniform(0, self.w[self.d])
            pre_c = u / v_conj_prod[self.d]
            for j in np.unique(self.d):
                c[j] = max(0, np.max(pre_c[self.d == j]))
            for k, dk in enumerate(self.d):
                for j in range(dk):
                    c_j_prime = 1 - u[k] * (1 - self.v[j]) / self.w[dk]
                    c_prime[j] = min(c_prime[j], c_j_prime)
            len_v = len(self.v)
            if self.a == 0:
                self._v_base[0] = self.rng.beta(1 + len(self.d),
                                                self.alpha + self.d.sum())
                self.v = np.repeat(self._v_base[0], len_v)
                len_v = 0
            for j in range(len_v):
                mask = (self._v_base > c[j]) & (self._v_base < c_prime[j])
                temp_v_base = self._v_base[mask]
                len_temp_v = len(temp_v_base)
                if len_temp_v == 0:
                    k = 0
                else:
                    p = np.array([1] * len_temp_v + [self.a],
                                 dtype=np.float64)
                    p /= p.sum()
                    k = self.rng.choice(range(len_temp_v + 1), p=p)
                if k < len_temp_v:
                    self.v[j] = temp_v_base[k]
                else:
                    trunc_beta = self.rng.uniform(
                        1 - np.power(1 - c[j], self.alpha),
                        1 - np.power(1 - c_prime[j], self.alpha)
                    )
                    trunc_beta = 1 - np.power(1 - trunc_beta, 1 / self.alpha)
                    self._v_base = np.append(self._v_base, trunc_beta)
                    self._d_base += [1]
                    self.v[j] = trunc_beta
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        return self.w

    def complete(self, size):
        if len(self._v_base) == 0:
            self._v_base = self.rng.beta(1, self.alpha, size=1)
            self._d_base += [1]
        while len(self.v) < size:
            p = np.array(self._d_base + [self.a], dtype=np.float64)
            p /= p.sum()
            jj = self.rng.choice(range(len(self._v_base) + 1), p=p)
            if jj <= len(self._v_base):
                self.v = np.append(self.v, self._v_base[jj])
            else:
                new_v_base = self.rng.beta(1, self.alpha)
                self._v_base = np.append(self._v_base, new_v_base)
                self._d_base += [1]
                self.v = np.append(self.v, new_v_base)
        self.w = self.v * np.cumprod(np.concatenate(([1],
                                                     1 - self.v[:-1])))
