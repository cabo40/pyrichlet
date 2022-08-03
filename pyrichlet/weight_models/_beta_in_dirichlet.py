import numpy as np
from scipy.special import loggamma

from ._base import BaseWeight
from ..mixture_models._utils import gumbel_max_sampling
from ..utils.functions import log_likelihood_beta, dirichlet_log_eppf


class BetaInDirichlet(BaseWeight):
    def __init__(self, alpha=1, a=0, rng=None):
        super().__init__(rng=rng)
        self.a = a
        self.alpha = alpha
        self.v = np.array([], dtype=np.float64)
        self._v_base = np.array([], dtype=np.float64)
        self._d_base = np.array([], dtype=np.int64)
        self._count_base = []

    def weighting_log_likelihood(self):
        v_unique, v_counts = np.unique(self.v, return_counts=True)
        ret = 0
        for vj in v_unique:
            ret += log_likelihood_beta(vj, 1, self.alpha)
        ret += dirichlet_log_eppf(self.a, v_counts)
        return ret

    def random(self, size=None, u=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is not None:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
        self.v = self.v[:0]
        self._v_base = self._v_base[:0]
        if len(self.d) == 0:
            self.complete(size)
            return self.w
        n = max(self.d) + 1
        if len(self._d_base) < n:
            self._d_base = np.concatenate(
                [self._d_base, [0] * (n - len(self._d_base))])
        elif len(self._d_base) > n:
            self._d_base = self._d_base[:n]
        k = max(self._d_base) + 1
        v_base = np.empty(k, dtype=np.float64)
        a_c = np.bincount(self.d)
        b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))
        # Update inner weights given the inner assignations
        for jj in range(k):
            a_c_base = np.sum(a_c[self._d_base == jj])
            b_c_base = np.sum(b_c[self._d_base == jj])
            v_base[jj] = self._rng.beta(a=1 + a_c_base,
                                        b=self.alpha + b_c_base)
        self._v_base = v_base
        # Update the inner assignations given inner weights and other
        # assignations. It is, update inner d_jj given d_{-jj}, inner v.
        for j in range(n):
            if self.a == 0:
                if j < len(self._d_base):
                    self._d_base[j] = 0
                else:
                    self._d_base = np.append(self._d_base, 0)
                continue
            d_base_reduced = np.delete(self._d_base, j)
            k = np.max(d_base_reduced)
            log_prob = []
            new_base = False
            for dd in range(k + 2):
                base_count = sum(d_base_reduced == dd)
                if base_count == 0 and not new_base:
                    new_base = True
                    temp_log_prob = (
                            np.log(self.a) + loggamma(1 + a_c[j])
                            + loggamma(self.alpha + b_c[j])
                            - loggamma(1 + self.alpha + a_c[j] + b_c[j]))
                    log_prob.append(temp_log_prob)
                    continue
                if base_count == 0:
                    if dd < k + 1:
                        log_prob.append(-np.inf)
                    continue
                temp_log_prob = (
                        np.log(base_count)
                        + a_c[j] * np.log(self._v_base[dd])
                        + b_c[j] * np.log(1 - self._v_base[dd]))
                log_prob.append(temp_log_prob)
            log_prob = np.array(log_prob)
            if j < len(self._d_base):
                self._d_base[j] = gumbel_max_sampling(log_prob, rng=self._rng)
            else:
                self._d_base = np.append(self._d_base,
                                         gumbel_max_sampling(log_prob,
                                                             rng=self._rng))
        self._count_base = [
            np.sum(self._d_base == j) for j in range(len(self._v_base))]
        self.v = self._v_base[self._d_base]
        self.w = self.v * np.cumprod(np.concatenate(([1],
                                                     1 - self.v[:-1])))
        return self.w

    def complete(self, size):
        if len(self._v_base) == 0:
            self._v_base = self._rng.beta(1, self.alpha, size=1)
            self._count_base = [1]
        while len(self.v) < size:
            p = np.array(self._count_base + [self.a], dtype=np.float64)
            p /= p.sum()
            jj = self._rng.choice(range(len(self._v_base) + 1), p=p)
            if jj < len(self._v_base):
                self.v = np.append(self.v, self._v_base[jj])
                self._count_base[jj] += 1
            else:
                new_v_base = self._rng.beta(1, self.alpha)
                self._v_base = np.append(self._v_base, new_v_base)
                self._count_base += [1]
                self.v = np.append(self.v, new_v_base)
            self._d_base = np.append(self._d_base, jj)
        self.w = self.v * np.cumprod(np.concatenate(([1],
                                                     1 - self.v[:-1])))
