from ._base import BaseWeight
import numpy as np
from scipy.stats import beta, binom
from scipy.optimize import minimize, brentq
from scipy.integrate import quad


class DGBProcess(BaseWeight):
    def __init__(self, p=0.5, theta=1, p_a=1, p_b=1, rng=None):
        super().__init__(rng=rng)
        self.p = p
        self.p_a = p_a
        self.p_b = p_b

        self.theta = theta

        self.v = np.array([], dtype=np.float64)
        self.bernoullis = np.array([], dtype=np.int)

    def structure_log_likelihood(self, v=None, bernoullis=None, p=None,
                                 theta=None):
        if v is None:
            v = self.v
        if bernoullis is None:
            bernoullis = self.bernoullis
        if p is None:
            p = self.p
        if theta is None:
            theta = self.theta
        log_likelihood = self.weight_log_likelihood(v=v, theta=theta)
        log_likelihood += self.persist_log_likelihood(bernoullis=bernoullis,
                                                      p=p)
        return log_likelihood

    def weight_log_likelihood(self, v=None, theta=None):
        if v is None:
            v = self.v
        if theta is None:
            theta = self.theta
        v = np.unique(v)
        log_likelihood = np.sum(beta.logpdf(v, a=1, b=theta))
        return log_likelihood

    def persist_log_likelihood(self, bernoullis=None, p=None):
        if bernoullis is None:
            bernoullis = self.bernoullis
        if p is None:
            p = self.p
        log_likelihood = binom.logpmf(k=np.sum(bernoullis),
                                      n=len(bernoullis),
                                      p=p)
        return log_likelihood

    def random(self, size=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is not None:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
        if len(self.d) == 0:
            # self.random_p()
            mask_change = self.rng.binomial(n=1, p=1 - self.p, size=size - 1)
            mask_change = np.concatenate(([0], np.cumsum(mask_change)))
            self.v = self.rng.beta(a=1, b=self.theta, size=mask_change[-1] + 1)
            self.v = self.v[mask_change]
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        else:
            self.random_p()
            self.random_bernoullis()
            remap = np.cumsum(self.bernoullis)
            a_c = np.bincount(self.d)
            b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))

            a_c = np.bincount(remap, a_c)
            b_c = np.bincount(remap, b_c)

            self.v = self.rng.beta(a=1 + a_c, b=self.theta + b_c)
            self.v = self.v[remap]
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
            if size is not None:
                self.complete(size)
        return self.w

    def complete(self, size):
        if type(size) is not int:
            raise TypeError("size parameter must be integer or None")
        if len(self.v) < size:
            if len(self.v) == 0:
                self.v = self.rng.beta(a=1, b=self.theta, size=1)
            mask_change = self.rng.binomial(n=1,
                                            p=1 - self.p,
                                            size=size - len(self.v))
            self.bernoullis = np.concatenate((self.bernoullis, mask_change))
            mask_change = np.cumsum(mask_change)
            temp_v = np.concatenate((
                [self.v[-1]],
                self.rng.beta(a=1, b=self.theta, size=mask_change[-1])))
            self.v = np.concatenate((self.v, temp_v[mask_change]))
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
            self.bernoullis = np.concatenate((self.bernoullis, [bool_new_val]))
            if bool_new_val:
                v_to_append = self.rng.beta(a=1, b=self.theta, size=1)
                self.v = np.concatenate((self.v, v_to_append))
            else:
                self.v = np.concatenate((self.v, [self.v[-1]]))
            self.w = np.concatenate((self.w, [(1 - sum(self.w)) * self.v[-1]]))
            w_sum += self.w[-1]
        return self.w

    def random_bernoullis(self):
        temp_bernoullis = self.rng.binomial(n=1, p=1 - self.p, size=max(self.d))
        self.bernoullis = np.concatenate(([0], temp_bernoullis))
        return self.bernoullis

    def random_p(self):
        self.p = self.rng.beta(a=self.p_a + sum(self.bernoullis),
                               b=self.p_b + sum(1 - self.bernoullis))
        return self.p

    def get_p(self):
        return self.p
