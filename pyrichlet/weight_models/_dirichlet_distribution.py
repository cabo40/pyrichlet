from ._base import BaseWeights
import numpy as np
from scipy.stats import dirichlet


class DirichletDistribution(BaseWeights):
    def __init__(self, n=1, theta=1, rng=None):
        super().__init__(rng=rng)
        assert type(n) == int, "parameter n must be of type int"
        self.n = n
        if type(theta) in (list, np.ndarray):
            assert self.n == len(theta), (
                "when passed as a list or an array theta must have length n"
            )
            self.theta = theta
        elif type(theta) in (int, float):
            self.theta = [theta] * self.n
        self.v = np.array([], dtype=np.float64)

    def structure_log_likelihood(self, v=None, theta=None):
        if v is None:
            v = self.v
        if theta is None:
            theta = self.theta
        return self.weight_log_likelihood(v=v, theta=theta)

    def weight_log_likelihood(self, v=None, theta=None):
        if v is None:
            v = self.v
        if theta is None:
            theta = self.theta
        if len(v) == 0:
            return 0
        return np.sum(dirichlet.logpdf(v, theta))

    def random(self, size=None):
        if len(self.d) > 0:
            if max(self.d) >= len(self.theta):
                raise ValueError(
                    'fitted structure is incompatible with this model'
                )
            else:
                a_c = np.bincount(self.d)
                a_c.resize(len(self.theta), refcheck=False)
                self.v = self.rng.dirichlet(self.theta + a_c)
        else:
            self.v = self.rng.dirichlet(self.theta)
        self.w = self.v * np.cumprod(np.concatenate(([1],
                                                     1 - self.v[:-1])))
        return self.w

    def complete(self, size=None):
        if len(self.w) == 0:
            self.random(None)
        return self.w

    def tail(self, x=None):
        if len(self.w) == 0:
            self.random(None)
        return self.w
