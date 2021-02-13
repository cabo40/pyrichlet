from ._base import BaseWeights
import numpy as np
from scipy.stats import dirichlet


class DirichletDistribution(BaseWeights):
    def __init__(self, n=1, alpha=1, rng=None):
        super().__init__(rng=rng)
        assert type(n) == int, "parameter n must be of type int"
        self.n = n
        if type(alpha) in (list, np.ndarray):
            self.n = n
            self.alpha = alpha
        elif type(alpha) in (int, float):
            self.alpha = [alpha] * self.n
        self.v = np.array([], dtype=np.float64)

    def structure_log_likelihood(self, v=None, alpha=None):
        if v is None:
            v = self.v
        if alpha is None:
            alpha = self.alpha
        return self.weight_log_likelihood(v=v, theta=alpha)

    def weight_log_likelihood(self, v=None, theta=None):
        if v is None:
            v = self.v
        if theta is None:
            theta = self.alpha
        if len(v) == 0:
            return 0
        return np.sum(dirichlet.logpdf(v, theta))

    def random(self, size=None):
        if len(self.d) > 0:
            if max(self.d) >= len(self.alpha):
                raise ValueError(
                    'fitted structure is incompatible with this model'
                )
            else:
                a_c = np.bincount(self.d)
                a_c.resize(len(self.alpha), refcheck=False)
                self.v = self.rng.dirichlet(self.alpha + a_c)
        else:
            self.v = self.rng.dirichlet(self.alpha)
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
