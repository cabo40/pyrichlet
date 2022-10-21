import numpy as np
from scipy.optimize import minimize, brentq
from scipy.integrate import quad
from scipy.stats import beta

from ._base import BaseWeight
from ..utils.functions import log_likelihood_beta


class BetaInBeta(BaseWeight):
    def __init__(self, x=0, alpha=1, a=1, b=1, p=0,
                 p_method="geometric",
                 p_optim_max_steps=10, rng=None):
        super().__init__(rng=rng)
        self.x = x
        self.a = a
        self.b = b
        self.alpha = alpha
        self.p = p

        self.p_method = p_method
        self.v = np.array([], dtype=np.float64)
        self.p_optim_max_steps = p_optim_max_steps
        self._validate_params()

    def weighting_log_likelihood(self):
        v = self.w[0]
        beta_a = 1 + self.x / (1 - self.x) * self.p
        beta_b = self.alpha + self.x / (1 - self.x) * (1 - self.p)
        ret = log_likelihood_beta(v, beta_a, beta_b)
        prod_v = 1 - v
        for wj in self.w[1:]:
            v = wj / prod_v
            ret += log_likelihood_beta(v, beta_a, beta_b)
            prod_v *= (1 - v)
        ret += self._internal_beta_log_likelihood()
        return ret

    def _internal_beta_log_likelihood(self):
        return log_likelihood_beta(self.p, self.a, self.b)

    def random(self, size=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        self.v = self.v[:0]
        if len(self.d) == 0:
            self.complete(size)
        else:
            if self.x > 0:
                self.random_p()
            a_c = np.bincount(self.d)
            b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))

            if size is None:
                size = len(a_c)
            elif size < len(a_c):
                a_c = a_c[:size]
                b_c = b_c[:size]
            else:
                pass

            if self.x < 1:
                self.v = self._rng.beta(
                    a=1 + self.x / (1 - self.x) * self.p + a_c,
                    b=self.alpha + self.x / (1 - self.x) * (1 - self.p) + b_c
                )
            else:
                self.v = np.repeat(self.p, size)
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
            if size is not None:
                self.complete(size)
        return self.w

    def complete(self, size):
        super().complete(size)
        if len(self.v == 0):
            self.random_p()
        if len(self.v) < size:
            if self.x < 1:
                concat_value = self._rng.beta(
                    a=1 + self.x / (1 - self.x) * self.p,
                    b=self.alpha + self.x / (1 - self.x) * (1 - self.p),
                    size=size - len(self.v)
                )
            else:
                concat_value = np.repeat(self.p, size - len(self.v))
            self.v = np.append(self.v, concat_value)
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        return self.w

    def random_p(self):
        if self.x == 1:
            self.p = self._rng.beta(a=self.a + len(self.d),
                                    b=self.b + self.d.sum())
            return self.p
        if self.x == 0:
            return self.p
        if len(self.d) == 0:
            self.p = self._rng.beta(a=self.a, b=self.b)
            return self.p
        # It's not Dirichlet or Gemetric and we need to fit it
        if self.p_method == "static":
            return self.p
        elif self.p_method == "independent":
            self.p = self._rng.beta(a=self.a, b=self.b)
            return self.p
        elif self.p_method == "geometric":
            self.p = self._rng.beta(a=self.a + len(self.d),
                                    b=self.b + self.d.sum())
            return self.p
        elif self.p_method == "max-likelihood":
            max_param = minimize(
                lambda p: -self._structure_log_likelihood(p=p),
                np.array([self.p]),
                bounds=[(0, 1)],
                options={'maxiter': self.p_optim_max_steps})
            if max_param.success:
                self.p = max_param.x[0]
            return self.p
        elif self.p_method == "inverse-sampling":
            unif = self._rng.uniform()

            def f(p):
                return np.exp(self._structure_log_likelihood(p=p))

            integral_normalization = quad(f, 0, 1)[0]

            def f_integral(p):
                return quad(f, 0, p)[0] / integral_normalization - unif

            try:
                self.p = brentq(f_integral, a=0, b=1)
            except ValueError:
                pass
            return self.p
        else:
            raise ValueError(f"unknown p-method")

    def _validate_params(self):
        accepted_methods = ["static", "independent", "geometric",
                            "rejection-sampling", "max-likelihood",
                            "inverse-sampling"]
        if self.p_method not in accepted_methods:
            raise ValueError(f"p_method must be one of {accepted_methods}")

    def _structure_log_likelihood(self, v=None, p=None, x=None, alpha=None):
        if v is None:
            v = self.v
        if p is None:
            p = self.p
        if x is None:
            x = self.x
        if alpha is None:
            alpha = self.alpha
        log_likelihood = self._weight_log_likelihood(v=v, p=p, x=x,
                                                     alpha=alpha)
        log_likelihood += self._p_log_likelihood(p=p)
        return log_likelihood

    def _weight_log_likelihood(self, v=None, p=None, x=None, alpha=None,
                               a=None, b=None):
        if v is None:
            v = self.v
        if p is None:
            p = self.p
        if x is None:
            x = self.x
        if alpha is None:
            alpha = self.alpha
        if a is None:
            alpha = self.alpha
        if b is None:
            alpha = self.alpha
        if x == 1:
            if len(v) == 0:
                return 0
            if np.all(v == v[0]):
                return 0
            else:
                return -np.inf
        return np.sum(
            beta.logpdf(v,
                        a=1 + x / (1 - x) * p,
                        b=alpha + x / (1 - x) * (1 - p)))

    def _p_log_likelihood(self, p=None, a=None, b=None):
        if p is None:
            p = self.p
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return beta.logpdf(p, a=a, b=b)
