from ._base import BaseWeight
import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize, brentq
from scipy.integrate import quad


class BetaInBeta(BaseWeight):
    def __init__(self, x=0.5, alpha=1, a=1, b=1, p=None, p_method="static",
                 p_optim_max_steps=10, rng=None):
        super().__init__(rng=rng)
        self.x = x
        self.a = a
        self.b = b
        self.alpha = alpha
        if p is None:
            self.p = self.rng.beta(a=self.a, b=self.b)
        else:
            self.p = p

        self.p_method = p_method
        self.v = np.array([], dtype=np.float64)
        self.p_optim_max_steps = p_optim_max_steps
        self._validate_params()

    def structure_log_likelihood(self, v=None, p=None, x=None, alpha=None):
        if v is None:
            v = self.v
        if p is None:
            p = self.p
        if x is None:
            x = self.x
        if alpha is None:
            alpha = self.alpha
        log_likelihood = self.weight_log_likelihood(v=v, p=p, x=x, alpha=alpha)
        log_likelihood += self.p_log_likelihood(p=p)
        return log_likelihood

    def weight_log_likelihood(self, v=None, p=None, x=None, alpha=None, a=None,
                              b=None):
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

    def p_log_likelihood(self, p=None, a=None, b=None):
        if p is None:
            p = self.p
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return beta.logpdf(p, a=a, b=b)

    def random(self, size=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is not None:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
        if len(self.d) == 0:
            self.p = self.rng.beta(a=self.a, b=self.b)
            self.v = self.rng.beta(
                a=1 + self.x / (1 - self.x) * self.p,
                b=self.alpha + self.x / (1 - self.x) * (1 - self.p),
                size=size
            )
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        else:
            self.random_p()
            a_c = np.bincount(self.d)
            b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))

            if size is not None and size < len(a_c):
                a_c = a_c[:size]
                b_c = b_c[:size]

            self.v = self.rng.beta(
                a=1 + self.x / (1 - self.x) * self.p + a_c,
                b=self.alpha + self.x / (1 - self.x) * (1 - self.p) + b_c
            )
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
                 self.rng.beta(
                     a=1 + self.x / (1 - self.x) * self.p,
                     b=self.alpha + self.x / (1 - self.x) * (1 - self.p),
                     size=size - len(self.v)))
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
                a=1 + self.x / (1 - self.x) * self.p,
                b=self.alpha + self.x / (1 - self.x) * (1 - self.p),
                size=1
            )
            self.v = np.concatenate((self.v, v_to_append))
            self.w = np.concatenate((self.w, [(1 - sum(self.w)) * self.v[-1]]))
            w_sum += self.w[-1]
        return self.w

    def random_p(self):
        if len(self.d) == 0:
            self.p = self.rng.beta(a=self.a, b=self.b)
            return self.p
        if self.x == 1:
            self.p = self.rng.beta(a=self.a + len(self.d),
                                   b=self.b + self.d.sum())
            return self.p
        elif self.x == 0:
            return self.p

        if self.p_method == "static":
            return self.p
        elif self.p_method == "independent":
            self.p = self.rng.beta(a=self.a, b=self.b)
            return self.p
        elif self.p_method == "geometric":
            self.p = self.rng.beta(a=self.a + len(self.d),
                                   b=self.b + self.d.sum())
            return self.p
        elif self.p_method == "rejection-sampling":
            prev_logp = self.structure_log_likelihood()
            curr_iter = 0
            while curr_iter < self.p_optim_max_steps:
                pass_var = self.rng.uniform(
                    low=0,
                    high=np.exp(self.weight_log_likelihood())
                )
                temp_p = self.rng.beta(a=self.a, b=self.b)
                curr_logp = self.structure_log_likelihood(p=temp_p)
                pass_condition = np.exp(curr_logp - prev_logp) > pass_var
                curr_iter += 1
                if pass_condition:
                    self.p = temp_p
                    return self.p
                prev_logp = curr_logp
            return self.p
        elif self.p_method == "max-likelihood":
            max_param = minimize(
                lambda p: -self.structure_log_likelihood(p=p),
                np.array([self.p]),
                bounds=[(0, 1)],
                options={'maxiter': self.p_optim_max_steps})
            if max_param.success:
                self.p = max_param.x[0]
            return self.p
        elif self.p_method == "inverse-sampling":
            unif = self.rng.uniform()

            def f(p):
                return np.exp(self.structure_log_likelihood(p=p))

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

    def update_method(self, p_method):
        self.p_method = p_method

    def get_p(self):
        return self.p

    def _validate_params(self):
        accepted_methods = ["static", "independent", "geometric",
                            "rejection-sampling", "max-likelihood",
                            "inverse-sampling"]
        if self.p_method not in accepted_methods:
            raise ValueError(f"p_method must be one of {accepted_methods}")
