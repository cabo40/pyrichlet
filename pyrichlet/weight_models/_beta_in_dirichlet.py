import numpy as np
from scipy.stats import beta
from collections import defaultdict
from scipy.special import loggamma
from scipy.optimize import minimize, brentq
from scipy.integrate import quad

from ._base import BaseWeight
from ..exceptions import NotFittedError
from ..utils.functions import mean_log_beta


class BetaInDirichlet(BaseWeight):
    def __init__(self, alpha=1, a=0, rng=None):
        super().__init__(rng=rng)
        self.a = a
        self.alpha = alpha
        self.v = np.array([], dtype=np.float64)

    def structure_log_likelihood(self, v=None, alpha=None, a=None):
        if v is None:
            v = self.v
        if alpha is None:
            alpha = self.alpha
        if a is None:
            a = self.a
        log_likelihood = self.weight_log_likelihood(v=v, alpha=alpha, a=a)
        return log_likelihood

    def weight_log_likelihood(self, v=None, alpha=None, a=None):
        if v is None:
            v = self.v
        if alpha is None:
            alpha = self.alpha
        if a is None:
            alpha = self.alpha
        return np.sum(
            beta.logpdf(v,
                        a=1,
                        b=alpha)
        )

    def random(self, size=None, u=None):
        if size is None and len(self.d) == 0:
            raise ValueError("Weight structure not fitted and `n` not passed.")
        if size is not None:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
        if len(self.d) == 0:
            inner_v = [self.rng.beta(1, self.alpha)]
            v = [inner_v[0]]
            inner_d = [1]
            len_v = 1
            while len_v < size:
                p = np.array(inner_d + [self.a], dtype=np.float64)
                p /= p.sum()
                j = self.rng.choice(range(len_v + 1), p=p)
                inner_d.append(j)
                if j <= len_v:
                    v.append(v[j])
                else:
                    new_v = self.rng.beta(1, self.alpha)
                    inner_v.append(new_v)
                    v.append(new_v)
                len_v += 1
            self.v = np.array(v, dtype=np.float64)
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        else:
            max_d = self.d.max()
            len_d = len(self.d)
            if u is None:
                u = self.rng.uniform(len_d)
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
                mask = np.full_like(self.v, True)
                mask[j] = False
                mask = mask & (self.v > c[j]) & (self.v < c_prime[j])
                temp_v = self.v[mask]
                len_temp_v = len(temp_v)
                p = np.array([1] * len_temp_v + [self.a],
                             dtype=np.float64)
                p /= p.sum()
                k = self.rng.choice(range(len_temp_v + 1), p=p)
                if k <= len_temp_v:
                    self.v[j] = temp_v[k]
                else:
                    trunc_beta = self.rng.uniform(
                        1 - np.power(1 - c[j], self.alpha),
                        1 - np.power(1 - c_prime[j], self.alpha)
                    )
                    trunc_beta = 1 - np.power(1 - trunc_beta, 1 / self.alpha)
                    self.v[j] = trunc_beta
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

    def fit_variational(self, variational_d):
        self.variational_d = variational_d
        self.variational_k = len(self.variational_d)
        self.variational_params = np.empty((self.variational_k, 2),
                                           dtype=np.float64)
        a_c = np.sum(self.variational_d, 1)
        b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))
        self.variational_params[:, 0] = 1 + self.x / (
                1 - self.x) * self.p + a_c
        self.variational_params[:, 1] = self.alpha + self.x / (
                1 - self.x) * (1 - self.p) + b_c

    def variational_mean_log_w_j(self, j):
        if self.variational_d is None:
            raise NotFittedError
        res = 0
        for jj in range(j):
            res += mean_log_beta(self.variational_params[jj][1],
                                 self.variational_params[jj][0])
        res += mean_log_beta(self.variational_params[j, 0],
                             self.variational_params[j, 1]
                             )
        return res

    def variational_mean_log_p_d__w(self, variational_d=None):
        if variational_d is None:
            _variational_d = self.variational_d
            if _variational_d is None:
                raise NotFittedError
        else:
            _variational_d = variational_d
        res = 0
        for j, nj in enumerate(np.sum(_variational_d, 1)):
            res += nj * self.variational_mean_log_w_j(j)
        return res

    def variational_mean_log_p_w(self):
        if self.variational_d is None:
            raise NotFittedError
        res = 0
        for j, params in enumerate(self.variational_params):
            res += mean_log_beta(params[0], params[1]) * self.x / (
                    1 - self.x) * self.p
            res += mean_log_beta(params[1], params[0]) * (
                    self.alpha + self.x / (1 - self.x) * (1 - self.p) - 1
            )
            res += loggamma(1 + self.x / (1 - self.x) + self.alpha)
            res -= loggamma(1 + self.x / (1 - self.x) * self.p)
            res -= loggamma(self.alpha + self.x / (1 - self.x) * (1 - self.p))
        return res

    def variational_mean_log_q_w(self):
        if self.variational_d is None:
            raise NotFittedError
        res = 0
        for params in self.variational_params:
            res += (params[0] - 1) * mean_log_beta(params[0], params[1])
            res += (params[1] - 1) * mean_log_beta(params[1], params[0])
            res += loggamma(params[0] + params[1])
            res -= loggamma(params[0]) + loggamma(params[1])
        return res

    def variational_mean_w(self, j):
        if j > self.variational_k:
            return 0
        res = 1
        for jj in range(j):
            res *= (self.variational_params[jj][1] /
                    self.variational_params[jj].sum())
        res *= self.variational_params[j, 0] / self.variational_params[j].sum()
        return res

    def variational_mode_w(self, j):
        if j > self.variational_k:
            return 0
        res = 1
        for jj in range(j):
            if self.variational_params[jj, 1] <= 1:
                if self.variational_params[jj, 0] <= 1:
                    raise ValueError('multimodal distribution')
                else:
                    return 0
            elif self.variational_params[jj, 0] <= 1:
                continue
            res *= ((self.variational_params[jj, 1] - 1) /
                    (self.variational_params[jj].sum() - 2))

        if self.variational_params[j, 0] <= 1:
            if self.variational_params[j, 1] <= 1:
                raise ValueError('multimodal distribution')
            else:
                return 0
        elif self.variational_params[j, 1] <= 1:
            return res
        res *= ((self.variational_params[j, 0] - 1) /
                (self.variational_params[j].sum() - 2))
        return res
