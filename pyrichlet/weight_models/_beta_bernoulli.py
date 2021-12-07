from ._base import BaseWeight
import numpy as np
from scipy.stats import beta, binom
from scipy.special import beta as betaf


class BetaBernoulli(BaseWeight):
    def __init__(self, p=1, alpha=1, rng=None):
        super().__init__(rng=rng)
        self.p = p
        self.alpha = alpha
        self.v = np.array([], dtype=np.float64)
        self.bernoullis = np.array([], dtype=np.int)

    def random(self, size=None):
        if size is None:
            if len(self.d) == 0:
                raise ValueError(
                    "Weight structure not fitted and `n` not passed.")
            size = 1
        else:
            if type(size) is not int:
                raise TypeError("size parameter must be integer or None")
        if len(self.d) == 0:
            self._random_bernoullis(size)
            mask_change = self.bernoullis
            mask_change = np.cumsum(mask_change)
            self.v = self.rng.beta(a=1, b=self.alpha, size=mask_change[-1] + 1)
            self.v = self.v[mask_change]
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        else:
            size = max(self.d.max(), len(self.v), size)
            self._random_bernoullis(size)
            mask_change = self.bernoullis
            mask_change = np.cumsum(mask_change)
            a_c = np.bincount(self.d)
            b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))

            a_c = np.bincount(mask_change, a_c)
            b_c = np.bincount(mask_change, b_c)

            self.v = self.rng.beta(a=1 + a_c, b=self.alpha + b_c)
            self.v = self.v[mask_change]
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
            bool_new_val = self.rng.binomial(n=1, p=self.p)
            self.bernoullis = np.append(self.bernoullis, bool_new_val)
            if bool_new_val:
                v_to_append = self.rng.beta(a=1, b=self.alpha)
                self.v = np.append(self.v, v_to_append)
            else:
                self.v = np.append(self.v, self.v[-1])
            self.w = np.append(self.w, (1 - sum(self.w)) * self.v[-1])
            w_sum += self.w[-1]
        return self.w

    def complete(self, size):
        if type(size) is not int:
            raise TypeError("size parameter must be integer or None")
        if len(self.v) < size:
            if len(self.v) == 0:
                self.v = self.rng.beta(a=1, b=self.alpha, size=1)
            mask_change = self.rng.binomial(n=1,
                                            p=self.p,
                                            size=size - len(self.v))
            self.bernoullis = np.concatenate((self.bernoullis, mask_change))
            mask_change = np.cumsum(mask_change)
            temp_v = np.concatenate((
                [self.v[-1]],
                self.rng.beta(a=1, b=self.alpha, size=mask_change[-1])))
            self.v = np.concatenate((self.v, temp_v[mask_change]))
            self.w = self.v * np.cumprod(np.concatenate(([1],
                                                         1 - self.v[:-1])))
        return self.w

    def structure_log_likelihood(self, v=None, bernoullis=None, p=None,
                                 alpha=None):
        if v is None:
            v = self.v
        if bernoullis is None:
            bernoullis = self.bernoullis
        if p is None:
            p = self.p
        if alpha is None:
            alpha = self.alpha
        log_likelihood = self.weight_log_likelihood(v=v, alpha=alpha)
        log_likelihood += self.persist_log_likelihood(bernoullis=bernoullis,
                                                      p=p)
        return log_likelihood

    def weight_log_likelihood(self, v=None, alpha=None):
        if v is None:
            v = self.v
        if alpha is None:
            alpha = self.alpha
        v = np.unique(v)
        log_likelihood = np.sum(beta.logpdf(v, a=1, b=alpha))
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

    def _random_bernoullis(self, size):
        if len(self.d) == 0:
            self.bernoullis = self.rng.binomial(n=1, p=self.p, size=size)
            self.bernoullis[0] = 0
        else:
            size = max(size, self.d.max())
            bernoullis = self.rng.binomial(n=1, p=self.p, size=size)
            a_c = np.bincount(self.d)
            b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))
            bernoullis[0] = 0
            for j in range(1, size):
                a_j_prime, b_j_prime, g_plus = 0, 0, 1
                k = j + 1
                for b in bernoullis[j + 1:]:
                    if b == 1:
                        break
                    a_j_prime += a_c[k]
                    b_j_prime += b_c[k]
                    k += 1
                    g_plus = betaf(a_j_prime, b_j_prime) * self.alpha
                a_j_prime += a_c[j]
                b_j_prime += b_c[j]
                k = j - 1
                for b in bernoullis[:j][::-1]:
                    if b == 1:
                        break
                    a_j_prime += a_c[k]
                    b_j_prime += b_c[k]
                    k -= 1
                g_minus = betaf(a_j_prime, b_j_prime) * self.alpha
                p = self.p
                p = p * g_plus / (p * g_plus + (1 - p) * g_minus)
                bernoullis[j] = self.rng.binomial(n=1, p=p)
            self.bernoullis = bernoullis
        return self.bernoullis

    def fit_variational(self, variational_d):
        raise NotImplementedError

    def variational_mean_log_w_j(self, j):
        raise NotImplementedError

    def variational_mean_log_p_d__w(self, variational_d=None):
        raise NotImplementedError

    def variational_mean_log_p_w(self):
        raise NotImplementedError

    def variational_mean_log_q_w(self):
        raise NotImplementedError

    def variational_mean_w(self, j):
        raise NotImplementedError

    def variational_mode_w(self, j):
        raise NotImplementedError
