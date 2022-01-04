from ._base import BaseWeight
import numpy as np
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
            self._random_bernoullis(self.d.max() + 1)
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
            self.complete(size)
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

    def _random_bernoullis(self, size):
        if len(self.d) == 0:
            self.bernoullis = self.rng.binomial(n=1, p=self.p, size=size)
            self.bernoullis[0] = 0
        else:
            size_fit = self.d.max()
            bernoullis = self.rng.binomial(n=1, p=self.p, size=size)
            a_c = np.bincount(self.d)
            b_c = np.concatenate((np.cumsum(a_c[::-1])[-2::-1], [0]))
            bernoullis[0] = 0
            for j in range(1, size_fit):
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
                p_times_plus = p * g_plus if p > 0 else 0
                not_p_times_minus = (1 - p) * g_minus if p < 1 else 0
                p = p_times_plus / (p_times_plus + not_p_times_minus)
                bernoullis[j] = self.rng.binomial(n=1, p=p)
            self.bernoullis = bernoullis
        return self.bernoullis
