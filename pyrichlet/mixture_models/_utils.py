from scipy.stats import invwishart, multivariate_normal
from scipy.special import digamma, loggamma
from itertools import repeat
import numpy as np


def random_normal_invw(mu, lam, psi, nu, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    ret_sigma = invwishart.rvs(nu, psi,
                               random_state=rng)
    ret_mu = multivariate_normal.rvs(mu, ret_sigma / lam,
                                     random_state=rng)
    return ret_mu, ret_sigma


def posterior_norm_invw_params(y, mu, lam, psi, nu):
    n, _ = y.shape
    ret_mu = (lam * mu + n * y.mean(axis=0)) / (lam + n)
    ret_lam = lam + n
    ret_psi = psi + n * np.cov(y.T, bias=True) + (
            (lam * n) / (lam + n) *
            ((y.mean(axis=0) - mu) @ (y.mean(axis=0) - mu)))
    ret_nu = nu + n
    return {"mu": ret_mu, "lambda": ret_lam, "psi": ret_psi, "nu": ret_nu}


def gumbel_max_sampling(logp, size=None, *, rng=None):
    if size is None:
        ret = np.argmax(logp -
                        np.log(-np.log(rng.uniform(size=logp.shape))), axis=0)
    else:
        ret = []
        for i in range(size):
            ret.append(np.argmax(logp -
                                 np.log(-np.log(rng.uniform(size=len(logp))))))
        ret = np.array(ret)
    return ret


def rejection_sample(f, max_y, a=0, b=1, size=None, *, rng=None):
    if size is None:
        x = rng.uniform(a, b)
        y = rng.uniform(0, max_y)
        while y > f(x):
            x = rng.uniform(a, b)
            y = rng.uniform(0, max_y)
        return x
    else:
        x = rng.uniform(a, b, size)
        y = rng.uniform(0, max_y, size)
        while np.any(y > f(x)):
            x[y > f(x)] = rng.uniform(a, b, np.sum(y > f(x)))
            y[y > f(x)] = rng.uniform(0, max_y, np.sum(y > f(x)))
        return x


def mixture_density(x, w, mu, sigma, u):
    k = len(w)

    ret = []
    for j in range(k):
        ret.append(multivariate_normal.pdf(x,
                                           mu[j],
                                           sigma[j],
                                           1))

    ret = np.array(ret).T
    mask = (np.array(list(repeat(u, k))) <
            np.array(list(repeat(w, len(u)))).transpose())

    ret = np.atleast_2d(ret.dot(mask / mask.sum(0))).mean(1)
    return ret


def cluster(x, w, mu, sigma, u=None):
    k = len(w)
    assign_prob = []
    for j in range(k):
        assign_prob.append(multivariate_normal.pdf(x,
                                                   mu[j],
                                                   sigma[j],
                                                   1))
    assign_prob = np.array(assign_prob).T
    if u is not None:
        mask = (np.array(list(repeat(u, k))) <
                np.array(list(repeat(w, len(u)))).transpose())

        w = (mask / mask.sum(0)).sum(1) / len(u)
    assign_prob = assign_prob * w
    grp = np.argmax(assign_prob, axis=1)
    uncertainty = 1 - assign_prob[range(len(x)), grp]
    u_grp, ret = np.unique(grp, return_inverse=True)
    return ret, uncertainty, w[u_grp], mu[u_grp], sigma[u_grp]


def log_wishart_normalization_term(precision, scale):
    dim = precision.shape[0]
    res = np.log(np.linalg.norm(precision)) * (-scale / 2)
    inverse_term = np.log(2) * (scale * dim / 2)
    inverse_term += np.log(np.pi) * (dim * (dim - 1) / 4)
    for i in range(dim):
        inverse_term += loggamma((scale - i) / 2)
    res -= inverse_term
    return res


def e_log_norm_wishart(precision, scale):
    dim = precision.shape[0]
    res = np.log(np.linalg.norm(precision))
    res += dim * np.log(2)
    for i in range(dim):
        res += digamma((scale - i) / 2)
    return res


def entropy_wishart(precision, scale):
    dim = precision.shape[0]
    res = -log_wishart_normalization_term(precision, scale)
    res -= (scale - dim - 1) / 2 * e_log_norm_wishart(precision, scale)
    res += scale * dim / 2
    return res
