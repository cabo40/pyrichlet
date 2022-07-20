import numpy as np
from scipy.special import digamma, loggamma
from scipy.stats import beta, betabinom, binom


def log_likelihood_beta(v, a, b):
    return beta.logpdf(v, a, b)


def log_likelihood_beta_binom(x, n, a, b):
    return betabinom.logpmf(x, n, a, b)


def log_likelihood_binom(x, n, p):
    return binom.logpmf(x, n, p)


def dirichlet_eppf(alpha, partition):
    return 0


def mean_log_beta(a, b):
    return digamma(a) - digamma(a + b)


def density_students_t(x, mu, precision, nu):
    dim = x.shape[1]
    density = np.exp(loggamma((nu + dim) / 2) - loggamma(nu / 2))
    density *= np.linalg.norm(precision) ** 0.5 / (nu * np.pi) ** (dim / 2)
    density *= (1 + np.einsum('ij,jk,ik->i',
                              x - mu, precision, x - mu
                              ) / nu) ** (-(nu + dim) / 2)
    return density


def density_normal(x, mu, precision):
    dim = x.shape[1]
    density = np.sqrt((2 * np.pi) ** -dim * np.linalg.norm(precision))
    density *= np.exp(- np.einsum('ij,jk,ik->i',
                                  x - mu, precision, x - mu
                                  ) / 2)
    return density
