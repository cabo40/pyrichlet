__all__ = ["mean_log_beta"]

from scipy.special import digamma


def mean_log_beta(a, b):
    return digamma(a) - digamma(a + b)
