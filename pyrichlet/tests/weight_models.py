import unittest
import numpy as np
import scipy.stats

from .. import mixture_models


class TestWeightModels(unittest.TestCase):
    def test_dirichlet_process(self):
        rng = np.random.default_rng(0)
        n = 100
        means = np.array([-10, 10])
        sds = np.array([1, 1])
        weights = np.array([0.5, 0.5])
        theta = rng.choice(range(len(weights)), size=n, p=weights)
        y = np.array([
            scipy.stats.multivariate_normal.rvs(means[j], sds[j],
                                                random_state=rng) for j in theta])

        dgp = mixture_models.BetaInBetaMixture(rng=rng)
        dgp.fit_gibbs(y)
        pass
