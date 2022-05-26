import unittest
import numpy as np
import scipy.stats


class BaseTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        n = 50
        means = np.array([-10, 10])
        sds = np.array([1, 1])
        weights = np.array([0.5, 0.5])
        theta = self.rng.choice(range(len(weights)), size=n, p=weights)
        self.y = np.array([
            scipy.stats.multivariate_normal.rvs(
                means[j], sds[j],
                random_state=self.rng
            ) for j in theta
        ])
        self.y_density = np.array([scipy.stats.multivariate_normal.pdf(
            self.y, means[j], sds[j]) * weights[j] for j in range(2)])
        self.y_density = self.y_density.sum(axis=0)
