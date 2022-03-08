import unittest
import numpy as np
import scipy.stats

from pyrichlet import mixture_models as mm


class TestMixtureModels(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        n = 100
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

    def test_dirichlet_distribution(self):
        n = 2
        mixture = mm.DirichletDistributionMixture(n=n, rng=self.rng)
        mixture.fit_variational(self.y)
        fitted_density = mixture.var_eap_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.004479156945755728)

    def test_dirichlet_process(self):
        mixture = mm.DirichletProcessMixture(rng=self.rng)
        mixture.fit_variational(self.y, n_groups=10)
        fitted_density = mixture.var_eap_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.004569792566361591)

    def test_pitman_yor_process(self):
        mixture = mm.PitmanYorMixture(rng=self.rng)
        mixture.fit_variational(self.y, n_groups=10)
        fitted_density = mixture.var_eap_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.004569836132405092)

    def test_geometric_process(self):
        mixture = mm.GeometricProcessMixture(rng=self.rng)
        mixture.fit_variational(self.y, n_groups=10)
        fitted_density = mixture.var_eap_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.006374035889285028)

    def test_equal_weighting(self):
        n = 2
        mixture = mm.EqualWeightedMixture(n=n, rng=self.rng)
        mixture.fit_variational(self.y)
        fitted_density = mixture.var_eap_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.0044931544153058315)

    def test_frequency_weighting(self):
        n = 2
        mixture = mm.FrequencyWeightedMixture(n=n, rng=self.rng)
        mixture.fit_variational(self.y)
        fitted_density = mixture.var_eap_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.004481288043345726)
