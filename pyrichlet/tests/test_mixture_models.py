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
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.003961566644610013)

    def test_dirichlet_process(self):
        mixture = mm.DirichletProcessMixture(rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.003554492133468814)

    def test_pitman_yor_process(self):
        mixture = mm.PitmanYorMixture(rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.003554492133468814)

    def test_geometric_process(self):
        mixture = mm.GeometricProcessMixture(rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.005647056599491388)

    def test_beta_in_dirichlet(self):
        mixture = mm.BetaInDirichletMixture(rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.015046881996052227)

    def test_beta_in_beta(self):
        mixture = mm.BetaInBetaMixture(rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.004105453049636178)

    def test_beta_bernoulli(self):
        mixture = mm.BetaBernoulliMixture(rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.00369608585646474)

    def test_beta_binomial(self):
        mixture = mm.BetaBinomialMixture(rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.0032883539259855087)

    def test_equal_weighting(self):
        n = 2
        mixture = mm.EqualWeightedMixture(n=n, rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.003825544022951682)

    def test_frequency_weighting(self):
        n = 2
        mixture = mm.FrequencyWeightedMixture(n=n, rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertEqual(mean_squared_error, 0.0038930404304534723)
