import numpy as np
from pyrichlet import mixture_models as mm

from ._base import BaseTest


class TestMixtureModels(BaseTest):
    def test_dirichlet_distribution(self):
        n = 2
        mixture = mm.DirichletDistributionMixture(n=n, rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)

    def test_dirichlet_process(self):
        mixture = mm.DirichletProcessMixture(rng=self.rng)
        mixture.fit_gibbs(self.y, init_groups=2)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)

    def test_pitman_yor_process(self):
        mixture = mm.PitmanYorMixture(rng=self.rng)
        mixture.fit_gibbs(self.y, init_groups=2)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)

    def test_geometric_process(self):
        mixture = mm.GeometricProcessMixture(rng=self.rng)
        mixture.fit_gibbs(self.y, init_groups=2)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)

    def test_beta_in_dirichlet(self):
        mixture = mm.BetaInDirichletMixture(rng=self.rng)
        mixture.fit_gibbs(self.y, init_groups=2)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)

    def test_beta_in_beta(self):
        mixture = mm.BetaInBetaMixture(rng=self.rng)
        mixture.fit_gibbs(self.y, init_groups=2)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)

    def test_beta_bernoulli(self):
        mixture = mm.BetaBernoulliMixture(rng=self.rng)
        mixture.fit_gibbs(self.y, init_groups=2)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)

    def test_beta_binomial(self):
        mixture = mm.BetaBinomialMixture(rng=self.rng)
        mixture.fit_gibbs(self.y, init_groups=2)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)

    def test_equal_weighting(self):
        n = 2
        mixture = mm.EqualWeightedMixture(n=n, rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)

    def test_frequency_weighting(self):
        n = 2
        mixture = mm.FrequencyWeightedMixture(n=n, rng=self.rng)
        mixture.fit_gibbs(self.y)
        fitted_density = mixture.gibbs_map_density()
        mean_squared_error = np.power(fitted_density - self.y_density,
                                      2).mean()
        self.assertAlmostEqual(mean_squared_error, 0, places=1)
