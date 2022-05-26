from pyrichlet import mixture_models as mm

from ._base import BaseTest


class TestMixtureModels(BaseTest):
    def test_dirichlet_distribution(self):
        n = 10
        mixture = mm.DirichletDistributionMixture(n=n, rng=self.rng)
        mixture.fit_variational(self.y)
        assert mixture.var_converged

    def test_dirichlet_process(self):
        n = 10
        mixture = mm.DirichletProcessMixture(rng=self.rng)
        mixture.fit_variational(self.y, n_groups=n)
        assert mixture.var_converged

    def test_pitman_yor_process(self):
        n = 10
        mixture = mm.PitmanYorMixture(rng=self.rng)
        mixture.fit_variational(self.y, n_groups=n)
        assert mixture.var_converged

    def test_geometric_process(self):
        n = 10
        mixture = mm.GeometricProcessMixture(rng=self.rng)
        mixture.fit_variational(self.y, n_groups=n)
        assert mixture.var_converged

    def test_equal_weighting(self):
        n = 10
        mixture = mm.EqualWeightedMixture(n=n)
        mixture.fit_variational(self.y)
        assert mixture.var_converged

    def test_frequency_weighting(self):
        n = 10
        mixture = mm.FrequencyWeightedMixture(n=n, rng=self.rng)
        mixture.fit_variational(self.y)
        assert mixture.var_converged
