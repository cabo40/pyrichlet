import numpy as np
from pyrichlet import weight_models as wm

from ._base import BaseTest


class TestWeightModels(BaseTest):
    n_sims = int(1e4)

    def test_dirichlet_distribution(self):
        n = 5
        weight_structure = wm.DirichletDistribution(n=n, rng=self.rng)
        w = np.array(
            [weight_structure.random()[:2] for _ in range(self.n_sims)])
        self.assertAlmostEqual(w[:, 0].mean(), 1 / n, places=1)
        self.assertAlmostEqual(w[:, 0].var(0), (n - 1) / n ** 2 / (n + 1),
                               places=1)
        self.assertAlmostEqual(np.corrcoef(w.T)[0, 1], -1 / (n - 1), places=1)

    def test_dirichlet_process(self):
        n = 2
        weight_structure = wm.DirichletProcess(rng=self.rng)
        w = np.array([weight_structure.random(n) for _ in range(self.n_sims)])
        self.assertAlmostEqual(w[:, 0].mean(), 1 / 2, places=1)
        cov_matrix = np.cov(w.T)
        self.assertAlmostEqual(cov_matrix[0, 0], 1 / 12, places=1)
        self.assertAlmostEqual(cov_matrix[0, 1], -1 / 24, places=1)

    def test_pitman_yor_process(self):
        n = 2
        weight_structure = wm.PitmanYorProcess(rng=self.rng)
        w = np.array([weight_structure.random(n) for _ in range(self.n_sims)])
        self.assertAlmostEqual(w[:, 0].mean(), 1 / 2, places=1)
        cov_matrix = np.cov(w.T)
        self.assertAlmostEqual(cov_matrix[0, 0], 1 / 12, places=1)
        self.assertAlmostEqual(cov_matrix[0, 1], -1 / 24, places=1)

    def test_geometric_process(self):
        n = 2
        weight_structure = wm.GeometricProcess(rng=self.rng)
        w = np.array([weight_structure.random(n) for _ in range(self.n_sims)])
        self.assertAlmostEqual(w[:, 0].mean(), 1 / 2, places=1)
        cov_matrix = np.cov(w.T)
        self.assertAlmostEqual(cov_matrix[0, 0], 1 / 12, places=1)
        self.assertAlmostEqual(cov_matrix[0, 1], 0, places=1)

    def test_beta_in_dirichlet(self):
        n = 2
        weight_structure = wm.BetaInDirichlet(rng=self.rng)
        w = np.array([weight_structure.random(n) for _ in range(self.n_sims)])
        self.assertAlmostEqual(w[:, 0].mean(), 1 / 2, places=1)
        cov_matrix = np.cov(w.T)
        self.assertAlmostEqual(cov_matrix[0, 0], 1 / 12, places=1)
        self.assertAlmostEqual(cov_matrix[0, 1], 0, places=1)

    def test_beta_in_beta(self):
        n = 2
        weight_structure = wm.BetaInBeta(rng=self.rng)
        w = np.array([weight_structure.random(n) for _ in range(self.n_sims)])
        self.assertAlmostEqual(w[:, 0].mean(), 1 / 2, places=1)
        cov_matrix = np.cov(w.T)
        self.assertAlmostEqual(cov_matrix[0, 0], 1 / 12, places=1)
        self.assertAlmostEqual(cov_matrix[0, 1], - 1 / 24, places=1)

    def test_beta_in_bernoulli(self):
        n = 2
        weight_structure = wm.BetaBernoulli(rng=self.rng)
        w = np.array([weight_structure.random(n) for _ in range(self.n_sims)])
        self.assertAlmostEqual(w[:, 0].mean(), 1 / 2, places=1)
        cov_matrix = np.cov(w.T)
        self.assertAlmostEqual(cov_matrix[0, 0], 1 / 12, places=1)
        self.assertAlmostEqual(cov_matrix[0, 1], - 1 / 24, places=1)

    def test_beta_binomial(self):
        n = 2
        weight_structure = wm.BetaBinomial(rng=self.rng)
        w = np.array([weight_structure.random(n) for _ in range(self.n_sims)])
        self.assertAlmostEqual(w[:, 0].mean(), 1 / 2, places=1)
        cov_matrix = np.cov(w.T)
        self.assertAlmostEqual(cov_matrix[0, 0], 1 / 12, places=1)
        self.assertAlmostEqual(cov_matrix[0, 1], - 1 / 24, places=1)

    def test_equal_weighting(self):
        n = 100
        weight_structure = wm.EqualWeighting(n=n, rng=self.rng)
        w = weight_structure.random()
        self.assertAlmostEqual(w.var(), 0)
        self.assertAlmostEqual(w[0], 1 / n)

    def test_frequency_weighting(self):
        n = 100
        weight_structure = wm.EqualWeighting(n=n, rng=self.rng)
        w = weight_structure.random()
        self.assertAlmostEqual(w.var(), 0)
        self.assertAlmostEqual(w[0], 1 / n)
