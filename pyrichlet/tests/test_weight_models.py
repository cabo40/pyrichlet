import unittest
import numpy as np

from pyrichlet import weight_models as wm


class TestWeightModels(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_dirichlet_distribution(self):
        n = 15
        weight_structure = wm.DirichletDistribution(n=n, rng=self.rng)
        w = weight_structure.random()
        self.assertEqual(w.var(), 0.005810073582979499)

    def test_dirichlet_process(self):
        n = 100
        weight_structure = wm.DirichletProcess(rng=self.rng)
        w = weight_structure.random(n)
        self.assertEqual(w.var(), 0.005356752112080659)

    def test_pitman_yor_process(self):
        n = 100
        weight_structure = wm.PitmanYorProcess(rng=self.rng)
        w = weight_structure.random(n)
        self.assertEqual(w.var(), 0.005356752112080659)

    def test_geometric_process(self):
        n = 100
        weight_structure = wm.GeometricProcess(rng=self.rng)
        w = weight_structure.random(n)
        self.assertEqual(w.var(), 0.00543480898073906)

    def test_beta_in_dirichlet(self):
        n = 100
        weight_structure = wm.BetaInDirichlet(rng=self.rng)
        w = weight_structure.random(n)
        self.assertEqual(w.var(), 0.005313877403931953)

    def test_beta_in_beta(self):
        n = 100
        weight_structure = wm.BetaInBeta(rng=self.rng)
        w = weight_structure.random(n)
        self.assertEqual(w.var(), 0.005356752112080659)

    def test_beta_in_bernoulli(self):
        n = 100
        weight_structure = wm.BetaBernoulli(rng=self.rng)
        w = weight_structure.random(n)
        self.assertEqual(w.var(), 0.004949629330580343)

    def test_beta_binomial(self):
        n = 100
        weight_structure = wm.BetaBinomial(rng=self.rng)
        w = weight_structure.random(n)
        self.assertEqual(w.var(), 0.005798198802257161)

    def test_equal_weighting(self):
        n = 15
        weight_structure = wm.EqualWeighting(n=n, rng=self.rng)
        w = weight_structure.random()
        self.assertEqual(w.var(), 0)

    def test_frequency_weighting(self):
        n = 15
        weight_structure = wm.EqualWeighting(n=n, rng=self.rng)
        w = weight_structure.random()
        self.assertEqual(w.var(), 0)
