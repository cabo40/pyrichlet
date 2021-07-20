from ._base import BaseGaussianMixture
from ..weight_models import DirichletDistribution


class DirichletDistributionMixture(BaseGaussianMixture):
    def __init__(self, *, n=1, theta=1, rng=None, **kwargs):
        weight_model = DirichletDistribution(n=n, alpha=theta, rng=rng)
        self.n = n
        super().__init__(weight_model=weight_model, rng=rng, **kwargs)
