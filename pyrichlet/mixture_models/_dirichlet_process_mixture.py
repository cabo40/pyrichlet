from ._base import BaseGaussianMixture
from ..weight_models import DirichletProcess


class DirichletProcessMixture(BaseGaussianMixture):
    def __init__(self, *, theta=1, rng=None, **kwargs):
        weight_model = DirichletProcess(alpha=theta, rng=rng)
        super().__init__(weight_model=weight_model, rng=rng, **kwargs)
