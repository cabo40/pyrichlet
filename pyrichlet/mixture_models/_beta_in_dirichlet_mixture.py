from ._base import BaseGaussianMixture
from ..weight_models import BetaInDirichlet


class BetaInDirichletMixture(BaseGaussianMixture):
    def __init__(self, *, alpha=1, a=0, rng=None, **kwargs):
        weight_model = BetaInDirichlet(alpha=alpha, a=a, rng=rng)
        super().__init__(weight_model=weight_model, rng=rng, **kwargs)
