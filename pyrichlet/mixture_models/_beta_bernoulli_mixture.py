from ._base import BaseGaussianMixture
from ..weight_models import BetaBernoulli


class BetaBernoulliMixture(BaseGaussianMixture):
    def __init__(self, *, p=1, alpha=1, rng=None, **kwargs):
        weight_model = BetaBernoulli(p=p, alpha=alpha, rng=rng)
        super().__init__(weight_model=weight_model, rng=rng, **kwargs)
