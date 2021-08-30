from ._base import BaseGaussianMixture
from ..weight_models import FrequencyWeighting


class FrequencyWeightedMixture(BaseGaussianMixture):
    def __init__(self, *, n=1, rng=None, **kwargs):
        weight_model = FrequencyWeighting(n=n, rng=rng)
        self.n = n
        super().__init__(weight_model=weight_model, rng=rng, **kwargs)
