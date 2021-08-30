from ._base import BaseGaussianMixture
from ..weight_models import PitmanYorProcess


class PitmanYorMixture(BaseGaussianMixture):
    def __init__(self, *, alpha=1, pyd=0, rng=None, **kwargs):
        weight_model = PitmanYorProcess(pyd=pyd, alpha=alpha, rng=rng)
        super().__init__(weight_model=weight_model, rng=rng, **kwargs)
