from ._base import BaseGaussianMixture
from ..weight_models import BetaBernoulli

import numpy as np


class BetaBernoulliMixture(BaseGaussianMixture):
    def __init__(self, *, p=0.5, theta=1, rng=None, **kwargs):
        weight_model = BetaBernoulli(p=p, alpha=theta, rng=rng)
        super().__init__(weight_model=weight_model, rng=rng, **kwargs)

    def _get_run_params(self):
        return {"w": self.weight_model.get_weights(),
                "mu": self.mu,
                "sigma": self.sigma,
                "u": self.u,
                "d": self.d,
                "p": self.weight_model.get_p()}
