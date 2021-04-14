from ._base import BaseGaussianMixture
from ..weight_models import GeometricProcess


class GeometricProcessMixture(BaseGaussianMixture):
    def __init__(self, *, a=1, b=1, rng=None, **kwargs):
        weight_model = GeometricProcess(a=a, b=b, rng=rng)
        super().__init__(weight_model=weight_model, rng=rng, **kwargs)

    def _get_run_params(self):
        return {"w": self.weight_model.get_weights(),
                "mu": self.mu,
                "sigma": self.sigma,
                "u": self.u,
                "d": self.d,
                "p": self.weight_model.get_p()}
