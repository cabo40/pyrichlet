from ._base import BaseGaussianMixture
from ..weight_models import GeometricProcess
import numpy as np


class GPMixture(BaseGaussianMixture):
    def __init__(self, *, a=1, b=1, mu_prior=None,
                 lambda_prior=1, psi_prior=None, nu_prior=None, total_iter=1000,
                 burn_in=100, subsample_steps=1, rng=None):
        weight_model = GeometricProcess(a=a, b=b, rng=rng)
        super().__init__(weight_model=weight_model, mu_prior=mu_prior,
                         lambda_prior=lambda_prior, psi_prior=psi_prior,
                         nu_prior=nu_prior, total_iter=total_iter,
                         burn_in=burn_in, subsample_steps=subsample_steps,
                         rng=rng)

    def _save_params(self):
        self.sim_params.append({"w": self.weight_model.get_weights(),
                                "mu": self.mu,
                                "sigma": self.sigma,
                                "u": self.u,
                                "d": self.d,
                                "p": self.weight_model.get_p()})
        self.n_groups.append(len(np.unique(self.d)))
        self.n_atoms.append(len(self.mu))
