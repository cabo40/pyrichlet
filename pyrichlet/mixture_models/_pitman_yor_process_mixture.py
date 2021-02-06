from ._base import BaseGaussianMixture
from ..weight_models import PitmanYorProcess


class PitmanYorMixture(BaseGaussianMixture):
    def __init__(self, *, pyd=0, theta=1, mu_prior=None,
                 lambda_prior=1, psi_prior=None, nu_prior=None, total_iter=1000,
                 burn_in=100, subsample_steps=1, rng=None):
        weight_model = PitmanYorProcess(pyd=pyd, theta=theta, rng=rng)
        super().__init__(weight_model=weight_model, mu_prior=mu_prior,
                         lambda_prior=lambda_prior, psi_prior=psi_prior,
                         nu_prior=nu_prior, total_iter=total_iter,
                         burn_in=burn_in, subsample_steps=subsample_steps,
                         rng=rng)
