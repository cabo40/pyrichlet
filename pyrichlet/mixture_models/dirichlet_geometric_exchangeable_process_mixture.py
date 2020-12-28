import scipy.stats
import numpy as np
from itertools import repeat

from ._base import BaseGaussianMixture
import scipy.optimize

from ..weight_models import DGEProcess


class DGEPMixture(BaseGaussianMixture):
    def __init__(self, *, x=0.5, a=1, b=1, theta=1, mu_prior=None,
                 lambda_prior=1, psi_prior=None, nu_prior=None,
                 p_method="static", total_iter=1000,
                 burn_in=100, subsample_steps=1, rng=None):
        weight_model = DGEProcess(x=x, a=a, b=b, theta=theta,
                                  p_method=p_method, rng=rng)
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


def _cluster(x, w, mu, sigma, u):
    k = len(w)
    ret = []
    for j in range(k):
        ret.append(scipy.stats.multivariate_normal.pdf(x,
                                                       mu[j],
                                                       sigma[j],
                                                       1))
    ret = np.array(ret).T
    weights = (np.array(list(repeat(u, k))) <
               np.array(list(repeat(w, len(u)))).transpose())

    weights = (weights / weights.sum(0)).sum(1) / len(u)
    ret = ret * weights
    grp = np.argmax(ret, axis=1)
    u_grp, ret = np.unique(grp, return_inverse=True)
    return ret, weights[u_grp], mu[u_grp], sigma[u_grp]
