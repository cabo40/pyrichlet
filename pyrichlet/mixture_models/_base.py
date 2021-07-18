from sklearn.cluster import SpectralClustering
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np

from abc import ABCMeta
from . import _utils
from ..weight_models import BaseWeight


class BaseGaussianMixture(metaclass=ABCMeta):
    """
    Base class for Gaussian Mixture Models

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    weight_model : pyrichlet.weight_model, default=None
        The weighting model for the mixing components
    mu_prior : {array, np.array}, default=None
        The prior centering parameter of the prior normal - inverse Wishart
        distribution. If None, the mean of the observations to fit will be used
    lambda_prior : float, default=1
        The precision parameter of the prior normal - inverse Wishart
        distribution.
    psi_prior : {array, np.array, np.matrix}, default=None
        The inverse scale matrix of the prior normal - inverse Wishart
        distribution. If None, the sample variance-covariance matrix will be
        used.
    nu_prior : float, default=None
        The degrees of freedom of the prior normal - inverse Wishart
        distribution. If None, the dimension of the scale matrix will be used.
    total_iter : int, default=1000
        The total number of steps in the Gibbs sampler algorithm.
    burn_in : int, default=100
        The number of steps in the Gibbs sampler to discard in expected a
        posteriori (EAP) estimations.
    subsample_steps : int, default=1
        The number of steps to draw before saving the realizations. The steps
        between savings will be discarded.
    show_progress : bool, default=False
        Whether to display the progress with tqdm.
     rng: {np.random.Generator, int}, default=None
        The PRNG to use for sampling.

    """

    def __init__(self, weight_model: BaseWeight = None, mu_prior=None,
                 lambda_prior=1, psi_prior=None, nu_prior=None,
                 total_iter=1000, burn_in=100, subsample_steps=1,
                 show_progress=False, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
        elif type(rng) is int:
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng

        assert total_iter > burn_in, (
            "total_iter must be greater than burn_in period")
        self.burn_in = int(burn_in)
        self.total_iter = int(total_iter)
        self.subsample_steps = int(subsample_steps)

        self.mu_prior = mu_prior
        self.lambda_prior = lambda_prior
        self.psi_prior = psi_prior
        self.nu_prior = nu_prior

        self.weight_model = weight_model

        self.y = np.array([])
        self.d = np.array([])
        self.mu = np.array([])
        self.sigma = np.array([[]])
        self.u = np.array([])

        self.affinity_matrix = np.array([])

        self.map_sim_params = None
        self.map_log_likelihood = -np.inf
        self.total_saved_steps = 0
        self.sim_params = []
        self.n_groups = []
        self.n_atoms = []
        self.n_log_likelihood = []
        self.show_progress = show_progress

    def fit_gibbs(self, y, warm_start=False, show_progress=None):
        """
        Fit posterior distribution using Gibbs sampling.

        This method does `self.total_iter` steps of the Gibbs sampler and
        stores the arising variables for a later computation of the expected a
        posteriori of the probability distribution density or of the clusters.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features)
            The input sample.

        warm_start : bool, default=False
            Whether to continue the sampling process from a past run or start
            over. If False, the sampling will start from the prior and saved
            states will be deleted.

        show_progress: bool, default=None
            If show_progress is True, a progress bar from the tqdm library is
            displayed.
        """
        self._initialize_params(y, warm_start=warm_start)
        if show_progress is not None:
            self.show_progress = show_progress
        if self.show_progress:
            print("Starting burn-in.")
            from tqdm import trange
            for _ in trange(self.burn_in):
                self._gibbs_step()
            print("Finished burn-in.")
            print("Starting training.")
            for i in trange(self.total_iter - self.burn_in):
                self._gibbs_step()
                if i % self.subsample_steps == 0:
                    self._save_params()
            print("Finished training.")
        else:
            for _ in range(self.burn_in):
                self._gibbs_step()
            for i in range(self.total_iter - self.burn_in):
                self._gibbs_step()
                if i % self.subsample_steps == 0:
                    self._save_params()

    def fit_variational(self, y, n=1, warm_start=False, show_progress=None,
                        tol=1e-8):
        if show_progress is not None:
            self.show_progress = show_progress
        self._initialize_params(y, max_groups=n, warm_start=warm_start)
        elbo = -np.inf
        elbo_diff = np.inf
        if self.show_progress:
            from tqdm import tqdm
            with tqdm() as t:
                while elbo_diff > tol:
                    self._maximize_variational()
                    prev_elbo = elbo
                    elbo = self._calc_elbo()
                    elbo_diff = abs(prev_elbo - elbo)
                    t.update()
        else:
            while elbo_diff > tol:
                self._maximize_variational()
                prev_elbo = elbo
                elbo = self._calc_elbo()
                elbo_diff = abs(prev_elbo - elbo)

    def _initialize_params(self, y, max_groups=None, warm_start=False):
        """
        Initialize the latent variables

        This method randomly initializes the number of groups, mean and
        variance variables and the assignation vector for a given vector of
        observed variables `y`.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features)
            The input sample.

        max_groups: int, default=None
            Maximum number of groups to assign in the initialization. If None,
            the  number of groups drawn from the weight model is not caped.

        warm_start : bool, default=False
            Whether to continue the sampling process from a past run or start
            over. If False, the sampling will start from the prior and saved
            states will be deleted.
        """
        if isinstance(y, pd.DataFrame):
            self.y = y.to_numpy()
        elif isinstance(y, list):
            self.y = np.array(y)
        elif isinstance(y, np.ndarray):
            self.y = y
        else:
            raise TypeError('type is not valid')

        if self.mu_prior is None:
            self.mu_prior = self.y.mean(axis=0)
        if self.psi_prior is None:
            self.psi_prior = np.atleast_2d(np.cov(self.y.T))
        if self.nu_prior is None:
            self.nu_prior = self.y.shape[1]
        self.mu = self.mu.reshape(0, *self.mu_prior.shape)
        self.sigma = self.sigma.reshape(0, *self.psi_prior.shape)

        if not warm_start:
            self.sim_params = []
            self.n_groups = []
            self.n_atoms = []
            self.total_saved_steps = 0
            self.affinity_matrix = np.zeros((len(self.y), len(self.y)))
            self.u = self.rng.uniform(0 + np.finfo(np.float64).eps, 1,
                                      len(self.y))
            if max_groups is None:
                self.weight_model.tail(1 - min(self.u))
            else:
                self.weight_model.complete(max_groups)
            self.d = self.weight_model.random_assignment(len(self.y))
            self._complete_atoms()

    def _maximize_variational(self):
        raise NotImplemented

    def _calc_elbo(self):
        raise NotImplemented

    def _update_weights(self):
        self.weight_model.fit(self.d)
        w = self.weight_model.random()
        self.u = self.rng.uniform(0 + np.finfo(np.float64).eps,
                                  w[self.d] + np.finfo(np.float64).eps)
        self.weight_model.tail(1 - min(self.u))

    def gibbs_eap_density(self, y=None, periods=None):
        if y is None:
            y = self.y
        y_sim = []
        if periods is None:
            for param in self.sim_params:
                y_sim.append(_utils.mixture_density(y,
                                                    param["w"],
                                                    param["mu"],
                                                    param["sigma"],
                                                    param["u"]))
        else:
            periods = min(periods, len(self.sim_params))
            for param in self.sim_params[-periods:]:
                y_sim.append(_utils.mixture_density(y,
                                                    param["w"],
                                                    param["mu"],
                                                    param["sigma"],
                                                    param["u"]))
        return np.array(y_sim).mean(axis=0)

    def gibbs_eap_affinity_matrix(self, y=None):
        if y is None:
            return self.affinity_matrix / self.total_saved_steps
        affinity_matrix = np.zeros((len(y), len(y)))
        for params in self.sim_params:
            grouping = _utils.cluster(y,
                                      params["w"],
                                      params["mu"],
                                      params["sigma"],
                                      params["u"])[0]
            affinity_matrix += np.equal(grouping, grouping[:, None])
        affinity_matrix /= len(self.sim_params)
        return affinity_matrix

    def gibbs_eap_spectral_consensus_cluster(self, y=None, n_clusters=1):
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        return sc.fit_predict(self.gibbs_eap_affinity_matrix(y))

    def gibbs_map_density(self, y=None):
        if y is None:
            y = self.y
        return _utils.mixture_density(y,
                                      self.map_sim_params["w"],
                                      self.map_sim_params["mu"],
                                      self.map_sim_params["sigma"],
                                      self.map_sim_params["u"])

    def gibbs_map_cluster(self, y=None, full=False):
        if y is None:
            y = self.y
        ret = _utils.cluster(y,
                             self.map_sim_params["w"],
                             self.map_sim_params["mu"],
                             self.map_sim_params["sigma"],
                             self.map_sim_params["u"])
        if not full:
            ret = ret[0]
        return ret

    def get_n_groups(self):
        return self.n_groups

    def get_n_theta(self):
        return self.n_atoms

    def get_sim_params(self):
        return self.sim_params

    def _get_run_params(self):
        return {"w": self.weight_model.get_weights(),
                "mu": self.mu,
                "sigma": self.sigma,
                "u": self.u,
                "d": self.d}

    def _save_params(self):
        self.sim_params.append(self._get_run_params())
        self.n_groups.append(len(np.unique(self.d)))
        self.n_atoms.append(len(self.mu))
        self._update_map_params()
        self.affinity_matrix += np.equal(self.d, self.d[:, None])
        self.total_saved_steps += 1

    def _update_map_params(self):
        run_log_likelihood = self._full_log_likelihood()
        if self.map_log_likelihood < run_log_likelihood:
            self.map_log_likelihood = run_log_likelihood
            self.map_sim_params = self._get_run_params()
        elif self.map_log_likelihood == -np.inf:
            # It's better than nothing
            self.map_sim_params = self._get_run_params()

    def _update_atoms(self):
        assert len(self.mu) == len(self.sigma)
        self.d = np.unique(self.d, return_inverse=True)[1]
        self.mu = []
        self.sigma = []
        for j in range(max(self.d) + 1):
            inj = (self.d == j).nonzero()[0]

            posterior_params = _utils.posterior_norm_invw_params(
                self.y[inj],
                mu=self.mu_prior,
                lam=self.lambda_prior,
                psi=self.psi_prior,
                nu=self.nu_prior)
            temp_mu, temp_sigma = _utils.random_normal_invw(
                mu=posterior_params["mu"],
                lam=posterior_params["lambda"],
                psi=posterior_params["psi"],
                nu=posterior_params["nu"],
                rng=self.rng)
            self.mu.append(temp_mu)
            self.sigma.append(temp_sigma)

        self.mu = np.array(self.mu).reshape(-1, *self.mu_prior.shape)
        self.sigma = np.array(self.sigma).reshape(-1, *self.psi_prior.shape)

    def _complete_atoms(self):
        missing_len = self.weight_model.get_size() - len(self.mu)
        for _ in range(missing_len):
            temp_mu, temp_sigma = _utils.random_normal_invw(
                mu=self.mu_prior,
                lam=self.lambda_prior,
                psi=self.psi_prior,
                nu=self.nu_prior,
                rng=self.rng
            )
            self.mu = np.concatenate((self.mu, [np.atleast_1d(temp_mu)]))
            self.sigma = np.concatenate((self.sigma,
                                         [np.atleast_2d(temp_sigma)]))

    def _update_d(self):
        log_prob = self._d_log_likelihood_vector()
        self.d = _utils.gumbel_max_sampling(log_prob, rng=self.rng)

    def _gibbs_step(self):
        self._update_atoms()
        self._update_weights()
        self._complete_atoms()
        self._update_d()

    def _d_log_likelihood_vector(self):
        with np.errstate(divide='ignore'):
            log_probability = np.array(
                [multivariate_normal.logpdf(self.y,
                                            self.mu[j],
                                            self.sigma[j],
                                            1)
                 for j in range(self.weight_model.get_size())]
            )
            log_probability += np.log(np.greater.outer(
                self.weight_model.get_weights(),
                self.u))
        return log_probability

    def _y_log_likelihood(self):
        ret = 0
        with np.errstate(divide='ignore'):
            for di in np.unique(self.d):
                ret += np.sum(multivariate_normal.logpdf(self.y[self.d == di],
                                                         self.mu[di],
                                                         self.sigma[di],
                                                         1))
        return ret

    def _mixture_log_likelihood(self):
        # TODO should return log likelihood of atoms
        ret_log_likelihood = self._y_log_likelihood()
        # Bellow is commented out since d's distribution is discrete uniform
        # ret_log_likelihood += np.sum(self._d_log_likelihood_vector()[self.d])
        return ret_log_likelihood

    def _full_log_likelihood(self):
        ret = self._mixture_log_likelihood()
        ret += self.weight_model.structure_log_likelihood()
        return ret
