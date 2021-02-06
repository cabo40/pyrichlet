from abc import ABCMeta
import numpy as np
from tqdm import trange
from scipy.stats import multivariate_normal
from . import _utils


class BaseGaussianMixture(metaclass=ABCMeta):
    def __init__(self, weight_model=None, mu_prior=None, lambda_prior=1,
                 psi_prior=None, nu_prior=None, total_iter=1000, burn_in=100,
                 subsample_steps=1, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
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

        self.map_sim_params = None
        self.map_log_likelihood = -np.inf
        self.sim_params = []
        self.n_groups = []
        self.n_atoms = []
        self.n_log_likelihood = []

    def fit(self, y, warm_start=False):
        self.y = y

        if self.mu_prior is None:
            self.mu_prior = self.y.mean(axis=0)
        if self.psi_prior is None:
            self.psi_prior = np.atleast_2d(np.cov(self.y.T))
        if self.nu_prior is None:
            _, self.nu_prior = self.y.shape

        if not warm_start:
            self.sim_params = []
            self.n_groups = []
            self.n_atoms = []

            self.u = self.rng.uniform(0 + np.finfo(np.float64).eps, 1, len(self.y))

            self.mu, self.sigma = _utils.random_normal_invw(
                mu=self.mu_prior,
                lam=self.lambda_prior,
                psi=self.psi_prior,
                nu=self.nu_prior,
                rng=self.rng)
            self.mu = np.array([self.mu])
            self.sigma = np.array([self.sigma])

            self.weight_model.tail(1 - min(self.u))
            self.d = self.weight_model.random_assignment(len(self.y))
            self._complete_atoms()
        self._train()

    def _update_weights(self):
        self.weight_model.fit(self.d)
        w = self.weight_model.random()
        self.u = self.rng.uniform(0 + np.finfo(np.float64).eps,
                                  w[self.d] + np.finfo(np.float64).eps)
        self.weight_model.tail(1 - min(self.u))

    def eap_density(self, y, periods=None):
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

    def map_density(self, y):
        return _utils.mixture_density(y,
                                      self.map_sim_params["w"],
                                      self.map_sim_params["mu"],
                                      self.map_sim_params["sigma"],
                                      self.map_sim_params["u"])

    def map_cluster(self, y):
        return _utils.cluster(y,
                              self.map_sim_params["w"],
                              self.map_sim_params["mu"],
                              self.map_sim_params["sigma"],
                              self.map_sim_params["u"])

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

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

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
            self.mu = np.concatenate((self.mu, [temp_mu]))
            self.sigma = np.concatenate((self.sigma, [temp_sigma]))

    def _update_d(self):
        logproba = self._d_log_likelihood_vector()
        self.d = _utils.gumbel_max_sampling(logproba, rng=self.rng)

    def _train(self):
        print("Starting burn-in.")
        for _ in trange(self.burn_in):
            self._gibbs_step()
        print("Finished burn-in.")
        print("Starting training.")
        for i in trange(self.total_iter - self.burn_in):
            self._gibbs_step()
            if i % self.subsample_steps == 0:
                self._save_params()
        print("Finished training.")

    def _gibbs_step(self):
        self._update_atoms()
        self._update_weights()
        self._complete_atoms()
        self._update_d()

    def _d_log_likelihood_vector(self):
        with np.errstate(divide='ignore'):
            logproba = np.array([multivariate_normal.logpdf(self.y,
                                                            self.mu[j],
                                                            self.sigma[j],
                                                            1)
                                 for j in range(self.weight_model.get_size())])
            logproba += np.log(np.greater.outer(self.weight_model.get_weights(),
                                                self.u))
        return logproba

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
        ret_log_likelihood = self._y_log_likelihood()
        # Bellow is commented out since d's distribution is discrete uniform
        # ret_log_likelihood += np.sum(self._d_log_likelihood_vector()[self.d])
        return ret_log_likelihood

    def _full_log_likelihood(self):
        ret = self._mixture_log_likelihood()
        ret += self.weight_model.structure_log_likelihood()
        return ret
