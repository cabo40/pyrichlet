from sklearn.cluster import SpectralClustering
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np

from abc import ABCMeta

from . import _utils
from ..exceptions import NotFittedError
from ..weight_models import BaseWeight
from ..utils.functions import density_students_t, density_normal


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

        # Variables used in Gibbs sampler
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

        # Variables used in variational method
        self.var_k = None
        self.var_d = None
        self.var_theta = None

        # Fitting flags
        self.gibbs_fitted = False
        self.var_fitted = False

        self.show_progress = show_progress

    def fit_gibbs(self, y, max_groups=None, warm_start=False,
                  show_progress=None, method="random"):
        """
        Fit posterior distribution using Gibbs sampling.

        This method does `self.total_iter` steps of the Gibbs sampler and
        stores the arising variables for a later computation of the expected a
        posteriori of the probability distribution density or of the clusters.

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

        show_progress: bool, default=None
            If show_progress is True, a progress bar from the tqdm library is
            displayed.

        method: str, default="random"
            "random": does a random initialization based on the prior models
            "kmeans": does a kmeans initialization
            "variational": fits the variational distribution an uses the MAP
                parameters as initialization
        """
        self._initialize_common_params(y)
        if not warm_start:
            self._initialize_gibbs_params(max_groups=max_groups, method=method)
        if show_progress is not None:
            self.show_progress = show_progress
        # Iterate the Gibbs steps with or without tqdm
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
        self.gibbs_fitted = True

    def fit_variational(self, y, n_groups=None, warm_start=False,
                        show_progress=None, tol=1e-8, max_iter=1e3,
                        method='kmeans'):
        if show_progress is not None:
            self.show_progress = show_progress
        self._initialize_common_params(y)
        if not warm_start:
            if n_groups is None:
                if hasattr(self, 'n'):
                    var_k = self.n
                else:
                    raise AttributeError("n_groups must be a positive integer")
            else:
                var_k = n_groups
            self._initialize_variational_params(var_k=var_k, method=method)
        elbo = -np.inf
        elbo_diff = np.inf
        iterations = 0
        if self.show_progress:
            from tqdm import tqdm
            with tqdm() as t:
                while elbo_diff > tol and iterations < max_iter:
                    self._maximize_variational()
                    prev_elbo = elbo
                    elbo = self._calc_elbo()
                    elbo_diff = abs(prev_elbo - elbo)
                    iterations += 1
                    t.update()
        else:
            while elbo_diff > tol and iterations < max_iter:
                self._maximize_variational()
                prev_elbo = elbo
                elbo = self._calc_elbo()
                elbo_diff = abs(prev_elbo - elbo)
                iterations += 1
        self.var_fitted = True

    def gibbs_eap_density(self, y=None, periods=None):
        if not self.gibbs_fitted:
            raise NotFittedError("Object must be fitted with gibbs_fit method")
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

    def gibbs_map_density(self, y=None):
        if not self.gibbs_fitted:
            raise NotFittedError("Object must be fitted with gibbs_fit method")
        if y is None:
            y = self.y
        return _utils.mixture_density(y,
                                      self.map_sim_params["w"],
                                      self.map_sim_params["mu"],
                                      self.map_sim_params["sigma"],
                                      self.map_sim_params["u"])

    def gibbs_eap_affinity_matrix(self, y=None):
        if not self.gibbs_fitted:
            raise NotFittedError("Object must be fitted with gibbs_fit method")
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
        if not self.gibbs_fitted:
            raise NotFittedError("Object must be fitted with gibbs_fit method")
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        return sc.fit_predict(self.gibbs_eap_affinity_matrix(y))

    def gibbs_map_cluster(self, y=None, full=False):
        if not self.gibbs_fitted:
            raise NotFittedError("Object must be fitted with gibbs_fit method")
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

    def var_eap_density(self, y=None):
        if not self.var_fitted:
            raise NotFittedError("Object must be fitted with fit_variational"
                                 " method")
        if y is None:
            _y = self.y
        else:
            if isinstance(y, pd.DataFrame):
                _y = y.to_numpy()
            elif isinstance(y, list):
                _y = np.array(y)
                if _y.ndim == 1:
                    _y = _y.reshape(-1, 1)
            elif isinstance(y, np.ndarray):
                if y.ndim == 1:
                    _y = y.copy()
                    _y = y.reshape(-1, 1)
                else:
                    _y = y
        dim = _y.shape[1]
        f_x = np.zeros(len(_y))
        for j, vt_j in enumerate(self.var_theta):
            v_mu_j, v_lambda_j, v_precision_j, v_scale_j = vt_j
            f_x += density_students_t(
                _y, v_mu_j,
                v_precision_j * (v_scale_j + 1 -
                                 dim) * v_lambda_j / (1 + v_lambda_j),
                v_scale_j + 1 - dim
            ) * self.weight_model.variational_mean_w(j)
        return f_x

    def var_map_density(self, y=None):
        if not self.var_fitted:
            raise NotFittedError("Object must be fitted with fit_variational"
                                 " method")
        if y is None:
            _y = self.y
        else:
            if isinstance(y, pd.DataFrame):
                _y = y.to_numpy()
            elif isinstance(y, list):
                _y = np.array(y)
                if _y.ndim == 1:
                    _y = _y.reshape(-1, 1)
            elif isinstance(y, np.ndarray):
                if y.ndim == 1:
                    _y = y.copy()
                    _y = y.reshape(-1, 1)
                else:
                    _y = y
        dim = _y.shape[1]
        f_x = np.zeros(len(_y))
        for j, vt_j in enumerate(self.var_theta):
            v_mu_j, v_lambda_j, v_precision_j, v_scale_j = vt_j
            map_mu = v_mu_j
            map_precision = (v_scale_j - dim) * v_precision_j
            f_x += density_normal(_y, map_mu, map_precision
                                  ) * self.weight_model.variational_mode_w(j)
        return f_x

    def var_map_cluster(self, y=None, full=False):
        if not self.var_fitted:
            raise NotFittedError("Object must be fitted with fit_variational"
                                 " method")
        if y is None:
            if not full:
                return self.var_d.argmax(0)
            else:
                d = self.var_d.argmax(0)
                return d, 1-self.var_d[d, range(len(d))]
        if isinstance(y, pd.DataFrame):
            _y = y.to_numpy()
        elif isinstance(y, list):
            _y = np.array(y)
            if _y.ndim == 1:
                _y = _y.reshape(-1, 1)
        else:
            _y = y
        dim = _y.shape[1]
        var_d = np.zeros((self.var_k, _y.shape[0]), dtype=np.float64)
        for j, vt_j in enumerate(self.var_theta):
            v_mu_j, v_lambda_j, v_precision_j, v_scale_j = vt_j
            log_d_ji = self.weight_model.variational_mean_log_w_j(j)
            log_d_ji += _utils.e_log_norm_wishart(v_precision_j, v_scale_j) / 2
            log_d_ji -= dim / (2 * v_lambda_j)
            log_d_ji -= (v_scale_j / 2 *
                         ((_y - v_mu_j).T * (
                                 v_precision_j @ (_y - v_mu_j).T)).sum(0)
                         )
            var_d[j, :] = log_d_ji
        var_d -= var_d.mean(0)
        var_d = np.exp(var_d)
        var_d += np.finfo(np.float64).eps
        var_d /= var_d.sum(0)
        if not full:
            return var_d.argmax(0)
        else:
            d = var_d.argmax(0)
            return d, var_d[d]

    def get_n_groups(self):
        return self.n_groups

    def get_n_theta(self):
        return self.n_atoms

    def get_sim_params(self):
        return self.sim_params

    def _initialize_common_params(self, y):
        """
        Initialize the prior variables if not given
        """
        if isinstance(y, (pd.DataFrame, pd.Series)):
            self.y = y.to_numpy()
        elif isinstance(y, list):
            self.y = np.array(y)
        elif isinstance(y, np.ndarray):
            self.y = y
        else:
            raise TypeError('type is not valid')
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)
        if self.mu_prior is None:
            self.mu_prior = self.y.mean(axis=0)
        else:
            self.mu_prior = np.array(self.mu_prior)
        if self.psi_prior is None:
            self.psi_prior = np.atleast_2d(np.cov(self.y.T))
        else:
            self.psi_prior = np.atleast_2d(self.psi_prior)
        if self.nu_prior is None:
            self.nu_prior = self.y.shape[1]

    def _initialize_gibbs_params(self, max_groups=None, method="random"):
        """
        Initialize the Gibbs sampler latent variables

        This method randomly initializes the number of groups, mean and
        variance variables and the assignation vector.

        Parameters
        ----------
        max_groups: int, default=None
            Maximum number of groups to assign in the initialization. If None,
            the  number of groups drawn from the weight model is not caped.
        method: str, default="random"
            "kmeans": does a kmeans initialization
            "random": does a random initialization based on the prior models
            "variational": fits the variational distribution an uses the MAP
                parameters as initialization
        """
        self.mu = np.empty((0, *self.mu_prior.shape))
        self.sigma = np.empty((0, *self.psi_prior.shape))
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

        if method == "kmeans":
            from sklearn.cluster import KMeans
            # TODO wait for sklearn to implement Generator as random_state
            # input https://github.com/scikit-learn/scikit-learn/issues/16988
            km = KMeans(
                n_clusters=self.var_k,
                random_state=np.random.RandomState(self.rng.bit_generator)
            )
            self.d = km.fit_predict(self.y)
        elif method == "random":
            self.d = self.weight_model.random_assignment(len(self.y))
        elif method == "variational":
            self.fit_variational(self.y,
                                 n_groups=self.weight_model.get_size())
            self.d = self.var_map_cluster()
        else:
            raise AttributeError("method param must be one of 'kmeans', "
                                 "'random', 'variational'")
        self._complete_atoms()

    def _initialize_variational_params(self, var_k, method="kmeans"):
        """
        Initialize the variational parameters for the variational distributions

        This method randomly initializes the parameters for the assignation
        vector distribution, assigns the variational Normal-Wishart parameters
        and fits the weight_model.

        Parameters
        ----------
        var_k: int
            Maximum number of groups to assign in the initialization. If None,
            the  number of groups drawn from the weight model is not caped.

        method: str
            "kmeans": initialize variational parameters using k-means algorithm
            "random": initialize variational parameters using a random
                assignment
        """
        self.var_k = var_k
        self.var_theta = []

        if method == "kmeans":
            from sklearn.cluster import KMeans
            # TODO wait for sklearn to implement Generator as random_state
            # input https://github.com/scikit-learn/scikit-learn/issues/16988
            km = KMeans(
                n_clusters=self.var_k,
                random_state=np.random.RandomState(self.rng.bit_generator)
            )
            d = km.fit_predict(self.y)
            dim = self.y.shape[1]
            var_d = np.zeros((self.var_k, self.y.shape[0]),
                             dtype=np.float64)
            for j in range(self.var_k):
                y_subset = self.y[d == j]
                if len(y_subset) == 0:
                    var_d[j, :] = np.finfo(np.float64).eps
                    continue
                mu_j = np.mean(y_subset, 0)
                precisions_j = np.linalg.inv(
                    np.atleast_2d(np.cov(y_subset.T))
                )
                mu_j = np.atleast_1d(mu_j)
                precisions_j = np.atleast_2d(precisions_j)
                # atoms initialization
                self.var_theta.append([mu_j, self.lambda_prior,
                                       precisions_j, self.nu_prior])
                # assignations initialization
                log_d_ji = _utils.e_log_norm_wishart(precisions_j,
                                                     self.nu_prior) / 2
                log_d_ji -= dim / (2 * self.lambda_prior)
                log_d_ji -= (self.nu_prior / 2 *
                             ((self.y - mu_j).T * (precisions_j @ (
                                     self.y - mu_j).T)).sum(0)
                             )
                var_d[j, :] = log_d_ji
            var_d -= var_d.max(0)
            var_d = np.exp(var_d)
            var_d += np.finfo(np.float64).eps
            var_d /= var_d.sum(0)
            self.var_d = var_d
        elif method == "random":
            for _ in range(self.var_k):
                mu_j, temp_psi = _utils.random_normal_invw(
                    mu=self.mu_prior,
                    lam=self.lambda_prior,
                    psi=self.psi_prior,
                    nu=self.nu_prior,
                    rng=self.rng
                )
                mu_j = np.atleast_1d(mu_j)
                temp_psi = np.atleast_2d(temp_psi)
                self.var_theta.append([mu_j, self.lambda_prior,
                                       np.linalg.inv(temp_psi), self.nu_prior])
            self.var_d = np.tile(1 / self.var_k, (self.var_k, self.y.shape[0]))
        else:
            raise AttributeError("method param must be one of 'kmeans', "
                                 "'random'")
        self.weight_model.fit_variational(self.var_d)

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
        run_log_likelihood = self._y_log_likelihood()
        if self.map_log_likelihood < run_log_likelihood:
            self.map_log_likelihood = run_log_likelihood
            self.map_sim_params = self._get_run_params()
        elif self.map_log_likelihood == -np.inf:
            # It's better than nothing
            self.map_sim_params = self._get_run_params()

    def _update_weights(self):
        self.weight_model.fit(self.d)
        w = self.weight_model.random()
        self.u = self.rng.uniform(0 + np.finfo(np.float64).eps,
                                  w[self.d] + np.finfo(np.float64).eps)
        self.weight_model.tail(1 - min(self.u))

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

    def _maximize_variational(self):
        self._update_var_d()
        self._update_var_w()
        self._update_var_theta()

    def _calc_elbo(self):
        ret = 0
        ret += self._e_q_log_p_x()
        ret += self._e_q_log_p_d__w()
        ret += self._e_log_p_w()
        ret += self._e_log_p_theta()
        ret -= self._e_log_q_d()
        ret -= self._e_loq_q_w()
        ret -= self._e_log_q_theta()
        return ret

    def _update_var_w(self):
        self.weight_model.fit_variational(self.var_d)

    def _update_var_theta(self):
        var_theta = []
        for vd_j in self.var_d:
            n_j = vd_j.sum()
            x_bar_j = (vd_j / n_j) @ self.y
            ns_j = (vd_j * (self.y - x_bar_j).T) @ (self.y - x_bar_j)
            v_lambda_j = self.lambda_prior + n_j
            v_scale_j = self.nu_prior + n_j
            v_mu_j = (self.lambda_prior * self.mu_prior +
                      n_j * x_bar_j) / v_lambda_j
            v_precision_j = (self.psi_prior + ns_j +
                             self.lambda_prior * n_j / (self.lambda_prior +
                                                        n_j) *
                             (x_bar_j - self.mu_prior) @ (x_bar_j -
                                                          self.mu_prior))
            v_precision_j = np.linalg.inv(v_precision_j)
            var_theta.append([v_mu_j, v_lambda_j, v_precision_j, v_scale_j])
        self.var_theta = var_theta

    def _update_var_d(self):
        dim = self.y.shape[1]
        var_d = np.zeros((self.var_k, self.y.shape[0]), dtype=np.float64)
        for j, vt_j in enumerate(self.var_theta):
            v_mu_j, v_lambda_j, v_precision_j, v_scale_j = vt_j
            log_d_ji = self.weight_model.variational_mean_log_w_j(j)
            log_d_ji += _utils.e_log_norm_wishart(v_precision_j, v_scale_j) / 2
            log_d_ji -= dim / (2 * v_lambda_j)
            log_d_ji -= (v_scale_j / 2 *
                         ((self.y - v_mu_j).T * (
                                 v_precision_j @ (self.y - v_mu_j).T)).sum(0)
                         )
            var_d[j, :] = log_d_ji
        var_d -= var_d.max(0)
        var_d = np.exp(var_d)
        var_d += np.finfo(np.float64).eps
        var_d /= var_d.sum(0)
        self.var_d = var_d

    def _e_q_log_p_x(self):
        dim = self.y.shape[1]
        res = 0
        for vd_j, vt_j in zip(self.var_d, self.var_theta):
            n_j = vd_j.sum()
            x_bar_j = (vd_j / n_j) @ self.y
            s_j = (vd_j / n_j * (self.y - x_bar_j).T) @ (self.y - x_bar_j)
            v_mu_j, v_lambda_j, v_precision_j, v_scale_j = vt_j
            res += _utils.e_log_norm_wishart(v_precision_j, v_scale_j)
            res -= dim / v_lambda_j
            res -= v_scale_j * np.einsum('ij,ji->', s_j, v_precision_j)
            res -= (v_scale_j *
                    (x_bar_j - v_mu_j) @ v_precision_j @ (x_bar_j - v_mu_j)
                    )
        res -= dim * np.log(2 * np.pi) * self.var_k
        res /= 2
        return res

    def _e_q_log_p_d__w(self):
        return self.weight_model.variational_mean_log_p_d__w(
            variational_d=self.var_d
        )

    def _e_log_p_w(self):
        return self.weight_model.variational_mean_log_p_w()

    def _e_log_p_theta(self):
        dim = self.y.shape[1]
        res = 0
        for vt_j in self.var_theta:
            v_mu_j, v_lambda_j, v_precision_j, v_scale_j = vt_j
            res += (self.nu_prior -
                    dim) * _utils.e_log_norm_wishart(v_precision_j, v_scale_j)
            res -= dim * self.lambda_prior / v_lambda_j
            res -= v_scale_j * np.einsum('ij,ji->',
                                         self.psi_prior, v_precision_j)
            res -= (self.lambda_prior * v_scale_j *
                    (v_mu_j - self.mu_prior) @ v_precision_j @ (v_mu_j -
                                                                self.mu_prior)
                    )
        res += dim * (np.log(self.lambda_prior / (2 * np.pi))) * self.var_k
        res /= 2
        res += self.var_k * _utils.log_wishart_normalization_term(
            np.linalg.inv(self.psi_prior), self.nu_prior
        )
        return res

    def _e_log_q_d(self):
        return np.sum(self.var_d * np.log(self.var_d))

    def _e_loq_q_w(self):
        return self.weight_model.variational_mean_log_p_w()

    def _e_log_q_theta(self):
        dim = self.y.shape[1]
        res = 0
        for vt_j in self.var_theta:
            _, v_lambda_j, v_precision_j, v_scale_j = vt_j
            res += _utils.e_log_norm_wishart(v_precision_j, v_scale_j)
            res += dim * (np.log(v_lambda_j / (2 * np.pi)))
            res -= _utils.entropy_wishart(v_precision_j, v_scale_j)
        res += dim * self.var_k
        res /= 2
        return res
