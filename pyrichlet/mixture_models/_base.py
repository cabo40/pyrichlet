from sklearn.cluster import SpectralClustering
from scipy.stats import multivariate_normal
from collections import defaultdict
import pandas as pd
import numpy as np

from abc import ABCMeta

from . import _utils
from ..exceptions import NotFittedError
from ..weight_models import BaseWeight
from ..utils.functions import density_students_t, density_normal
from ..utils.validators import rng_parser


class BaseGaussianMixture(metaclass=ABCMeta):
    """
    Base class for Gaussian Mixture Models

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    weight_model : BaseWeight, default=None
        The weighting model for the mixing components
    mu_prior : {float, array, np.array}, default=None
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
        The total number of steps in the Gibbs sampler algorithm or the max
        number of steps for the variational algorithm.
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

    def __init__(self, weight_model: BaseWeight, mu_prior=None,
                 lambda_prior=1, psi_prior=None, nu_prior=None,
                 total_iter=1000, burn_in=100, subsample_steps=1,
                 show_progress=False, rng=None):

        assert total_iter > burn_in, (
            "total_iter must be greater than burn_in period")
        self.rng = rng_parser(rng)
        self.burn_in = int(burn_in)
        self.total_iter = int(total_iter)
        self.subsample_steps = int(subsample_steps)

        self.mu_prior = mu_prior
        self.lambda_prior = lambda_prior
        self.psi_prior = psi_prior
        self.nu_prior = nu_prior

        self.weight_model = weight_model
        self.y = np.array([])
        self._column_names = None

        # Variables used in Gibbs sampler
        self.d = np.array([])
        self.theta = {}
        self.u = np.array([])
        self.affinity_matrix = np.array([])
        self.map_sim_params = None
        self.map_log_likelihood = -np.inf
        self.total_saved_steps = 0
        self.sim_params = []
        self.n_groups = []
        self.n_atoms = []
        self.n_log_likelihood = []

        # Extra variables used in variational methods
        self.var_k = None
        self.var_d = None
        self.var_theta = None

        # Fitting flags
        self.gibbs_fitted = False
        self.var_fitted = False
        self.var_converged = False

        self.show_progress = show_progress

    def fit_gibbs(self, y, init_groups=None, warm_start=False,
                  show_progress=None, init_method="kmeans"):
        """
        Fit posterior distribution using Gibbs sampling.

        This method does `self.total_iter` steps of the Gibbs sampler and
        stores the arising variables for a later computation of the expected a
        posteriori of the probability distribution density or of the clusters.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features)
            The input sample.

        init_groups: int, default=None
            Maximum number of groups to assign in the initialization. If None,
            the  initial number of groups is drawn from the attribute n.

        warm_start : bool, default=False
            Whether to continue the sampling process from a past run or start
            over. If False, the sampling will start from the prior and saved
            states will be deleted.

        show_progress: bool, default=None
            If show_progress is True, a progress bar from the tqdm library is
            displayed.

        init_method: str, default="random"
            "random": does a random initialization based on the prior models
            "kmeans": does a kmeans initialization
            "variational": fits the variational distribution an uses the MAP
            parameters as initialization
        """
        self._initialize_common_params(y)
        if not warm_start:
            self._initialize_gibbs_params(init_groups=init_groups,
                                          method=init_method)
            self._update_map_params()
        if show_progress is not None:
            self.show_progress = show_progress

        # Iterate the Gibbs steps with or without tqdm
        burn_in_iterator = range(self.burn_in)
        range_iterator = range(self.total_iter - self.burn_in)
        if self.show_progress:
            from tqdm import tqdm
            burn_in_iterator = tqdm(burn_in_iterator)
            range_iterator = tqdm(range_iterator)
            print("Starting burn-in.")
        for _ in burn_in_iterator:
            self._gibbs_step()
            self._update_map_params()
        if self.show_progress:
            print("Finished burn-in.")
            print("Starting training.")
        for i in range_iterator:
            self._gibbs_step()
            self._update_map_params()
            if i % self.subsample_steps == 0:
                self._save_params()
        if self.show_progress:
            print("Finished training.")
        self.gibbs_fitted = True

    def fit_variational(self, y, n_groups=None, warm_start=False,
                        show_progress=None, tol=1e-8, init_method='kmeans'):
        """
        Fit posterior variational distribution using mean field theory.

        This method does up to `self.total_iter` steps of the gradient descent
        algorithm to fit the variational distributions of weights, atoms and
        assignations of the mixture.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features)
            The input sample.

        n_groups : int, default=None
            The number of groups of the truncated variational distribution.
            If None, the  number of groups will be deduced from the weighting
            structure if possible.

        warm_start : bool, default=False
            Whether to continue the sampling process from a past run or start
            over. If False, the sampling will start from the prior parameters
            and any previous calculations will be discarded.

        show_progress : bool, default=None
            If show_progress is True, a progress bar from the tqdm library is
            displayed.

        tol: float, default=1e-8
            The tolerance of change in the evidence lower bound (ELBO) between
            iterations. The process finishes when the change is less than
            `tol`.

        init_method : str, default="kmeans"
            "kmeans": initialize variational parameters using k-means algorithm
            "random": initialize variational parameters using a random
                assignment
        """
        if show_progress is not None:
            self.show_progress = show_progress
        self._initialize_common_params(y)
        if not warm_start:
            if hasattr(self, 'n') and n_groups is None:
                var_k = self.n
            elif n_groups is None:
                raise AttributeError("n_groups must be a positive integer")
            else:
                var_k = n_groups
            self._initialize_variational_params(var_k=var_k,
                                                init_method=init_method)
        elbo = -np.inf
        elbo_diff = np.inf
        iterations = 0
        t = None
        if self.show_progress:
            from tqdm import tqdm
            t = tqdm()
        while elbo_diff > tol and iterations < self.total_iter:
            self._maximize_variational()
            prev_elbo = elbo
            elbo = self._calc_elbo()
            elbo_diff = abs(prev_elbo - elbo)
            iterations += 1
            if t is not None:
                t.update()
        self.var_fitted = True
        if iterations < self.total_iter:
            self.var_converged = True

    def gibbs_eap_density(self, y=None, dim=None, component=None,
                          periods=None):
        """
        Returns the (Gibbs fitted) expected a posteriori density at y

        This method must be called after fitting a dataset with `fit_gibbs`.
        It returns the density at `y` as defined by the average of the mixture
        at every saved Gibbs step.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The data points over which to evaluate the EAP density. If `None`
            the data used at fitting is used.
        dim: int, {array-like} default=None
            The desired dimension index for which to marginalize the density,
            if None, all dimensions are used.
        component: int default=None
            Only returns the scaled density for a particular component.
        periods : int, default=None
            The number of saved periods to use counting backwards from the
            last Gibbs step. If `None`, all saved periods are used.
        """
        if not self.gibbs_fitted:
            raise NotFittedError("Object must be fitted with fit_gibbs method")
        _y = self._cast_observations(y)
        y_sim = []
        if periods is None:
            sim_params = self.sim_params
        else:
            i_start = min(periods, len(self.sim_params))
            sim_params = self.sim_params[-i_start:]
        for param in sim_params:
            y_sim.append(_utils.mixture_density(_y,
                                                param["w"],
                                                param["theta"],
                                                dim=dim,
                                                component=component))
        return np.array(y_sim).mean(axis=0)

    def gibbs_map_density(self, y=None, dim=None, component=None):
        """
        Returns the (Gibbs fitted) maximum a posteriori density at y

        This method must be called after fitting a dataset with `fit_gibbs`.
        It returns the density at `y` as defined by the random mixture within
        the Gibbs steps having the highest likelihood.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The data points over which to evaluate the MAP density. If `None`
            the data used at fitting is used.
        dim: int, {array-like} default=None
            The desired dimension index for which to marginalize the density,
            if None, all dimensions are used.
        component: int default=None
            Only returns the scaled density for a particular component.
        """
        if not self.gibbs_fitted:
            raise NotFittedError("Object must be fitted with fit_gibbs method")
        _y = self._cast_observations(y)
        return _utils.mixture_density(_y,
                                      self.map_sim_params["w"],
                                      self.map_sim_params["theta"],
                                      dim=dim,
                                      component=component)

    def gibbs_eap_affinity_matrix(self, y=None):
        """
        Returns the (Gibbs fitted) affinity matrix for the observations y

        This method must be called after fitting a dataset with `fit_gibbs`.
        It returns an affinity matrix for `y`. The entry (i,j) of the returned
        matrix denotes the proportion of draws where the observation i shared
        the same group as the observation j.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The data points for which to get an affinity matrix. If `None`
            the data used at fitting is used.
        """
        if not self.gibbs_fitted:
            raise NotFittedError(
                "Object must be fitted with the fit_gibbs method")
        if y is None:
            return self.affinity_matrix / self.total_saved_steps
        _y = self._cast_observations(y)
        affinity_matrix = np.zeros((len(_y), len(_y)))
        for params in self.sim_params:
            grouping = _utils.cluster(_y, params["w"], params["theta"])[0]
            affinity_matrix += np.equal(grouping, grouping[:, None])
        affinity_matrix /= len(self.sim_params)
        return affinity_matrix

    def gibbs_eap_spectral_consensus_cluster(self, y=None, n_clusters=1):
        """
        Returns the (Gibbs fitted) expected a posteriori cluster for y

        This method must be called after fitting a dataset with
        `fit_gibbs`.
        It returns the EAP consensus clustering for the observations `y`.
        It uses the spectral clustering algorithm over the EAP affinity matrix
         as consensus algorithm.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The data points to cluster. If `None`
            the data used at fitting is used.
        n_clusters: int, default=1
            The number of clusters to output.
        """
        if not self.gibbs_fitted:
            raise NotFittedError(
                "Object must be fitted with the fit_gibbs method")
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        return sc.fit_predict(self.gibbs_eap_affinity_matrix(y))

    def gibbs_map_cluster(self, y=None, full=False):
        """
        Returns the (Gibbs fitted) maximum a posteriori cluster for y

        This method is called after fitting a dataset with `fit_gibbs`.
        It returns the clustering for `y` using the mixture within the Gibbs
        steps with the greatest likelihood.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The data points to cluster. If `None`
            the data used at fitting is used.
        full: bool, default=False
            if full is false, only a vector with the clustering output is
            returned. If true, a tuple with the clusters and assignation
            uncertainties is returned.
        """
        if not self.gibbs_fitted:
            raise NotFittedError(
                "Object must be fitted with fit_gibbs method")
        _y = self._cast_observations(y)
        ret = _utils.cluster(_y,
                             self.map_sim_params["w"],
                             self.map_sim_params["theta"])
        if not full:
            ret = ret[0]
        return ret

    def gibbs_map_pairplot(self):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from scipy.stats import chi2
        from scipy.linalg import ldl

        names = self._column_names
        n_feats = self.y.shape[1]
        grp = self.gibbs_map_cluster()

        fig, axes = plt.subplots(nrows=n_feats, ncols=n_feats, sharex='col',
                                 figsize=(n_feats * 2, n_feats * 2))
        fig.set_dpi(150)
        alpha = 0.05
        alpha_radius = np.sqrt(chi2.ppf(1 - alpha, 2))
        color = plt.get_cmap('tab10')
        if max(grp) > 10:
            color = color(np.linspace(0, 1, len(np.unique(grp)) + 1))
        else:
            color = color(np.linspace(0, 1, 10))

        # We set labels for each subplot on the left and lower borders
        # and set a shared y-axis for all subplots except those in the diagonal
        for it in range(n_feats):
            ax = axes[-1, it]
            ax.set_xlabel(names[it])
            ax = axes[it, 0]
            ax.set_ylabel(names[it])
            for it2 in range(n_feats - 1):
                if it == it2:
                    continue
                it3 = it2 + 1
                if it == it3:
                    it3 += 1
                if it3 == n_feats:
                    continue
                ax = axes[it, it2]
                ax.sharey(axes[it, it3])

        # Iterates over all subplots and does a density plot over the diagonal
        # and scatters off the diagonal
        for it in range(n_feats):
            for it2 in range(n_feats):
                ax = axes[it2, it]
                ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=4))
                ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=4))
                ax.tick_params(axis='y', labelrotation=90)
                ax.tick_params(labelsize=8)
                ticks = ax.xaxis.get_major_ticks()
                ticks[0].label1.set_visible(False)
                ticks[-1].label1.set_visible(False)
                ticks = ax.yaxis.get_major_ticks()
                ticks[0].label1.set_visible(False)
                ticks[-1].label1.set_visible(False)
                ax.grid(color='lightgray', linestyle='--', alpha=0.5)
                if it == it2:
                    y_min = self.y[:, it].min()
                    y_max = self.y[:, it].max()
                    y_ptp = y_max - y_min
                    y_range = np.linspace(y_min - y_ptp / 4,
                                          y_max + y_ptp / 4,
                                          100)
                    for j in np.unique(grp):
                        dens = self.gibbs_map_density(y_range, dim=it,
                                                      component=j)
                        # ax.plot(y_range, dens, c=color[j])
                        ax.fill_between(y_range, dens, interpolate=True,
                                        color=color[j], alpha=0.5)
                else:
                    ax.scatter(self.y[:, it], self.y[:, it2], s=10,
                               c=color[grp], alpha=0.5)
                    for j in np.unique(grp):
                        mu = self.map_sim_params['theta'][j][0]
                        mu = mu[[it, it2]]
                        sigma = self.map_sim_params['theta'][j][1]
                        sigma = sigma[[it, it2], :][:, [it, it2]]
                        ldl_sigma = ldl(sigma)
                        sig_sq = ldl_sigma[0] @ np.sqrt(
                            ldl_sigma[1]) @ ldl_sigma[0].T
                        circle_points = np.array(
                            [np.cos(np.linspace(0, 2 * np.pi, 100)),
                             np.sin(np.linspace(0, 2 * np.pi,
                                                100))]).T * alpha_radius
                        circle_points = np.matmul(circle_points, sig_sq) + mu
                        ax.plot(circle_points[:, 0], circle_points[:, 1],
                                linestyle='--', c=color[j])
                        # break
            axes[it, it].tick_params(axis='y', colors='white')

    def var_eap_density(self, y=None, dim=None, component=None):
        """
        Returns the expected a posteriori density at y using variational
        inference

        This method is called after fitting a dataset with
        `fit_variational`.
        It returns the density at `y` as described by the fitted variational
        distributions using the expected density at each point.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The points at which to draw the variational EAP density. If `None`
            the data used at fitting is used.
        dim: int, {array-like} default=None
            The desired dimension index for which to marginalize the density,
            if None, all dimensions are used.
        component: int default=None
            Only returns the scaled density for a particular component.
        """
        if not self.var_fitted:
            raise NotFittedError("Object must be fitted with fit_variational"
                                 " method")
        _y = self._cast_observations(y)
        if dim is None:
            dim = np.arange(_y.shape[1])
        if isinstance(dim, int):
            len_dim = 1
        else:
            len_dim = len(dim)
        f_x = np.zeros(len(_y))
        if component is None:
            iterator = enumerate(self.var_theta)
        else:
            iterator = [(component, self.var_theta[component])]
        for j, vt_j in iterator:
            v_mu_j, v_lambda_j, v_precision_j, v_scale_j = vt_j
            v_mu_j = v_mu_j[dim]
            v_precision_j = v_precision_j[:, dim][dim, :]
            f_x += density_students_t(
                _y, v_mu_j,
                v_precision_j * (v_scale_j + 1 -
                                 len_dim) * v_lambda_j / (1 + v_lambda_j),
                v_scale_j + 1 - len_dim
            ) * self.weight_model.variational_mean_w_j(j)
        return f_x

    def var_map_density(self, y=None, dim=None, component=None):
        """
        Returns the maximum a posteriori density at y using variational
        inference

        This method is called after fitting a dataset with `fit_variational`.
        It returns the density at `y` as described by the fitted variational
        distributions using the maximum likelihood density at each point.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The points at which to draw the variational MAP density. If `None`
            the data used at fitting is used.
        """
        if not self.var_fitted:
            raise NotFittedError("Object must be fitted with fit_variational"
                                 " method")
        _y = self._cast_observations(y)
        if isinstance(dim, int):
            len_dim = 1
        else:
            len_dim = len(dim)
        f_x = np.zeros(len(_y))
        if component is None:
            iterator = enumerate(self.var_theta)
        else:
            iterator = [(component, self.var_theta[component])]
        for j, vt_j in iterator:
            v_mu_j, v_lambda_j, v_precision_j, v_scale_j = vt_j
            v_mu_j = v_mu_j[dim]
            v_precision_j = v_precision_j[:, dim][dim, :]
            map_mu = v_mu_j
            map_precision = (v_scale_j - len_dim) * v_precision_j
            f_x += density_normal(_y, map_mu, map_precision
                                  ) * self.weight_model.variational_mode_w_j(j)
        return f_x

    def var_eap_affinity_matrix(self, y=None):
        """
        Returns the (Variational fitted) affinity matrix for the observations y

        This init_method must be called after fitting a dataset with
        `fit_variational`.
        It returns an affinity matrix for `y`. The entry (it,it2) of the
        returned matrix denotes the variational probability of draws in the
        assignation of y[it] and y[it2].

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The data points for which to get an affinity matrix. If `None`
            the data used at fitting is used.
        """
        if not self.var_fitted:
            raise NotFittedError(
                "Object must be fitted with the fit_variational method")
        if y is None:
            var_d = self.var_d
        else:
            _y = self._cast_observations(y)
            dim = y.shape[1]
            var_d = np.zeros((self.var_k, y.shape[0]), dtype=np.float64)
            for j, vt_j in enumerate(self.var_theta):
                v_mu_j, v_lambda_j, v_precision_j, v_scale_j = vt_j
                log_d_ji = self.weight_model.variational_mean_log_w_j(j)
                log_d_ji += _utils.e_log_norm_wishart(v_precision_j,
                                                      v_scale_j) / 2
                log_d_ji -= dim * np.log(2 * np.pi) / 2
                log_d_ji -= dim / v_lambda_j / 2
                log_d_ji -= (v_scale_j *
                             ((y - v_mu_j).T * (
                                     v_precision_j @ (y - v_mu_j).T)).sum(
                                 0)
                             ) / 2
                var_d[j, :] = log_d_ji
            var_d -= var_d.max(axis=0, initial=-np.inf)
            var_d = np.exp(var_d)
            var_d += np.finfo(np.float64).eps
            var_d /= var_d.sum(axis=0)
        affinity_matrix = var_d.T @ var_d
        return affinity_matrix

    def var_eap_spectral_consensus_cluster(self, y=None, n_clusters=1):
        """
        Returns the (Variational fitted) expected a posteriori cluster for y

        This init_method must be called after fitting a dataset with
        `fit_variational`.
        It returns the EAP consensus clustering for the observations `y`.
        It uses the spectral clustering algorithm over the EAP affinity matrix
        as the consensus algorithm.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The data points to cluster. If `None`
            the data used at fitting is used.
        n_clusters: int, default=1
            The number of clusters to output.
        """
        if not self.var_fitted:
            raise NotFittedError(
                "Object must be fitted with fit_variational init_method")
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        return sc.fit_predict(self.var_eap_affinity_matrix(y))

    def var_map_cluster(self, y=None, full=False):
        """
        Returns the maximum a posteriori clustering for y using variational
        inference

        This method is called after fitting a dataset with `fit_variational`.
        It returns a clustering for `y` using the fitted variational
        distributions and the assignations with greater likelihood.

        Parameters
        ----------
        y : {array-like} of shape (n_samples, n_features), default=None
            The points to cluster using the MAP assignations. If `None`
            the data used at fitting is used.
        full: bool, default=False
            If False (default), only the maximum a posteriori clustering is
            returned. If True, the variational assignation probabilty is also
            returned.
        """
        if not self.var_fitted:
            raise NotFittedError("Object must be fitted with fit_variational"
                                 " method")
        if y is None:
            if not full:
                return self.var_d.argmax(0)
            else:
                d = self.var_d.argmax(0)
                return d, 1 - self.var_d[d, range(len(d))]
        _y = self._cast_observations(y)
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
        if isinstance(y, pd.DataFrame):
            self.y = y.to_numpy()
            self._column_names = y.columns
        elif isinstance(y, pd.Series):
            self.y = y.to_numpy()
            if y.name is None:
                self._column_names = np.array([0])
            else:
                self._column_names = np.array([y.name])
        elif isinstance(y, list):
            self.y = np.array(y)
            if self.y.ndim == 1:
                self.y = self.y.reshape(-1, 1)
                self._column_names = np.array([0])
            else:
                self._column_names = np.arange(self.y.shape[1])
        elif isinstance(y, np.ndarray):
            self.y = np.copy(y)
            if self.y.ndim == 1:
                self.y = self.y.reshape(-1, 1)
                self._column_names = np.array([0])
            else:
                self._column_names = np.arange(self.y.shape[1])
        else:
            raise TypeError('Invalid type for variable y')

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

    def _initialize_gibbs_params(self, init_groups=None, method="kmeans"):
        """
        Initialize the Gibbs sampler latent variables

        This method randomly initializes the number of groups, mean and
        variance variables and the assignation vector.

        Parameters
        ----------
        init_groups: int, default=None
            Maximum number of groups to assign in the initialization. If None,
            the  initial number of groups is drawn from the attribute n.
        method: str
            "kmeans": does a kmeans initialization
            "random": does a random initialization based on the prior models
            "variational": fits the variational distribution an uses the MAP
                parameters as initialization
        """

        def atom_generator():
            mu, sigma = _utils.random_normal_invw(
                mu=self.mu_prior,
                lam=self.lambda_prior,
                psi=self.psi_prior,
                nu=self.nu_prior,
                rng=self.rng
            )
            return np.atleast_1d(mu), np.atleast_2d(sigma)

        self.sim_params = []
        self.n_groups = []
        self.n_atoms = []
        self.total_saved_steps = 0

        self.theta = defaultdict(atom_generator)
        self.affinity_matrix = np.zeros((len(self.y), len(self.y)))
        self.u = self.rng.uniform(0 + np.finfo(np.float64).eps, 1,
                                  len(self.y))
        if init_groups is None:
            self.weight_model.tail(1 - min(self.u))
        else:
            self.weight_model.complete(init_groups)

        if method == "kmeans":
            if hasattr(self, 'n'):
                n = self.n
            else:
                if init_groups is not None:
                    n = init_groups
                else:
                    raise ValueError("init_groups must be an integer for"
                                     " k-means initialization")
            self.d = _utils.kmeans_cluster_size_biased(self.y, n, self.rng)
        elif method == "random":
            self.d = self.weight_model.random_assignment(len(self.y))
        elif method == "variational":
            self.fit_variational(self.y,
                                 n_groups=self.weight_model.get_size())
            self.d = self.var_map_cluster()
        else:
            raise AttributeError("init_method param must be one of 'kmeans', "
                                 "'random', 'variational'")

    def _initialize_variational_params(self, var_k, init_method="kmeans"):
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

        init_method: str
            "kmeans": initialize variational parameters using k-means algorithm
            "random": initialize variational parameters using a random
                assignment
        """
        self.var_k = var_k
        self.var_theta = []

        if init_method == "kmeans":
            d = _utils.kmeans_cluster_size_biased(self.y, self.var_k,
                                                  self.rng)
            var_d = np.zeros((self.var_k, self.y.shape[0]),
                             dtype=np.float64)
            var_d[d, range(len(d))] = 1
            self.var_d = var_d
            self.weight_model.fit_variational(np.empty(shape=(self.var_k, 0)))
            self._update_var_theta()
            self._update_var_d()
        elif init_method == "random":
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
            raise AttributeError("init_method param must be one of 'kmeans', "
                                 "'random'")
        self.weight_model.fit_variational(self.var_d)

    def _get_run_params(self):
        return {"w": self.weight_model.get_weights(),
                "theta": dict(self.theta),
                "u": self.u,
                "d": self.d}

    def _save_params(self):
        self.sim_params.append(self._get_run_params())
        self.n_groups.append(len(np.unique(self.d)))
        self.n_atoms.append(len(self.theta))
        self.affinity_matrix += np.equal(self.d, self.d[:, None])
        self.total_saved_steps += 1

    def _update_map_params(self):
        """Calc the likelihood and parameters of the run. Update MAP if the
        likelihood is greater
        """
        run_log_likelihood = self._run_log_likelihood()
        if self.map_log_likelihood < run_log_likelihood:
            self.map_log_likelihood = run_log_likelihood
            self.map_sim_params = self._get_run_params()
        elif self.map_log_likelihood == -np.inf:
            # Save the params to get something to compare
            self.map_sim_params = self._get_run_params()

    def _update_weights(self):
        self.weight_model.fit(self.d)
        w = self.weight_model.random()
        self.u = self.rng.uniform(0 + np.finfo(np.float64).eps,
                                  w[self.d] + np.finfo(np.float64).eps)
        self.weight_model.tail(1 - min(self.u))

    def _update_atoms(self):
        for j in np.unique(self.d):
            mask_j = self.d == j
            posterior_params = _utils.posterior_norm_invw_params(
                self.y[mask_j],
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
            temp_mu = np.atleast_1d(temp_mu)
            temp_sigma = np.atleast_2d(temp_sigma)
            self.theta[j] = (temp_mu, temp_sigma)

    def _update_d(self):
        log_prob = self._d_log_likelihood_vector()
        self.d = _utils.gumbel_max_sampling(log_prob, rng=self.rng)

    def _gibbs_step(self):
        self._update_atoms()
        self._update_weights()
        self._update_d()

    def _d_log_likelihood_vector(self):
        with np.errstate(divide='ignore'):
            log_probability = np.array(
                [multivariate_normal.logpdf(self.y,
                                            self.theta[j][0],
                                            self.theta[j][1],
                                            1)
                 for j in range(self.weight_model.get_size())]
            )
            log_probability += np.log(np.greater.outer(
                self.weight_model.get_weights(),
                self.u))
        return log_probability

    def _run_log_likelihood(self):
        ret = 0
        ret += self._y_log_likelihood()
        ret += self._d_log_likelihood()
        return ret

    def _y_log_likelihood(self):
        """returns the loglikelihood of f(y|d, w, theta)"""
        ret = 0
        with np.errstate(divide='ignore'):
            for j in np.unique(self.d):
                ret += np.sum(multivariate_normal.logpdf(self.y[self.d == j],
                                                         self.theta[j][0],
                                                         self.theta[j][1],
                                                         1))
        return ret

    def _d_log_likelihood(self):
        """returns the loglikelihood of f(d|w)"""
        return self.weight_model.assignation_log_likelihood(self.d)

    def _w_log_likelihood(self):
        """returns the loglikelihood of f(w)"""
        return self.weight_model.weighting_log_likelihood()

    def _theta_log_likelihood(self):
        """returns the loglikelihood of f(theta)"""
        res = 0
        for j in np.unique(self.d):
            mu, sigma = self.theta[j]
            res += _utils.log_likelihood_normal_invw(
                mu=mu,
                sigma=sigma,
                mu0=self.mu_prior,
                lam0=self.lambda_prior,
                psi0=self.psi_prior,
                nu0=self.nu_prior
            )
        return res

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
            ns_j = np.einsum('i,ij,ik->jk',
                             vd_j, (self.y - x_bar_j), (self.y - x_bar_j)
                             ) / n_j
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
            log_d_ji -= dim * np.log(2 * np.pi) / 2
            log_d_ji -= dim / v_lambda_j / 2
            log_d_ji -= (v_scale_j *
                         ((self.y - v_mu_j).T * (
                                 v_precision_j @ (self.y - v_mu_j).T)).sum(0)
                         ) / 2
            var_d[j, :] = log_d_ji
        var_d -= var_d.max(axis=0, initial=-np.inf)
        var_d = np.exp(var_d)
        var_d += np.finfo(np.float64).eps
        var_d /= var_d.sum(axis=0)
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

    def _cast_observations(self, y):
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
            elif isinstance(y, (int, float)):
                _y = np.array([[y]])
            else:
                raise TypeError("Invalid type for variable y")
        return _y
