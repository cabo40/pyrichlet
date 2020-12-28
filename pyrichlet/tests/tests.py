import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from pyrichlet import mixture_models
import scipy.stats


def test_bayesian_mixture_weights():
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 2

    X = rng.rand(n_samples, n_features)

    # Case Dirichlet distribution for the weight concentration prior type
    # bgmm = BayesianGaussianMixture(
    #     weight_concentration_prior_type="dirichlet_distribution",
    #     n_components=3, random_state=rng).fit(X)

    dpgmm = BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        n_components=3, random_state=rng).fit(X)


def test_base_mixture():
    rng = np.random.default_rng(0)
    n_samples, n_features = 10, 2

    means = np.array([[0, -4],
                      [3, 3],
                      [-3, 2]])
    sds = np.array([[[1, 0.7],
                     [0.7, 1]],
                    [[1, -0.6],
                     [-0.6, 1]],
                    [[1, 0],
                     [0, 1]]])

    weights = np.array([0.4, 0.3, 0.3])
    weights = weights / weights.sum()

    n = 300
    theta = rng.choice(range(len(weights)), size=n, p=weights)
    y = np.array([
        scipy.stats.multivariate_normal.rvs(means[j], sds[j],
                                            random_state=rng) for j in theta])

    dgp = mixture_models.DGEPMixture(total_iter=1e7, rng=rng)
    dgp.fit(y)
    import matplotlib.pyplot as plt
    nplot = 50
    x1 = np.linspace(y[:, 0].min() - 1, y[:, 0].max() + 1, nplot)
    x2 = np.linspace(y[:, 1].min() - 1, y[:, 1].max() + 1, nplot)
    x_plt = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
    ret = dgp.density(x_plt)
    plt.contour(x1, x2, ret.reshape(nplot, -1))
