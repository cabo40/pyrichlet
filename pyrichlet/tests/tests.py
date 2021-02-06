import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from .. import mixture_models


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

    dgp = mixture_models.BetaInBetaMixture(rng=rng)
    dgp.fit(y)
    nplot = 50
    x1 = np.linspace(y[:, 0].min() - 1, y[:, 0].max() + 1, nplot)
    x2 = np.linspace(y[:, 1].min() - 1, y[:, 1].max() + 1, nplot)
    x_plt = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
    ret = dgp.eap_density(x_plt)
    plt.contour(x1, x2, ret.reshape(nplot, -1))
