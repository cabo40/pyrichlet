from sklearn.mixture import BayesianGaussianMixture


class DirichletProcessMixture(BayesianGaussianMixture):
    def __init__(self, theta=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, init_params='kmeans',
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10):
        super(self).__init__(self)
