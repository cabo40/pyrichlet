[![Documentation Status](https://readthedocs.org/projects/pyrichlet/badge/?version=main)](https://pyrichlet.readthedocs.io/en/main/?badge=main)
![PyPI - Version](https://img.shields.io/pypi/v/pyrichlet)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrichlet)
<!---[Static Badge](https://img.shields.io/badge/DOI-doi-blue)-->

# Project description

Pyrichlet is a package for doing density estimation and clustering using 
Gaussian mixtures with BNP weighting models

# Installation

With pip:

```
pip install pyrichlet
```

For a specific version:

```
pip install pyrichlet==0.0.9
```


# Usage

This is a quick guide. For a more detailed usage see
https://pyrichlet.readthedocs.io/en/main/index.html.


The mixture models that this package implements are

- `DirichletDistributionMixture`
- `DirichletProcessMixture`
- `PitmanYorMixture`
- `GeometricProcessMixture`
- `BetaInBetaMixture`
- `BetaInDirichletMixture`
- `BetaBernoulliMixture`
- `BetaBinomialMixture`

They can be fitted for an array or dataframe using a Gibbs sampler or
variational Bayes methods,

```python
from pyrichlet import mixture_models

mm = mixture_models.DirichletProcessMixture()
y = [1, 2, 3, 4]
mm.fit_gibbs(y, init_groups=2)

mm.fit_variational(y, n_groups=2)
```

and use the fitted class to do density estimation

```python
x = 2.5
f_x = mm.gibbs_eap_density(x)
f_x = mm.var_eap_density(x)
```

or clustering

```python
mm.var_map_cluster()
mm.gibbs_map_cluster()
mm.gibbs_eap_spectral_consensus_cluster()
```