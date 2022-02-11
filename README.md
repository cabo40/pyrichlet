# Project description

Pyrichlet is a package for doing data analysis via density estimation
and clustering using Gaussian mixtures of several weighting structures.

# Installation

With pip:

```
pip install pyrichlet
```

For a specific version:

```
pip install pyrichlet==0.0.3
```


# Usage

This is a quick guide. For a more detailed usage see
https://pyrichlet.readthedocs.io/en/latest/index.html.

The weighting structure models that this package implements are

- `DirichletDistribution`
- `DirichletProcess`
- `PitmanYorProcess`
- `GeometricProcess`
- `BetaInBetaProcess`
- `BetaInDirichletProcess`
- `BetaBernoulliProcess`
- `BetaBinomialProcess`

They can be imported and initialized as

```python
from pyrichlet import weight_models

wm = weight_models.DirichletProcess()
wm.fit([0, 0, 1])
wm.random(10)
wm.random_assignment(100)
wm.reset()
wm.random(10)
wm.random_assignment(100)
```

For each weighting structure there is an associated Gaussian mixture model, formerly

- `DirichletDistributionMixture`
- `DirichletProcessMixture`
- `PitmanYorMixture`
- `GeometricProcessMixture`
- `BetaInBetaMixture`
- `BetaInDirichletMixture`
- `BetaBernoulliMixture`
- `BetaBinomialMixture`

The mixture models can fit array or dataframe data for density estimation

```python
from pyrichlet import mixture_models

mm = mixture_models.DirichletProcessMixture()
y = [1, 2, 3, 4]
mm.fit_gibbs(y)
x = 2.5
f_x = mm.gibbs_eap_density(x)

mm.fit_variational(y, n_groups=2)
f_x = mm.var_eap_density(x)
```

or for cluster estimation

```python
mm.var_map_cluster()
mm.gibbs_map_cluster()
mm.gibbs_eap_spectral_consensus_cluster()
```