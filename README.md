# Project description

Pyrichlet is a package useful for doing data analysis via density estimation and clustering using gaussian mixtures with
weighting structures generated with the stick-breaking method.

# Installation

This project is hosted in pip, it's enought to do

```
pip install pyrichlet
```

# Usage

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
```

For each weighting structure there is an associated gaussian mixture model, formerly

- `DirichletDistributionMixture`
- `DirichletProcessMixture`
- `PitmanYorMixture`
- `GeometricProcessMixture`
- `BetaInBetaMixture`
- `BetaInDirichletProcess`
- `BetaBernoulliMixture`
- `BetaBinomialMixture`

The mixture models can fit array or dataframe data

```python
from pyrichlet import mixture_models

mm = mixture_models.DirichletProcessMixture()
y = [1, 2, 3, 4]
mm.fit_gibbs(y)

x = 2.5
f_x = mm.gibbs_eap_density(x)
```