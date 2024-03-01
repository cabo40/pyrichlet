Usage
=====
.. note::
    The weighting structure models that this package implements are listed under
    :doc:`models`.

.. hint::
    Code blocks using the ``rng`` parameter use it only for reproducibility
    purposes.

There are two families of classes in `pyrichlet`, weighting models and mixture
models.
A Weighting model object can be imported and initialized as:

.. doctest::

    >>> from pyrichlet import weight_models
    >>> wm = weight_models.DirichletProcess(rng=0)
    >>> wm = weight_models.DirichletProcess(rng=0)

then `wm` can be used to draw a sample vector of weights with length `10` from
its prior distribution

.. doctest::

        >>> wm.random(10)
        array([7.02467947e-01, 2.12012016e-01, 8.52339410e-02, 2.75312557e-04,
               8.69186887e-06, 8.68128787e-07, 2.27208506e-07, 6.14190832e-07,
               6.03943713e-08, 2.02785792e-07])

we can get a new realization for the vector by running ``wm.random`` again, or
get more weights for the same realization with

.. doctest::

        >>> wm.complete(12)
        array([7.02467947e-01, 2.12012016e-01, 8.52339410e-02, 2.75312557e-04,
               8.69186887e-06, 8.68128787e-07, 2.27208506e-07, 6.14190832e-07,
               6.03943713e-08, 2.02785792e-07, 7.66088536e-08, 2.75049683e-08])

or get `100` independent random assignations using the current truncated
structure

.. doctest::

        >>> wm.random_assignment(100)
        array([0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 2, 0, 1, 0, 0, 1, 0, 1, 1, 2,
               0, 1, 2, 2, 0, 1, 2, 2, 0, 2, 1, 1, 0, 0, 1, 2, 0, 0, 0, 2, 0, 1,
               0, 0, 1, 0, 1, 0, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0])

this methods can be applied to the posterior distribution after fitting a
database of assignations.
Once fitted the minimum vector size can be inferred from the data

.. doctest::

        >>> wm.fit([0, 1, 2, 3, 3, 3, 5])
        >>> wm.random()
        array([0.22802716, 0.14438267, 0.2624806 , 0.3200459 , 0.0360923 ,
               0.00697771])
        >>> wm.random_assignment(20)
        array([1, 3, 3, 2, 1, 5, 1, 0, 3, 3, 3, 4, 3, 3, 3, 1, 0, 3, 3, 0])


The fitted data can be replaced by calling again the ``fit`` method, or by
resetting the weighting structure

.. doctest::

        >>> wm.reset()

For each weighting structure there is an associated Gaussian mixture model.

.. doctest::

        >>> from pyrichlet import mixture_models
        >>> mm = mixture_models.DirichletProcessMixture(rng=0)

The mixture models can fit data represented in an array or as a dataframe

.. doctest::

        >>> mm.fit_gibbs([1, 2, 3, 4], init_groups=2)

we can get the EAP density at a single point

.. doctest::

        >>> mm.gibbs_eap_density(2.5)
        array([0.25341657])

or at several

.. doctest::

        >>> mm.gibbs_eap_density([1.5, 2.5, 3.5])
        array([0.18017642, 0.25341657, 0.19041053])

weighting structures can also be fitted using variational inference, to which
we can calculate the EAP density

.. doctest::

        >>> mm.fit_variational([1, 2, 3, 4], n_groups=2)
        >>> mm.var_eap_density([1.5, 2.5, 3.5])
        array([0.20426694, 0.32282514, 0.20426694])

mixture models can also be used for clustering

.. doctest::

        >>> mm.var_map_cluster()
        array([0, 0, 1, 1])
        >>> mm.gibbs_map_cluster()
        array([0, 0, 0, 0])
        >>> mm.gibbs_eap_spectral_consensus_cluster()
        array([0, 0, 0, 0], dtype=int32)

Depending on the database, fitting can take a noticeable time to finish.
To show the progress of the fitting method, the parameter `show_progress` can
be set

.. code-block:: python

        >>> mm.fit_gibbs([1, 2, 3, 4], init_groups=2, show_progress=True)
