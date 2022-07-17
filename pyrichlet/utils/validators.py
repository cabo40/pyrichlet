import numpy as np


def rng_parser(rng):
    if rng is None:
        return np.random.default_rng()
    if type(rng) is int:
        return np.random.default_rng(rng)
    if type(rng) is np.random.Generator:
        return rng
    raise TypeError("Invalid random number generator")
