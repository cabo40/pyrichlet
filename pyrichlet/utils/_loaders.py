try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
import pandas as pd
from . import _data


def load_chaetocnema():
    df = pd.read_csv(pkg_resources.resource_string(_data, "chaetocnema.csv"),
                     usecols=list(range(8)),
                     index_col=0)
    return df.iloc[:, :-1], df.iloc[:, -1].to_numpy()


def load_penguins():
    df = pd.read_csv(pkg_resources.resource_string(_data, "penguins.csv"),
                     usecols=[0, 2, 3, 4, 5])
    df = df.dropna()
    return df.iloc[:, 1:], df.iloc[:, 0].to_numpy()
