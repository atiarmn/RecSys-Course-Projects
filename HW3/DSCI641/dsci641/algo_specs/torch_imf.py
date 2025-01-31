"""
Implicit-feedback matrix factorization with PyTorch (userwise).
"""

from scipy.stats import zipfian, loguniform, uniform, randint
from ..algorithms.torchimf import TorchImplicitMFUserMSE

space = [
    # log-uniform (Zipf) distribution [5, 250]
    ("n_features", zipfian(1, 246, loc=4)),
    ("confweight", uniform(1, 1000)),
    ("reg", loguniform(1.0e-6, 10)),
    ("lr", loguniform(1.0e-6, 1)),
    ("epochs", randint(5, 20)),
]


def default():
    return TorchImplicitMFUserMSE(50)


def from_params(n_features, **kwargs):
    args = {k: v for (k, v) in kwargs.items() if k in [n for (n, d) in space]}
    return TorchImplicitMFUserMSE(n_features, **args)
