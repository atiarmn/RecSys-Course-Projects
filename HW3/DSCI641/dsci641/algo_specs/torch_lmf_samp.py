"""
Logistic matrix factorization with PyTorch (sampled).
"""

from scipy.stats import zipfian, loguniform, randint
from ..algorithms.torchimfsamp import TorchSampledMF

space = [
    # log-uniform (Zipf) distribution [5, 250]
    ("n_features", zipfian(1, 246, loc=4)),
    ("reg", loguniform(1.0e-6, 10)),
    ("lr", loguniform(1.0e-6, 1)),
    ("epochs", randint(5, 20)),
]


def default():
    return TorchSampledMF(50, loss="logistic")


def from_params(n_features, **kwargs):
    args = {k: v for (k, v) in kwargs.items() if k in [n for (n, d) in space]}
    return TorchSampledMF(n_features, loss="logistic", **args)
