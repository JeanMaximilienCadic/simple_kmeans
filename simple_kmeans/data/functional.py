from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import random


def generate_data(n_samples, n_clusters):
    """
    Generate a set of clusters composed of normal 2d data
    :param n_samples:
    :param n_clusters:
    :return:
    """
    k_samples = n_samples//n_clusters
    xy, labels = None, None
    for k in range(n_clusters):
        _xy, _labels = make_gaussian_quantiles(mean=(random.randrange(-10, 10), random.randrange(-10, 10)),
                                               cov=1,
                                               n_samples=k_samples,
                                               n_features=2,
                                               n_classes=1)[0], \
                       np.ones(k_samples, dtype=int) * k
        if xy is None:
            xy = _xy
            labels = _labels
        else:
            xy = np.concatenate((xy, _xy), axis=0)
            labels = np.concatenate((labels, _labels), axis=0)
    return xy, labels