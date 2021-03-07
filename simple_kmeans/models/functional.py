import random
from math import floor
import numpy as np


def naive_sharding(ds, k):
    """
    Optimized approach to initialize the centroids.

    :param ds:
    :param k:
    :return:
    """
    n = np.shape(ds)[1]
    m = np.shape(ds)[0]
    centroids = np.mat(np.zeros((k, n)))

    # Sum all elements of each row, add as col to original dataset, sort
    composite = np.mat(np.sum(ds, axis=1))
    ds = np.append(composite.T, ds, axis=1)
    ds.sort(axis=0)

    # Step value for dataset sharding
    step = floor(m / k)

    # Vectorize mean ufunc for numpy array
    vfunc = np.vectorize(_get_mean)

    # Divide matrix rows equally by k-1 (so that there are k matrix shards)
    # Sum columns of shards, get means; these columnar means are centroids
    for j in range(k):
        if j == k - 1:
            centroids[j:] = vfunc(np.sum(ds[j * step:, 1:], axis=0), step)
        else:
            centroids[j:] = vfunc(np.sum(ds[j * step:(j + 1) * step, 1:], axis=0), step)

    return centroids


def _get_mean(sums, step):
    return sums / step


def rcentroid():
    return np.array([random.random(), random.random()])