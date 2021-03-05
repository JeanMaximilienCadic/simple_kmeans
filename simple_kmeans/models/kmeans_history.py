import random
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.linalg import norm
import pickle
from simple_kmeans.visualization import display


class KMeansHistory(dict):
    def __init__(self, data, niters):
        super(KMeansHistory, self).__init__()
        self._data = data
        self._niters = niters
        self._n_clusters = 2
        self.completed = False
        self._centroids, self._preds = self.init()

    def init(self):
        self.clear()
        self._iter = 0
        _centroids, _preds = self.init_centroids(), None
        return _centroids, _preds

    def init_centroids(self):
        def new_centroid():
            return np.array([random.random(), random.random()])
        return np.array([new_centroid() for _ in range(self._n_clusters)])

    def calculate_centroids(self, preds):
        _centroids = np.array([np.mean(self._data[preds == k], axis=0) for k in range(2)])
        try:
            assert not np.max(np.isnan(_centroids))
        except AssertionError:
            raise ValueError
        return _centroids

    def step(self, tolerance=10e-4):
        try:
            assert self._iter<self._niters
            D = pairwise_distances(self._data, self._centroids)
            _preds = np.argmin(D, axis=1)
            _centroids = self.calculate_centroids(_preds)
            assert norm(self._centroids - _centroids) > tolerance
            self._centroids, self._preds = _centroids, _preds
            self.setdefault(self._iter, (self._centroids, self._preds))
            self._iter += 1
        except AssertionError:
            self.completed = True
            self._iter-=1
        except ValueError:
            self.init()

    def show(self, step=-1):
        step = self._iter if step==-1 else step
        assert step >=0 and step<=self._iter
        centroids, preds = self.__getitem__(step)
        pickle.dump(self._data, open("../../etc/data.pkl", "wb"))
        pickle.dump(centroids, open("../../etc/centroids.pkl", "wb"))
        pickle.dump(preds, open("../../etc/preds.pkl", "wb"))
        display(self._data, centroids, preds)

