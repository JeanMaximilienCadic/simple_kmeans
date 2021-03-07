from scipy.linalg import norm
import numpy as np
from .functional import rcentroid, naive_sharding


class KMeansHistory(dict):
    def __init__(self, method, data, n_clusters, max_iter, init):
        """
        Store the results of the KMeans implementat at each iteration.

        :param method:
        :param data:
        :param n_clusters:
        :param max_iter:
        :param init:
        """
        super(KMeansHistory, self).__init__()
        self._method = method
        self._data = data
        self._max_iter = max_iter
        self._n_clusters = n_clusters
        self._init = init
        self.completed = False
        self.cluster_centers_ =None
        self.labels_= None
        self.init()

    def init(self):
        self.clear()
        self._iter = 0
        self.labels_ = None
        self.initcluster_centers_()

    def initcluster_centers_(self):
        if self._init == "sharding":
            self.cluster_centers_ = naive_sharding(self._data, self._n_clusters)
        else:
            self.cluster_centers_ = np.array([rcentroid() for _ in range(self._n_clusters)])

    def step(self, tolerance=10e-4):
        try:
            assert self._iter<self._max_iter
            cluster_centers_, labels_ = self._method(self._data, self.cluster_centers_)
            assert norm(self.cluster_centers_ - cluster_centers_) > tolerance
            self.cluster_centers_, self.labels_ = cluster_centers_, labels_
            self.setdefault(self._iter, (self.cluster_centers_, self.labels_))
            self._iter += 1
            return True
        except AssertionError:
            self.completed = True
            self._iter-=1
            return True
        except ValueError:
            self._init = "random"
            self.init()
            return False


