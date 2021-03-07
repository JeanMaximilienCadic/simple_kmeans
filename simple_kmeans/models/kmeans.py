import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from simple_kmeans.models.kmeans_history import KMeansHistory
from matplotlib import cm
from sklearn.metrics import pairwise_distances
import random
import pandas as pd
import multiprocessing as mp


class KMeans:
    def __init__(self,
                 n_clusters=8,
                 init='sharding',
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 verbose=0,
                 random_state=0,
                 copy_x=True,
                 n_jobs=-1):
        """
        Main KMeans reimplementation

        :param n_clusters:
        :param init:
        :param n_init:
        :param max_iter:
        :param tol:
        :param precompute_distances:
        :param verbose:
        :param random_state:
        :param copy_x:
        :param n_jobs:
        :param algorithm:
        """
        random.seed(random_state)
        self._n_clusters = n_clusters
        self._init = init
        self._n_init = n_init
        self._max_iter = max_iter
        self._tol = tol
        self._verbose = verbose
        self._random_state = random_state
        self._copy_x = copy_x
        self._n_jobs = n_jobs if n_jobs>0 else mp.cpu_count()
        self._colors = cm.get_cmap('viridis', self._n_clusters).colors
        self._ms = None
        self._Ms = None
        self._history = None
        self._data = None
        self.cluster_centers_ = None
        self.labels_ = None

    def set_normalized_data(self, data):
        self._ms = np.min(data, axis=0)
        data -= self._ms
        self._Ms = np.max(data, axis=0)
        data /= self._Ms
        self._data = data

    def fit(self, data):
        self.set_normalized_data(data)
        candidates = []
        if self._init == "sharding":
            candidates.append(self.run(self.step,
                                       copy.deepcopy(self._data),
                                       self._n_clusters,
                                       self._max_iter,
                                       self._init))
        else:
            with ProcessPoolExecutor(self._n_jobs) as e:
                fs = [e.submit(self.run,
                               method=self.step,
                               data=copy.deepcopy(self._data),
                               n_clusters=self._n_clusters,
                               max_iter=self._max_iter,
                               init=self._init) for _ in range(self._n_init)]
                for f in as_completed(fs):
                    assert f.exception() is None
                    candidates.append(f.result())

        self.cluster_centers_, self.labels_ = self.step(self._data, self.electcluster_centers_(candidates))

    @staticmethod
    def step(data, cluster_centers_):
        D = pairwise_distances(data, cluster_centers_)
        labels_ = np.argmin(D, axis=1)
        cluster_centers_ = KMeans.calculatecluster_centers_(data, labels_, len(cluster_centers_))
        return cluster_centers_, labels_

    @staticmethod
    def calculatecluster_centers_(data, labels_, n_clusters):
        try:
            assert len(set(labels_))==n_clusters
            df = pd.DataFrame.from_records(np.array([labels_, data[:, 0], data[:, 1]]).T, columns=["labels_", "X", "Y"])
            cluster_centers_ = np.array(df.groupby('labels_').mean())
            return cluster_centers_
        except AssertionError:
            raise ValueError


    def electcluster_centers_(self, candidates):
        # Averate the cluster_centers_ per cluster
        cluster_centers__inits = np.array([h.cluster_centers_ for h in candidates]).reshape(-1, 2)
        cluster_centers__inits, counts = np.unique(np.array(cluster_centers__inits/self._tol, dtype=int), axis=0, return_counts=True)
        inds = np.argsort(counts)[::-1]
        cluster_centers_ = cluster_centers__inits[inds[:self._n_clusters]]*self._tol
        return cluster_centers_

    @staticmethod
    def run(method, data, n_clusters, max_iter, init):
        history = KMeansHistory(method=method,
                                init=init,
                                data=data,
                                n_clusters=n_clusters,
                                max_iter=max_iter)

        # Converge
        while not history.completed:
            history.step()

        return history


