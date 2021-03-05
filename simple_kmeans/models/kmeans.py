import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import copy
from simple_kmeans.models import KMeansHistory


class KMeans:
    def __init__(self,
                 n_clusters=8,
                 init='k-means++',
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 precompute_distances='auto',
                 verbose=0,
                 random_state=None,
                 copy_x=True,
                 n_jobs=1,
                 algorithm='auto'):

        self._n_clusters = n_clusters
        self._init = init
        self._n_init = n_init
        self._max_iter = max_iter
        self._tol = tol
        self._precompute_distances = precompute_distances
        self._verbose = verbose
        self._random_state = random_state
        self._copy_x = copy_x
        self._n_jobs = n_jobs
        self._algorithm = algorithm
        self._ms = None
        self._Ms = None
        self._history = None


    def normalize(self, X):
        self._ms = np.min(X, axis=0)
        X -= self._ms
        self._Ms = np.max(X, axis=0)
        X /= self._Ms
        return X

    def denormalize(self, X):
        # Post processing
        X *= self._Ms
        X += self._ms
        return X


    def fit(self, X):
        X = self.normalize(X)
        candidates = []
        with ProcessPoolExecutor(self._n_jobs) as e:
            fs = [e.submit(self._fit, copy.deepcopy(X)) for _ in range(self._n_init)]
            for f in tqdm(as_completed(fs), total=len(fs), desc="KMeans"):
                try:
                    if f._exception is None:
                        candidates.append(f._result)
                except Exception as e:
                    a=1

        # # Averate the centroids per cluster
        # all_centers = np.array(all_centers)
        # centroids = np.unique(np.array(all_centers.reshape(-1, 2)/self._tol, dtype=np.int), axis=0)*self._tol
        # assert len(centroids)==self._n_clusters
        # self.centroids = centroids
        # # dcenters = pairwise_distances(unique_centers, unique_centers)
        # # Average the centers
        # centroids = self.denormalize(centroids)
        self._centroids = candidates[0]
        return self._centroids


    def _fit(self, X):
        # Initialize the clusters
        history = KMeansHistory(data=X, niters=self._max_iter)

        # Converge
        while not history.completed:
            history.step()

        # Return historical results
        return history
