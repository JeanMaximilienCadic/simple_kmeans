from simple_kmeans.models import KMeans
import warnings
from sklearn.datasets import make_gaussian_quantiles
import numpy as np

if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as w:
        # Construct dataset
        X1, y1 = make_gaussian_quantiles(cov=2.,
                                         n_samples=200, n_features=2,
                                         n_classes=2, random_state=1)
        X2, y2 = make_gaussian_quantiles(mean=(3, 5), cov=1.5,
                                         n_samples=300, n_features=2,
                                         n_classes=2, random_state=1)

        X = np.concatenate((X1, X2))
        kmeans = KMeans(n_clusters=2, n_init=1)
        history = kmeans.fit(np.concatenate((X1, X2), axis=0))
        history.show()
