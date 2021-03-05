from sklearn.cluster import KMeans
import numpy as np


def _scikit(X, n_clusters = 2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    preds = kmeans.predict(X)
    cluster_centers = kmeans.cluster_centers_
    return preds, np.array(cluster_centers)


