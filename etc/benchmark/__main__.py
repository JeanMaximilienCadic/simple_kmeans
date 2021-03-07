from simple_kmeans.models import KMeans as simpleKmeans
from sklearn.cluster import KMeans as sklearnKmeans
import numpy as np
import time
from datetime import timedelta
from simple_kmeans.visualization import serialize_results
from matplotlib import cm
from simple_kmeans.metrics import accuracy
from simple_kmeans.data import generate_data
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def run_exp(n_samples, n_clusters, max_iter, model, init):
    t0 = time.time()
    xy, labels = generate_data(n_samples=n_samples, n_clusters=n_clusters)
    map_gt = serialize_results(simpleKmeans.calculatecluster_centers_(xy, labels, n_clusters), labels)
    kmeans = model(n_clusters=n_clusters, max_iter=max_iter, init=init, n_jobs=12)
    kmeans.fit(xy)
    _map = serialize_results(kmeans.cluster_centers_, kmeans.labels_)
    acc = accuracy(map_gt, _map, n_clusters)
    return model.__module__, init, timedelta(seconds=time.time()-t0), acc


if __name__ == "__main__":
    n_samples, n_clusters, max_iter, n_exp = 600, 3, 300, 500
    colors = cm.get_cmap('viridis', n_clusters).colors
    scores = {}
    from itertools import product
    with ProcessPoolExecutor() as e:
        params = [(simpleKmeans,  "sharding"), (sklearnKmeans, "random")]
        fs = [e.submit(run_exp,
                       n_samples,
                       n_clusters,
                       max_iter,
                       model,
                       init)
              for _, (model, init) in product(range(n_exp), params)]
        for f in tqdm(as_completed(fs), total=len(fs), desc="Processing"):
            assert f._exception is None
            name, init, duration, acc = f._result
            try:
                scores[name].append(acc)
            except:
                scores[name] = [acc]

    plt.xlabel('iters')
    plt.ylabel('scores')
    diff_score = np.array(scores["simple_kmeans.models.kmeans"]) - np.array(scores["sklearn.cluster._kmeans"])
    plt.plot(diff_score, label=f"simple_kmeans vs kmeans {int(np.mean(diff_score)*100)/100}%")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
