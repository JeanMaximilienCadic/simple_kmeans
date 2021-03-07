from simple_kmeans.visualization import display, serialize_results
from simple_kmeans.metrics import accuracy
from simple_kmeans.data import generate_data
from simple_kmeans.models import KMeans
from matplotlib import cm
import matplotlib
matplotlib.use('TkAgg')

def main(visualization=True):
    n_samples, n_clusters, max_iter, init = 3000, 5, 300, "sharding"
    colors = cm.get_cmap('viridis', n_clusters).colors
    xy, labels = generate_data(n_samples=3000, n_clusters=n_clusters)
    map_gt = serialize_results(KMeans.calculatecluster_centers_(xy, labels, n_clusters), labels)
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, max_iter=max_iter, init=init)
    model.fit(xy)
    print(model.labels_)
    print(model.cluster_centers_)
    if visualization:
        _map = serialize_results(model.cluster_centers_, model.labels_)
        display(xy, _map, colors, title=f"KMeans_{KMeans.__module__}", score=accuracy(map_gt, _map, n_clusters))


if __name__ == "__main__":
    main(visualization=True)