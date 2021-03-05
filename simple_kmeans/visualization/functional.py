import matplotlib.pyplot as plt
import numpy as np

def display(data, centroids, preds):
    plot_colors = "br"
    class_names = "01"

    plt.figure(figsize=(8, 8))

    padding = 0.2
    x_min, x_max = data[:, 0].min() - padding, data[:, 0].max() + padding
    y_min, y_max = data[:, 1].min() - padding, data[:, 1].max() + padding
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(preds == i)
        plt.scatter(data[idx, 0],
                    data[idx, 1],
                    c=c,
                    cmap=plt.cm.Paired,
                    s=20,
                    edgecolor='k',
                    label="Cluster %s" % n)
    for centroid, c in zip(centroids, ["blue", "red"]):
        # plt.scatter(centers[:, 0], centers[:, 1], c=c, s=200, alpha=0.5)
        plt.scatter(centroid[0], centroid[1], c=c, s=200, alpha=0.5)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('KMeans')

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.35)
    plt.show()

