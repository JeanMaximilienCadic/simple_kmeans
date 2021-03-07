import matplotlib.pyplot as plt
import numpy as np

def serialize_results(centroids, preds):
    """
    Keep a consitent order for the cluster's id values by sorting the y coordinate of centroids.
    :param centroids:
    :param preds:
    :return:
    """
    # Reorganize
    classes = np.unique(preds)
    map = dict([(k, (centroids[k], np.argwhere([preds == k])[:,1])) for k in np.unique(preds)])
    inds = np.argsort(centroids[:, 1])[::-1]
    map = dict([(idx, map[inds[idx]]) for idx in classes])
    return map


def display(data, map, colors, title, score=None):
    """
    Display the results of the last iteration.

    :param data:
    :param map:
    :param colors:
    :param title:
    :param score:
    :return:
    """
    title = title if score is None else f"{title} | Score: {score}"
    padding = 0.2
    x_min, x_max = -padding, 1 + padding
    y_min, y_max = -padding, 1 + padding
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    for idx, color in enumerate(colors):
        centroid, inds = map[idx]
        plt.scatter(data[inds][:, 0],
                    data[inds][:, 1],
                    color=color,
                    cmap=plt.cm.Paired,
                    s=20,
                    edgecolor='k',
                    label="Cluster %s" % idx)
        plt.scatter(centroid[0], centroid[1], color=color, s=200, alpha=0.5)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

