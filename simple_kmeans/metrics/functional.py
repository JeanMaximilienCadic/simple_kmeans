import numpy as np


def accuracy(map_gt, _map, n_clusters):
    """
    Get the average accuracy per cluster.

    :param map_gt:
    :param _map:
    :param n_clusters:
    :return:
    """
    acc = np.mean([len(set(map_gt[k][1]).intersection(set(_map[k][1])))/len(set(map_gt[k][1]))
                   for k in range(n_clusters)])
    acc = int(acc*10000)/100
    return acc
