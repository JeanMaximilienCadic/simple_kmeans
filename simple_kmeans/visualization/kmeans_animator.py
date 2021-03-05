import matplotlib.animation as animation
# from sklearn.cluster import KMeans
from simple_kmeans.models import KMeans
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
import random
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.linalg import norm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import copy
import pickle




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


class KMeansAnimator(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, numpoints=500):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c,
                                    cmap="jet", edgecolor="k")
        self.ax.axis([-0.2, 1.2, -0.2, 1.2])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        while True:
            c =  np.zeros(len(data))
            c[preds==1] = 0.5
            inds = random.sample(list(range(len(c))), 200)
            c[inds]=0
            yield np.c_[data[:, 0], data[:, 1], c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        self.scat.set_offsets(data[:, :2])
        self.scat.set_array(data[:, 2])
        return self.scat,


if __name__ == '__main__':
    a = KMeansAnimator()
    plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import pickle
#
# plot_colors = "br"
# class_names = "01"

# xdata, ydata = [], []
# fig, ax = plt.subplots(figsize=(8, 8))
#
# padding = 0.2
# x_min, x_max = data[:, 0].min() - padding, data[:, 0].max() + padding
# y_min, y_max = data[:, 1].min() - padding, data[:, 1].max() + padding
#
# def init():
#     return
#
# def update(frame):
#     for i, n, c in zip(range(2), class_names, plot_colors):
#         idx = np.where(preds == i)
#         plt.scatter(data[idx, 0],
#                     data[idx, 1],
#                     c=c,
#                     cmap=plt.cm.Paired,
#                     s=20,
#                     edgecolor='k',
#                     label="Cluster %s" % n)
#
#
# ani = FuncAnimation(fig,
#                     update,
#                     frames=np.linspace(0, 2 * np.pi, 128),
#                     init_func=init,
#                     blit=True)
# plt.show()
#
#
# a=1
# # kmeans.show()