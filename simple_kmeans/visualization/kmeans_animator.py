"""
DISCLAIMER
This code has been modified from this link accessible online
https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from simple_kmeans.data import generate_data


class KMeansAnimator(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, model, nclusters=5, numpoints=500):
        self._model = model
        self.nclusters = nclusters
        self.numpoints = numpoints
        self.stream = self.data_stream()
        self.frame=0
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=10,
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        plt.title("KMeans evolution through movement noise.")
        self.ax.axis([-0.2, 1.2 , -0.2, 1.2])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        from simple_kmeans.visualization import serialize_results
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        xy, _ = generate_data(self.numpoints, self.nclusters)

        s = np.ones(self.numpoints)
        c = np.zeros((self.numpoints, ))
        while True:
            if self.frame>100:
                xy, _ = generate_data(self.numpoints, self.nclusters)
                self.frame = 0
            if random.random()<=0.5:
                coef = 0.01
            else:
                coef = -0.01

            xy += coef * (np.random.random((self.numpoints, 2)) - 0.5)
            kmeans = self._model(n_clusters=self.nclusters, n_init=10, n_jobs=12)
            kmeans.fit(xy)
            _map = serialize_results(kmeans.cluster_centers_, kmeans.labels_)
            for idx , color in enumerate(kmeans._colors):
                c[_map[idx][1]] = color[0]
            # c += 0.02 * (np.random.random(self.numpoints) - 0.5)
            self.frame+=1
            yield np.c_[xy[:,0], xy[:,1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(30 * abs(data[:, 2])**1.5 + 10)
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


