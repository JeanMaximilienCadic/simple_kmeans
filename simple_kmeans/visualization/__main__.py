from matplotlib import pyplot as plt
from .kmeans_animator import KMeansAnimator


if __name__ == '__main__':
    from simple_kmeans.models import KMeans
    a = KMeansAnimator(model=KMeans)
    plt.show()