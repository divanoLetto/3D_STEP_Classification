import numpy as np
from sklearn.neighbors import NearestNeighbors


def euclidean_distance(descx, centrs, n=10):
    if (descx.shape[0] == 0):
        H2 = np.zeros((centrs.shape[0],))
        return H2
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(centrs)
    distances, cluster_ids = nbrs.kneighbors(descx)
    H2, edges = np.histogram(cluster_ids.ravel(), bins=centrs.shape[0], range=(0, centrs.shape[0]), density=True)
    return H2


def cosine_distance(descx, centrs, n=10):
    if (descx.shape[0] == 0):
        H2 = np.zeros((centrs.shape[0],))
        return H2
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine').fit(centrs)
    distances, cluster_ids = nbrs.kneighbors(descx)
    H2, edges = np.histogram(cluster_ids.ravel(), bins=centrs.shape[0], range=(0, centrs.shape[0]), density=True)
    return H2


def cosine_distance2(descx, centrs, n=10):
    if (descx.shape[0] == 0):
        H2 = np.zeros((centrs.shape[0],))
        return H2
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine').fit(centrs)
    distances, cluster_ids = nbrs.kneighbors(descx)
    H2, edges = np.histogram(cluster_ids.ravel(), bins=centrs.shape[0], range=(0, centrs.shape[0]), density=True)

    return H2


def histogram_intersection(a, b):
    return 1 - np.sum(np.minimum(a / (0.0001 + np.sum(a)), b / (0.00001 + np.sum(b))))


def histogram_intersection_distance(descx, centrs):  # make better
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=histogram_intersection).fit(centrs)
    distances, cluster_ids = nbrs.kneighbors(descx)
    H2, edges = np.histogram(cluster_ids.ravel(), bins=centrs.shape[0], range=(0, centrs.shape[0]), density=True)
    return H2