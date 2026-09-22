"""Thin wrappers for sklearn/kmedoids clusterers with a unified constructor API."""

from kmedoids import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin
from sklearn.mixture import GaussianMixture


class AgglomerativeClusteringWrapper(AgglomerativeClustering):
    """Wrapper to make AgglomerativeClustering accept ``random_state`` for consistency."""

    def __init__(self, random_state, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state


class GaussianMixtureWrapper(GaussianMixture):
    """Wrapper to make GaussianMixture accept ``n_clusters`` for consistency."""

    def __init__(self, n_clusters, **kwargs):
        super().__init__(n_components=n_clusters, **kwargs)
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        super().fit(X, y)
        self.cluster_centers_ = self.means_
        return self

    def fit_predict(self, X, y=None):
        labels = super().fit_predict(X, y)
        self.cluster_centers_ = self.means_
        return labels


class KMedoidsWrapper(KMedoids):
    """Wrapper to fix ``predict`` when metric is not "precomputed".

    https://github.com/kno10/python-kmedoids/blob/ef2b3b68ec0aaa9ee63d6e4bf9724cdfff874d17/kmedoids/__init__.py#L966
    """

    def predict(self, X):
        Y = self.cluster_centers_
        X = pairwise_distances_argmin(X, Y=Y, metric=self.metric)
        return X
