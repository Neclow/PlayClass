"""Core clustering interface: fit, predict, and k-ablation."""

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.clustering._metrics import gap_score
from src.clustering._wrappers import (
    AgglomerativeClusteringWrapper,
    GaussianMixtureWrapper,
    KMedoidsWrapper,
)


@dataclass
class ClusterResult:
    """Result of a single clustering fit: the fitted clusterer, assigned labels, and cluster sizes."""

    clusterer: ClusterMixin
    labels: np.ndarray
    sizes: list


CLUSTERERS = {
    "k-means": partial(KMeans, n_init=1),
    "k-medoids": partial(KMedoidsWrapper, method="fasterpam", metric="euclidean"),
    "hac": partial(AgglomerativeClusteringWrapper, linkage="ward", metric="euclidean"),
    "gmm": partial(GaussianMixtureWrapper, n_init=1),
}


def fit_predict(
    X: np.ndarray, solver: str, k: int, random_state: int, metric=None, **kwargs
) -> ClusterResult:
    """Fit a clusterer from CLUSTERERS and return a ClusterResult."""
    try:
        clusterer = CLUSTERERS[solver](
            n_clusters=k, random_state=random_state, **kwargs
        )
    except KeyError as exc:
        raise ValueError(
            (
                f"Unsupported solver: {solver}.\n"
                f"Supported solvers are: {list(CLUSTERERS.keys())}"
            )
        ) from exc

    labels = clusterer.fit_predict(X)

    return ClusterResult(
        clusterer=clusterer,
        labels=labels,
        sizes=np.bincount(labels, minlength=k).tolist(),
    )


def _shift(arr, num, fill_value=np.nan):
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))


def ablate_k(X, solver, k_range, random_state, with_gap=False, **kwargs):
    """Sweep k values for a given solver, returning silhouette (and optionally gap) scores."""
    results = {}
    sils = np.zeros((len(k_range),))
    gaps, sks = np.zeros((len(k_range),)), np.zeros((len(k_range),))
    for i, k in enumerate(tqdm(k_range, desc=solver)):
        res = fit_predict(X, solver, k, random_state)

        if with_gap:
            gap, sk, method = gap_score(
                res.clusterer, X, res.labels, random_state=random_state, **kwargs
            )
            gaps[i] = gap
            sks[i] = sk

        sil = silhouette_score(
            X,
            res.labels,
            metric=kwargs.get("metric", "euclidean"),
            random_state=random_state,
        )

        sils[i] = sil
        results[k] = res
    df = pd.DataFrame.from_dict(results, orient="index")

    df["silhouette"] = sils
    if with_gap:
        sks_shifted = _shift(sks, -1)
        gaps_shifted = _shift(gaps, -1)
        diff = gaps - gaps_shifted + sks_shifted
        df["gap"] = gaps
        df["gap_diff"] = diff

    return df
