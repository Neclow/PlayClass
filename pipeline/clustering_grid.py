"""k × solver silhouette grid over four imputation/PCA variants.

Writes grid CSVs, scaled arrays, and labels to data/results/clustering/.
Run once per dataset release; the figure notebook reads the outputs.

    pixi run -e classifier python -m script.clustering_grid
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.cluster import CLUSTERERS, ablate_k

K_RANGE = range(2, 13)
RANDOM_SEED = 42
OUT = Path("data/results/clustering")


def grid_search(X):
    """Run a k x solver silhouette grid search over all CLUSTERERS."""
    solvers = list(CLUSTERERS.keys())
    rows = []
    for solver in solvers:
        sens_k = ablate_k(X, solver=solver, k_range=K_RANGE, random_state=RANDOM_SEED)
        for k_val in K_RANGE:
            rows.append(
                {
                    "solver": solver,
                    "k": k_val,
                    "silhouette": sens_k.loc[k_val, "silhouette"],
                }
            )
    grid = pd.DataFrame(rows)

    pivot = grid.pivot(index="k", columns="solver", values="silhouette").round(3)
    print(pivot.to_markdown())

    mean_sil = grid.groupby("solver")["silhouette"].mean()
    best_solver = mean_sil.idxmax()
    print(
        f"Best solver by mean silhouette: {best_solver} ({mean_sil[best_solver]:.3f})\n"
    )
    return grid, best_solver


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    KEY = ["video_id", "bird_id", "window"]
    lab = pd.read_parquet("data/dataset/labels.parquet", columns=KEY + ["behav_label"])
    lab = lab[lab.behav_label != "social"].reset_index(drop=True)

    fw = lab.merge(
        pd.read_parquet("data/dataset/features_windowed.parquet"), on=KEY, how="left"
    )
    feat_cols = [c for c in fw.columns if c not in {*KEY, "behav_label", "n_frames"}]

    X = fw[feat_cols].astype(float)
    y = lab.behav_label.to_numpy()
    print(f"X: {X.shape}, y: {y.shape}\n")

    X_dropna = X.dropna()
    Z_dropna = StandardScaler().fit_transform(X_dropna)
    Z_dropna_pca = PCA(50).fit_transform(Z_dropna)

    X_fillna_median = X.fillna(X.median())
    Z_fillna_median = StandardScaler().fit_transform(X_fillna_median)
    Z_fillna_median_pca = PCA(50).fit_transform(Z_fillna_median)

    print(f"dropna:        {X_dropna.shape[0]} rows -> PCA {Z_dropna_pca.shape}")
    print(
        f"fillna_median: {X_fillna_median.shape[0]} rows -> PCA {Z_fillna_median_pca.shape}\n"
    )

    np.save(OUT / "Z_dropna.npy", Z_dropna)
    np.save(OUT / "Z_dropna_pca.npy", Z_dropna_pca)
    np.save(OUT / "Z_fillna_median.npy", Z_fillna_median)
    np.save(OUT / "Z_fillna_median_pca.npy", Z_fillna_median_pca)
    np.save(OUT / "y.npy", y)
    print(f"Saved scaled arrays to {OUT}/\n")

    datasets = {
        "dropna": Z_dropna,
        "dropna_pca": Z_dropna_pca,
        "fillna_median": Z_fillna_median,
        "fillna_median_pca": Z_fillna_median_pca,
    }

    results = {}
    for name, data in datasets.items():
        print(f"{'=' * 60}\n  {name}  ({data.shape})\n{'=' * 60}\n")
        grid, best_solver = grid_search(data)
        grid.to_csv(OUT / f"grid_{name}.csv", index=False)
        results[name] = best_solver

    print(f"\n{'=' * 60}\n  Summary\n{'=' * 60}")
    for name, best in results.items():
        print(f"  {name:25s} -> {best}")

    pd.DataFrame([{"variant": k, "best_solver": v} for k, v in results.items()]).to_csv(
        OUT / "summary.csv", index=False
    )
    print(f"\nAll results saved to {OUT}/")


if __name__ == "__main__":
    main()
