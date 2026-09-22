# Analysis

Analysis notebooks live in `notebook/`. Open and run them in Jupyter — all paths
are relative to `notebook/`, all outputs go to `img/`. The `classifier` pixi
environment has everything needed (sklearn, matplotlib, torch, xgboost, shap).

## Notebooks

| Notebook                    | Data inputs                                                                                                                                                | Outputs (`img/`)                                                                                                                                                     |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fig2_clustering.ipynb`     | `data/dataset/labels.parquet`, `data/dataset/features_windowed.parquet`, `data/results/clustering/Z_dropna.npy`, `data/results/clustering/grid_dropna.csv` | `fig2_cluster_heatmap.pdf`, `fig2_tsne_*.pdf/png`, `fig2_radar.pdf`, `figS_loadings.pdf`, `figS_cluster_vs_k.pdf`, `data/results/clustering/cluster_assignments.csv` |
| `fig3_classification.ipynb` | `data/dataset/`, `data/results/eval_classification/`                                                                                                       | `fig3_metrics.pdf`, `fig3_heatmap*.pdf`, `fig3_shap_feature_groups.pdf`                                                                                              |
| `figS_tracker_eval.ipynb`   | `data/results/eval_tracking/results/`                                                                                                                      | `figS_tracker_eval_hota.pdf`, `figS_tracker_eval_metrics_agg.pdf`                                                                                                    |

## Clustering grid

Runs the k × solver silhouette grid for the dropna variant and writes
`Z_dropna.npy` and `grid_dropna.csv` to `data/results/clustering/`. Run this
before the notebook:

```sh
pixi run -e classifier python -m pipeline.clustering_grid
```

## Classification evaluation and tables

Assemble the segment sweep (Supp Table 3) and ablation (Supp Table 4) CSVs from
existing `loco_summary.csv` files (no GPU needed):

```sh
pixi run -e classifier python -m pipeline.eval_classification tables
```

To backfill per-fold confusion matrices, recall, and per-sample predictions from
checkpoints (requires GPU):

```sh
pixi run -e classifier python -m pipeline.eval_classification evaluate --all
```
