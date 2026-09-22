Ethogram for all observed play behaviours inside the test arenas divided into the corresponding categories of locomotor, social and object play. Definitions have been adapted from references [1-4].

Research data and generated results are excluded from Git; publication on Zenodo
is pending. Git retains the data documentation and manually edited
postprocessing corrections.

The 30 video recordings analysed in the accompanying paper are part of a study
of play behaviour in young chickens by Oscarsson et al. (2026).

1 Baxter, M., Bailie, C. L. & O&apos; Connell, N. E. Play behaviour, fear responses and activity levels in commercial broiler chickens provided with preferred environmental enrichments. *animal* **13**, 171-179, doi:10.1017/S1751731118001118 (2019).

- `videos/`: 30 raw 15-minute recordings from Oscarsson et al. (2026), days 28
  and 29 of rearing
- `labels/`: human-expert ethogram annotations (Excel format)
- `dataset/`: pipeline outputs — cleaned tracks, computed morphokinematic
  features, and model embeddings
  - `labels.parquet`: ethogram annotations aligned to tracking windows
  - `tracks.parquet`: postprocessed tracks (segmented bounding-boxs) with
    protocol bird IDs and window column
  - `features_all.parquet`: per-frame morphokinematic descriptors (spatial,
    temporal, pairwise)
  - `features_windowed.parquet`: morphokinematic descriptors summarised over
    5-second windows
  - `embeddings_{backbone}_{size}[_{variant}].pt` — embeddings per (video, bird,
    window), for variants DINOv3, VideoPrism, and V-JEPA 2/2.1 and
    backbone/sizes ViT-B and ViT-L
- `postprocessing/`: per-video human-audit JSONs for resolving identity switches
  and removing erroneous tracks
- `results/`: outputs from tracking, classification, and data analysis
  - `tracking/`: raw tracking pipeline outputs
  - `eval_tracking/`: tracking pipeline evaluation
  - `eval_classification/`: play category classification evaluation
  - `clustering/`: unsupervised clustering results on morphokinematic features

**Reference:** Oscarsson, R., Hedlund, L., Rutkauskaite, A., & Jensen, P.
(2026). Individual variation in play in young chickens – assessment and
connection to affective state and personality. _Scientific Reports_, 16(1), 634.
https://doi.org/10.1038/s41598-025-34437-x
