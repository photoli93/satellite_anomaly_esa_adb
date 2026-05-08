## 1. Using Kaggle ESA-ADB (Mission1) as the initial dataset

**Decision.**  
For the first version of the project, the primary working dataset is the Kaggle ESA-ADB challenge data based on ESA-Mission1, provided as `train.parquet` and `test.parquet` with an `is_anomaly` column.

**Reasons :**

- The Kaggle dataset is a **simplified and preprocessed** version of ESA-Mission1 from the ESA Anomaly Dataset (ESA-AD), following the official ESA-ADB preprocessing pipeline. This removes the need to re-implement complex raw-data preparation early in the project.
- It includes a clean `is_anomaly` label, where each time index is marked 1 if any channel is affected by an anomaly or rare event, making it straightforward to train and evaluate anomaly detectors.
- For a beginner, starting from `train.parquet`:
  - Reduces engineering overhead (no custom timestamp handling, no raw file stitching).
  - Keeps the focus on model design, evaluation and edge deployment.
- The dataset is still **official ESA-ADB data**: it is derived from real ESA telemetry and referenced in the benchmark and related publications so results remain relevant for ESA operations.

**Extension path.**

- Once the pipeline works well on Kaggle Mission1, the next step is to re-run the same design on raw ESA-Mission1 (and optionally Mission2) from ESA-AD, using the preprocessing scripts provided in the official ESA-ADB GitHub repository.