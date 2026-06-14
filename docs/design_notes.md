## 1. Using the Kaggle Mission1 challenge files as the initial working dataset

**Decision :**

For the first version of the project, the main working data comes from the ESA-ADB Kaggle challenge files for Mission1, stored locally in `data/raw/ESA-Mission1/` as `train.parquet`, `test.parquet` and `target_channels.csv`

**Choice making :**

- These files provide a practical entry point for a first prototype because they are already structured for anomaly-detection experimentation, which avoids spending too much time on raw telemetry preparation at the beginning
- The training data includes an `is_anomaly` label, which makes it easier to build and compare first baseline models before moving to more advanced event- or channel-level approaches

**Current data layout :**

- `data/raw/ESA-Mission1/train.parquet`: training split with anomaly labels.
- `data/raw/ESA-Mission1/test.parquet`: test split for inference and evaluation experiments.
- `data/raw/ESA-Mission1/target_channels.csv`: target channel information for the Mission1 challenge setting

**Limits of this first choice :**

- The Kaggle Mission1 files are useful for a first implementation but they are still only the starting point of the project scope
- They simplify the original telemetry problem so results from this phase should be presented as prototype results rather than as a full reproduction of the complete ESA-ADB benchmark

**Extension path :**

- Once the pipeline is stable on the Mission1 challenge files, the next step is to reuse the same preprocessing, training and inference structure on the wider ESA anomaly dataset and if relevant, on additional missions
- At that stage, the official ESA-ADB codebase can be used as the main reference for aligning preprocessing and evaluation more closely with the published benchmark setup

## 2. Initial project scope

For the first iteration, the goal is not to reproduce the full ESA-ADB benchmark. The goal is to build a small, reproducible pipeline on Mission1 challenge data : explore the files, preprocess telemetry windows, train one baseline model and one lightweight deep-learning model and prepare a simple inference path for Jetson Nano.

At this stage, Mission1 is the only active dataset used in experiments. The `ESA-Mission2` and `ESA-Mission3` folders are kept in the repository structure for future extension but they are not part of the first development milestone