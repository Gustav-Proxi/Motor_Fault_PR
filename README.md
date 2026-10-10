# Bandwidth-Limited Pattern Recognition

**Which induction-motor fault signatures survive consumer-grade vibration sensing?**

CSE 455/555 Applied Pattern Recognition, Fall 2026 — Vaishak G Kumar, Sanika Vilas Najan.
Track Option 2: Dimensionality Reduction vs. Noise & Out-of-Distribution Data.

A smartphone IMU samples at 100 Hz, so nothing above 50 Hz survives. This project measures which of six motor health states (H, B1, B2, B3, V, R) remain separable under that truncation, which projection (PCA vs. LDA) preserves them, and how separability degrades under noise, sample dropout and gain drift. Classical pattern recognition only: hand-written features, no pre-trained models.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas pyarrow scikit-learn matplotlib pytest
python scripts/download_data.py        # 36 CSVs (~190 MB) -> data/raw/
```

## Run

```bash
python scripts/run_baseline.py                   # clean grid -> results/baseline/
python scripts/run_corruption.py --corrupt both  # train + test corrupted -> results/corruption_both/
python scripts/run_corruption.py --corrupt test  # train clean, test corrupted -> results/corruption_test/
python -m pytest                                 # tests
```

`run_baseline.py` caches features at `data/processed/features_clean.parquet`; delete it after changing feature or windowing code. Each results folder gets a `runlog.json` with seed, parameters, library versions and dataset version.

## Layout

| Path | Contents |
|---|---|
| `src/motorpr/data.py` | Loading, 1024-sample windows (50% overlap), per-window DC removal |
| `src/motorpr/features.py` | 21 time/frequency features per channel × 6 channels = 126 |
| `src/motorpr/corruption.py` | Noise, dropout and gain drift on raw recordings |
| `src/motorpr/projections.py` | Hand-written PCA, LDA (Ledoit–Wolf) and Mahalanobis whitening |
| `src/motorpr/evaluation.py` | Recording-grouped CV, model grid, metrics, separability |
| `scripts/` | Download and experiment entry points |
| `tests/` | Unit tests on synthetic signals |

## Validation

Overlapping windows from one recording are highly correlated, so all reported results use cross-validation grouped by recording. A naive random split is reported only to show the leakage: on clean data it scores about 1.00 macro-F1, against 0.26–0.49 for grouped splits.

## Data

Ertarğın, M., Orhan, A., Yıldırım, Ö., Chauhan, S., & Vashishtha, G. (2026). A smartphone-based vibration dataset for induction motor fault diagnosis under different speed and load conditions. *Data in Brief, 67*, 112916. https://doi.org/10.1016/j.dib.2026.112916 — data: https://doi.org/10.17632/rs4vz8n3t5 (CC BY 4.0). Not redistributed here; use the download script.
