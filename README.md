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

Everything below regenerates every number, table and figure in the report from the raw data.

```bash
# 1. features (cached) + the clean model grid, seed 0
python scripts/run_baseline.py                   # -> results/baseline/

# 2. every experiment over 5 seeds (seed 0 -> results/<name>/, seed k -> results/<name>_seed<k>/)
for s in 0 1 2 3 4; do
  [ $s -gt 0 ] && python scripts/run_baseline.py --seed $s
  python scripts/run_corruption.py --corrupt both --seed $s   # train + test corrupted
  python scripts/run_corruption.py --corrupt test --seed $s   # train clean, test corrupted
  python scripts/run_ood.py --seed $s                         # cross-speed / cross-load splits
  python scripts/run_sa.py --seed $s                          # Subspace Alignment vs controls
done
python scripts/aggregate_seeds.py                # mean ± std over seeds -> results/multiseed/

# 3. separability analysis, DTW and nested tuning (seed 0)
python scripts/run_analysis.py                   # -> results/analysis/
python scripts/run_dtw.py                        # -> results/dtw/
python scripts/run_tuning.py                     # -> results/tuning/

# 4. report tables and figures, then the PDF
python scripts/make_report_figures.py            # separability, DTW, Kernel PCA, projection + confusion figures; tables
python scripts/make_methods_tables.py            # model grid, splits, OOD, SA tables; corruption figure
(cd report && latexmk -pdf main.tex)             # report/main.pdf

python scripts/export_ui_data.py                 # data bundle for the interactive UI -> ui/public/data/
python -m pytest                                 # tests
```

`run_baseline.py` caches features at `data/processed/features_clean.parquet`; delete it after changing feature or windowing code. Run it once before anything else: every other script reads that cache (the corruption sweeps re-extract features from the corrupted signal). The corruption sweeps and `run_sa.py` run their cells in parallel (`--n-jobs`, default all cores); `run_dtw.py` and `run_tuning.py` use one or two cores and save partial results, so an interrupted run resumes where it stopped. A seed changes the grouped fold assignment, the random-forest seed, the Kernel PCA subsample and the corruption noise. Results can differ slightly between machines (near-ties in the tuning grid search can flip), so the report's numbers all come from one run. Each results folder gets a `runlog.json` with seed, parameters, library versions and dataset version. On a 16-core node the whole sequence takes about an hour; on a laptop, several hours.

## Interactive UI

`ui/` is a single-page site that walks through the project. It covers the motor and its fault signatures, all 36 recordings with live spectra, aliasing, the validation protocol, a 3-D view of the PCA / LDA / Kernel PCA feature spaces, separability, the model grid, corruptions, DTW and tuning. It reads one data bundle exported from the results above:

```bash
python scripts/export_ui_data.py       # needs the feature cache and results/ (one core, a few seconds)
cd ui && npm install && npm run dev    # http://localhost:5173
```

`npm run build` writes a static site to `ui/dist/` (`npm run preview` serves it). The k-NN, corruption, DTW and spectrum demos run in the browser on the exported windows, and animations pause when they are off screen. The bundle includes the corruption-sweep curves once `run_corruption.py` results exist.

## Layout

| Path | Contents |
|---|---|
| `src/motorpr/data.py` | Loading, 1024-sample windows (50% overlap), per-window DC removal |
| `src/motorpr/features.py` | 21 time/frequency features per channel × 6 channels = 126 |
| `src/motorpr/corruption.py` | Noise, dropout and gain drift on raw recordings |
| `src/motorpr/projections.py` | Hand-written PCA, LDA (Ledoit–Wolf) and Mahalanobis whitening |
| `src/motorpr/evaluation.py` | Recording-grouped CV, model grid, metrics, separability |
| `src/motorpr/kernel_pca.py` | Hand-written RBF Kernel PCA front-end (`kpca5`) |
| `src/motorpr/analysis.py` | Held-out pair Fisher scores, rotor-bar sideband and aliasing checks |
| `src/motorpr/dtw.py` | Window envelopes and DTW k-NN |
| `src/motorpr/tuning.py` | Nested recording-grouped hyperparameter tuning |
| `src/motorpr/domain.py` | Subspace Alignment and per-domain z-scoring for unseen operating conditions |
| `scripts/` | Download and experiment entry points |
| `report/` | Report figures, tables and section sources |
| `ui/` | Interactive project site (React, Three.js); data from `scripts/export_ui_data.py` |
| `tests/` | Unit tests on synthetic signals |

## Validation

Overlapping windows from one recording are highly correlated, so all reported results use cross-validation grouped by recording. A naive random split is reported only to show the leakage: on clean data it scores about 1.00 macro-F1, against 0.26–0.49 for grouped splits.

## Data

Ertarğın, M., Orhan, A., Yıldırım, Ö., Chauhan, S., & Vashishtha, G. (2026). A smartphone-based vibration dataset for induction motor fault diagnosis under different speed and load conditions. *Data in Brief, 67*, 112916. https://doi.org/10.1016/j.dib.2026.112916 — data: https://doi.org/10.17632/rs4vz8n3t5 (CC BY 4.0). Not redistributed here; use the download script.
