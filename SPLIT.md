# Work Split

Vaishak and Sanika, agreed 2026-10-22. Report + code due **Nov 25, 2026**. Item numbers refer to `ROADMAP.md`.

The two tracks touch different files, so both can run at the same time without merge conflicts. Stick to the files you own; if you need a change in the other person's files, ask.

## Vaishak — infrastructure and extensions (items 1, 3, 5)

**Owns:** `scripts/run_baseline.py`, `scripts/run_corruption.py`, `src/motorpr/evaluation.py`, `src/motorpr/projections.py`, `src/motorpr/corruption.py`, plus new modules for the OOD splits and Subspace Alignment.

- [ ] **1a.** Corruption sweep: compute the clean (level 0) case once, not once per mode.
- [ ] **1b.** Run corruption levels in parallel (target: full sweep in ~5–8 min, down from ~30).
- [ ] **1c.** Rerun the baseline and both sweeps over ~5 seeds; report mean ± std.
- [ ] **3a.** Cross-speed split (train 30 + 50 Hz, test 40 Hz) and cross-load split (train unloaded, test loaded).
- [ ] **3b.** Subspace Alignment across supply frequencies.
- [ ] **5a.** Kernel PCA (RBF) as a non-linear front-end.
- [ ] **5b.** Nested hyperparameter tuning (currently fixed: k=5, C=10, 300 trees).

## Sanika — analysis and DTW (items 2, 4)

**Owns:** new `src/motorpr/analysis.py`, new `scripts/run_analysis.py`, `notebooks/`, and output under `results/analysis/`.
Use the cached features (`data/processed/features_clean.parquet`, made by `run_baseline.py`) and the helpers in `evaluation.py` (`recording_folds`, `fisher_criterion`, `separability`) **read-only**.

- [ ] **2a.** Per-feature, per-class-pair Fisher scores (6 classes → 15 pairs × 126 features). Which features make B1 separable? Score on held-out recordings with `transfer_fisher`. Plain Fisher on a test fold only measures how distinct the recordings are (see `CLAUDE.md`).
- [ ] **2b.** R vs B2: do the rotor-bar sideband features actually differ between R and the other classes?
- [ ] **2c.** B3 and V: is any separability coming from aliased energy or from low-frequency secondary effects? Lead: the loaded 30 Hz recordings show strong 28/42 Hz peaks.
- [ ] **4a.** Write up the two DTW options with a recommendation: (a) drop it with a justification, or (b) use it as a DTW k-NN baseline on window envelopes.
- [ ] **4b.** If (b) is cheap, prototype it. Final call made together.

## Shared

- **Report sections:** Vaishak writes methods, validation and robustness; Sanika writes separability analysis and DTW. Each reviews the other's.
- **Hand-offs:** Sanika's numbers are single-seed until 1c is merged; rerun them after. Item 3's OOD discussion reuses the confusion explanations from item 2, so share findings as they come in.
- **Rules (from `AGENTS.md`):** hand-written features and classical methods only, recording-grouped CV for every reported number, projections fit on training folds only.

## Git workflow

1. Get collaborator access to `Gustav-Proxi/Motor_Fault_PR`, then clone it and follow the setup in `README.md`.
2. Branch from `main`: `vaishak/<topic>` or `sanika/<topic>` (e.g. `sanika/fisher-pairs`).
3. Short, imperative commit messages (`Add per-pair Fisher scores`).
4. Open a PR into `main`; the other person reviews. Run `python -m pytest` before opening it.
5. `data/` and `results/` are gitignored. Put key figures and tables in the PR description, or commit them under `report/` once that exists.
