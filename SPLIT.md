# Work Split

Vaishak and Sanika, agreed 2026-10-22. **Revised 2026-11-09 to balance the work 50/50; Sanika agreed 2026-11-13.** Report + code due **Nov 25, 2026**. Item numbers refer to `ROADMAP.md`.

The two tracks touch different files, so both can run at the same time without merge conflicts. Stick to the files you own; if you need a change in the other person's files, ask.

## Vaishak — infrastructure and robustness (items 1, 3)

**Owns:** `scripts/run_baseline.py`, `scripts/run_corruption.py`, `src/motorpr/evaluation.py`, `src/motorpr/projections.py`, `src/motorpr/corruption.py`, plus new modules for the OOD splits and Subspace Alignment.

- [ ] **1a.** Corruption sweep: compute the clean (level 0) case once, not once per mode.
- [ ] **1b.** Run corruption levels in parallel (target: full sweep in ~5–8 min, down from ~30).
- [ ] **1c.** Rerun the baseline and both sweeps over ~5 seeds; report mean ± std.
- [ ] **3a.** Cross-speed split (train 30 + 50 Hz, test 40 Hz) and cross-load split (train unloaded, test loaded).
- [ ] **3b.** Subspace Alignment across supply frequencies.
- [ ] **R1.** Report: methods, validation and robustness sections.

## Sanika — analysis, DTW and extensions (items 2, 4, 5)

**Owns:** new `src/motorpr/analysis.py`, new `scripts/run_analysis.py`, new `src/motorpr/kernel_pca.py`, new `src/motorpr/tuning.py`, their tests, `notebooks/`, and output under `results/analysis/`.
Use the cached features (`data/processed/features_clean.parquet`, made by `run_baseline.py`) and the helpers in `evaluation.py` (`recording_folds`, `fisher_criterion`, `separability`) **read-only**. One exception: 5a adds its entry to `front_ends()`.

- [x] **2a.** Per-feature, per-class-pair Fisher scores (6 classes → 15 pairs × 126 features). Which features make B1 separable? Score on held-out recordings with `transfer_fisher`. Plain Fisher on a test fold only measures how distinct the recordings are.
- [x] **2b.** R vs B2: do the rotor-bar sideband features actually differ between R and the other classes?
- [ ] **2c.** B3 and V: is any separability coming from aliased energy or from low-frequency secondary effects? Lead: the loaded 30 Hz recordings show strong 28/42 Hz peaks.
- [ ] **4a.** Write up the two DTW options with a recommendation: (a) drop it with a justification, or (b) use it as a DTW k-NN baseline on window envelopes.
- [ ] **4b.** If (b) is cheap, prototype it. Final call made together. If we choose (a), this slot becomes a fresh-clone run of the full pipeline, fixing whatever breaks.
- [x] **5a.** Kernel PCA (RBF) as a non-linear front-end, with its tests. Merge before 1c so the multi-seed rerun includes it.
- [ ] **5b.** Nested hyperparameter tuning (currently fixed: k=5, C=10, 300 trees).
- [ ] **T1.** Tests for `analysis.py`.
- [ ] **F1.** Final analysis figures and tables for the report.
- [ ] **R2.** Report: separability analysis section.
- [ ] **R3.** Report: DTW section.
- [ ] **R4.** Report: Kernel PCA and tuning results.

## Shared

- **Report sections:** assigned above (R1 to R4). Each reviews the other's.
- **Hand-offs:** Sanika's numbers are single-seed until 1c is merged; rerun them after. Item 3's OOD discussion reuses the confusion explanations from item 2, so share findings as they come in.
- **Rules (from `AGENTS.md`):** hand-written features and classical methods only, recording-grouped CV for every reported number, projections fit on training folds only.

## Git workflow

1. Get collaborator access to `Gustav-Proxi/Motor_Fault_PR`, then clone it and follow the setup in `README.md`.
2. Branch from `main`: `vaishak/<topic>` or `sanika/<topic>` (e.g. `sanika/fisher-pairs`).
3. Short, imperative commit messages (`Add per-pair Fisher scores`).
4. Open a PR into `main`; the other person reviews. Run `python -m pytest` before opening it.
5. **Squash-merge every PR**, so one task is one commit on `main`, credited to the PR's author. Tick the task's checkbox inside that PR.
6. `data/` and `results/` are gitignored. Put key figures and tables in the PR description, or commit them under `report/` once that exists.

## Commit balance

Target: **12 commits each on `main`**, one per task. Check with `git shortlog -sn --no-merges origin/main`.

| | Done | Still to do | Final |
|---|---|---|---|
| Vaishak | 6 (base pipeline, roadmap, 3 split docs, this revision) | 1a, 1b, 1c, 3a, 3b, R1 = 6 | **12** |
| Sanika | 0 | 2a, 2b, 2c, 4a, 4b, 5a, 5b, T1, F1, R2, R3, R4 = 12 | **12** |

- Everyone commits only work they did themselves, from their own account. Set `git config user.email` to an email on your GitHub account or the commits won't be credited to you.
- No empty, formatting-only or split-up commits to move the count.
- Review feedback is fixed by the PR's author, inside that PR.
- If a real extra fix is needed, the person who is behind in the count takes it when they can. If one person runs out of time, a whole task moves to the other and we say so here.

## Target dates

| By | Vaishak | Sanika |
|---|---|---|
| Nov 12 | 1a, 1b | 2a, T1 |
| Nov 15 | 3a | 5a, 2b |
| Nov 18 | 1c, 3b | 2c, 4a, 4b |
| Nov 21 | R1 | 5b, F1 |
| Nov 23 | Review Sanika's sections | R2, R3, R4 |
| Nov 24 | Final tally and check | Review Vaishak's sections |
