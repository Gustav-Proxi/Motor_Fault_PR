# Roadmap

Status as of 2026-10-10. Report + code due **Nov 25, 2026**.

## Where we are

The base pipeline runs end to end (see `README.md`): 6,264 windows × 126 hand-written features, hand-written PCA/LDA, 6 classifiers, recording-grouped CV, and corruption sweeps. All numbers below are **single-seed** and use grouped CV (chance macro-F1 = 0.17).

- **Leakage gap.** A naive random split scores ≈1.00 macro-F1; holding out whole recordings gives 0.26–0.49. Each class×condition is a single recording, so every grouped split is also a cross-condition test for the held-out class.
- **Confusions.** B1 separates best (recall 0.74–1.00). R is *not* easy, contrary to our hypothesis: it is confused with B2 in both directions. H, B3 and V blur together, which fits the Nyquist argument.
- **LDA fits recording identity.** At k=1 its in-sample Fisher J is 206, but on held-out recordings it is 1.8. At k=5, PCA and LDA both reach ≈16 held-out.
- **Corruption, train and test both corrupted:** noise and dropout barely matter even at 50%. Gain drift hurts, and LDA most of all (0.49 → 0.25).
- **Corruption, train clean and test corrupted:** the reverse. 10% noise or dropout already drops LDA to ≈0.2, and at 50% it falls below chance. Gain drift barely matters. Raw features with no projection degrade least.

## Next steps (in order)

### 1. Make the numbers trustworthy
- [ ] Corruption sweep: compute the clean (level 0) case once instead of once per mode.
- [ ] Run corruption levels in parallel; a full sweep should drop from ~30 min to ~5–8 min.
- [ ] Rerun the baseline and both sweeps over ~5 seeds and report mean ± std. Fold-to-fold std is up to ±0.2, so differences below ~0.05 are noise until then.

### 2. Explain the surprising results (core contribution: the Nyquist boundary)
- [ ] Per-feature, per-class-pair Fisher scores: what makes B1 separable?
- [ ] R vs B2 confusion: do the rotor-bar sideband features actually differ between R and the other classes?
- [ ] B3 and V: does any separability come from aliased energy or from low-frequency secondary effects? Lead: the loaded 30 Hz recordings show strong 28/42 Hz peaks.

### 3. Address the condition confound
- [ ] Explicit cross-speed split (train 30 + 50 Hz, test 40 Hz) and cross-load split (train unloaded, test loaded).
- [ ] Subspace Alignment across supply frequencies.

### 4. Decide on DTW (needs both of us)
The proposal applies DTW to load- and speed-transition segments, but every recording is a steady 15-minute run with no transitions. Options:
- (a) drop it, with a justification written in the report;
- (b) reuse it as a DTW k-NN baseline on window envelopes.

### 5. Remaining extensions
- [ ] Kernel PCA (RBF) as a non-linear front-end.
- [ ] Nested hyperparameter tuning; models currently use fixed defaults (k=5, C=10, 300 trees).

## Splitting the work

Agreed split, 2026-10-22; full task list and git workflow in `SPLIT.md`. The two tracks touch different files, so both can run in parallel on separate branches and merge into `main` through PRs.

### Vaishak — infrastructure and extensions (items 1, 3, 5)
Branch `vaishak/<topic>`. Owns `scripts/run_baseline.py`, `scripts/run_corruption.py`, `src/motorpr/evaluation.py`, `src/motorpr/projections.py`, and new modules for the OOD splits and Subspace Alignment.
- [ ] 1. Clean case computed once, parallel corruption levels, 5-seed reruns with mean ± std.
- [ ] 3. Cross-speed and cross-load splits; Subspace Alignment.
- [ ] 5. Kernel PCA (RBF); nested hyperparameter tuning.

### Sanika — analysis (items 2, 4)
Branch `sanika/<topic>`. Owns a new `src/motorpr/analysis.py`, `scripts/run_analysis.py`, `notebooks/`, and `results/analysis/`. Uses the existing feature cache and `evaluation.py` read-only; if something there needs to change, ask Vaishak instead of editing it.
- [ ] 2. Per-feature, per-class-pair Fisher scores; the R vs B2 sideband check; aliasing vs. low-frequency effects for B3 and V.
- [ ] 4. Write up DTW options (a) and (b) with a recommendation, and prototype (b) if it is cheap. We decide together.

### Hand-offs
- Sanika's numbers are single-seed until item 1 lands; rerun them once the multi-seed harness is merged.
- Item 3's OOD splits will reuse the confusion explanations from item 2, so share findings as they come in.
- Report sections follow the same split: Vaishak writes methods, validation and robustness; Sanika writes separability analysis and DTW. Each reviews the other's.
