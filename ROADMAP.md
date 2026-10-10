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

Items 2 and 4 don't depend on 1 and 3, so one person can take the analysis (2, 4) while the other takes the infrastructure and extensions (1, 3, 5). To be agreed.
