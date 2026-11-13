# DTW: options and recommendation

ROADMAP item 4, task 4a in `SPLIT.md`. We make the final call together.

## What the proposal promised

> Dynamic Time Warping is applied to compare load-transition and speed-change segments across conditions, where variable-length alignment is required.
> (`CSE455_Proposal_VaishakGKumar.pdf`, Sec. III, Classifiers)

## Why that plan cannot run on this dataset

- **There are no transitions.** Each of the 36 recordings is a steady 15-minute run at one supply frequency and one load. Speed and load never change within a file, and nothing is recorded between conditions. So there are no load-transition or speed-change segments to align.
- **There is no variable-length alignment problem.** Every sample is a fixed 1024-sample window at a fixed 100 Hz. DTW is useful when the same event unfolds at different speeds. Here nothing has a start or an end to line up.
- **Steady-state vibration has no shared phase.** Two windows from the same recording begin at arbitrary points in the shaft cycle, so their raw waveforms do not line up and are not meant to. Any information in a window is in its *statistics* (level, spectrum, impulsiveness), which the hand-written features already measure.

## Option (a): drop DTW, with a justification in the report

Text for the report: *"The proposal planned DTW for load- and speed-transition segments. The published recordings are steady-state runs at fixed conditions, with no transition segments and fixed-length windows, so there is no time-alignment problem for DTW to solve. We therefore do not use it."*

- Cost: none.
- Risk: a method promised in the proposal disappears. The justification above is factual and short, which keeps that risk small.

## Option (b): DTW k-NN baseline on window envelopes

Use DTW as a *distance* for a k-NN classifier, applied to a short sequence derived from each window, not as an alignment tool for transitions.

- **Sequence.** Frame-RMS envelope of the three gravity-compensated axes: non-overlapping 16-sample frames give a (64, 3) series at 6.25 Hz, log-scaled.
- **Distance.** Multivariate DTW (Euclidean local cost) with a Sakoe–Chiba band of ±8 frames (12.5%).
- **Classifier.** 1-NN and 5-NN. DTW is O(T·band) per pair, so both sides are subsampled. The references are every 12th window of each training recording (~450). The queries are every 4th window of each held-out recording (1,584 in total). Windows overlap by 50%, so neighbouring windows are near-duplicates.
- **Validation.** The same grouped folds as everything else (`recording_folds`, one held-out recording per class).
- **Controls.**
  1. The same k-NN with `band = 0` (no warping). If DTW helps, it must beat this.
  2. Euclidean k-NN on the 126 standardised hand-written features, using the same references.
- **Expected cost.** 1,584 queries × ~450 references × (64 × 17) cells ≈ 8·10⁸ cell updates. Vectorised along anti-diagonals in NumPy, that is about a minute on one CPU core, so it counts as cheap.
- **Expected result.** Warping should add little or nothing over `band = 0`, for the phase reason above. The envelope mostly carries the RMS level, which separates lubrication faults (energy) and not much else. That makes it a useful *negative control*: it shows that the separability we report comes from spectral and statistical features, not from waveform shape.

## Recommendation

**(b), kept small, with (a)'s justification text reused as its framing.** Reasons:

1. It keeps a method the proposal promised and the TA may look for, but presents it honestly: DTW applied where it can apply (as a distance between window envelopes), with the reason the original use is impossible.
2. It answers a real question cheaply: *does elastic alignment of the amplitude envelope carry class information that the hand-written features miss?* Comparing against `band = 0` isolates the effect of warping itself.
3. The report cost is one paragraph and one table row (R3).

A result where DTW ties `band = 0` and trails the feature baselines still counts: that is the negative control working.

**Fall back to (a)** only if the 4b prototype is too slow to rerun (more than ~10 minutes for the full grouped CV) or too unstable to report (fold-to-fold spread larger than the gaps it is meant to show). In that case the 4b slot becomes the fresh-clone run of the whole pipeline, as `SPLIT.md` says.
