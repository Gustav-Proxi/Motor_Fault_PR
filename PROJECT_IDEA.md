# Bandwidth-Limited Pattern Recognition

**Which induction-motor fault signatures survive consumer-grade vibration sensing?**

- **Course:** CSE 455/555 Applied Pattern Recognition, Fall 2026
- **Team:** Vaishak G Kumar (vaishakg@buffalo.edu), Sanika Vilas Najan (snajan@buffalo.edu)
- **Track:** Option 2, Dimensionality Reduction vs. Noise & Out-of-Distribution Data
- **Constraint:** classical PR only, no pre-trained models. Features are written by hand so every value can be derived and defended.
- **Source:** `../CSE455_Proposal.pdf` (submitted) and vault note `01-projects/CSE455-Vibration-PR`

---

## 1. Problem

This is a **six-class supervised classification** task. Given a short window of triaxial vibration from an induction motor, recorded by a consumer smartphone, assign it to one machine health state. Then find which linear or non-linear projection best keeps the states separable as the signal is corrupted and the operating conditions shift.

**Why it matters.** Electric motors used 53% of global electricity in 2023 (72% in industry, 87% in agriculture). Vibration monitoring catches faults before failure, but it normally needs industrial accelerometers plus a DAQ chain plus an analyst, which costs thousands of dollars per machine. Small workshops, farms, and pumping stations skip it and run machines until they fail. A smartphone is already at every one of those machines, but it is a much worse sensor.

**The core idea: Nyquist truncation.** Phone IMUs log at about 100 Hz, so nothing above 50 Hz can be recovered. For this motor (30/40/50 Hz supply, running speed about 14.5/19.3/24 Hz):

| Signature | Frequency | Below 50 Hz? |
|---|---|---|
| Broken rotor bar sidebands (1× ± 2sf) | 1–3 Hz spacing around the running speed | Yes |
| Lubrication loss (broadband energy, impulsiveness) | Captured by time-domain statistics | Yes |
| BPFO (outer race) | about 44–86 Hz | Mostly no |
| BPFI (inner race) | about 72–130 Hz | No |
| Voltage imbalance (2× line frequency) | 60/80/100 Hz | No |

So the usual question ("which classifier is best?") flips. The sampling rate truncates the feature space before any classifier sees it. The real question is **which class structure survives the truncation, and which projection preserves what is left.** The contribution is a *boundary*, not an accuracy number. That boundary matters because commercial smartphone monitoring products claim bearing-defect detection at sampling rates that physically cannot support it, which means missed faults with real safety consequences.

## 2. Dataset

Ertarğın et al. (2026), *Data in Brief*, 67, 112916. Mendeley: https://doi.org/10.17632/rs4vz8n3t5.1 (CC-BY 4.0).

| Property | Value |
|---|---|
| Sensor | iPhone 15 Pro Max IMU on a squirrel-cage induction motor |
| Sampling rate | 100 Hz (Nyquist 50 Hz) |
| Channels | 6: gx, gy, gz (raw) + guserx, gusery, guserz (gravity-compensated) |
| Classes | Healthy (H), insufficient lubrication (B1), severely insufficient lubrication (B2), cracked outer ring (B3), voltage imbalance (V), broken rotor bar (R) |
| Conditions | 6 classes × 3 speeds (30/40/50 Hz) × 2 loads = 36 recordings |
| Duration | 15 min per recording, about 90k samples per channel |
| Size | under 1 GB, fits in laptop memory |

- **Windows:** 1024 samples (10.24 s) at 50% overlap gives about 175 windows per recording and **about 6,300 labelled windows** in total. The 0.098 Hz frequency resolution is enough to resolve the 1–3 Hz rotor-bar sidebands.
- **Balance:** uniform, with 6 recordings per class.
- **Limitation:** one motor in a controlled lab. The dataset is new and uncited, so there is no classical PR baseline yet. Findings will be framed with that in mind.

## 3. Method

**Preprocessing:** per-channel DC removal and normalization, then sliding windows.

**Features (about 100–150 per window), per channel:**
- *Time domain:* RMS, variance, skewness, kurtosis, crest factor, peak-to-peak, shape factor, impulse factor
- *Frequency domain* (Hann periodogram and Welch): band power relative to running frequency, energy at 1× and 2× running speed, sideband energy in 1× ± 3 Hz, spectral centroid, spread, spectral kurtosis, spectral entropy

**Dimensionality reduction (the core comparison):**
- **PCA:** eigendecomposition of the sample covariance
- **LDA:** generalized eigenproblem on S_B and S_W. rank(S_B) caps it at C − 1 = 5 directions. Ledoit–Wolf shrinkage handles a near-singular S_W.
- **Kernel PCA (RBF):** tests whether a non-linear projection keeps the boundaries better under heavy noise
- **Subspace Alignment:** aligns projected bases across supply frequencies for cross-condition transfer

**Classifiers:** k-NN (Euclidean and Mahalanobis), soft-margin SVM (linear and RBF), Random Forest, Gaussian Naive Bayes (baseline).

**Corruption sweep (0% → 50%), applied to the raw signal:**
1. Additive Gaussian noise scaled to per-channel variance
2. Random windowed sample dropout (missing values, BLE packet gaps)
3. Per-axis gain drift (IMU miscalibration)

**OOD protocol:**
- Cross-speed: train on 30 and 50 Hz, test zero-shot on 40 Hz
- Cross-load: train unloaded, test loaded

The full pipeline is re-run at every corruption level and OOD split. This gives a degradation curve for every projection × classifier pair.

**Validation:** overlapping windows from the same recording are highly correlated, so a random split leaks information. Use **grouped CV with recording ID as the group** (6 folds, one recording per class per fold). Also report the naive random-split score, and treat the gap between the two as a finding in its own right.

**Tooling:** NumPy, SciPy (signal), pandas, scikit-learn, Matplotlib, all on CPU.

## 4. Evaluation

- **Classification:** accuracy, macro-F1, per-class precision and recall, 6×6 confusion matrices, one-vs-rest ROC/AUC. The confusion structure matters most, because the Nyquist argument predicts *specific* confusions rather than uniform degradation.
- **Separability:** Fisher criterion in raw, PCA, KPCA, and LDA spaces at bottleneck k ∈ [1, 5] (95%-variance PCA as the upper reference). Also scree plots, 2D projection scatters, Silhouette, and Davies–Bouldin.
- **Robustness and OOD:** accuracy and macro-F1 against corruption level for each mode, compared across front-ends and classifiers. Also the size of the cross-speed and cross-load drops.

## 5. Hypotheses

| Class | Expectation | Reason |
|---|---|---|
| R (broken rotor bar) | Well separated | Sidebands are below Nyquist |
| B1, B2 (lubrication) | Separate from H, but confused with each other | Differ in degree, not kind (energy and impulsiveness) |
| B3 (outer ring), V (voltage imbalance) | Hardest | Defining frequencies are above 50 Hz |

If B3 or V turn out to be separable anyway, inspect the discriminating features to check whether they rely on **aliased energy** or on **low-frequency secondary effects**. A negative or partial result is still a real finding about the limits of consumer-grade sensing.

## 6. Logistics

- **Proposal:** submitted (Sept 2026)
- **Report + code due:** Nov 25, 2026. Presentation in finals week.
- **Grading:** 10% proposal · 10% TA check-in · 30% report + code · 10% reproducibility · 40% presentation (defended individually)
- **Budget:** about 65 hrs over 12 weeks

## 7. References

1. Antoni, J., & Randall, R. B. (2006). The spectral kurtosis: Application to the vibratory surveillance and diagnostics of rotating machines. *MSSP*, 20(2), 308–331.
2. Duda, R. O., Hart, P. E., & Stork, D. G. (2001). *Pattern Classification* (2nd ed.). Wiley.
3. Ertarğın, M., et al. (2026). A smartphone-based vibration dataset for induction motor fault diagnosis under different speed and load conditions. *Data in Brief*, 67, 112916. https://doi.org/10.1016/j.dib.2026.112916
4. IEA 4E EMSA. (2025). Policy brief #9: Electric motor systems — why are they important? https://www.iea-4e.org/emsa/
5. Randall, R. B., & Antoni, J. (2011). Rolling element bearing diagnostics — A tutorial. *MSSP*, 25(2), 485–520.
6. Wheat, D., et al. (2024). Impact of data leakage in vibration signals used for bearing fault diagnosis. *IEEE Access*, 12, 169885–169895.
7. Welch, P. D. (1967). The use of FFT for the estimation of power spectra. *IEEE Trans. Audio Electroacoust.*, 15(2), 70–73.
