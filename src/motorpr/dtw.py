"""Dynamic Time Warping k-NN baseline on window envelopes (ROADMAP item 4, option b).

Each 1024-sample window is reduced to an amplitude envelope: the RMS of non-overlapping
``frame``-sample frames on the three gravity-compensated axes (gUserX/Y/Z), log-scaled.
With the default 16-sample frames that is a (64, 3) sequence at 6.25 Hz.

DTW aligns two envelopes with a Sakoe–Chiba band of half-width ``band`` frames:

    D(i, j) = ||a_i - b_j|| + min(D(i-1, j), D(i, j-1), D(i-1, j-1)),  |i - j| <= band,

and the distance is D(T, T). ``band=0`` forbids warping and reduces to the sum of
frame-wise Euclidean distances, the no-alignment control.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

USER_CHANNELS = (3, 4, 5)   # gUserX, gUserY, gUserZ in data.CHANNELS
EPS = 1e-12


def envelope(w: np.ndarray, frame: int = 16, channels: tuple[int, ...] = USER_CHANNELS
             ) -> np.ndarray:
    """Windows (N, L, 6) in g -> log10 frame-RMS envelopes (N, L // frame, len(channels))."""
    x = w[:, : (w.shape[1] // frame) * frame, list(channels)]
    N, L, C = x.shape
    rms = np.sqrt((x.reshape(N, L // frame, frame, C) ** 2).mean(axis=2))
    return np.log10(rms + EPS)


def dtw_to_many(q: np.ndarray, R: np.ndarray, band: int) -> np.ndarray:
    """DTW distance from one sequence q (T, c) to each of M sequences R (M, T, c) -> (M,).

    The recursion is filled one anti-diagonal (i + j = s) at a time: every cell on a
    diagonal depends only on the previous two, so each step is vectorised over the
    diagonal and over all M references.
    """
    M, T, _ = R.shape
    band = max(int(band), 0)
    cost = np.sqrt(((q[None, :, None, :] - R[:, None, :, :]) ** 2).sum(axis=-1))  # (M, T, T)
    D = np.full((M, T + 1, T + 1), np.inf)
    D[:, 0, 0] = 0.0
    for s in range(2, 2 * T + 1):
        i = np.arange(max(1, s - T), min(T, s - 1) + 1)
        j = s - i
        keep = np.abs(i - j) <= band
        i, j = i[keep], j[keep]
        if len(i) == 0:
            continue
        best = np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
        D[:, i, j] = cost[:, i - 1, j - 1] + best
    return D[:, T, T]


def dtw_distances(A: np.ndarray, B: np.ndarray, band: int) -> np.ndarray:
    """Pairwise DTW distances between sequences A (nA, T, c) and B (nB, T, c) -> (nA, nB)."""
    return np.stack([dtw_to_many(a, B, band) for a in A])


def knn_predict(D: np.ndarray, y_ref: np.ndarray, k: int = 1) -> np.ndarray:
    """Majority vote of the k nearest references per row of D (ties: nearest wins)."""
    nn = np.argsort(D, axis=1, kind="stable")[:, :k]
    out = []
    for row in nn:
        votes = Counter(y_ref[row])
        top = max(votes.values())
        out.append(next(y_ref[r] for r in row if votes[y_ref[r]] == top))
    return np.array(out)


def reference_subset(rec_ids: np.ndarray, stride: int) -> np.ndarray:
    """Indices of every ``stride``-th window within each recording (keeps 1/stride of them)."""
    rec_ids = np.asarray(rec_ids)
    idx = []
    for r in dict.fromkeys(rec_ids):
        where = np.flatnonzero(rec_ids == r)
        idx.append(where[::stride])
    return np.concatenate(idx)
