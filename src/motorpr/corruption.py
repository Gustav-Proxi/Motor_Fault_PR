"""Signal corruptions applied to a whole raw recording (n_samples, 6) before windowing.

``level`` runs 0.0–0.5 for every mode. Channels i and i+3 (gX/gUserX, ...) are the same
physical axis, so axis-level effects (gain) are applied to both.
"""
from __future__ import annotations

import numpy as np

MODES = ("noise", "dropout", "gain")


def add_noise(x: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise with variance = level x per-channel signal variance.

    level 0.5 -> SNR of 3 dB; level 0.1 -> 10 dB.
    """
    sigma = np.sqrt(level) * x.std(axis=0)
    return x + rng.standard_normal(x.shape) * sigma


def dropout(x: np.ndarray, level: float, rng: np.random.Generator,
            gap_range: tuple[int, int] = (5, 50)) -> np.ndarray:
    """Remove ~level of samples in contiguous gaps (BLE packet loss), then linearly interpolate.

    Gaps hit all channels at once since a dropped packet carries every axis. Interpolation
    mimics what a logging app would do; it erases the gap's high-frequency content.
    """
    n = len(x)
    missing = np.zeros(n, dtype=bool)
    target = int(level * n)
    while missing.sum() < target:
        g = rng.integers(*gap_range, endpoint=True)
        s = rng.integers(0, n - g)
        missing[s:s + g] = True
    if not missing.any():
        return x.copy()
    keep = ~missing
    t = np.arange(n)
    return np.column_stack([np.interp(t, t[keep], x[keep, c]) for c in range(x.shape[1])])


def gain_drift(x: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Per-axis gain that drifts linearly across the recording: g(t) = 1 + level*(a + (b-a)t/T).

    a, b ~ U(-1, 1) per physical axis, so at level 0.5 an axis can read 50–150% of truth.
    """
    a, b = rng.uniform(-1, 1, size=(2, 3))
    ramp = np.linspace(0.0, 1.0, len(x))[:, None]
    g = 1.0 + level * (a + (b - a) * ramp)          # (n, 3)
    return x * np.concatenate([g, g], axis=1)


def corrupt(x: np.ndarray, mode: str, level: float, rng: np.random.Generator) -> np.ndarray:
    if level == 0:
        return x
    if mode == "noise":
        # Noise is scaled to the vibration, not to gravity, so measure std after DC removal.
        dc = x.mean(axis=0)
        return add_noise(x - dc, level, rng) + dc
    if mode == "dropout":
        return dropout(x, level, rng)
    if mode == "gain":
        return gain_drift(x, level, rng)
    raise ValueError(f"unknown corruption mode {mode!r}")
