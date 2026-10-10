"""Hand-written time- and frequency-domain features, 21 per channel (126 per window).

Frequency features use two spectral estimates:
- Hann periodogram over the full 1024-sample window (0.098 Hz bins) for narrow-band
  quantities — the 1x/2x running-speed peaks and the rotor-bar sidebands.
- Welch PSD (256-sample Hann segments, 50% overlap) for smooth spectral-shape statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from .data import CHANNELS, FS

TIME_FEATURES = ["rms", "var", "skew", "kurt", "crest", "p2p", "shape", "impulse"]
FREQ_FEATURES = [
    "e1x", "e2x", "sideband",                       # narrow-band energies (fraction of total)
    "band_lo", "band_1x", "band_2x", "band_hi",     # band power in running-speed units
    "centroid", "spread", "entropy",                # Welch spectral shape
    "sk_max", "sk_mean",                            # Antoni spectral kurtosis summary
    "slip",                                         # 1 - f1 / f_sync
]
PEAK_HALF_BW = 0.3   # Hz, half-width of the 1x / 2x integration bands
SIDEBAND = (0.5, 3.0)  # Hz offset from 1x; covers 2sf pole-pass sidebands
EPS = 1e-20


def feature_names() -> list[str]:
    return [f"{c}_{f}" for c in CHANNELS for f in TIME_FEATURES + FREQ_FEATURES]


def time_features(w: np.ndarray) -> np.ndarray:
    """w: (N, L, C) zero-mean windows -> (N, C, 8)."""
    absw = np.abs(w)
    rms = np.sqrt((w ** 2).mean(axis=1))
    peak = absw.max(axis=1)
    mean_abs = absw.mean(axis=1)
    return np.stack([
        rms,
        w.var(axis=1),
        stats.skew(w, axis=1),
        stats.kurtosis(w, axis=1, fisher=False),
        peak / (rms + EPS),
        w.max(axis=1) - w.min(axis=1),
        rms / (mean_abs + EPS),
        peak / (mean_abs + EPS),
    ], axis=-1)


def _band(freqs: np.ndarray, p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sum p over freqs in [lo, hi) with per-window bounds. p: (N, C, F); lo/hi: (N, 1)."""
    m = (freqs[None, :] >= lo) & (freqs[None, :] < hi)      # (N, F)
    return np.einsum("ncf,nf->nc", p, m.astype(p.dtype))


def running_speed(freqs: np.ndarray, p: np.ndarray, supply_hz: np.ndarray) -> np.ndarray:
    """Estimate shaft speed f1 per window as the strongest peak near synchronous speed.

    The motor is 4-pole (sync = supply/2). Search 0.80–1.03 x sync on the channel-summed
    periodogram; the upper margin tolerates drive-frequency error seen in the recordings.
    """
    sync = supply_hz / 2.0
    lo, hi = 0.80 * sync, 1.03 * sync
    m = (freqs[None, :] >= lo[:, None]) & (freqs[None, :] <= hi[:, None])
    ps = np.where(m, p.sum(axis=1), -np.inf)
    return freqs[np.argmax(ps, axis=1)]


def spectral_kurtosis(w: np.ndarray, nperseg: int = 64) -> np.ndarray:
    """Antoni's STFT spectral kurtosis SK(f) = <|X|^4>/<|X|^2>^2 - 2. Returns (N, C, F)."""
    _, _, Z = signal.stft(w, fs=FS, nperseg=nperseg, noverlap=nperseg * 3 // 4, axis=1,
                          boundary=None, padded=False)
    # Z: (N, F, C, T) since the time axis was 1
    a2 = np.abs(Z) ** 2
    sk = (a2 ** 2).mean(axis=-1) / (a2.mean(axis=-1) ** 2 + EPS) - 2.0
    return np.moveaxis(sk, 1, 2)[..., 1:]   # drop DC bin


def freq_features(w: np.ndarray, supply_hz: np.ndarray) -> np.ndarray:
    """w: (N, L, C), supply_hz: (N,) -> (N, C, 13)."""
    N, L, C = w.shape
    f_p, p = signal.periodogram(w, fs=FS, window="hann", axis=1)    # (N, F, C)
    p = np.moveaxis(p, 1, 2)                                         # (N, C, F)
    total = p[..., 1:].sum(axis=-1) + EPS

    f1 = running_speed(f_p, p, supply_hz)[:, None]                   # (N, 1)
    e1 = _band(f_p, p, f1 - PEAK_HALF_BW, f1 + PEAK_HALF_BW) / total
    e2 = _band(f_p, p, 2 * f1 - PEAK_HALF_BW, 2 * f1 + PEAK_HALF_BW) / total
    side = (_band(f_p, p, f1 - SIDEBAND[1], f1 - SIDEBAND[0])
            + _band(f_p, p, f1 + SIDEBAND[0], f1 + SIDEBAND[1])) / total
    zero = np.zeros_like(f1)
    bands = [
        _band(f_p, p, zero + f_p[1], 0.5 * f1),
        _band(f_p, p, 0.5 * f1, 1.5 * f1),
        _band(f_p, p, 1.5 * f1, 2.5 * f1),
        _band(f_p, p, 2.5 * f1, zero + np.inf),
    ]
    bands = [b / total for b in bands]

    f_w, pw = signal.welch(w, fs=FS, nperseg=256, axis=1)
    pw = np.moveaxis(pw, 1, 2)[..., 1:]
    f_w = f_w[1:]
    pn = pw / (pw.sum(axis=-1, keepdims=True) + EPS)
    centroid = (pn * f_w).sum(axis=-1)
    spread = np.sqrt((pn * (f_w - centroid[..., None]) ** 2).sum(axis=-1))
    entropy = -(pn * np.log(pn + EPS)).sum(axis=-1) / np.log(pn.shape[-1])

    sk = spectral_kurtosis(w)
    slip = np.broadcast_to(1.0 - f1 / (supply_hz[:, None] / 2.0), (N, C))

    return np.stack([e1, e2, side, *bands, centroid, spread, entropy,
                     sk.max(axis=-1), sk.mean(axis=-1), slip], axis=-1)


def extract_features(w: np.ndarray, supply_hz: np.ndarray, batch: int = 512) -> pd.DataFrame:
    """Windows (N, L, 6) + per-window supply frequency -> DataFrame (N, 126)."""
    supply_hz = np.asarray(supply_hz, dtype=float)
    out = []
    for i in range(0, len(w), batch):
        wb, sb = w[i:i + batch], supply_hz[i:i + batch]
        f = np.concatenate([time_features(wb), freq_features(wb, sb)], axis=-1)  # (n, C, 21)
        out.append(f.reshape(len(wb), -1))
    return pd.DataFrame(np.concatenate(out), columns=feature_names())
