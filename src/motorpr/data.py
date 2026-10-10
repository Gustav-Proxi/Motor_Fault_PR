"""Loading recordings and cutting them into labelled windows.

File names follow ``<Class>_<Load>_<Speed>Hz.csv`` (load 1 = loaded, 0 = unloaded).
Each CSV is semicolon-separated with columns gX;gY;gZ;gUserX;gUserY;gUserZ in g, at 100 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FS = 100.0
CLASSES = ["H", "B1", "B2", "B3", "V", "R"]
CHANNELS = ["gX", "gY", "gZ", "gUserX", "gUserY", "gUserZ"]
WINDOW = 1024
STEP = 512

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"


@dataclass
class Recording:
    rec_id: str          # e.g. "B3_1_40Hz" — the grouping key for CV
    label: str           # one of CLASSES
    load: int            # 1 loaded, 0 unloaded
    supply_hz: int       # 30, 40 or 50
    signal: np.ndarray   # (n_samples, 6), raw g units


def parse_name(stem: str) -> tuple[str, int, int]:
    label, load, speed = stem.split("_")
    return label, int(load), int(speed.removesuffix("Hz"))


def load_recording(path: Path) -> Recording:
    label, load, supply = parse_name(path.stem)
    df = pd.read_csv(path, sep=";")
    return Recording(path.stem, label, load, supply, df[CHANNELS].to_numpy(dtype=np.float64))


def load_all(raw_dir: Path = RAW_DIR) -> list[Recording]:
    paths = sorted(raw_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no CSVs in {raw_dir}; run scripts/download_data.py")
    return [load_recording(p) for p in paths]


def make_windows(x: np.ndarray, size: int = WINDOW, step: int = STEP) -> np.ndarray:
    """Slice (n_samples, n_ch) into (n_windows, size, n_ch) with per-window DC removal.

    DC is removed per window rather than per recording so the gravity vector (and any slow
    orientation drift) is dropped without using statistics from outside the window.
    Amplitude is deliberately not normalised here: absolute vibration energy is a
    discriminative feature, and scaling is fit later inside each CV fold.
    """
    n = (len(x) - size) // step + 1
    idx = np.arange(size)[None, :] + step * np.arange(n)[:, None]
    w = x[idx]
    return w - w.mean(axis=1, keepdims=True)


def windows_with_meta(recs: list[Recording], size: int = WINDOW, step: int = STEP
                      ) -> tuple[np.ndarray, pd.DataFrame]:
    """Window every recording; returns windows (N, size, 6) and per-window metadata."""
    ws, metas = [], []
    for r in recs:
        w = make_windows(r.signal, size, step)
        ws.append(w)
        metas.append(pd.DataFrame({
            "rec_id": r.rec_id, "label": r.label, "load": r.load,
            "supply_hz": r.supply_hz, "win_idx": np.arange(len(w)),
        }))
    return np.concatenate(ws), pd.concat(metas, ignore_index=True)
