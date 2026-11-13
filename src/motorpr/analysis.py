"""Separability analysis: which features, and which parts of the spectrum, separate which classes.

Every score here is measured on held-out recordings. The grouped folds from
``evaluation.recording_folds`` hold out one recording per class, and ``transfer_fisher``
takes the class means from the training fold and the spread from the held-out windows
around them. A plain Fisher ratio on a test fold would only say how distinct six individual
recordings are, not how well a class generalises to a recording it has not seen.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler

from .data import CLASSES, DATA_DIR, FS
from .evaluation import fisher_criterion, transfer_fisher
from .projections import LDA

META_COLS = ["rec_id", "label", "load", "supply_hz", "win_idx"]
FEATURE_CACHE = DATA_DIR / "processed" / "features_clean.parquet"
PAIRS = list(combinations(CLASSES, 2))          # 15 unordered class pairs


def load_feature_cache(path: Path = FEATURE_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the clean-feature cache written by ``scripts/run_baseline.py`` (read-only).

    Returns (meta, F): per-window metadata and the (N, 126) feature table.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/run_baseline.py first")
    df = pd.read_parquet(path)
    return df[META_COLS].reset_index(drop=True), df.drop(columns=META_COLS).reset_index(drop=True)


def split_feature_name(name: str) -> tuple[str, str]:
    """'gUserX_kurt' -> ('gUserX', 'kurt')."""
    channel, feature = name.split("_", 1)
    return channel, feature


# ---------------------------------------------------------------- 2a: per-feature pair Fisher

def pair_fisher_table(F: pd.DataFrame, meta: pd.DataFrame,
                      folds: list[tuple[np.ndarray, np.ndarray]],
                      pairs: list[tuple[str, str]] = PAIRS) -> pd.DataFrame:
    """Univariate two-class Fisher ratio for every feature, class pair and fold.

    For a single feature and two classes, J = sum_c p_c (m_c - m)^2 / s_W^2.
    j_heldout uses ``transfer_fisher``: m_c from the training fold, s_W^2 from held-out
    windows around those training means. j_train is the in-sample ratio on the training
    fold (5 recordings per class), for comparison. Both are invariant to feature scaling,
    so no standardisation is needed.

    Returns one row per (fold, class_a, class_b, feature).
    """
    X, y = F.to_numpy(dtype=float), meta["label"].to_numpy()
    names = list(F.columns)
    rows = []
    for k, (tr, te) in enumerate(folds):
        for a, b in pairs:
            mtr = np.isin(y[tr], (a, b))
            mte = np.isin(y[te], (a, b))
            Xtr, ytr = X[tr][mtr], y[tr][mtr]
            Xte, yte = X[te][mte], y[te][mte]
            for j, name in enumerate(names):
                rows.append((k, a, b, name,
                             transfer_fisher(Xtr[:, j], ytr, Xte[:, j], yte),
                             fisher_criterion(Xtr[:, j], ytr)))
    out = pd.DataFrame(rows, columns=["fold", "class_a", "class_b", "feature",
                                      "j_heldout", "j_train"])
    chan_feat = out["feature"].map(split_feature_name)
    out["channel"] = chan_feat.str[0]
    out["feat_type"] = chan_feat.str[1]
    return out


def summarise_pair_fisher(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (class pair, feature), sorted by held-out J.

    j_heldout is the *median* over the 6 folds. Each fold's S_W comes from a single held-out
    recording per class, and one recording that happens to sit next to the training mean
    makes S_W tiny and that fold's J explode (e.g. 88 against 0.5-2 elsewhere), so the
    mean is dominated by luck. j_heldout_min is the worst held-out fold.
    """
    keys = ["class_a", "class_b", "feature", "channel", "feat_type"]
    s = (table.groupby(keys)
         .agg(j_heldout=("j_heldout", "median"), j_heldout_min=("j_heldout", "min"),
              j_heldout_mean=("j_heldout", "mean"), j_train=("j_train", "mean"))
         .reset_index())
    # in-sample / held-out: large when a feature mostly tells individual recordings apart
    s["overfit_ratio"] = s["j_train"] / (s["j_heldout"] + 1e-12)
    return s.sort_values(["class_a", "class_b", "j_heldout"], ascending=[True, True, False],
                         ignore_index=True)


def class_vs_rest(summary: pd.DataFrame, classes: list[str] = CLASSES) -> pd.DataFrame:
    """For each class and feature: the weakest and mean held-out J over its 5 pairs.

    A feature that separates class c from *every* other class has a high ``j_min``; one
    that only separates c from some classes has a high ``j_mean`` but a low ``j_min``.
    """
    rows = []
    for c in classes:
        d = summary[(summary.class_a == c) | (summary.class_b == c)]
        g = d.groupby(["feature", "channel", "feat_type"]).j_heldout
        agg = g.agg(j_min="min", j_mean="mean").reset_index()
        agg["weakest_vs"] = [
            _other(d.loc[d[d.feature == f].j_heldout.idxmin()], c) for f in agg.feature]
        agg.insert(0, "cls", c)
        rows.append(agg)
    return (pd.concat(rows, ignore_index=True)
            .sort_values(["cls", "j_min"], ascending=[True, False], ignore_index=True))


def _other(row: pd.Series, c: str) -> str:
    return row.class_b if row.class_a == c else row.class_a


def by_feature_type(summary: pd.DataFrame) -> pd.DataFrame:
    """Best channel per (pair, feature type): which *kind* of feature separates each pair."""
    idx = summary.groupby(["class_a", "class_b", "feat_type"]).j_heldout.idxmax()
    return (summary.loc[idx, ["class_a", "class_b", "feat_type", "channel", "j_heldout"]]
            .sort_values(["class_a", "class_b", "j_heldout"], ascending=[True, True, False],
                         ignore_index=True))


def pair_separability(F: pd.DataFrame, meta: pd.DataFrame,
                      folds: list[tuple[np.ndarray, np.ndarray]],
                      pairs: list[tuple[str, str]] = PAIRS) -> pd.DataFrame:
    """Held-out separability of each class pair using all 126 features at once.

    Per fold: standardise on the training windows of the two classes, fit the two-class
    Fisher LDA direction (1-D, Ledoit–Wolf) on them, and score held-out windows along that
    direction with ``transfer_fisher``. Scoring in the full 126-D space instead is
    meaningless: one held-out recording per class leaves S_W with almost no spread in most
    directions and J runs into the thousands. Returns the median and worst fold per pair;
    a low J predicts that the two classes will be confused.
    """
    X, y = F.to_numpy(dtype=float), meta["label"].to_numpy()
    rows = []
    for k, (tr, te) in enumerate(folds):
        for a, b in pairs:
            itr, ite = tr[np.isin(y[tr], (a, b))], te[np.isin(y[te], (a, b))]
            sc = StandardScaler().fit(X[itr])
            lda = LDA(1).fit(sc.transform(X[itr]), y[itr])
            Ztr, Zte = lda.transform(sc.transform(X[itr])), lda.transform(sc.transform(X[ite]))
            rows.append(dict(fold=k, class_a=a, class_b=b,
                             j_heldout=transfer_fisher(Ztr, y[itr], Zte, y[ite]),
                             j_train=fisher_criterion(Ztr, y[itr])))
    d = pd.DataFrame(rows)
    return (d.groupby(["class_a", "class_b"])
            .agg(j_heldout=("j_heldout", "median"), j_heldout_min=("j_heldout", "min"),
                 j_train=("j_train", "mean"))
            .reset_index().sort_values("j_heldout", ignore_index=True))


def pair_matrix(pairs_df: pd.DataFrame, value: str = "j_heldout",
                classes: list[str] = CLASSES) -> pd.DataFrame:
    """Symmetric class x class matrix from a per-pair table (diagonal NaN)."""
    M = pd.DataFrame(np.nan, index=classes, columns=classes)
    for r in pairs_df.itertuples():
        M.loc[r.class_a, r.class_b] = M.loc[r.class_b, r.class_a] = getattr(r, value)
    return M


# ---------------------------------------------------------------- 2b: rotor-bar sidebands

def recording_medians(F: pd.DataFrame, meta: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-recording median of each feature in ``cols``, with label / load / supply_hz."""
    d = pd.concat([meta[["rec_id", "label", "load", "supply_hz"]], F[cols]], axis=1)
    return (d.groupby(["rec_id", "label", "load", "supply_hz"])[cols].median()
            .reset_index())


def condition_ranks(rec: pd.DataFrame, col: str, cls: str = "R") -> pd.DataFrame:
    """Where does ``cls`` sit among the 6 classes at each (load, supply) condition?

    Each condition has exactly one recording per class, so comparing within a condition
    removes the speed/load effect. rank 1 = largest value. z is cls's distance from the
    other recordings in units of their spread. Conditions where ``cls`` or fewer than two
    other classes are present (e.g. filtered out as unresolved) are skipped.
    """
    rows = []
    for (load, hz), d in rec.groupby(["load", "supply_hz"]):
        v = d.set_index("label")[col]
        if cls not in v.index or len(v) < 3:
            continue
        others = v.drop(cls)
        rows.append(dict(load=load, supply_hz=hz, value=v[cls], others_median=others.median(),
                         rank=int((v > v[cls]).sum() + 1),
                         z=(v[cls] - others.mean()) / (others.std() + 1e-12)))
    return pd.DataFrame(rows)


def recording_spectrum(x: np.ndarray, fs: float = FS, nperseg: int = 8192
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of a whole recording (n_samples, n_ch), summed over channels.

    8192-sample Hann segments give 0.012 Hz bins, fine enough to resolve pole-pass
    sidebands a few tenths of a hertz from the running speed. Each segment is mean-removed,
    which drops gravity the same way per-window DC removal does.
    """
    f, p = signal.welch(x, fs=fs, nperseg=nperseg, axis=0, detrend="constant")
    return f, p.sum(axis=1)


def refine_running_speed(f: np.ndarray, p: np.ndarray, supply_hz: float) -> float:
    """Shaft speed f1: strongest peak in 0.80-1.03 x synchronous speed (as in features.py),
    refined by a parabola through the log-power of the peak bin and its neighbours."""
    sync = supply_hz / 2.0
    m = np.flatnonzero((f >= 0.80 * sync) & (f <= 1.03 * sync))
    i = m[np.argmax(p[m])]
    a, b, c = np.log(p[i - 1:i + 2] + 1e-30)
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return float(f[i] + np.clip(shift, -0.5, 0.5) * (f[1] - f[0]))


def _band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    return float(p[(f >= lo) & (f <= hi)].sum())


def sideband_levels(f: np.ndarray, p: np.ndarray, supply_hz: float, half_bw: float = 0.1,
                    floor_span: float = 4.0) -> dict:
    """Rotor-bar sidebands at the physically predicted place, f1 +- f_p.

    slip s = 1 - f1 / f_sync and the pole-pass frequency f_p = 2 s f_supply (4-pole motor).
    Powers are summed over f +- half_bw.
      sbr_db   mean sideband power relative to the 1x carrier (dB)
      snr_db   mean sideband power relative to the local floor: the median PSD bin within
               f1 +- floor_span, scaled to the same bandwidth (dB)
      resolved False when f_p < 3 half_bw, i.e. the sidebands sit inside the carrier band
    """
    f1 = refine_running_speed(f, p, supply_hz)
    slip = 1.0 - f1 / (supply_hz / 2.0)
    fp = 2.0 * slip * supply_hz
    carrier = _band_power(f, p, f1 - half_bw, f1 + half_bw)
    lower = _band_power(f, p, f1 - fp - half_bw, f1 - fp + half_bw)
    upper = _band_power(f, p, f1 + fp - half_bw, f1 + fp + half_bw)
    near = (f >= f1 - floor_span) & (f <= f1 + floor_span)
    n_bins = int(((f >= f1 - half_bw) & (f <= f1 + half_bw)).sum())
    floor = float(np.median(p[near])) * n_bins
    sb = 0.5 * (lower + upper)
    return dict(f1=f1, slip=slip, f_pole=fp, resolved=bool(abs(fp) >= 3 * half_bw),
                sbr_db=10 * np.log10(sb / carrier), snr_db=10 * np.log10(sb / floor),
                lower_db=10 * np.log10(lower / floor), upper_db=10 * np.log10(upper / floor))


def alias(freq: float | np.ndarray, fs: float = FS) -> float | np.ndarray:
    """Apparent frequency of a tone at ``freq`` after sampling at fs (folds into [0, fs/2])."""
    return np.abs(freq - fs * np.round(np.asarray(freq) / fs))


def sideband_alias_overlap(f1: float, f_pole: float, fs: float = FS, kmax: int = 8,
                           tol: float = 0.1) -> str:
    """Aliased shaft harmonics k x f1 (above Nyquist) that land within ``tol`` Hz of a
    predicted sideband f1 +- f_pole, e.g. "upper=3x, lower=5x"; "" if none.

    For a 4-pole motor f_pole = 2 f_supply - 4 f1, so f1 + f_pole = 2 f_supply - 3 f1 and
    f1 - f_pole = 5 f1 - 2 f_supply. At a 50 Hz supply with fs = 100 Hz these are exactly
    the aliases of the 3rd and 5th shaft harmonics, whatever the slip.
    """
    hits = []
    for side, f_sb in (("upper", f1 + f_pole), ("lower", f1 - f_pole)):
        ks = [k for k in range(2, kmax + 1)
              if k * f1 > fs / 2 and abs(alias(k * f1, fs) - f_sb) <= tol]
        if ks:
            hits.append(f"{side}={'/'.join(f'{k}x' for k in ks)}")
    return ", ".join(hits)
