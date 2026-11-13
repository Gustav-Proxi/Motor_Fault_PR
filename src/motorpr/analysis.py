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
from sklearn.preprocessing import StandardScaler

from .data import CLASSES, DATA_DIR
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
