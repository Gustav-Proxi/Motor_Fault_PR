"""Cross-validation splits, model grid, metrics and separability measures.

Leakage rule: every learned step (scaler, projection, classifier) lives in one sklearn
Pipeline that is fit on the training fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, davies_bouldin_score,
                             f1_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import CLASSES
from .projections import LDA, PCA, Whitener

N_FOLDS = 6


# ---------------------------------------------------------------- splits

def recording_folds(meta: pd.DataFrame, mode: str = "grouped", seed: int = 0
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) pairs.

    grouped   — each class's 6 recordings are shuffled and dealt one per fold, so every
                test fold holds one whole recording per class (proposal protocol).
    condition — fold k holds every class's recording at speed/load condition k
                (leave-one-condition-out: test condition never seen in training).
    naive     — stratified random split over windows; leaks overlapping windows of the same
                recording across train/test. Reported only to measure that leakage.
    """
    if mode == "naive":
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        return list(skf.split(meta, meta["label"]))

    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    if mode == "grouped":
        for _, recs in meta.groupby("label")["rec_id"]:
            ids = np.array(sorted(recs.unique()))
            for k, rid in enumerate(rng.permutation(ids)):
                fold_of[rid] = k
    elif mode == "condition":
        conds = sorted(meta[["load", "supply_hz"]].drop_duplicates().itertuples(index=False))
        cidx = {tuple(c): k for k, c in enumerate(conds)}
        for rid, load, hz in meta[["rec_id", "load", "supply_hz"]].drop_duplicates().itertuples(index=False):
            fold_of[rid] = cidx[(load, hz)]
    else:
        raise ValueError(mode)

    f = meta["rec_id"].map(fold_of).to_numpy()
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(N_FOLDS)]


# ---------------------------------------------------------------- model grid

def front_ends(pca_k: int | float = 0.95) -> dict[str, list]:
    """Projection steps placed after standardisation."""
    return {
        "raw": [],
        f"pca{pca_k}": [("proj", PCA(pca_k))],
        "pca5": [("proj", PCA(5))],
        "lda5": [("proj", LDA(5))],
    }


def classifiers(seed: int = 0) -> dict[str, list]:
    return {
        "knn_euc": [("clf", KNeighborsClassifier(5))],
        "knn_mah": [("white", Whitener()), ("clf", KNeighborsClassifier(5))],
        "svm_lin": [("clf", SVC(kernel="linear", C=1.0))],
        "svm_rbf": [("clf", SVC(kernel="rbf", C=10.0, gamma="scale"))],
        "rf": [("clf", RandomForestClassifier(300, n_jobs=-1, random_state=seed))],
        "gnb": [("clf", GaussianNB())],
    }


def make_pipeline(front: list, clf: list) -> Pipeline:
    from sklearn.base import clone
    return Pipeline([("scale", StandardScaler())] + [(n, clone(s)) for n, s in front + clf])


# ---------------------------------------------------------------- metrics

def fisher_criterion(Z: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> float:
    """J = tr(S_W^{-1} S_B), the multi-class Fisher ratio. Ridge keeps S_W invertible."""
    Z = np.atleast_2d(Z.T).T
    mu = Z.mean(axis=0)
    d = Z.shape[1]
    S_W, S_B = np.zeros((d, d)), np.zeros((d, d))
    for c in np.unique(y):
        Zc = Z[y == c]
        mc = Zc.mean(axis=0)
        S_W += (Zc - mc).T @ (Zc - mc)
        S_B += len(Zc) * np.outer(mc - mu, mc - mu)
    S_W /= len(Z)
    S_B /= len(Z)
    S_W += ridge * np.trace(S_W) / d * np.eye(d)
    return float(np.trace(np.linalg.solve(S_W, S_B)))


def transfer_fisher(Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray, y_te: np.ndarray,
                    ridge: float = 1e-6) -> float:
    """Fisher ratio that generalises to unseen recordings.

    S_B comes from the training class means; S_W is the spread of *held-out* windows around
    those same training means. A test fold has one recording per class, so its own S_W lacks
    recording-to-recording variation and the plain held-out J is wildly inflated.
    """
    Z_tr, Z_te = np.atleast_2d(Z_tr.T).T, np.atleast_2d(Z_te.T).T
    d = Z_tr.shape[1]
    mu = Z_tr.mean(axis=0)
    S_B = np.zeros((d, d))
    resid = np.empty_like(Z_te)
    for c in np.unique(y_tr):
        mc = Z_tr[y_tr == c].mean(axis=0)
        S_B += (y_tr == c).mean() * np.outer(mc - mu, mc - mu)
        resid[y_te == c] = Z_te[y_te == c] - mc
    # Six test recordings leave S_W near-singular in high d (features constant within a
    # recording, e.g. slip, get ~0 spread), so shrink it like every other covariance here.
    S_W, _ = ledoit_wolf(resid, assume_centered=True)
    S_W += ridge * np.trace(S_W) / d * np.eye(d)
    return float(np.trace(np.linalg.solve(S_W, S_B)))


def separability(Z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "fisher": fisher_criterion(Z, y),
        "silhouette": float(silhouette_score(Z, y, sample_size=min(2000, len(Z)), random_state=0)),
        "davies_bouldin": float(davies_bouldin_score(Z, y)),
    }


def run_cv(pipe: Pipeline, X: np.ndarray, y: np.ndarray,
           folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Fit/predict per fold. Returns pooled predictions plus per-fold scores."""
    y_pred = np.empty_like(y)
    fold_acc, fold_f1 = [], []
    for tr, te in folds:
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[te])
        y_pred[te] = p
        fold_acc.append(accuracy_score(y[te], p))
        fold_f1.append(f1_score(y[te], p, average="macro"))
    # Folds may not cover every row (e.g. stacked clean-train / corrupted-test data), so
    # pool only the rows that were actually predicted.
    tested = np.unique(np.concatenate([te for _, te in folds]))
    return {
        "acc": float(np.mean(fold_acc)), "acc_std": float(np.std(fold_acc)),
        "macro_f1": float(np.mean(fold_f1)), "macro_f1_std": float(np.std(fold_f1)),
        "y_pred": y_pred,
        "confusion": confusion_matrix(y[tested], y_pred[tested], labels=CLASSES),
    }


def projected_separability(front: list, X: np.ndarray, y: np.ndarray,
                           folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Fit scaler+projection per fold and report, averaged over folds:

    fisher_heldout — transfer_fisher (held-out spread around training class means)
    *_train        — Fisher / Silhouette / Davies–Bouldin on the training fold itself.
                     In-sample, so optimistic for LDA, which maximises J on exactly this data.
    """
    rows = []
    for tr, te in folds:
        pipe = make_pipeline(front, [])
        pipe.fit(X[tr], y[tr])
        Z_tr, Z_te = pipe.transform(X[tr]), pipe.transform(X[te])
        s = {f"{k}_train": v for k, v in separability(Z_tr, y[tr]).items()}
        s["fisher_heldout"] = transfer_fisher(Z_tr, y[tr], Z_te, y[te])
        rows.append(s)
    return pd.DataFrame(rows).mean().to_dict()
