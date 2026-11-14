"""Nested, recording-grouped hyperparameter tuning (ROADMAP item 5b).

Outer loop: the usual 6 grouped folds (one held-out recording per class). Inner loop: the
outer-training recordings (5 per class) are dealt into 5 grouped folds the same way, and a
grid search picks the hyperparameters with the best inner macro-F1. The winner is refit on
the whole outer-training fold and scored once on the outer test recordings, which the
search never saw. Reported numbers therefore carry no selection bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from .evaluation import make_pipeline, recording_folds

# Defaults in evaluation.classifiers(): k=5, linear C=1, RBF C=10 / gamma="scale", RF 300
# fully grown trees (min_samples_leaf=1), GNB var_smoothing=1e-9. Each grid contains that
# default. Grids are kept small so the whole search runs on a laptop.
CLASSIFIER_GRIDS: dict[str, dict[str, list]] = {
    "knn_euc": {"clf__n_neighbors": [1, 5, 15, 45]},
    "knn_mah": {"clf__n_neighbors": [1, 5, 15, 45]},
    "svm_lin": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    # "scale" is ~0.008 on the 126 raw features and ~0.2 after a 5-D projection
    "svm_rbf": {"clf__C": [1.0, 10.0, 100.0], "clf__gamma": ["scale", 0.01, 0.1]},
    # n_jobs=1 per forest: the grid search runs its own workers
    "rf": {"clf__min_samples_leaf": [1, 10], "clf__n_jobs": [1]},
    "gnb": {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
}
# RBF width of the Kernel PCA front-end; "scale" = 1/126 ~ 0.008 on standardised features
FRONT_GRIDS: dict[str, dict[str, list]] = {
    "kpca5": {"proj__gamma": ["scale", 0.002, 0.03]},
}


def inner_folds(meta_tr: pd.DataFrame, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped folds inside an outer-training set (positional indices into meta_tr).

    ``recording_folds`` deals each class's recordings one per fold; with 5 recordings per
    class only 5 folds are filled, so the empty sixth is dropped.
    """
    meta_tr = meta_tr.reset_index(drop=True)
    return [(tr, te) for tr, te in recording_folds(meta_tr, "grouped", seed) if len(te)]


def param_grid(front_name: str, clf_name: str) -> dict[str, list]:
    return {**FRONT_GRIDS.get(front_name, {}), **CLASSIFIER_GRIDS[clf_name]}


def nested_cv(front_name: str, front: list, clf_name: str, clf: list,
              X: np.ndarray, y: np.ndarray, meta: pd.DataFrame,
              outer: list[tuple[np.ndarray, np.ndarray]], seed: int = 0, n_jobs: int = 2,
              train_stride: int = 1, memory=None) -> pd.DataFrame:
    """Nested CV for one front-end x classifier. One row per outer fold with the tuned and
    default macro-F1 on the outer test recordings and the chosen parameters.

    train_stride: fit on every train_stride-th window of each training recording (windows
                  overlap by 50%, so neighbours are near-duplicates); test on all windows.
                  Tuned and default models use the same subset.
    memory:       joblib.Memory (or path) for caching fitted front-ends across grid points,
                  so e.g. a Kernel PCA is fit once per (gamma, fold) instead of once per
                  classifier setting.
    """
    grid = param_grid(front_name, clf_name)
    win = meta["win_idx"].to_numpy()
    rows = []
    for k, (tr, te) in enumerate(outer):
        tr = tr[win[tr] % train_stride == 0]
        inner = inner_folds(meta.iloc[tr], seed + k)
        pipe = make_pipeline(front, clf).set_params(memory=memory)
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=n_jobs, refit=True)
        gs.fit(X[tr], y[tr])
        default = make_pipeline(front, clf).fit(X[tr], y[tr])
        rows.append(dict(
            fold=k, front=front_name, clf=clf_name,
            f1_tuned=f1_score(y[te], gs.predict(X[te]), average="macro"),
            f1_default=f1_score(y[te], default.predict(X[te]), average="macro"),
            f1_inner_best=gs.best_score_,
            params=_short(gs.best_params_),
        ))
    return pd.DataFrame(rows)


def _short(params: dict) -> str:
    """'clf__C' -> 'C', 'proj__gamma' -> 'kpca_gamma' (so it never collides with an SVM's)."""
    def name(k: str) -> str:
        step, p = k.split("__", 1)
        return f"kpca_{p}" if step == "proj" else p
    return ", ".join(f"{name(k)}={v}" for k, v in sorted(params.items())
                     if not k.endswith("n_jobs"))
