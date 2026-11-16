"""Subspace Alignment for unsupervised domain adaptation (Fernando et al., 2013).

Source = training operating conditions, target = an unseen speed or load. The target's
*labels* are never used, but its windows are (transductive): that is the setting SA is
defined for, and a deployed sensor would have unlabeled data from its own machine.

1. z-score each domain with its own mean and std.
2. P_s, P_t = top-k PCA eigenvectors (d x k) of each domain.
3. M = P_s^T P_t aligns the source basis to the target basis.
4. Source windows map to z_s P_s M, target windows to z_t P_t; a classifier trained on the
   first is applied to the second.

``project`` also returns two controls on the same k: "none" (one scaler and one PCA fit on
the source, applied to the target, as in every other experiment) and "zscore" (per-domain
z-scoring, then the source PCA for both). "zscore" isolates how much of any gain comes from
re-centring the target alone, before any alignment.
"""
from __future__ import annotations

import numpy as np

from .projections import PCA

METHODS = ("none", "zscore", "sa")


def _zscore(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (X - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-12)


def project(Xs: np.ndarray, Xt: np.ndarray, k: int, method: str
            ) -> tuple[np.ndarray, np.ndarray]:
    """Map source Xs and target Xt into a shared k-D space. -> (Zs, Zt)"""
    if method == "none":
        zs, zt = _zscore(Xs, Xs), _zscore(Xt, Xs)
        p = PCA(k).fit(zs)
        return p.transform(zs), p.transform(zt)
    zs, zt = _zscore(Xs, Xs), _zscore(Xt, Xt)
    ps = PCA(k).fit(zs)
    if method == "zscore":
        return ps.transform(zs), ps.transform(zt)
    if method == "sa":
        pt = PCA(k).fit(zt)
        M = ps.components_.T @ pt.components_                      # (k, k)
        return ps.transform(zs) @ M, pt.transform(zt)
    raise ValueError(f"unknown method {method!r}")
