"""Hand-implemented projections as scikit-learn transformers, so they fit inside CV folds.

PCA and LDA are written out (rather than imported) so every step can be derived:
- PCA: eigendecomposition of the sample covariance.
- LDA: generalised eigenproblem S_B v = lambda S_W v, with Ledoit–Wolf shrinkage of S_W.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import ledoit_wolf


class PCA(BaseEstimator, TransformerMixin):
    """n_components: int, or float in (0, 1) for the smallest k reaching that variance."""

    def __init__(self, n_components: int | float = 0.95):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X - self.mean_, rowvar=False)
        evals, evecs = linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = np.clip(evals[order], 0, None), evecs[:, order]
        self.explained_variance_ratio_full_ = evals / evals.sum()
        if isinstance(self.n_components, float):
            k = int(np.searchsorted(np.cumsum(self.explained_variance_ratio_full_),
                                    self.n_components) + 1)
        else:
            k = self.n_components
        self.k_ = k
        self.components_ = evecs[:, :k]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_


class LDA(BaseEstimator, TransformerMixin):
    """Fisher LDA projection to at most C-1 directions.

    shrinkage: "ledoit-wolf" (S_W shrunk toward a scaled identity, intensity estimated from
    the pooled within-class residuals), a float in [0, 1], or None for the plain estimate.
    """

    def __init__(self, n_components: int | None = None, shrinkage: str | float | None = "ledoit-wolf"):
        self.n_components = n_components
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        C, d = len(self.classes_), X.shape[1]
        mu = X.mean(axis=0)
        resid = np.empty_like(X)
        S_B = np.zeros((d, d))
        for c in self.classes_:
            m = y == c
            mc = X[m].mean(axis=0)
            resid[m] = X[m] - mc
            diff = (mc - mu)[:, None]
            S_B += m.sum() * diff @ diff.T
        S_B /= len(X)

        if self.shrinkage == "ledoit-wolf":
            S_W, self.shrinkage_ = ledoit_wolf(resid, assume_centered=True)
        else:
            S_W = resid.T @ resid / len(X)
            a = float(self.shrinkage or 0.0)
            S_W = (1 - a) * S_W + a * np.trace(S_W) / d * np.eye(d)
            self.shrinkage_ = a

        evals, evecs = linalg.eigh(S_B, S_W)
        order = np.argsort(evals)[::-1]
        k = min(self.n_components or C - 1, C - 1)   # rank(S_B) <= C-1
        self.eigenvalues_ = evals[order][:k]
        self.scalings_ = evecs[:, order][:, :k]
        self.mean_ = mu
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.scalings_


class Whitener(BaseEstimator, TransformerMixin):
    """Map x -> Sigma^{-1/2}(x - mu). Euclidean k-NN after this is Mahalanobis k-NN.

    Sigma is the Ledoit–Wolf shrunk covariance so the inverse exists in 126 dimensions.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        cov, _ = ledoit_wolf(X - self.mean_, assume_centered=True)
        evals, evecs = linalg.eigh(cov)
        self.W_ = evecs / np.sqrt(np.clip(evals, 1e-12, None))
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.W_
