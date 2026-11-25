"""Hand-written RBF Kernel PCA as a scikit-learn transformer, so it fits inside CV folds.

Steps (Schölkopf et al., 1998):
1. K_ij = exp(-gamma * ||x_i - x_j||^2) on the n fit points.
2. Centre in feature space: K~ = K - 1_n K - K 1_n + 1_n K 1_n  (1_n = all entries 1/n).
3. Eigendecompose K~ = A diag(lambda) A^T; keep the top k eigenpairs.
4. A new point x projects onto component j as  z_j = sum_i A_ij k~(x, x_i) / sqrt(lambda_j),
   where k~ is the kernel row centred with the fit-set statistics. On the fit points this
   reduces to z_j = sqrt(lambda_j) A_ij, the usual kernel principal components.

The kernel matrix is n x n, so for speed the fit uses at most ``max_fit`` training windows,
drawn at random (seeded). Every window, train or test, is still projected through the
kernel against that fit set, and nothing from the test fold is ever used to fit.

Cost: with an integer ``n_components`` only the top k eigenpairs are computed. Fits are
also memoised on (training data, parameters): the model grids fit the same front-end on
the same fold once per classifier, and a fit is deterministic, so later ones reuse the
first. A different fold, corruption level or seed is different data or parameters and is
fitted afresh, so the cache cannot leak anything between folds.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin

_FIT_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_FIT_CACHE_SIZE = 32          # each entry holds ~max_fit x d floats (~1.5 MB at defaults)


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """exp(-gamma * ||a - b||^2) for every row pair. A: (n, d), B: (m, d) -> (n, m)."""
    sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.clip(sq, 0.0, None))


class KernelPCA(BaseEstimator, TransformerMixin):
    """RBF Kernel PCA.

    n_components: int, or float in (0, 1) for the smallest k whose eigenvalues reach that
                  fraction of the centred kernel's total variance (mirrors ``PCA``).
    gamma:        RBF width in 1 / (feature units)^2, or "scale" for 1 / (d * Var(X)),
                  the same default scikit-learn uses for RBF SVMs. On standardised
                  features Var(X) = 1, so "scale" gives 1/d.
    max_fit:      at most this many training rows are used to build the kernel matrix.
    random_state: seed for that subsample. Downstream scores move with it (kpca5 + linear
                  SVM: 0.26-0.35 macro-F1 over 5 seeds), so multi-seed runs should vary
                  it, e.g. via ``front_ends(seed=...)``.
    cache:        reuse an identical earlier fit (same data and parameters).
    """

    def __init__(self, n_components: int | float = 5, gamma: float | str = "scale",
                 max_fit: int = 1500, random_state: int = 0, cache: bool = True):
        self.n_components = n_components
        self.gamma = gamma
        self.max_fit = max_fit
        self.random_state = random_state
        self.cache = cache

    def _cache_key(self, X: np.ndarray) -> tuple:
        digest = hashlib.blake2b(np.ascontiguousarray(X).tobytes(), digest_size=16).hexdigest()
        return (digest, X.shape, self.n_components, self.gamma, self.max_fit, self.random_state)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        key = self._cache_key(X) if self.cache else None
        if key is not None and key in _FIT_CACHE:
            _FIT_CACHE.move_to_end(key)
            self.__dict__.update(_FIT_CACHE[key])
            return self

        if len(X) > self.max_fit:
            rng = np.random.default_rng(self.random_state)
            X = X[np.sort(rng.choice(len(X), self.max_fit, replace=False))]
        n, d = X.shape
        if self.gamma == "scale":
            var = X.var()
            self.gamma_ = 1.0 / (d * var) if var > 0 else 1.0
        else:
            self.gamma_ = float(self.gamma)

        K = rbf_kernel(X, X, self.gamma_)
        self.K_col_mean_ = K.mean(axis=0)            # mean_i k(x_i, x_j), per fit point j
        self.K_mean_ = self.K_col_mean_.mean()
        Kc = K - self.K_col_mean_[None, :] - self.K_col_mean_[:, None] + self.K_mean_
        total = float(np.trace(Kc))                  # = sum of all eigenvalues

        if isinstance(self.n_components, float):
            evals, evecs = linalg.eigh(Kc)
            evals, evecs = np.clip(evals[::-1], 0.0, None), evecs[:, ::-1]
            self.explained_variance_ratio_full_ = evals / evals.sum()
            k = int(np.searchsorted(np.cumsum(self.explained_variance_ratio_full_),
                                    self.n_components) + 1)
        else:
            k = min(int(self.n_components), n)
            evals, evecs = linalg.eigh(Kc, subset_by_index=[n - k, n - 1])
            evals, evecs = np.clip(evals[::-1], 0.0, None), evecs[:, ::-1]
        k = min(k, int((evals > 1e-12 * evals[0]).sum()))   # drop null directions
        self.k_ = k
        self.eigenvalues_ = evals[:k]
        self.explained_variance_ratio_ = self.eigenvalues_ / total
        self.alphas_ = evecs[:, :k] / np.sqrt(evals[:k])     # A_ij / sqrt(lambda_j)
        self.X_fit_ = X

        if key is not None:
            _FIT_CACHE[key] = {a: v for a, v in self.__dict__.items() if a.endswith("_")}
            if len(_FIT_CACHE) > _FIT_CACHE_SIZE:
                _FIT_CACHE.popitem(last=False)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Kx = rbf_kernel(X, self.X_fit_, self.gamma_)                    # (m, n)
        Kx = Kx - Kx.mean(axis=1, keepdims=True) - self.K_col_mean_[None, :] + self.K_mean_
        return Kx @ self.alphas_
