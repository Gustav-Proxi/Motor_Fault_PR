import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA as SkPCA

from motorpr.projections import LDA, PCA, Whitener

rng = np.random.default_rng(0)


def _blobs(n=200, d=10, C=4):
    means = rng.normal(0, 3, (C, d))
    X = np.concatenate([rng.normal(m, 1, (n, d)) for m in means])
    y = np.repeat(np.arange(C), n)
    return X, y


def test_pca_matches_sklearn_up_to_sign():
    X, _ = _blobs()
    ours = PCA(3).fit_transform(X)
    ref = SkPCA(3).fit_transform(X)
    for j in range(3):
        assert np.allclose(np.abs(ours[:, j]), np.abs(ref[:, j]), atol=1e-8)


def test_pca_variance_threshold():
    X, _ = _blobs()
    p = PCA(0.9).fit(X)
    cum = np.cumsum(p.explained_variance_ratio_full_)
    assert cum[p.k_ - 1] >= 0.9 and (p.k_ == 1 or cum[p.k_ - 2] < 0.9)


def test_lda_capped_at_c_minus_1():
    X, y = _blobs(C=4)
    assert LDA(10).fit_transform(X, y).shape[1] == 3


def test_lda_subspace_matches_sklearn():
    X, y = _blobs(C=4)
    ours = LDA(shrinkage=None).fit_transform(X, y)
    ref = LinearDiscriminantAnalysis(solver="eigen").fit_transform(X, y)
    # same subspace: regress one on the other, residual ~ 0
    coef, res, *_ = np.linalg.lstsq(np.c_[ours, np.ones(len(X))], ref, rcond=None)
    assert np.allclose(np.c_[ours, np.ones(len(X))] @ coef, ref, atol=1e-6)


def test_whitener_gives_identity_covariance_without_shrinkage_effects():
    X = rng.multivariate_normal([0, 0, 0], [[4, 1, 0], [1, 2, 0], [0, 0, 1]], 20_000)
    Z = Whitener().fit_transform(X)
    np.testing.assert_allclose(np.cov(Z, rowvar=False), np.eye(3), atol=0.05)
