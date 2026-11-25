import numpy as np
import pytest
from sklearn.decomposition import KernelPCA as SkKernelPCA

from motorpr.evaluation import classifiers, fisher_criterion, front_ends, make_pipeline
from motorpr.kernel_pca import KernelPCA, rbf_kernel
from motorpr.projections import PCA

rng = np.random.default_rng(0)


def _rings(n=300, noise=0.05):
    """Two concentric circles (radius 1 and 3): not linearly separable in 2-D."""
    t = rng.uniform(0, 2 * np.pi, 2 * n)
    r = np.repeat([1.0, 3.0], n)
    X = np.c_[r * np.cos(t), r * np.sin(t)] + rng.normal(0, noise, (2 * n, 2))
    return X, np.repeat([0, 1], n)


def test_rbf_kernel_values():
    A = np.array([[0.0, 0.0], [1.0, 0.0]])
    K = rbf_kernel(A, A, gamma=0.5)
    np.testing.assert_allclose(np.diag(K), 1.0)
    assert K[0, 1] == pytest.approx(np.exp(-0.5))


def test_matches_sklearn_up_to_sign():
    X = rng.normal(size=(200, 6))
    X_new = rng.normal(size=(50, 6))
    ours = KernelPCA(4, gamma=0.2).fit(X)
    ref = SkKernelPCA(4, kernel="rbf", gamma=0.2, eigen_solver="dense").fit(X)
    for a, b in ((ours.transform(X), ref.transform(X)),
                 (ours.transform(X_new), ref.transform(X_new))):
        for j in range(4):
            assert np.allclose(np.abs(a[:, j]), np.abs(b[:, j]), atol=1e-8)


def test_fit_points_project_to_scaled_eigenvectors():
    X = rng.normal(size=(120, 3))
    kp = KernelPCA(3, gamma=0.3).fit(X)
    Z = kp.transform(X)
    # columns are uncorrelated and carry the kernel eigenvalues as squared norms
    G = Z.T @ Z
    np.testing.assert_allclose(np.diag(G), kp.eigenvalues_, rtol=1e-8)
    np.testing.assert_allclose(G - np.diag(np.diag(G)), 0, atol=1e-8)


def test_unfolds_concentric_rings_where_pca_cannot():
    X, y = _rings()
    j_pca = fisher_criterion(PCA(1).fit_transform(X), y)
    j_kpca = fisher_criterion(KernelPCA(1, gamma=0.5).fit_transform(X), y)
    assert j_pca < 0.1
    assert j_kpca > 20 * j_pca


def test_float_n_components_reaches_variance_fraction():
    X = rng.normal(size=(150, 5))
    kp = KernelPCA(0.8, gamma=0.1).fit(X)
    cum = np.cumsum(kp.explained_variance_ratio_full_)
    assert cum[kp.k_ - 1] >= 0.8 and (kp.k_ == 1 or cum[kp.k_ - 2] < 0.8)


def test_gamma_scale_is_inverse_dim_times_variance():
    X = rng.normal(0, 2.0, size=(100, 8))
    kp = KernelPCA(2).fit(X)
    assert kp.gamma_ == pytest.approx(1.0 / (8 * X.var()))


def test_subsample_bounds_kernel_size_and_is_seeded():
    X = rng.normal(size=(500, 4))
    a = KernelPCA(3, max_fit=100, random_state=1).fit(X)
    b = KernelPCA(3, max_fit=100, random_state=1).fit(X)
    assert a.X_fit_.shape == (100, 4)
    np.testing.assert_array_equal(a.X_fit_, b.X_fit_)
    assert a.transform(X).shape == (500, 3)


def test_kpca_front_end_runs_in_pipeline():
    X, y = _rings(n=100)
    pipe = make_pipeline(front_ends()["kpca5"], classifiers()["knn_euc"])
    pipe.fit(X[::2], y[::2])
    assert pipe.named_steps["proj"].k_ == 5
    assert (pipe.predict(X[1::2]) == y[1::2]).mean() > 0.95


def test_top_k_path_matches_full_spectrum():
    X = rng.normal(size=(150, 4))
    top = KernelPCA(3, gamma=0.3, cache=False).fit(X)
    full = KernelPCA(0.9999, gamma=0.3, cache=False).fit(X)
    np.testing.assert_allclose(top.eigenvalues_, full.eigenvalues_[:3], rtol=1e-8)
    np.testing.assert_allclose(top.explained_variance_ratio_,
                               full.explained_variance_ratio_full_[:3], rtol=1e-6)
    Zt, Zf = top.transform(X), full.transform(X)[:, :3]
    for j in range(3):
        assert np.allclose(np.abs(Zt[:, j]), np.abs(Zf[:, j]), atol=1e-8)


def test_fit_cache_reuses_only_identical_fits():
    X = rng.normal(size=(300, 4))
    a = KernelPCA(3, max_fit=100).fit(X)
    b = KernelPCA(3, max_fit=100).fit(X)                 # same data and parameters
    assert b.alphas_ is a.alphas_
    c = KernelPCA(3, max_fit=100).fit(X + 1e-9)          # different training data
    d = KernelPCA(3, max_fit=100, random_state=1).fit(X)  # different subsample
    assert c.alphas_ is not a.alphas_ and d.alphas_ is not a.alphas_
    e = KernelPCA(3, max_fit=100, cache=False).fit(X)
    assert e.alphas_ is not a.alphas_
    np.testing.assert_allclose(e.transform(X), a.transform(X))


def test_front_end_seed_sets_kpca_subsample():
    assert front_ends()["kpca5"][0][1].random_state == 0
    assert front_ends(seed=3)["kpca5"][0][1].random_state == 3
