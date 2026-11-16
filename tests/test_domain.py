import numpy as np
import pytest

from motorpr.domain import METHODS, _zscore, project
from motorpr.projections import PCA


def _domains(shift: float, rotate: bool, seed: int = 0):
    """Two Gaussian classes in 10-D. The class shift is shared by three dimensions, so it
    stays the top principal direction after z-scoring. The target is shifted, rescaled and
    (optionally) carries the class shift on three other dimensions."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], 300)
    Xs = rng.normal(0, 1, (600, 10))
    Xs[:, 0:3] += 4 * y[:, None]
    Xt = rng.normal(0, 1, (600, 10))
    Xt[:, (3, 4, 5) if rotate else (0, 1, 2)] += 4 * y[:, None]
    return Xs, (Xt * 3 + shift), y


def _transfer_acc(method: str, shift: float) -> float:
    Xs, Xt, y = _domains(shift, rotate=False)
    Zs, Zt = project(Xs, Xt, 2, method)
    mu = np.stack([Zs[y == c].mean(axis=0) for c in (0, 1)])
    pred = np.argmin(((Zt[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)
    return float((pred == y).mean())


def test_shapes_and_unknown_method():
    Xs, Xt, _ = _domains(0, False)
    for m in METHODS:
        Zs, Zt = project(Xs, Xt, 3, m)
        assert Zs.shape == (600, 3) and Zt.shape == (600, 3)
    with pytest.raises(ValueError):
        project(Xs, Xt, 3, "bogus")


def test_identical_domains_need_no_adaptation():
    Xs, _, _ = _domains(0, False)
    Zs_n, Zt_n = project(Xs, Xs, 2, "none")
    Zs_a, Zt_a = project(Xs, Xs, 2, "sa")
    np.testing.assert_allclose(Zt_n, Zs_n)
    # same subspace on both sides: M is diagonal +-1, so SA only flips signs
    np.testing.assert_allclose(np.abs(Zt_a), np.abs(Zs_a), atol=1e-8)


def test_zscore_fixes_a_shift_that_breaks_no_adaptation():
    assert _transfer_acc("none", shift=20) < 0.7
    for m in ("zscore", "sa"):
        assert _transfer_acc(m, shift=20) > 0.95


def test_sa_uses_the_least_squares_alignment_of_the_bases():
    # M = P_s^T P_t is the minimiser of ||P_s M - P_t||_F (P_s has orthonormal columns)
    Xs, Xt, _ = _domains(5, True)
    # same z-scoring as the module: the target's 2nd/3rd eigenvalues are nearly equal, so
    # even a 1e-12 difference in the input rotates those eigenvectors
    zs, zt = _zscore(Xs, Xs), _zscore(Xt, Xt)
    ps, pt = PCA(3).fit(zs), PCA(3).fit(zt)
    M, *_ = np.linalg.lstsq(ps.components_, pt.components_, rcond=None)
    Zs, Zt = project(Xs, Xt, 3, "sa")
    np.testing.assert_allclose(Zs, ps.transform(zs) @ M, atol=1e-8)
    np.testing.assert_allclose(Zt, pt.transform(zt), atol=1e-8)
