import numpy as np
import pytest

from motorpr.corruption import corrupt

rng = np.random.default_rng(0)
X = rng.standard_normal((20_000, 6)) * [1, 2, 3, 1, 2, 3] + [0, 0, -1, 0, 0, 0]


def test_level_zero_is_identity():
    for m in ("noise", "dropout", "gain"):
        np.testing.assert_array_equal(corrupt(X, m, 0.0, np.random.default_rng(1)), X)


def test_noise_variance_scales_with_level():
    y = corrupt(X, "noise", 0.5, np.random.default_rng(1))
    added = (y - X).var(axis=0)
    np.testing.assert_allclose(added / X.var(axis=0), 0.5, rtol=0.05)


def test_dropout_fraction_and_finite():
    y = corrupt(X, "dropout", 0.3, np.random.default_rng(1))
    changed = (y != X).any(axis=1).mean()
    assert 0.28 < changed < 0.36
    assert np.isfinite(y).all()


def test_gain_applies_same_factor_to_paired_axes():
    y = corrupt(X, "gain", 0.5, np.random.default_rng(1))
    g = y / X
    np.testing.assert_allclose(g[:, :3], g[:, 3:])
    assert g.min() >= 0.5 - 1e-9 and g.max() <= 1.5 + 1e-9


def test_unknown_mode():
    with pytest.raises(ValueError):
        corrupt(X, "bogus", 0.1, rng)
