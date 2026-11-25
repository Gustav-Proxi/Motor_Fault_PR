import numpy as np
import pytest

from motorpr.dtw import dtw_distances, dtw_to_many, envelope, knn_predict, reference_subset

rng = np.random.default_rng(0)


def _dtw_reference(a, b, band):
    """Textbook O(T^2) recursion, for checking the vectorised version."""
    T = len(a)
    D = np.full((T + 1, T + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, T + 1):
        for j in range(1, T + 1):
            if abs(i - j) <= band:
                c = np.linalg.norm(a[i - 1] - b[j - 1])
                D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[T, T]


@pytest.mark.parametrize("band", [0, 1, 3, 20])
def test_matches_textbook_recursion(band):
    q = rng.normal(size=(12, 2))
    R = rng.normal(size=(4, 12, 2))
    ours = dtw_to_many(q, R, band)
    ref = [_dtw_reference(q, r, band) for r in R]
    np.testing.assert_allclose(ours, ref)


def test_zero_band_is_framewise_distance():
    q, r = rng.normal(size=(10, 3)), rng.normal(size=(10, 3))
    assert dtw_to_many(q, r[None], 0)[0] == pytest.approx(np.linalg.norm(q - r, axis=1).sum())


def test_identical_is_zero_and_distance_is_symmetric():
    A = rng.normal(size=(3, 16, 2))
    D = dtw_distances(A, A, band=4)
    np.testing.assert_allclose(np.diag(D), 0, atol=1e-12)
    np.testing.assert_allclose(D, D.T)


def test_warping_absorbs_a_time_shift():
    t = np.linspace(0, 4 * np.pi, 64)
    a = np.sin(t)[:, None]
    b = np.sin(t - 0.4)[:, None]          # same shape, shifted by ~2 frames
    no_warp = dtw_to_many(a, b[None], 0)[0]
    warp = dtw_to_many(a, b[None], 5)[0]
    assert warp < 0.3 * no_warp


def test_envelope_shape_and_level():
    w = np.zeros((2, 1024, 6))
    w[0, :, 3:] = 2.0 * np.cos(2 * np.pi * 10 * np.arange(1024) / 100)[:, None]
    E = envelope(w, frame=16)
    assert E.shape == (2, 64, 3)
    # frame RMS of a 10 Hz cosine over 16 samples is close to a / sqrt(2)
    assert np.median(E[0]) == pytest.approx(np.log10(2 / np.sqrt(2)), abs=0.05)
    assert (E[0] > E[1]).all()


def test_knn_vote_and_tie_break():
    D = np.array([[0.1, 0.2, 0.3], [0.5, 0.1, 0.2]])
    y = np.array(["a", "b", "b"])
    assert list(knn_predict(D, y, k=1)) == ["a", "b"]
    assert list(knn_predict(D, y, k=3)) == ["b", "b"]
    # two-way tie at k=2: the nearer neighbour's class wins
    assert list(knn_predict(D, y, k=2)) == ["a", "b"]


def test_reference_subset_keeps_every_recording():
    rec = np.repeat(["r1", "r2", "r3"], [10, 7, 4])
    idx = reference_subset(rec, stride=3)
    assert set(rec[idx]) == {"r1", "r2", "r3"}
    assert len(idx) == 4 + 3 + 2
