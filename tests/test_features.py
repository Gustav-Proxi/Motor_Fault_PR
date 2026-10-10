import numpy as np
import pytest

from motorpr.data import FS, make_windows
from motorpr.features import extract_features, feature_names, freq_features, time_features

L = 1024
t = np.arange(L) / FS


def _win(*chans):
    return np.stack(chans, axis=-1)[None]   # (1, L, C)


def test_window_count_and_dc_removal():
    x = np.random.default_rng(0).standard_normal((10_000, 6)) + 5.0
    w = make_windows(x)
    assert w.shape == ((10_000 - 1024) // 512 + 1, 1024, 6)
    np.testing.assert_allclose(w.mean(axis=1), 0, atol=1e-12)


def test_window_boundaries():
    x = np.arange(3000, dtype=float)[:, None].repeat(6, axis=1)
    w = make_windows(x)
    # second window starts at sample 512; after DC removal its first value is -mean offset
    np.testing.assert_allclose(w[1, :, 0] + (512 + 511.5), np.arange(512, 512 + 1024))


def test_sine_time_features():
    a = 2.0
    # cosine so the sampled grid hits the true peak (crest factor uses the sampled max)
    s = a * np.cos(2 * np.pi * 10 * t)
    f = time_features(_win(s))[0, 0]
    rms, var, skew, kurt, crest, p2p, shape, impulse = f
    assert rms == pytest.approx(a / np.sqrt(2), rel=1e-2)
    assert var == pytest.approx(a ** 2 / 2, rel=1e-2)
    assert kurt == pytest.approx(1.5, rel=1e-2)          # sine kurtosis
    assert crest == pytest.approx(np.sqrt(2), rel=1e-2)
    # 10 samples/cycle: mean|cos| on the grid is 0.647, not the continuous 2/pi
    grid_mean_abs = np.abs(np.cos(2 * np.pi * np.arange(10) / 10)).mean()
    assert shape == pytest.approx((1 / np.sqrt(2)) / grid_mean_abs, rel=1e-3)
    assert abs(skew) < 1e-2


def test_running_speed_and_narrowband_energy():
    # 4-pole motor on 40 Hz supply, 3% slip -> 19.4 Hz shaft speed
    f1 = 19.4
    s = np.sin(2 * np.pi * f1 * t)
    f = freq_features(_win(*[s] * 6), np.array([40.0]))[0, 0]
    e1x, slip = f[0], f[-1]
    assert slip == pytest.approx(0.03, abs=0.01)
    assert e1x > 0.9


def test_sideband_energy_detects_rotor_bar_signature():
    f1 = 24.0
    carrier = np.sin(2 * np.pi * f1 * t)
    sb = 0.5 * (np.sin(2 * np.pi * (f1 - 2) * t) + np.sin(2 * np.pi * (f1 + 2) * t))
    clean = freq_features(_win(*[carrier] * 6), np.array([50.0]))[0, 0, 2]
    faulty = freq_features(_win(*[carrier + sb] * 6), np.array([50.0]))[0, 0, 2]
    assert faulty > 10 * max(clean, 1e-6)


def test_spectral_kurtosis_flags_impulses():
    rng = np.random.default_rng(1)
    gauss = rng.standard_normal(L)
    imp = gauss.copy()
    # sparse transients: SK responds to non-stationary energy, not to a dense impulse train
    imp[[200, 600, 900]] += 15.0
    f_g = freq_features(_win(*[gauss] * 6), np.array([30.0]))[0, 0]
    f_i = freq_features(_win(*[imp] * 6), np.array([30.0]))[0, 0]
    assert f_i[-3] > f_g[-3]   # sk_max


def test_extract_shape_and_names():
    w = np.random.default_rng(0).standard_normal((7, L, 6))
    F = extract_features(w, np.full(7, 30.0), batch=3)
    assert F.shape == (7, 126)
    assert list(F.columns) == feature_names()
    assert np.isfinite(F.to_numpy()).all()
