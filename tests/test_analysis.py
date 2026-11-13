import numpy as np
import pandas as pd
import pytest

from motorpr import analysis as an
from motorpr.data import CLASSES, FS
from motorpr.evaluation import recording_folds

rng = np.random.default_rng(0)


def _dataset(wins=20):
    """36 synthetic recordings (6 classes x 3 speeds x 2 loads) with three features:

    gX_sep    B1 sits 4 sd above every other class (the planted signature)
    gX_noise  pure noise
    gY_rec    a per-recording offset, so it identifies recordings but not classes
    """
    meta, rows = [], []
    for c in CLASSES:
        for load in (0, 1):
            for hz in (30, 40, 50):
                rid = f"{c}_{load}_{hz}Hz"
                offset = rng.normal(0, 5)
                for k in range(wins):
                    meta.append(dict(rec_id=rid, label=c, load=load, supply_hz=hz, win_idx=k))
                    rows.append([rng.normal(4.0 if c == "B1" else 0.0, 1.0),
                                 rng.normal(0, 1), offset + rng.normal(0, 0.1)])
    return pd.DataFrame(meta), pd.DataFrame(rows, columns=["gX_sep", "gX_noise", "gY_rec"])


@pytest.fixture(scope="module")
def fisher():
    meta, F = _dataset()
    folds = recording_folds(meta, "grouped", seed=0)
    table = an.pair_fisher_table(F, meta, folds)
    return meta, F, folds, table, an.summarise_pair_fisher(table)


# ---------------------------------------------------------------- 2a

def test_split_feature_name():
    assert an.split_feature_name("gUserX_band_lo") == ("gUserX", "band_lo")


def test_table_covers_every_fold_pair_and_feature(fisher):
    _, F, _, table, _ = fisher
    assert len(table) == 6 * 15 * F.shape[1]
    assert set(table.channel) == {"gX", "gY"}


def test_planted_feature_tops_every_b1_pair(fisher):
    *_, s = fisher
    b1 = s[(s.class_a == "B1") | (s.class_b == "B1")]
    top = b1.groupby(["class_a", "class_b"]).head(1)
    assert (top.feature == "gX_sep").all()
    sep = b1[b1.feature == "gX_sep"].j_heldout
    noise = b1[b1.feature == "gX_noise"].j_heldout
    assert sep.min() > 2.0 and noise.max() < 0.2


def test_recording_identity_feature_does_not_transfer(fisher):
    *_, s = fisher
    rec = s[s.feature == "gY_rec"]
    sep = s[((s.class_a == "B1") | (s.class_b == "B1")) & (s.feature == "gX_sep")]
    # differs a lot between recordings, but that is not class information
    assert rec.j_heldout.median() < 0.5
    assert rec.j_heldout.max() < sep.j_heldout.min()


def test_class_vs_rest_uses_weakest_pair(fisher):
    *_, s = fisher
    cvr = an.class_vs_rest(s)
    b1 = cvr[cvr.cls == "B1"].iloc[0]
    assert b1.feature == "gX_sep" and b1.j_min > 2.0
    h = cvr[(cvr.cls == "H") & (cvr.feature == "gX_sep")].iloc[0]
    # gX_sep only separates H from B1, so its weakest H pair is near zero
    assert h.j_min < 0.2 and h.weakest_vs != "B1"


def test_by_feature_type_keeps_best_channel(fisher):
    *_, s = fisher
    ft = an.by_feature_type(s)
    assert len(ft) == 15 * 3
    row = ft[(ft.class_a == "H") & (ft.class_b == "B1") & (ft.feat_type == "sep")].iloc[0]
    assert row.channel == "gX"


def test_pair_separability_and_matrix(fisher):
    meta, F, folds, *_ = fisher
    sep = an.pair_separability(F, meta, folds)
    assert len(sep) == 15
    b1 = sep[(sep.class_a == "B1") | (sep.class_b == "B1")].j_heldout
    rest = sep[(sep.class_a != "B1") & (sep.class_b != "B1")].j_heldout
    assert b1.min() > 5 * rest.max()
    M = an.pair_matrix(sep)
    assert np.isnan(np.diag(M.to_numpy())).all()
    np.testing.assert_allclose(M.to_numpy(), M.to_numpy().T, equal_nan=True)


# ---------------------------------------------------------------- 2b

def test_condition_ranks_put_a_planted_class_first():
    meta, F = _dataset(wins=5)
    F = F.assign(gX_sb=np.where(meta.label == "R", 10.0, 0.0) + rng.normal(0, 1, len(F)))
    rec = an.recording_medians(F, meta, ["gX_sb"])
    assert len(rec) == 36
    ranks = an.condition_ranks(rec, "gX_sb", "R")
    assert len(ranks) == 6 and (ranks["rank"] == 1).all() and (ranks.z > 3).all()


def _tone_recording(f1, supply_hz, sideband_amp=0.0, n=60_000, noise=0.01):
    t = np.arange(n) / FS
    slip = 1 - f1 / (supply_hz / 2)
    fp = 2 * slip * supply_hz
    x = np.sin(2 * np.pi * f1 * t)
    x += sideband_amp * (np.sin(2 * np.pi * (f1 - fp) * t) + np.sin(2 * np.pi * (f1 + fp) * t))
    x = x[:, None] + rng.normal(0, noise, (n, 3))
    return an.recording_spectrum(x)


def test_running_speed_is_refined_below_bin_width():
    f, p = _tone_recording(24.2731, 50)
    assert an.refine_running_speed(f, p, 50) == pytest.approx(24.2731, abs=0.004)


def test_sideband_levels_find_pole_pass_sidebands():
    clean = an.sideband_levels(*_tone_recording(24.25, 50), supply_hz=50)
    faulty = an.sideband_levels(*_tone_recording(24.25, 50, sideband_amp=0.1), supply_hz=50)
    assert faulty["f_pole"] == pytest.approx(3.0, abs=0.02)
    assert faulty["resolved"]
    assert faulty["sbr_db"] == pytest.approx(-20, abs=1.5)      # 0.1 amplitude = -20 dB
    assert faulty["snr_db"] > clean["snr_db"] + 20


# ---------------------------------------------------------------- 2c

def test_alias_folds_into_nyquist_band():
    np.testing.assert_allclose(an.alias(np.array([43.0, 60.0, 80.0, 100.0, 130.0])),
                               [43.0, 40.0, 20.0, 0.0, 30.0])


def test_candidate_sources_mark_aliased_line_harmonic():
    src = an.candidate_sources(f1=19.3, supply_hz=40).set_index("source")
    assert src.loc["2x line", "apparent_hz"] == pytest.approx(20.0)
    assert src.loc["2x line", "aliased"] and not src.loc["1x f1", "aliased"]
    assert src.loc["BPFO", "true_hz"] == pytest.approx(an.BPFO_ORDER * 19.3)


def test_explain_frequency():
    src = an.candidate_sources(f1=14.2, supply_hz=30)
    assert an.explain_frequency(28.4, src) == "2x f1"
    assert an.explain_frequency(40.0, src) == "alias:2x line"
    assert an.explain_frequency(33.3, src) == ""


def test_spectral_peaks_and_excess():
    t = np.arange(60_000) / FS
    x = (np.sin(2 * np.pi * 12.0 * t) + 0.3 * np.sin(2 * np.pi * 31.0 * t))[:, None]
    f, p = an.recording_spectrum(x + rng.normal(0, 0.01, (len(t), 2)))
    peaks = an.spectral_peaks(f, p, n=2)
    np.testing.assert_allclose(sorted(peaks.freq), [12.0, 31.0], atol=0.02)
    assert an.excess_over_reference(f, 2 * p, p, 12.0) == pytest.approx(10 * np.log10(2))




def test_sidebands_at_50hz_supply_sit_on_aliased_shaft_harmonics():
    # 4-pole motor: f1 + f_p = 2 f_supply - 3 f1, which is alias(3 f1) when f_supply = fs/2
    for f1 in (24.05, 24.3, 23.9):
        fp = 2 * 50 - 4 * f1
        assert an.sideband_alias_overlap(f1, fp) == "upper=3x, lower=5x"
    # at 30 Hz supply under load the sidebands land on no aliased harmonic
    assert an.sideband_alias_overlap(13.97, 2 * 30 - 4 * 13.97) == ""


def test_peak_category_prefers_in_band_sources():
    assert an.peak_category("") == "unexplained"
    assert an.peak_category("2x f1 / 1x line") == "in-band shaft"
    assert an.peak_category("alias:4x f1 / alias:2x line") == "aliased"
    assert an.peak_category("1x f1 / alias:2x line") == "in-band shaft"
    assert an.peak_category("1x line") == "in-band line"


def test_region_masks_partition_the_band_without_overlap():
    freqs = np.fft.rfftfreq(1024, 1 / FS)
    masks = an.region_masks(freqs, np.array([14.0, 19.9, 24.0]), np.array([30, 40, 50]))
    stacked = np.stack(list(masks.values()))              # (regions, N, F)
    assert (stacked.sum(axis=0) <= 1).all()                # no bin counted twice
    covered = stacked.any(axis=0)
    np.testing.assert_array_equal(covered, (freqs >= 0.5)[None, :].repeat(3, axis=0))
    # 30 Hz supply: 2x line (60 Hz) appears at 40 Hz. 50 Hz supply: 2x line folds to DC and
    # is lost; only 3x line (150 Hz -> 50 Hz, at Nyquist) leaves any line_alias bins.
    k40 = np.argmin(np.abs(freqs - 40.0))
    assert masks["line_alias"][0, k40]
    assert freqs[masks["line_alias"][2]].min() >= 50 - 0.3
    # 3 x 24 = 72 Hz is above Nyquist and appears at 28 Hz
    k28 = np.argmin(np.abs(freqs - 28.0))
    assert masks["shaft_alias"][2, k28] and masks["shaft"][0, np.argmin(np.abs(freqs - 42.0))]


def test_region_energies_find_aliased_harmonic_energy():
    t = np.arange(1024) / FS
    f1 = 24.0
    base = np.sin(2 * np.pi * f1 * t)
    w = np.zeros((2, 1024, 2))
    w[0, :, 0] = base
    w[1, :, 0] = base + np.sin(2 * np.pi * (100 - 3 * f1) * t)   # 3 x f1 folded to 28 Hz
    R = an.region_energies(w, np.array([50.0, 50.0]))
    assert list(R.columns) == [f"region_{k}" for k in an.REGIONS] + ["region_total"]
    assert R.region_shaft_alias[1] > R.region_shaft_alias[0] + 2      # > 100x larger share
    assert R.region_shaft[0] > -0.01                                   # nearly all at 1x
    shares = 10 ** R[[f"region_{k}" for k in an.REGIONS]].to_numpy()
    assert (shares.sum(axis=1) <= 1 + 1e-6).all()
