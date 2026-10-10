import numpy as np
import pytest
import pandas as pd

from motorpr.data import CLASSES
from motorpr.evaluation import fisher_criterion, recording_folds, transfer_fisher


def _meta(wins=10):
    rows = []
    for c in CLASSES:
        for load in (0, 1):
            for hz in (30, 40, 50):
                rid = f"{c}_{load}_{hz}Hz"
                rows += [dict(rec_id=rid, label=c, load=load, supply_hz=hz)] * wins
    return pd.DataFrame(rows)


def test_grouped_folds_never_share_recordings():
    m = _meta()
    for mode in ("grouped", "condition"):
        folds = recording_folds(m, mode, seed=3)
        assert len(folds) == 6
        seen = set()
        for tr, te in folds:
            assert set(m.rec_id.iloc[tr]).isdisjoint(m.rec_id.iloc[te])
            te_recs = set(m.rec_id.iloc[te])
            assert te_recs.isdisjoint(seen)
            seen |= te_recs
            # exactly one recording per class in every test fold
            assert m.iloc[te].groupby("label").rec_id.nunique().eq(1).all()
            assert m.iloc[te].label.nunique() == 6
        assert len(seen) == 36


def test_condition_folds_hold_out_one_condition():
    m = _meta()
    for _, te in recording_folds(m, "condition"):
        assert len(m.iloc[te][["load", "supply_hz"]].drop_duplicates()) == 1


def test_naive_folds_do_leak():
    m = _meta()
    tr, te = recording_folds(m, "naive")[0]
    assert not set(m.rec_id.iloc[tr]).isdisjoint(m.rec_id.iloc[te])


def test_fisher_higher_for_separated_classes():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 500)
    near = rng.normal(0, 1, (1000, 2)) + y[:, None] * 0.5
    far = rng.normal(0, 1, (1000, 2)) + y[:, None] * 5
    assert fisher_criterion(far, y) > 10 * fisher_criterion(near, y)


def test_transfer_fisher_penalises_shifted_test_recordings():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 500)
    tr = rng.normal(0, 1, (1000, 2)) + y[:, None] * 5
    same = rng.normal(0, 1, (1000, 2)) + y[:, None] * 5
    shifted = same + 4.0          # test recordings drift away from the training centres
    assert transfer_fisher(tr, y, same, y) == pytest.approx(fisher_criterion(tr, y), rel=0.2)
    assert transfer_fisher(tr, y, shifted, y) < 0.2 * transfer_fisher(tr, y, same, y)
