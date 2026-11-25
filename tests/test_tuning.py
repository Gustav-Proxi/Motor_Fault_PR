import numpy as np
import pandas as pd

from motorpr.data import CLASSES
from motorpr.evaluation import classifiers, front_ends, recording_folds
from motorpr.tuning import CLASSIFIER_GRIDS, inner_folds, nested_cv, param_grid


def _meta(wins=8):
    rows = []
    for c in CLASSES:
        for load in (0, 1):
            for hz in (30, 40, 50):
                rows += [dict(rec_id=f"{c}_{load}_{hz}Hz", label=c, load=load, supply_hz=hz,
                              win_idx=k) for k in range(wins)]
    return pd.DataFrame(rows)


def test_inner_folds_stay_inside_outer_training_set_and_group_by_recording():
    m = _meta()
    for tr, te in recording_folds(m, "grouped", seed=1):
        m_tr = m.iloc[tr].reset_index(drop=True)
        inner = inner_folds(m_tr, seed=2)
        assert len(inner) == 5
        outer_test = set(m.rec_id.iloc[te])
        seen = set()
        for itr, ite in inner:
            assert ite.max() < len(m_tr)
            tr_ids, te_ids = set(m_tr.rec_id.iloc[itr]), set(m_tr.rec_id.iloc[ite])
            assert tr_ids.isdisjoint(te_ids)
            assert te_ids.isdisjoint(outer_test) and tr_ids.isdisjoint(outer_test)
            assert m_tr.iloc[ite].groupby("label").rec_id.nunique().eq(1).all()
            seen |= te_ids
        assert len(seen) == 30


def test_every_grid_contains_the_current_default():
    defaults = {"knn_euc": ("clf__n_neighbors", 5), "knn_mah": ("clf__n_neighbors", 5),
                "svm_lin": ("clf__C", 1.0), "svm_rbf": ("clf__C", 10.0),
                "rf": ("clf__min_samples_leaf", 1), "gnb": ("clf__var_smoothing", 1e-9)}
    for name, (key, value) in defaults.items():
        assert value in CLASSIFIER_GRIDS[name][key]
    assert "proj__gamma" in param_grid("kpca5", "knn_euc")
    assert "proj__gamma" not in param_grid("pca5", "knn_euc")


def test_nested_cv_reports_every_outer_fold():
    rng = np.random.default_rng(0)
    m = _meta(wins=12)
    y = m.label.to_numpy()
    centres = {c: rng.normal(0, 3, 4) for c in CLASSES}
    X = np.stack([centres[c] for c in y]) + rng.normal(0, 1, (len(m), 4))
    outer = recording_folds(m, "grouped", seed=0)
    res = nested_cv("raw", front_ends()["raw"], "knn_euc", classifiers()["knn_euc"],
                    X, y, m, outer, n_jobs=1)
    assert list(res.fold) == list(range(6))
    assert (res.f1_tuned > 0.8).all()
    assert res.params.str.startswith("n_neighbors=").all()


def test_train_stride_halves_training_windows_but_scores_all_test_windows(tmp_path):
    rng = np.random.default_rng(1)
    m = _meta(wins=12)
    y = m.label.to_numpy()
    centres = {c: rng.normal(0, 3, 4) for c in CLASSES}
    X = np.stack([centres[c] for c in y]) + rng.normal(0, 1, (len(m), 4))
    outer = recording_folds(m, "grouped", seed=0)[:2]
    res = nested_cv("kpca5", front_ends()["kpca5"], "knn_euc", classifiers()["knn_euc"],
                    X, y, m, outer, n_jobs=1, train_stride=2, memory=str(tmp_path))
    assert len(res) == 2 and (res.f1_tuned > 0.6).all()
    assert res.params.str.contains("gamma=").all()
    assert any(tmp_path.iterdir())          # front-end fits were cached
