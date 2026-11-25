"""DTW k-NN baseline on window envelopes (ROADMAP item 4b; options in DTW.md).

For each grouped fold, held-out windows are classified by k-NN against a subsample of
training windows, with three distances on the same references:

  dtw       DTW on log frame-RMS envelopes, Sakoe-Chiba band --band frames
  nowarp    the same envelopes with band 0 (no warping): isolates what warping adds
  features  Euclidean distance on the 126 standardised hand-written features

Kept light enough for a laptop: one core, BLAS limited to one thread, references are every
--stride-th window of each training recording and queries every --test-stride-th window
of each held-out recording (windows overlap by 50%, so neighbours are near-duplicates).
Each finished fold is saved to results/dtw/fold_<k>.pkl, and a rerun skips those folds,
so an interrupted run resumes where it stopped (use --fresh to recompute).

Outputs (results/dtw/): metrics.csv (per fold), summary.csv, confusion.npz, runlog.json
Usage:   python scripts/run_dtw.py [--band 8] [--stride 12] [--test-stride 4] [--seed 0]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motorpr.analysis import load_feature_cache  # noqa: E402
from motorpr.data import CLASSES, load_all, make_windows  # noqa: E402
from motorpr.dtw import dtw_distances, envelope, knn_predict, reference_subset  # noqa: E402
from motorpr.evaluation import recording_folds  # noqa: E402
from motorpr.runlog import write_runlog  # noqa: E402

KS = (1, 5)


def envelopes(frame: int) -> tuple[np.ndarray, list[str]]:
    """Envelope of every window, one recording at a time (never holds all windows)."""
    E, ids = [], []
    for r in load_all():
        e = envelope(make_windows(r.signal), frame)
        E.append(e)
        ids += [r.rec_id] * len(e)
    return np.concatenate(E), ids


def run_fold(k: int, tr: np.ndarray, te: np.ndarray, E: np.ndarray, X: np.ndarray,
             y: np.ndarray, rec: np.ndarray, band: int, stride: int, test_stride: int
             ) -> list[dict]:
    ref = tr[reference_subset(rec[tr], stride)]
    qry = te[reference_subset(rec[te], test_stride)]
    sc = StandardScaler().fit(X[ref])
    Zr, Zq = sc.transform(X[ref]), sc.transform(X[qry])
    dists = {
        "dtw": dtw_distances(E[qry], E[ref], band),
        "nowarp": dtw_distances(E[qry], E[ref], 0),
        "features": np.sqrt(np.clip((Zq ** 2).sum(1)[:, None] + (Zr ** 2).sum(1)[None, :]
                                    - 2 * Zq @ Zr.T, 0, None)),
    }
    rows = []
    for name, D in dists.items():
        for kk in KS:
            p = knn_predict(D, y[ref], kk)
            rows.append(dict(fold=k, distance=name, k=kk, n_ref=len(ref), n_query=len(qry),
                             macro_f1=f1_score(y[qry], p, average="macro"),
                             acc=float((p == y[qry]).mean()), y_pred=p, te=qry))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=int, default=8)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--test-stride", type=int, default=4)
    ap.add_argument("--frame", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore saved folds")
    args = ap.parse_args()
    out = ROOT / "results" / "dtw"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    E, ids = envelopes(args.frame)
    meta, F = load_feature_cache()
    assert list(meta.rec_id) == ids, "feature cache and recordings are out of sync"
    X, y, rec = F.to_numpy(), meta["label"].to_numpy(), meta["rec_id"].to_numpy()
    folds = recording_folds(meta, "grouped", args.seed)
    print(f"envelopes {E.shape}, {len(folds)} folds, band {args.band}, "
          f"stride {args.stride}, test stride {args.test_stride}", flush=True)

    rows = []
    for k, (tr, te) in enumerate(folds):
        ckpt = out / f"fold_{k}.pkl"
        if ckpt.exists() and not args.fresh:
            saved = pickle.loads(ckpt.read_bytes())
            if saved["args"] == vars(args) | {"fresh": saved["args"]["fresh"]}:
                rows += saved["rows"]
                print(f"fold {k}: loaded from {ckpt.name}", flush=True)
                continue
        t1 = time.time()
        fold_rows = run_fold(k, tr, te, E, X, y, rec, args.band, args.stride, args.test_stride)
        ckpt.write_bytes(pickle.dumps({"args": vars(args), "rows": fold_rows}))
        rows += fold_rows
        f1 = {(r["distance"], r["k"]): r["macro_f1"] for r in fold_rows}
        print(f"fold {k}: dtw {f1['dtw', 1]:.2f}  nowarp {f1['nowarp', 1]:.2f}  "
              f"features {f1['features', 1]:.2f} (k=1)  [{time.time() - t1:.0f}s]", flush=True)

    cms = {}
    for (name, kk), grp in pd.DataFrame(rows).groupby(["distance", "k"]):
        te = np.concatenate(grp.te.tolist())
        p = np.concatenate(grp.y_pred.tolist())
        cms[f"{name}|k{kk}"] = confusion_matrix(y[te], p, labels=CLASSES)
    np.savez(out / "confusion.npz", **cms)

    df = pd.DataFrame(rows).drop(columns=["y_pred", "te"])
    df.to_csv(out / "metrics.csv", index=False)
    summary = (df.groupby(["distance", "k"])
               .agg(macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                    acc=("acc", "mean"), n_ref=("n_ref", "mean"), n_query=("n_query", "sum"))
               .reset_index())
    for key, cm in cms.items():
        name, kk = key.split("|")
        rec_c = np.diag(cm) / cm.sum(axis=1)
        for c, r in zip(CLASSES, rec_c):
            summary.loc[(summary.distance == name) & (summary.k == int(kk[1:])),
                        f"recall_{c}"] = r
    summary.to_csv(out / "summary.csv", index=False)
    elapsed = time.time() - t0
    write_runlog(out, **vars(args), ks=KS, n_windows=len(y), envelope_shape=E.shape[1:],
                 elapsed_s=round(elapsed))
    print(summary.round(3).to_string(index=False))
    print(f"total {elapsed:.0f}s")


if __name__ == "__main__":
    main()
