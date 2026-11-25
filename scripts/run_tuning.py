"""Nested, recording-grouped hyperparameter tuning (ROADMAP item 5b). Needs the feature cache.

For every front-end x classifier: outer 6 grouped folds, inner 5 grouped folds on the
outer-training recordings, grid search on inner macro-F1 (grids in src/motorpr/tuning.py).
Each outer fold also scores the current fixed defaults on the same split and the same
training windows, so the two columns are directly comparable.

Kept light enough for a laptop: --n-jobs workers (default 2) with BLAS limited to one
thread, fits on every --train-stride-th window of each training recording (windows overlap
by 50%), and fitted Kernel PCA front-ends are cached across grid points. Each finished
front-end x classifier is saved under results/tuning/parts/ and skipped on a rerun, so an
interrupted run resumes (use --fresh to recompute).

Outputs (results/tuning/):
  nested_folds.csv   one row per (front, clf, outer fold): tuned / default F1, chosen params
  summary.csv        mean +- std over outer folds, gain, and how often the top choice won
  runlog.json
Usage: python scripts/run_tuning.py [--fronts raw lda5 kpca5] [--clfs ...] [--seed 0]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from joblib import Memory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motorpr.analysis import load_feature_cache  # noqa: E402
from motorpr.evaluation import classifiers, front_ends, recording_folds  # noqa: E402
from motorpr.runlog import write_runlog  # noqa: E402
from motorpr.tuning import CLASSIFIER_GRIDS, FRONT_GRIDS, nested_cv  # noqa: E402


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def top(params: pd.Series) -> pd.Series:
        vc = params.value_counts()
        return pd.Series({"top_params": vc.index[0], "top_params_folds": int(vc.iloc[0])})

    g = df.groupby(["front", "clf"], sort=False)
    s = g.agg(f1_tuned=("f1_tuned", "mean"), f1_tuned_std=("f1_tuned", "std"),
              f1_default=("f1_default", "mean"), f1_default_std=("f1_default", "std"),
              f1_inner_best=("f1_inner_best", "mean")).reset_index()
    s["gain"] = s.f1_tuned - s.f1_default
    # inner CV picks on 5 recordings per class; the gap to outer F1 is the selection optimism
    s["inner_optimism"] = s.f1_inner_best - s.f1_tuned
    tops = pd.DataFrame([dict(front=f, clf=c, **top(d.params)) for (f, c), d in g])
    return s.merge(tops, on=["front", "clf"])


def main() -> None:
    fronts_all, clfs_all = front_ends(), classifiers()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fronts", nargs="+", default=["raw", "lda5", "kpca5"],
                    choices=list(fronts_all))
    ap.add_argument("--clfs", nargs="+", default=list(clfs_all), choices=list(clfs_all))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--fresh", action="store_true", help="ignore saved parts")
    args = ap.parse_args()
    out = ROOT / "results" / "tuning"
    parts_dir = out / "parts" / f"seed{args.seed}_stride{args.train_stride}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    memory = Memory(ROOT / "results" / "cache" / "tuning", verbose=0)

    meta, F = load_feature_cache()
    X, y = F.to_numpy(), meta["label"].to_numpy()
    outer = recording_folds(meta, "grouped", args.seed)
    clfs = classifiers(args.seed)

    parts = []
    for fname in args.fronts:
        for cname in args.clfs:
            part = parts_dir / f"{fname}__{cname}.csv"
            if part.exists() and not args.fresh:
                parts.append(pd.read_csv(part))
                print(f"{fname:6s} {cname:8s} loaded", flush=True)
                continue
            t0 = time.time()
            r = nested_cv(fname, fronts_all[fname], cname, clfs[cname], X, y, meta, outer,
                          seed=args.seed, n_jobs=args.n_jobs, train_stride=args.train_stride,
                          memory=memory if fname in FRONT_GRIDS else None)
            r.to_csv(part, index=False)
            parts.append(r)
            print(f"{fname:6s} {cname:8s} tuned {r.f1_tuned.mean():.3f}±{r.f1_tuned.std():.3f}  "
                  f"default {r.f1_default.mean():.3f}  [{time.time() - t0:.0f}s]  "
                  f"{r.params.value_counts().index[0]}", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out / "nested_folds.csv", index=False)
    s = summarise(df)
    s.to_csv(out / "summary.csv", index=False)
    write_runlog(out, **vars(args), outer_split="grouped", inner_split="grouped (5 folds)",
                 classifier_grids=CLASSIFIER_GRIDS, front_grids=FRONT_GRIDS)
    print(s.drop(columns=["f1_inner_best"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
