"""Subspace Alignment across operating conditions (ROADMAP item 3b).

Source = recordings at the training conditions, target = an unseen condition. Targets:
  speed30 / speed40 / speed50   train on the other two supply frequencies (both loads)
  load1                         train unloaded, test loaded (all speeds)
For each target, k and classifier, three projections (src/motorpr/domain.py):
  none    one scaler + PCA fit on the source, applied to the target (the usual pipeline)
  zscore  each domain z-scored with its own statistics, then the source PCA
  sa      Subspace Alignment: source PCA basis aligned to the target's own PCA basis
SA and zscore use the target's unlabeled windows (transductive); labels never leave the
source. Scores are macro-F1 on the target and held-out Fisher J (transfer_fisher) in the
shared space.

Outputs (results/subspace_alignment/):
  metrics.csv   one row per (target, k, method, classifier): macro-F1, acc, per-class recall
  fisher.csv    held-out Fisher J per (target, k, method)
  summary.csv   mean over the three speed targets, and the load target, per (k, method, clf)
  runlog.json
Usage: python scripts/run_sa.py [--ks 5 10 20] [--n-jobs N] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_baseline import clean_features  # noqa: E402
from motorpr.data import CLASSES  # noqa: E402
from motorpr.domain import METHODS, project  # noqa: E402
from motorpr.evaluation import classifiers, transfer_fisher  # noqa: E402
from motorpr.runlog import write_runlog  # noqa: E402

TARGETS = ("speed30", "speed40", "speed50", "load1")


def target_mask(meta: pd.DataFrame, target: str) -> np.ndarray:
    if target.startswith("speed"):
        return (meta["supply_hz"] == int(target[5:])).to_numpy()
    return (meta["load"] == int(target[4:])).to_numpy()


def run_cell(X: np.ndarray, y: np.ndarray, test: np.ndarray, target: str, k: int,
             method: str, seed: int) -> tuple[list[dict], dict]:
    Zs, Zt = project(X[~test], X[test], k, method)
    ys, yt = y[~test], y[test]
    rows = []
    for cname, steps in classifiers(seed).items():
        # projected spaces are already centred; no extra scaler, so SA's scaling is kept
        p = Pipeline([(n, clone(s)) for n, s in steps]).fit(Zs, ys).predict(Zt)
        rec = recall_score(yt, p, labels=CLASSES, average=None, zero_division=0)
        rows.append(dict(target=target, k=k, method=method, clf=cname,
                         macro_f1=f1_score(yt, p, average="macro"),
                         acc=accuracy_score(yt, p),
                         **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}))
    fisher = dict(target=target, k=k, method=method,
                  fisher_heldout=transfer_fisher(Zs, ys, Zt, yt))
    return rows, fisher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / "results" / ("subspace_alignment" if args.seed == 0 else f"subspace_alignment_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    meta, F = clean_features()
    X, y = F.to_numpy(), meta["label"].to_numpy()
    cells = [(t, k, m) for t in TARGETS for k in args.ks for m in METHODS]
    res = Parallel(n_jobs=args.n_jobs)(
        delayed(run_cell)(X, y, target_mask(meta, t), t, k, m, args.seed) for t, k, m in cells)

    metrics = pd.DataFrame([r for rows, _ in res for r in rows])
    fisher = pd.DataFrame([f for _, f in res])
    metrics.to_csv(out / "metrics.csv", index=False)
    fisher.to_csv(out / "fisher.csv", index=False)

    metrics["group"] = np.where(metrics.target.str.startswith("speed"), "speed (mean of 3)",
                                "load1")
    summary = (metrics.groupby(["group", "k", "method", "clf"])
               .agg(macro_f1=("macro_f1", "mean"), macro_f1_min=("macro_f1", "min"))
               .reset_index())
    summary.to_csv(out / "summary.csv", index=False)

    write_runlog(out, **vars(args), targets=TARGETS, methods=METHODS)
    best = summary.loc[summary.groupby(["group", "k", "method"]).macro_f1.idxmax()]
    print(best.pivot_table(index=["group", "k"], columns="method", values="macro_f1")
          .round(3).to_string())
    print("\nheld-out Fisher J:\n", fisher.pivot_table(index=["target", "k"], columns="method",
                                                     values="fisher_heldout").round(2).to_string())


if __name__ == "__main__":
    main()
