"""Out-of-distribution splits: does a model trained at some operating conditions transfer
to a speed or load it has never seen? (ROADMAP item 3a)

cross_speed  train on 30 + 50 Hz supply, test on 40 Hz (both loads): interpolation in speed
cross_load   train unloaded, test loaded (all speeds): the larger shift, since load changes
             slip and the whole harmonic comb moves

Each split is a single train/test partition over recordings, so there is no fold-to-fold
spread here; seed-to-seed spread comes from the multi-seed reruns (1c). Needs the feature
cache from run_baseline.py (built here if missing).

Outputs (results/ood/):
  metrics.csv          accuracy / macro-F1 per (split, front-end, classifier)
  per_class.csv        per-class recall and precision for every model
  separability.csv     held-out Fisher J (transfer_fisher) per (split, front-end)
  confusion.npz        6x6 confusion matrices, keyed "<split>|<front>|<clf>"
  confusion_best.png   best model per (split, front-end)
  runlog.json
Usage: python scripts/run_ood.py [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_baseline import clean_features, plot_confusions  # noqa: E402
from motorpr.data import CLASSES  # noqa: E402
from motorpr.evaluation import (classifiers, front_ends, make_pipeline,  # noqa: E402
                                projected_separability, recording_folds, run_cv)
from motorpr.runlog import write_runlog  # noqa: E402

SPLITS = ("cross_speed", "cross_load")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / "results" / "ood"
    out.mkdir(parents=True, exist_ok=True)

    meta, F = clean_features()
    X, y = F.to_numpy(), meta["label"].to_numpy()
    fronts, clfs = front_ends(), classifiers(args.seed)

    rows, per_class, sep, cms = [], [], [], {}
    for split in SPLITS:
        folds = recording_folds(meta, split, args.seed)
        tr, te = folds[0]
        print(f"{split}: train {meta.rec_id.iloc[tr].nunique()} recordings, "
              f"test {meta.rec_id.iloc[te].nunique()}")
        for fname, front in fronts.items():
            sep.append(dict(split=split, front=fname,
                            **projected_separability(front, X, y, folds)))
            for cname, clf in clfs.items():
                t0 = time.time()
                r = run_cv(make_pipeline(front, clf), X, y, folds)
                key = f"{split}|{fname}|{cname}"
                cm = cms[key] = r["confusion"]
                rows.append(dict(split=split, front=fname, clf=cname,
                                 acc=r["acc"], macro_f1=r["macro_f1"]))
                for i, c in enumerate(CLASSES):
                    per_class.append(dict(split=split, front=fname, clf=cname, cls=c,
                                          recall=cm[i, i] / cm[i].sum(),
                                          precision=cm[i, i] / max(cm[:, i].sum(), 1)))
                print(f"{key:40s} acc {r['acc']:.3f}  F1 {r['macro_f1']:.3f}  "
                      f"({time.time() - t0:.1f}s)", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(sep).to_csv(out / "separability.csv", index=False)
    np.savez(out / "confusion.npz", **cms)

    best = metrics.loc[metrics.groupby(["split", "front"], sort=False).macro_f1.idxmax()]
    plot_confusions(cms, [f"{r.split}|{r.front}|{r.clf}" for r in best.itertuples()],
                    out / "confusion_best.png")

    write_runlog(out, seed=args.seed, splits=SPLITS, n_windows=len(X), n_features=X.shape[1],
                 front_ends=list(fronts), classifiers={k: str(v) for k, v in clfs.items()})
    print("\nmacro-F1:\n", metrics.pivot_table(index=["front", "clf"], columns="split",
                                               values="macro_f1").round(3).to_string())


if __name__ == "__main__":
    main()
