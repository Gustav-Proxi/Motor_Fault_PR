"""Corruption sweep: re-run features + every front-end x classifier at each corruption level.

Corruption is applied to each whole raw recording before windowing (src/motorpr/corruption.py).

--corrupt both  (default) train and test recordings are corrupted: intrinsic separability
                of the corrupted signal, matching the proposal's "re-run the full pipeline".
--corrupt test  train clean, test corrupted: a model deployed on a degraded sensor.

Outputs: results/corruption_<corrupt>/metrics.csv, degradation_<mode>.png, runlog.json
Levels run in parallel (--n-jobs worker processes). Each level's noise stream is keyed by
(seed, mode, level, recording), so results do not depend on the number of workers.

Usage:   python scripts/run_corruption.py [--corrupt both|test] [--levels 0 .1 .2 .3 .4 .5]
                                          [--n-jobs N]
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motorpr.corruption import MODES, corrupt  # noqa: E402
from motorpr.data import Recording, load_all, windows_with_meta  # noqa: E402
from motorpr.evaluation import (classifiers, front_ends, make_pipeline,  # noqa: E402
                                projected_separability, recording_folds, run_cv)
from motorpr.features import extract_features  # noqa: E402
from motorpr.runlog import write_runlog  # noqa: E402


def features_for(recs: list[Recording], mode: str, level: float, seed: int
                 ) -> tuple[pd.DataFrame, np.ndarray]:
    cr = []
    for r in recs:
        # independent, reproducible stream per (seed, mode, level, recording)
        key = zlib.crc32(f"{seed}|{mode}|{level}|{r.rec_id}".encode())
        x = corrupt(r.signal, mode, level, np.random.default_rng(key))
        cr.append(Recording(r.rec_id, r.label, r.load, r.supply_hz, x))
    W, meta = windows_with_meta(cr)
    return meta, extract_features(W, meta["supply_hz"].to_numpy()).to_numpy()


def evaluate(fronts: dict, clfs: dict, X: np.ndarray, y: np.ndarray,
             folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Every front-end x classifier on one feature matrix."""
    rows = []
    for fname, front in fronts.items():
        sep = projected_separability(front, X, y, folds)
        for cname, clf in clfs.items():
            r = run_cv(make_pipeline(front, clf), X, y, folds)
            rows.append(dict(front=fname, clf=cname, acc=r["acc"], macro_f1=r["macro_f1"],
                             fisher_heldout=sep["fisher_heldout"]))
    return rows


def run_level(recs: list[Recording], mode: str, level: float, args: argparse.Namespace,
              X_clean: np.ndarray, y: np.ndarray,
              folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[str, float, list[dict], float]:
    """One (mode, level) cell of the sweep. Level 0 is the clean run, shared by every mode."""
    t0 = time.time()
    fronts, clfs = front_ends(seed=args.seed), classifiers(args.seed)
    if level == 0:
        res = evaluate(fronts, clfs, X_clean, y, folds)
    elif args.corrupt == "both":
        _, X_cor = features_for(recs, mode, level, args.seed)
        res = evaluate(fronts, clfs, X_cor, y, folds)
    else:
        _, X_cor = features_for(recs, mode, level, args.seed)
        # stack clean (train) and corrupted (test) copies; fold indices point into each half
        n = len(X_clean)
        res = evaluate(fronts, clfs, np.vstack([X_clean, X_cor]), np.concatenate([y, y]),
                       [(tr, te + n) for tr, te in folds])
    return mode, level, res, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrupt", choices=("both", "test"), default="both")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--split", default="grouped", choices=("grouped", "condition"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1, help="worker processes (-1 = all cores)")
    args = ap.parse_args()
    name = f"corruption_{args.corrupt}"
    out = ROOT / "results" / (name if args.seed == 0 else f"{name}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    recs = load_all()
    meta0, X_clean = features_for(recs, "noise", 0.0, args.seed)
    y = meta0["label"].to_numpy()
    folds = recording_folds(meta0, args.split, args.seed)
    fronts, clfs = front_ends(seed=args.seed), classifiers(args.seed)

    # corrupt() is the identity at level 0, so the clean run is computed once for all modes
    cells = ([(args.modes[0], 0.0)] if 0.0 in args.levels else [])
    cells += [(m, lv) for m in args.modes for lv in args.levels if lv != 0]
    done = {}
    jobs = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(run_level)(recs, m, lv, args, X_clean, y, folds) for m, lv in cells)
    for mode, level, res, secs in jobs:
        done[mode, level] = res
        best = max(res, key=lambda r: r["macro_f1"])
        label = "clean" if level == 0 else f"{mode:8s} {level:.1f}"
        print(f"{label:12s}  best F1 {best['macro_f1']:.3f} "
              f"({best['front']}+{best['clf']})  [{secs:.0f}s]", flush=True)

    clean = done.get((args.modes[0], 0.0))
    rows = [dict(mode=m, level=lv, **r)
            for m in args.modes for lv in args.levels
            for r in (clean if lv == 0 else done[m, lv])]
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    for mode, d in df.groupby("mode"):
        fig, axes = plt.subplots(1, len(fronts), figsize=(4 * len(fronts), 3.4), sharey=True)
        for ax, (fname, dd) in zip(axes, d.groupby("front", sort=False)):
            for cname, c in dd.groupby("clf"):
                ax.plot(c.level, c.macro_f1, "o-", ms=3, label=cname)
            ax.set_title(fname)
            ax.set_xlabel(f"{mode} level")
            ax.axhline(1 / 6, color="grey", ls=":", lw=1)
        axes[0].set_ylabel("macro-F1 (grouped CV)")
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"{mode} — corrupt {args.corrupt}")
        fig.tight_layout()
        fig.savefig(out / f"degradation_{mode}.png", dpi=150)
        plt.close(fig)

    write_runlog(out, **vars(args), front_ends=list(fronts), classifiers=list(clfs))
    print(df.pivot_table(index=["mode", "level"], columns="front", values="macro_f1",
                         aggfunc="max").round(3).to_string())


if __name__ == "__main__":
    main()
