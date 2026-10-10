"""Clean-signal baseline: every front-end x classifier under grouped, condition and naive CV.

Outputs (results/baseline/):
  metrics.csv          accuracy / macro-F1 per (split, front-end, classifier)
  per_class.csv        per-class recall and precision (grouped split)
  separability.csv     Fisher / Silhouette / Davies–Bouldin on held-out folds
  fisher_vs_k.csv      Fisher criterion for PCA and LDA at k = 1..5
  confusion.npz        6x6 confusion matrices, keyed "<split>|<front>|<clf>"
  *.png                figures

Usage: python scripts/run_baseline.py [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motorpr.data import CLASSES, DATA_DIR, load_all, windows_with_meta  # noqa: E402
from motorpr.evaluation import (classifiers, front_ends, make_pipeline,  # noqa: E402
                                projected_separability, recording_folds, run_cv)
from motorpr.features import extract_features  # noqa: E402
from motorpr.projections import LDA, PCA  # noqa: E402
from motorpr.runlog import write_runlog  # noqa: E402

SPLITS = ("grouped", "condition", "naive")


def clean_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = DATA_DIR / "processed" / "features_clean.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        W, meta = windows_with_meta(load_all())
        df = pd.concat([meta, extract_features(W, meta["supply_hz"].to_numpy())], axis=1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    meta_cols = ["rec_id", "label", "load", "supply_hz", "win_idx"]
    return df[meta_cols], df.drop(columns=meta_cols)


def plot_confusions(cms: dict, keys: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.6))
    for ax, k in zip(np.atleast_1d(axes), keys):
        cm = cms[k]
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(6), CLASSES)
        ax.set_yticks(range(6), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_title(k.split("|", 1)[1].replace("|", " + "), fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / "results" / "baseline"
    out.mkdir(parents=True, exist_ok=True)

    meta, F = clean_features()
    X, y = F.to_numpy(), meta["label"].to_numpy()
    print(f"{X.shape[0]} windows x {X.shape[1]} features from {meta.rec_id.nunique()} recordings")

    fronts, clfs = front_ends(), classifiers(args.seed)
    folds = {s: recording_folds(meta, s, args.seed) for s in SPLITS}

    rows, per_class, cms = [], [], {}
    for split in SPLITS:
        for fname, front in fronts.items():
            for cname, clf in clfs.items():
                t0 = time.time()
                r = run_cv(make_pipeline(front, clf), X, y, folds[split])
                key = f"{split}|{fname}|{cname}"
                cms[key] = r["confusion"]
                rows.append(dict(split=split, front=fname, clf=cname,
                                 **{k: r[k] for k in ("acc", "acc_std", "macro_f1", "macro_f1_std")}))
                if split == "grouped":
                    cm = r["confusion"]
                    for i, c in enumerate(CLASSES):
                        per_class.append(dict(front=fname, clf=cname, cls=c,
                                              recall=cm[i, i] / cm[i].sum(),
                                              precision=cm[i, i] / max(cm[:, i].sum(), 1)))
                print(f"{key:40s} acc {r['acc']:.3f}±{r['acc_std']:.3f}  "
                      f"F1 {r['macro_f1']:.3f}  ({time.time() - t0:.1f}s)")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
    np.savez(out / "confusion.npz", **cms)

    # separability of each front-end on held-out recordings
    sep = [dict(front=f, **projected_separability(fr, X, y, folds["grouped"]))
           for f, fr in fronts.items()]
    pd.DataFrame(sep).to_csv(out / "separability.csv", index=False)

    fk = []
    for k in range(1, 6):
        for name, proj in (("pca", PCA(k)), ("lda", LDA(k))):
            s = projected_separability([("proj", proj)], X, y, folds["grouped"])
            fk.append(dict(proj=name, k=k, **s))
    fk = pd.DataFrame(fk)
    fk.to_csv(out / "fisher_vs_k.csv", index=False)

    # figures
    g = metrics[metrics.split == "grouped"]
    best = g.loc[g.groupby("front").macro_f1.idxmax()]
    plot_confusions(cms, [f"grouped|{r.front}|{r.clf}" for r in best.itertuples()],
                    out / "confusion_best_per_front.png")

    piv = metrics.pivot_table(index=["front", "clf"], columns="split", values="macro_f1")
    fig, ax = plt.subplots(figsize=(10, 4))
    piv[list(SPLITS)].plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("macro-F1")
    ax.set_title("Grouped vs condition vs naive split (naive leaks overlapping windows)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "split_gap.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, col, title in zip(axes, ("fisher_train", "fisher_heldout"),
                              ("training fold (in-sample)", "held-out recordings")):
        for name, d in fk.groupby("proj"):
            ax.plot(d.k, d[col], "o-", label=name.upper())
        ax.set_xlabel("projected dimensions k")
        ax.set_ylabel("Fisher J")
        ax.set_title(title)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "fisher_vs_k.png", dpi=150)
    plt.close(fig)

    # 2-D projections fit on fold-0 train, plotted on fold-0 test
    tr, te = folds["grouped"][0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, proj) in zip(axes, (("PCA", PCA(2)), ("LDA", LDA(2)))):
        p = make_pipeline([("proj", proj)], []).fit(X[tr], y[tr])
        Z = p.transform(X[te])
        for c in CLASSES:
            m = y[te] == c
            ax.scatter(Z[m, 0], Z[m, 1], s=4, label=c, alpha=0.6)
        ax.set_title(f"{name} (held-out recordings)")
    axes[1].legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(out / "projection_2d.png", dpi=150)
    plt.close(fig)

    write_runlog(out, seed=args.seed, splits=SPLITS, n_windows=len(X), n_features=X.shape[1],
                 front_ends=list(fronts), classifiers={k: str(v) for k, v in clfs.items()})
    print("\nGrouped-split macro-F1:\n", piv.round(3).to_string())
    print("\nSeparability (held-out):\n", pd.DataFrame(sep).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
