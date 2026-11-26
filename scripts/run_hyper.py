"""Accuracy vs hyperparameters: grouped-CV macro-F1 as one hyperparameter is swept, 5 seeds.

Three sweeps, every other setting at the defaults of the main grid (evaluation.py):
  dims  projected dimensions k for PCA (1-20), LDA (1-5) and Kernel PCA (1-10), linear SVM
  C     SVM regularisation C (1e-3 to 10) on the raw features, linear and RBF kernel
  knn   neighbour count (1-45) for k-NN on the raw features and after LDA-5

Outputs: results/hyper/{per_seed,metrics}.csv, runlog.json; report/figures/fig_hyper.pdf.
Needs the feature cache from run_baseline.py.

Usage: python scripts/run_hyper.py [--n-jobs -1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_baseline import clean_features  # noqa: E402
from motorpr.evaluation import make_pipeline, recording_folds, run_cv  # noqa: E402
from motorpr.kernel_pca import KernelPCA  # noqa: E402
from motorpr.projections import LDA, PCA  # noqa: E402
from motorpr.runlog import write_runlog  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # same palette as make_report_figures.py
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def configs(seed: int) -> list[tuple[str, str, float, list, list]]:
    """(sweep, series, value, front steps, classifier steps) for one seed."""
    lin = [("clf", SVC(kernel="linear", C=1.0))]
    out = [("dims", "PCA", k, [("proj", PCA(k))], lin) for k in (1, 2, 3, 5, 8, 12, 20)]
    out += [("dims", "LDA", k, [("proj", LDA(k))], lin) for k in range(1, 6)]
    out += [("dims", "Kernel PCA", k, [("proj", KernelPCA(k, random_state=seed))], lin)
            for k in (1, 2, 3, 5, 8, 10)]
    for C in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        out.append(("C", "linear SVM", C, [], [("clf", SVC(kernel="linear", C=C))]))
        out.append(("C", "RBF SVM", C, [], [("clf", SVC(kernel="rbf", C=C, gamma="scale"))]))
    for n in (1, 3, 5, 9, 15, 25, 45):
        out.append(("knn", "raw", n, [], [("clf", KNeighborsClassifier(n))]))
        out.append(("knn", "LDA-5", n, [("proj", LDA(5))], [("clf", KNeighborsClassifier(n))]))
    return out


def evaluate(X, y, meta, seed, sweep, series, value, front, clf) -> dict:
    r = run_cv(make_pipeline(front, clf), X, y, recording_folds(meta, "grouped", seed=seed))
    return {"seed": seed, "sweep": sweep, "series": series, "value": value,
            "macro_f1": r["macro_f1"]}


def plot(m: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 1.3, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    panels = (("dims", "projected dimensions $k$", ("PCA", "LDA", "Kernel PCA"), False),
              ("C", "SVM $C$", ("linear SVM", "RBF SVM"), True),
              ("knn", "neighbours", ("raw", "LDA-5"), True))
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.5), sharey=True)
    for ax, (sweep, xlabel, series, logx) in zip(axes, panels):
        for i, s in enumerate(series):
            d = m[(m.sweep == sweep) & (m.series == s)].sort_values("value")
            ax.fill_between(d.value, d.f1_mean - d.f1_std, d.f1_mean + d.f1_std,
                            color=SERIES[i], alpha=0.12, lw=0)
            ax.plot(d.value, d.f1_mean, marker="o", ms=2.5, color=SERIES[i], label=s)
        if logx:
            ax.set_xscale("log")
        ax.axhline(1 / 6, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel(xlabel)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, handlelength=1.0,
                  borderaxespad=0.1, labelspacing=0.15)
    axes[0].set_ylabel("macro-F1")
    axes[0].set_ylim(0.1, 0.52)
    axes[0].set_yticks([0.2, 0.3, 0.4, 0.5])
    fig.tight_layout(w_pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--plot-only", action="store_true", help="redraw from results/hyper/metrics.csv")
    args = ap.parse_args()
    out = ROOT / "results" / "hyper"
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(pd.read_csv(out / "metrics.csv"), ROOT / "report" / "figures" / "fig_hyper.pdf")
        return

    meta, F = clean_features()
    X, y = F.to_numpy(), meta["label"].to_numpy()
    jobs = [(s, *c) for s in SEEDS for c in configs(s)]
    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(evaluate)(X, y, meta, *j) for j in jobs)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    m = (per_seed.groupby(["sweep", "series", "value"]).macro_f1
         .agg(f1_mean="mean", f1_std="std").reset_index())
    m.to_csv(out / "metrics.csv", index=False)
    plot(m, ROOT / "report" / "figures" / "fig_hyper.pdf")
    write_runlog(out, seeds=list(SEEDS), split="grouped", n_configs=len(jobs))
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
