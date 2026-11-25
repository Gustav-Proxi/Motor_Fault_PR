"""Tables and the corruption figure for the methods / validation / robustness sections (R1).

Reads results/multiseed/ (scripts/aggregate_seeds.py) and writes, under report/:
  tables/tab_grid.tex        grouped-CV macro-F1, every front-end x classifier, mean ± std
  tables/tab_splits.tex      best model per split (naive / condition / grouped)
  tables/tab_ood.tex         cross-speed and cross-load, best classifier per front-end
  tables/tab_sa.tex          Subspace Alignment vs per-domain z-score vs no adaptation
  figures/fig_corruption.pdf best macro-F1 per front-end against corruption level, ±1 std

Usage: python scripts/make_methods_tables.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "results" / "multiseed"
TAB = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"

FRONTS = {"raw": "Raw 126-D", "pca0.95": "PCA (95\\% var.)", "pca5": "PCA-5",
          "lda5": "LDA-5", "kpca5": "Kernel PCA-5"}
CLFS = {"knn_euc": "$k$-NN", "knn_mah": "$k$-NN (Mah.)", "svm_lin": "Lin.\\ SVM",
        "svm_rbf": "RBF SVM", "rf": "RF", "gnb": "GNB"}
PLAIN_CLFS = {"knn_euc": "k-NN", "knn_mah": "k-NN (Mahalanobis)", "svm_lin": "linear SVM",
              "svm_rbf": "RBF SVM", "rf": "random forest", "gnb": "Gaussian NB"}


def pm(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def write(name: str, lines: list[str]) -> None:
    (TAB / name).write_text("\n".join(lines) + "\n")


def best_per_front(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby("front", sort=False).macro_f1_mean.idxmax()]


def tab_grid(base: pd.DataFrame) -> None:
    g = base[base.split == "grouped"]
    best = g.macro_f1_mean.max()
    lines = [r"\begin{tabular}{l" + "c" * len(CLFS) + "}", r"\toprule",
             "Front-end & " + " & ".join(CLFS.values()) + r" \\", r"\midrule"]
    for front, label in FRONTS.items():
        cells = []
        for clf in CLFS:
            r = g[(g.front == front) & (g.clf == clf)].iloc[0]
            cell = pm(r.macro_f1_mean, r.macro_f1_std)
            cells.append(f"\\textbf{{{cell}}}" if r.macro_f1_mean == best else cell)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    write("tab_grid.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def tab_splits(base: pd.DataFrame) -> None:
    lines = [r"\begin{tabular}{llc}", r"\toprule",
             r"Split & Best model & macro-F1 \\", r"\midrule"]
    for split, label in [("naive", "Naive (random windows)"),
                         ("condition", "Leave one condition out"),
                         ("grouped", "Grouped by recording")]:
        r = base[base.split == split].sort_values("macro_f1_mean").iloc[-1]
        lines.append(f"{label} & {FRONTS[r.front]} + {CLFS[r.clf]} & "
                     f"{pm(r.macro_f1_mean, r.macro_f1_std)} \\\\")
    write("tab_splits.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def tab_ood(ood: pd.DataFrame, per_class: pd.DataFrame) -> None:
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Cross-speed (test 40\,Hz)} & "
             r"\multicolumn{2}{c}{Cross-load (test loaded)} \\",
             r"Front-end & macro-F1 & classifier & macro-F1 & classifier \\", r"\midrule"]
    best = {split: best_per_front(ood[ood.split == split]).set_index("front")
            for split in ("cross_speed", "cross_load")}
    for front, label in FRONTS.items():
        s, l = best["cross_speed"].loc[front], best["cross_load"].loc[front]
        lines.append(f"{label} & {pm(s.macro_f1_mean, s.macro_f1_std)} & {CLFS[s.clf]} & "
                     f"{pm(l.macro_f1_mean, l.macro_f1_std)} & {CLFS[l.clf]} \\\\")
    lines += [r"\midrule"]
    top = ood[ood.split == "cross_load"].sort_values("macro_f1_mean").iloc[-1]
    recall = per_class[(per_class.split == "cross_load") & (per_class.front == top.front)
                       & (per_class.clf == top.clf)].set_index("cls").recall_mean
    lines.append(r"\multicolumn{5}{l}{Cross-load recall, " + f"{FRONTS[top.front]} + "
                 f"{CLFS[top.clf]}: " + ", ".join(f"{c} {recall[c]:.2f}" for c in
                                                  ["H", "B1", "B2", "B3", "V", "R"]) + r"} \\")
    write("tab_ood.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def tab_sa(sa: pd.DataFrame) -> None:
    sa = sa.assign(group=sa.target.where(sa.target == "load1", "speed"))
    # mean over the three speed targets, then over seeds already done; best classifier per cell
    cell = (sa.groupby(["group", "k", "method", "clf"]).macro_f1_mean.mean()
            .groupby(["group", "k", "method"]).max().unstack("method"))
    avg = sa.groupby(["group", "k", "method"]).macro_f1_mean.mean().unstack("method")
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{Best classifier} & \multicolumn{3}{c}{Mean of 6 classifiers} \\",
             r"Target & $k$ & none & z-score & SA & none & z-score & SA \\", r"\midrule"]
    for group, label in [("speed", "Unseen speed (mean of 3)"), ("load1", "Unseen load")]:
        for k in sorted(sa.k.unique()):
            b, a = cell.loc[(group, k)], avg.loc[(group, k)]
            lines.append(f"{label if k == sa.k.min() else ''} & {k} & "
                         + " & ".join(f"{b[m]:.2f}" for m in ("none", "zscore", "sa")) + " & "
                         + " & ".join(f"{a[m]:.2f}" for m in ("none", "zscore", "sa")) + r" \\")
    write("tab_sa.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def fig_corruption(both: pd.DataFrame, test: pd.DataFrame) -> None:
    modes = [("noise", "Additive noise"), ("dropout", "Sample dropout"), ("gain", "Gain drift")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey=True)
    for row, (df, title) in enumerate([(both, "train and test corrupted"),
                                       (test, "train clean, test corrupted")]):
        for col, (mode, mode_label) in enumerate(modes):
            ax = axes[row, col]
            d = df[df["mode"] == mode]
            for front, label in FRONTS.items():
                rows = d[d.front == front]
                best = rows.loc[rows.groupby("level").macro_f1_mean.idxmax()].sort_values("level")
                ax.plot(best.level, best.macro_f1_mean, "o-", ms=3,
                        label=label.replace("\\%", "%"))  # LaTeX escape is not mathtext
                ax.fill_between(best.level, best.macro_f1_mean - best.macro_f1_std,
                                best.macro_f1_mean + best.macro_f1_std, alpha=0.15)
            ax.axhline(1 / 6, color="grey", ls=":", lw=1)
            ax.set_title(f"{mode_label}: {title}", fontsize=9)
            if row == 1:
                ax.set_xlabel("corruption level")
            if col == 0:
                ax.set_ylabel("macro-F1 (grouped CV)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_corruption.pdf")
    plt.close(fig)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(MS / "baseline__metrics.csv")
    tab_grid(base)
    tab_splits(base)
    tab_ood(pd.read_csv(MS / "ood__metrics.csv"), pd.read_csv(MS / "ood__per_class.csv"))
    tab_sa(pd.read_csv(MS / "subspace_alignment__metrics.csv"))
    fig_corruption(pd.read_csv(MS / "corruption_both__metrics.csv"),
                   pd.read_csv(MS / "corruption_test__metrics.csv"))
    print("wrote tab_grid, tab_splits, tab_ood, tab_sa, fig_corruption")


if __name__ == "__main__":
    main()
