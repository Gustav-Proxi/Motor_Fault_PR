"""Figures and LaTeX tables for Sanika's report sections (separability, DTW, KPCA + tuning).

Reads the CSVs written by run_analysis.py, run_dtw.py, run_tuning.py and run_baseline.py,
and the feature cache (for the Kernel PCA Fisher-vs-k curve). Writes, under report/:
  figures/fig_pair_j.pdf      held-out J for every class pair (2a)
  figures/fig_sidebands.pdf   predicted rotor-bar sidebands vs the feature's band (2b)
  figures/fig_regions.pdf     held-out J per spectral region for B3 / V pairs (2c)
  figures/fig_kpca.pdf        held-out Fisher J vs k over 5 seeds; default vs tuned macro-F1 (5a, 5b)
  figures/fig_projection.pdf  PCA / LDA scatter of a held-out fold; confusion of the best model
  figures/fig_fisher_k.pdf    held-out Fisher J vs k, PCA / LDA / Kernel PCA, 5 seeds (column width)
  tables/tab_b1.tex, tab_peaks.tex, tab_dtw.tex, tab_tuning.tex
and results/analysis/kpca_fisher_vs_k.csv, fisher_vs_k_seeds.csv, confusion_best_seed<k>.npy.
The 5-seed quantities are recomputed here (one projection fit per fold; the 30 linear SVMs of
the confusion matrix are cached), so this script does not need the other seeds' result folders.

Usage: python scripts/make_report_figures.py [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motorpr import analysis as an  # noqa: E402
from motorpr.data import CLASSES, RAW_DIR, load_recording  # noqa: E402
from motorpr.evaluation import (classifiers, front_ends, make_pipeline,  # noqa: E402
                                projected_separability, recording_folds, run_cv,
                                transfer_fisher)
from motorpr.kernel_pca import KernelPCA  # noqa: E402
from motorpr.projections import LDA, PCA  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"

# Reference palette, light mode only (print). Categorical slots 1-3 validate all-pairs for
# colour-vision deficiency; slot 3 is below 3:1 contrast, so bars carry value labels.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
COL1, COL2 = 3.5, 7.16          # IEEE single / double column width, inches
SEEDS = (0, 1, 2, 3, 4)          # the 5-seed protocol of the report
# Classes use categorical slots 1-6 (validated on adjacent pairs); in a scatter all six are on
# screen together, so each class also gets its own marker.
CLS_COLOR = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
CLS_MARKER = dict(zip(CLASSES, ["o", "s", "^", "D", "P", "X"]))
BEST = ("raw", "svm_lin")        # best grouped model over 5 seeds (tab_grid.tex)


def style() -> None:
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(RES / "analysis" / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def hbars(ax, labels: list[str], groups: dict[str, np.ndarray], fmt: str = "{:.2f}") -> None:
    """Grouped horizontal bars with a 2-point gap between neighbours and value labels."""
    n, k = len(labels), len(groups)
    h = 0.8 / k
    y = np.arange(n)
    for i, (name, vals) in enumerate(groups.items()):
        pos = y - 0.4 + h * (i + 0.5)
        ax.barh(pos, vals, height=h, color=SERIES[i], label=name,
                edgecolor="white", linewidth=1.0)
        for p, v in zip(pos, vals):
            ax.text(v, p, " " + fmt.format(v), va="center", ha="left", fontsize=6, color=INK2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------- 2a

def fig_pair_j() -> None:
    sep = pd.read_csv(RES / "analysis" / "pair_separability.csv")
    M = an.pair_matrix(sep).to_numpy()
    fig, ax = plt.subplots(figsize=(COL1 * 0.78, COL1 * 0.68))
    vmax = np.nanmax(M)
    im = ax.imshow(M, cmap=BLUES, vmin=0, vmax=vmax)
    for i in range(6):
        for j in range(6):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.55 * vmax else INK)
    ax.set_xticks(range(6), CLASSES)
    ax.set_yticks(range(6), CLASSES)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("held-out Fisher J (median of 6 folds)")
    cb.outline.set_visible(False)
    save(fig, "fig_pair_j")


def tab_b1() -> None:
    s = pd.read_csv(RES / "analysis" / "pair_fisher.csv")
    b1 = s[(s.class_a == "B1") | (s.class_b == "B1")]
    top = b1.groupby(["class_a", "class_b"], sort=False).head(2)
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Pair & Feature & $J_\text{held-out}$ & worst fold & $J_\text{train}$ \\",
             r"\midrule"]
    for r in top.itertuples():
        feat = r.feature.replace("_", r"\_")
        lines.append(f"{r.class_a}--{r.class_b} & \\texttt{{{feat}}} & {r.j_heldout:.2f} & "
                     f"{r.j_heldout_min:.2f} & {r.j_train:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_b1.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 2b

def fig_sidebands() -> None:
    sb = pd.read_csv(RES / "analysis" / "sidebands.csv")
    rec = load_recording(RAW_DIR / "R_1_50Hz.csv")
    f, p = an.recording_spectrum(rec.signal)
    row = sb.set_index("rec_id").loc["R_1_50Hz"]
    f1, fp = row.f1, row.f_pole

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    m = (f > f1 - 7) & (f < f1 + 7)
    db = 10 * np.log10(p[m])
    for lo, hi in ((f1 - 3, f1 - 0.5), (f1 + 0.5, f1 + 3)):
        a.axvspan(lo, hi, color=GRID, lw=0, label="feature's sideband band" if lo < f1 else None)
    a.plot(f[m], db, color=SERIES[0], lw=0.9, label="R, loaded, 50 Hz supply")
    for x, txt in ((f1 - fp, f"$f_1-f_p$ = alias($5f_1$)"), (f1 + fp, f"$f_1+f_p$ = alias($3f_1$)")):
        a.axvline(x, color=SERIES[1], lw=1.0, ls="--")
        a.text(x, db.max() + 1, txt, ha="center", va="bottom", fontsize=6.5, color=INK)
    a.set_xlabel("frequency (Hz)")
    a.set_ylabel("PSD (dB, 6 channels summed)")
    a.set_ylim(top=db.max() + 7)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    for i, (load, name, marker) in enumerate(((0, "unloaded", "o"), (1, "loaded", "^"))):
        d = sb[sb.load == load]
        x = np.array([CLASSES.index(c) for c in d.label]) + (i - 0.5) * 0.25
        b.scatter(x, d.f_pole.abs().clip(lower=0.02), s=14, marker=marker, color=SERIES[i],
                  edgecolor="white", linewidth=0.6, label=name, zorder=3)
    b.axhspan(0.5, 3.0, color=GRID, lw=0)
    b.text(2.5, 1.2, "feature's 0.5–3 Hz band", ha="center", va="center", fontsize=6.5,
           color=INK2)
    b.set_yscale("log")
    b.set_xticks(range(6), CLASSES)
    b.set_ylabel("pole-pass spacing $f_p$ (Hz)")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_sidebands")


# ---------------------------------------------------------------- 2c

def fig_regions() -> None:
    s = pd.read_csv(RES / "analysis" / "regions.csv")
    names = {"region_subsync": "sub-synchronous", "region_shaft": "shaft harmonics",
             "region_shaft_alias": "aliased shaft harm.", "region_line_alias": "aliased line harm.",
             "region_other": "other bins", "region_total": "total level", "mech_slip": "slip"}
    pairs = {"H-B3": "H vs B3", "H-V": "H vs V", "B3-V": "B3 vs V"}
    piv = s.pivot(index="feature", columns="pair", values="j_heldout").loc[list(names)]
    fig, ax = plt.subplots(figsize=(COL1, 3.4))
    hbars(ax, list(names.values()), {lab: piv[p].to_numpy() for p, lab in pairs.items()})
    ax.set_xlabel("held-out Fisher J (median of 6 folds)")
    ax.set_xlim(0, piv[list(pairs)].to_numpy().max() * 1.2)
    ax.legend(loc="lower right", borderaxespad=0.2)
    save(fig, "fig_regions")


def tab_peaks() -> None:
    pk = pd.read_csv(RES / "analysis" / "peaks.csv")
    spec = pk[pk.label.isin(["B3", "V"]) & (pk.over_h_db >= 10)]
    cats = ["in-band shaft", "aliased", "unexplained"]
    cnt = spec.groupby(["label", "category"]).size().unstack(fill_value=0).reindex(
        columns=cats, fill_value=0)
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Class & in-band shaft & aliased & unexplained & total \\", r"\midrule"]
    for c, r in cnt.iterrows():
        lines.append(f"{c} & " + " & ".join(str(int(v)) for v in r) + f" & {int(r.sum())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_peaks.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 4b

def tab_dtw() -> None:
    d = pd.read_csv(RES / "dtw" / "summary.csv")
    label = {"dtw": "DTW (band $\\pm$8)", "nowarp": "no warping (band 0)",
             "features": "126 hand features"}
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Distance & $k$ & macro-F1 & B1 recall & R recall \\", r"\midrule"]
    for r in d.sort_values(["k", "distance"]).itertuples():
        lines.append(f"{label[r.distance]} & {r.k} & {r.macro_f1:.2f} $\\pm$ {r.macro_f1_std:.2f}"
                     f" & {r.recall_B1:.2f} & {r.recall_R:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_dtw.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 5a, 5b

def kpca_fisher_vs_k(seed: int) -> pd.DataFrame:
    meta, F = an.load_feature_cache()
    X, y = F.to_numpy(), meta["label"].to_numpy()
    folds = recording_folds(meta, "grouped", seed)
    rows = []
    for k in range(1, 6):
        s = projected_separability([("proj", KernelPCA(k))], X, y, folds)
        rows.append(dict(proj="kpca", k=k, **s))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "analysis" / "kpca_fisher_vs_k.csv", index=False)
    return d


def fisher_vs_k_seeds(seeds=SEEDS) -> pd.DataFrame:
    """Held-out (transfer) Fisher J of PCA, LDA and Kernel PCA for k = 1..5, per seed.

    The first k components of a 5-component PCA / LDA / Kernel PCA fit are exactly the k-component
    fit, so each projection is fitted once per fold and sliced. Fold mean per seed, as
    run_baseline.py reports it.
    """
    path = RES / "analysis" / "fisher_vs_k_seeds.csv"
    if path.exists():
        return pd.read_csv(path)
    meta, F = an.load_feature_cache()
    X, y = F.to_numpy(), meta["label"].to_numpy()
    rows = []
    for seed in seeds:
        per = []
        for tr, te in recording_folds(meta, "grouped", seed):
            for name, proj in (("pca", PCA(5)), ("lda", LDA(5)),
                               ("kpca", KernelPCA(5, random_state=seed))):
                pipe = make_pipeline([("proj", proj)], []).fit(X[tr], y[tr])
                Z_tr, Z_te = pipe.transform(X[tr]), pipe.transform(X[te])
                for k in range(1, 6):
                    per.append(dict(proj=name, k=k, fisher_heldout=transfer_fisher(
                        Z_tr[:, :k], y[tr], Z_te[:, :k], y[te])))
        d = pd.DataFrame(per).groupby(["proj", "k"], as_index=False).fisher_heldout.mean()
        rows.append(d.assign(seed=seed))
        print(f"  Fisher vs k, seed {seed} done", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def best_confusion(seeds=SEEDS) -> np.ndarray:
    """Grouped-CV confusion of the best model (BEST), summed over seeds.

    Uses results/baseline[_seed<k>]/confusion.npz when the 5-seed run is present, otherwise fits
    the model here and caches it in results/analysis/confusion_best_seed<k>.npy.
    """
    key = f"grouped|{BEST[0]}|{BEST[1]}"
    total = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    meta = X = y = None
    for seed in seeds:
        npz = RES / ("baseline" if seed == 0 else f"baseline_seed{seed}") / "confusion.npz"
        cache = RES / "analysis" / f"confusion_best_seed{seed}.npy"
        if npz.exists() and key in np.load(npz).files:
            cm = np.load(npz)[key]
        elif cache.exists():
            cm = np.load(cache)
        else:
            if meta is None:
                meta, F = an.load_feature_cache()
                X, y = F.to_numpy(), meta["label"].to_numpy()
            pipe = make_pipeline(front_ends(seed=seed)[BEST[0]], classifiers(seed)[BEST[1]])
            cm = run_cv(pipe, X, y, recording_folds(meta, "grouped", seed))["confusion"]
            np.save(cache, cm)
            print(f"  confusion, seed {seed} done", flush=True)
        total += cm
    return total


def fig_projection(seed: int) -> None:
    """(a, b) PCA and LDA fitted on the training recordings of grouped fold 1, every window
    projected; (c) row-normalised confusion of the best model over 5 seeds."""
    meta, F = an.load_feature_cache()
    X, y = F.to_numpy(), meta["label"].to_numpy()
    tr, te = recording_folds(meta, "grouped", seed)[0]
    fig = plt.figure(figsize=(COL2, 1.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.92], wspace=0.32, bottom=0.27, top=0.9)
    for i, (name, proj) in enumerate((("PCA", PCA(2)), ("LDA", LDA(2)))):
        ax = fig.add_subplot(gs[i])
        Z = make_pipeline([("proj", proj)], []).fit(X[tr], y[tr]).transform(X)
        for c in CLASSES:
            m_tr = tr[y[tr] == c][::4]
            m_te = te[y[te] == c][::2]
            ax.scatter(Z[m_tr, 0], Z[m_tr, 1], s=2, color=CLS_COLOR[c], alpha=0.25, lw=0,
                       rasterized=True)
            ax.scatter(Z[m_te, 0], Z[m_te, 1], s=9, color=CLS_COLOR[c], marker=CLS_MARKER[c],
                       edgecolor="white", linewidth=0.3, alpha=0.9, label=c, rasterized=True)
        pts = Z[np.concatenate([tr[::4], te[::2]])]
        lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
        pad = 0.06 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xlabel(f"{name} 1")
        ax.set_ylabel(f"{name} 2")
        ax.set_title(f"({'ab'[i]}) {name}, held-out fold", loc="left")
        ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[2])
    cm = best_confusion()
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap=BLUES, vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            if cmn[i, j] >= 0.005:
                ax.text(j, i, f"{cmn[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else INK)
    ax.set_xticks(range(6), CLASSES)
    ax.set_yticks(range(6), CLASSES)
    ax.grid(False)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("(c) Raw + linear SVM, 5 seeds", loc="left")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=2, color=INK2, alpha=0.5))
    labels.append("training windows")
    fig.legend(handles, labels, loc="lower center", ncol=7, bbox_to_anchor=(0.36, -0.01),
               markerscale=1.4, handletextpad=0.2, columnspacing=1.0)
    save(fig, "fig_projection")


def fig_fisher_k() -> None:
    """Held-out Fisher J against k for PCA, LDA and Kernel PCA, mean +- std over 5 seeds."""
    fk = fisher_vs_k_seeds().groupby(["proj", "k"]).fisher_heldout.agg(["mean", "std"]).reset_index()
    fig, a = plt.subplots(figsize=(COL1, 1.1))
    for i, (name, lab) in enumerate((("pca", "PCA"), ("lda", "LDA"), ("kpca", "Kernel PCA"))):
        d = fk[fk.proj == name].sort_values("k")
        a.fill_between(d.k, d["mean"] - d["std"], d["mean"] + d["std"], color=SERIES[i],
                       alpha=0.12, lw=0)
        a.plot(d.k, d["mean"], marker="o", ms=4, color=SERIES[i], label=lab,
               markeredgecolor="white", markeredgewidth=0.6)
        a.annotate(f"{d['mean'].iloc[-1]:.1f}", (5, d["mean"].iloc[-1]), xytext=(5, -2 if name != "kpca" else 3),
                   textcoords="offset points", fontsize=6.5, color=INK2, va="center")
    a.set_xlabel("projected dimensions $k$")
    a.set_ylabel("held-out Fisher $J$")
    a.set_xticks(range(1, 6))
    a.set_xlim(0.8, 5.6)
    a.legend(loc="upper left")
    save(fig, "fig_fisher_k")


def fig_kpca(seed: int) -> None:
    kpca_fisher_vs_k(seed)                                   # seed-0 curve for the UI bundle
    fk = fisher_vs_k_seeds().groupby(["proj", "k"]).fisher_heldout.agg(["mean", "std"]).reset_index()
    tun = pd.read_csv(RES / "tuning" / "summary.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.0))
    for i, (name, lab) in enumerate((("pca", "PCA"), ("lda", "LDA"), ("kpca", "Kernel PCA"))):
        d = fk[fk.proj == name].sort_values("k")
        a.fill_between(d.k, d["mean"] - d["std"], d["mean"] + d["std"], color=SERIES[i],
                       alpha=0.12, lw=0)
        a.plot(d.k, d["mean"], marker="o", ms=4, color=SERIES[i], label=lab,
               markeredgecolor="white", markeredgewidth=0.6)
    a.set_xlabel("projected dimensions $k$")
    a.set_ylabel("held-out Fisher J (5 seeds)")
    a.set_xticks(range(1, 6))
    a.legend()

    best = tun.loc[tun.groupby("front").f1_tuned.idxmax()].set_index("front")
    best_d = tun.loc[tun.groupby("front").f1_default.idxmax()].set_index("front")
    nice = {"raw": "raw (126)", "pca5": "PCA-5", "lda5": "LDA-5", "kpca5": "Kernel PCA-5"}
    order = [f for f in ("raw", "pca5", "lda5", "kpca5") if f in best.index]
    hbars(b, [nice[f] for f in order],
          {"default (best classifier)": best_d.loc[order, "f1_default"].to_numpy(),
           "nested-tuned (best classifier)": best.loc[order, "f1_tuned"].to_numpy()})
    b.axvline(1 / 6, color=INK2, lw=0.8, ls=":")          # chance; named in the caption
    b.set_xlabel("grouped-CV macro-F1 (dotted: chance)")
    b.set_xlim(0, 0.6)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    a.legend(loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_kpca")


def tab_tuning() -> None:
    t = pd.read_csv(RES / "tuning" / "summary.csv")
    lines = [r"\begin{tabular}{llrrrl}", r"\toprule",
             r"Front & Classifier & default & tuned & gain & most chosen (folds) \\",
             r"\midrule"]
    for r in t.itertuples():
        params = str(r.top_params).replace("_", r"\_")
        # gain from the rounded scores, so each row reads consistently; + 0.0 avoids -0.00
        gain = round(round(r.f1_tuned, 2) - round(r.f1_default, 2), 2) + 0.0
        lines.append(f"{r.front} & {r.clf.replace('_', chr(92) + '_')} & {r.f1_default:.2f} & "
                     f"{r.f1_tuned:.2f} & {gain:+.2f} & {params} "
                     f"({r.top_params_folds}/6) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_tuning.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", help="run only these figure/table functions")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    style()
    if args.only:
        for name in args.only:
            fn = globals()[name]
            fn(args.seed) if name in ("fig_kpca", "fig_projection") else fn()  # noqa: B018
        return
    fig_pair_j()
    tab_b1()
    fig_sidebands()
    fig_regions()
    tab_peaks()
    tab_dtw()
    fig_kpca(args.seed)
    fig_projection(args.seed)
    fig_fisher_k()
    tab_tuning()
    print("tables in", TAB)


if __name__ == "__main__":
    main()
