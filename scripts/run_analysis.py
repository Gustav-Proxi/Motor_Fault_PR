"""Separability analysis (ROADMAP item 2). Needs the feature cache from run_baseline.py.

Stages:
  fisher    per-feature, per-class-pair held-out Fisher scores (2a)

Outputs (results/analysis/):
  pair_fisher_folds.csv   every (fold, pair, feature) score
  pair_fisher.csv         per (pair, feature): median / worst / mean held-out J over folds,
                          and the in-sample J on the training folds for comparison
  class_vs_rest.csv       per class and feature: weakest and mean held-out J over its 5 pairs
  feature_types.csv       best channel per (pair, feature type)
  pair_separability.csv   held-out J per pair along its 2-class LDA direction (126 features)
  runlog.json

Held-out scores always come from evaluation.transfer_fisher on grouped folds.

Usage: python scripts/run_analysis.py [--stages fisher] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motorpr import analysis as an  # noqa: E402
from motorpr.evaluation import recording_folds  # noqa: E402
from motorpr.runlog import write_runlog  # noqa: E402

OUT = ROOT / "results" / "analysis"
STAGES = ("fisher",)


def stage_fisher(meta: pd.DataFrame, F: pd.DataFrame, seed: int) -> None:
    folds = recording_folds(meta, "grouped", seed)
    t0 = time.time()
    table = an.pair_fisher_table(F, meta, folds)
    table.to_csv(OUT / "pair_fisher_folds.csv", index=False)
    summary = an.summarise_pair_fisher(table)
    summary.to_csv(OUT / "pair_fisher.csv", index=False)
    cvr = an.class_vs_rest(summary)
    cvr.to_csv(OUT / "class_vs_rest.csv", index=False)
    an.by_feature_type(summary).to_csv(OUT / "feature_types.csv", index=False)
    sep = an.pair_separability(F, meta, folds)
    sep.to_csv(OUT / "pair_separability.csv", index=False)
    print(f"[fisher] {len(table)} scores in {time.time() - t0:.0f}s")

    print("\nHeld-out J per class pair along its 2-class LDA direction (low = confusable):")
    print(an.pair_matrix(sep).round(2).to_string())
    print("\nFeatures that separate each class from every other class (top 5 by weakest pair):")
    for c, d in cvr.groupby("cls", sort=False):
        print(f"  {c}: " + ", ".join(f"{r.feature} ({r.j_min:.2f} vs {r.weakest_vs})"
                                      for r in d.head(5).itertuples()))
    b1 = summary[(summary.class_a == "B1") | (summary.class_b == "B1")]
    print("\nTop 3 features for each B1 pair (median held-out J, in-sample J):")
    for (a, b), d in b1.groupby(["class_a", "class_b"]):
        print(f"  {a}-{b}: " + ", ".join(f"{r.feature} ({r.j_heldout:.2f}, {r.j_train:.2f})"
                                         for r in d.head(3).itertuples()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta, F = an.load_feature_cache()
    print(f"{len(F)} windows x {F.shape[1]} features from {meta.rec_id.nunique()} recordings")
    if "fisher" in args.stages:
        stage_fisher(meta, F, args.seed)
    write_runlog(OUT, **vars(args), split="grouped", n_windows=len(F))


if __name__ == "__main__":
    main()
