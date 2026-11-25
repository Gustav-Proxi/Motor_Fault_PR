"""Mean ± std over seeds for every result table (ROADMAP item 1c).

Each experiment script writes seed 0 to results/<name>/ and seed k to results/<name>_seed<k>/.
This reads every seed it finds and writes results/multiseed/<name>__<table>.csv with one row
per model / condition: mean, std and the number of seeds. A seed changes the grouped fold
assignment, the random-forest seed, the Kernel PCA subsample and the corruption noise.

Usage: python scripts/aggregate_seeds.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motorpr.runlog import write_runlog  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "multiseed"

# (experiment, table, grouping keys, value columns)
TABLES = [
    ("baseline", "metrics", ["split", "front", "clf"], ["macro_f1", "acc"]),
    ("baseline", "per_class", ["front", "clf", "cls"], ["recall", "precision"]),
    ("baseline", "fisher_vs_k", ["proj", "k"], ["fisher_heldout", "fisher_train"]),
    ("baseline", "separability", ["front"], ["fisher_heldout"]),
    ("corruption_both", "metrics", ["mode", "level", "front", "clf"], ["macro_f1", "fisher_heldout"]),
    ("corruption_test", "metrics", ["mode", "level", "front", "clf"], ["macro_f1", "fisher_heldout"]),
    ("ood", "metrics", ["split", "front", "clf"], ["macro_f1", "acc"]),
    ("ood", "per_class", ["split", "front", "clf", "cls"], ["recall"]),
    ("subspace_alignment", "metrics", ["target", "k", "method", "clf"], ["macro_f1"]),
    ("subspace_alignment", "fisher", ["target", "k", "method"], ["fisher_heldout"]),
]


def seed_dirs(name: str) -> dict[int, Path]:
    dirs = {0: RESULTS / name} if (RESULTS / name).is_dir() else {}
    for path in RESULTS.glob(f"{name}_seed*"):
        match = re.fullmatch(rf"{re.escape(name)}_seed(\d+)", path.name)
        if match and path.is_dir():
            dirs[int(match.group(1))] = path
    return dict(sorted(dirs.items()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    found = {}
    for name, table, keys, values in TABLES:
        frames = []
        for seed, path in seed_dirs(name).items():
            csv = path / f"{table}.csv"
            if csv.exists():
                frames.append(pd.read_csv(csv).assign(seed=seed))
        if not frames:
            print(f"{name}/{table}: no seeds found")
            continue
        df = pd.concat(frames, ignore_index=True)
        cols = [c for c in values if c in df.columns]
        agg = df.groupby(keys, sort=False)[cols].agg(["mean", "std"])
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        agg["n_seeds"] = df.groupby(keys, sort=False)["seed"].nunique()
        agg.reset_index().to_csv(OUT / f"{name}__{table}.csv", index=False)
        found[f"{name}/{table}"] = sorted(df.seed.unique().tolist())
        print(f"{name}/{table}: seeds {found[f'{name}/{table}']}")
    write_runlog(OUT, tables=found)

    grouped = OUT / "baseline__metrics.csv"
    if grouped.exists():
        g = pd.read_csv(grouped)
        g = g[g.split == "grouped"].sort_values("macro_f1_mean", ascending=False)
        print("\nGrouped CV, best per front-end (mean ± std over seeds):")
        for front, d in g.groupby("front", sort=False):
            r = d.iloc[0]
            print(f"  {front:8s} {r.clf:8s} {r.macro_f1_mean:.3f} ± {r.macro_f1_std:.3f} (n={r.n_seeds})")


if __name__ == "__main__":
    main()
