"""Export the data bundle that the project UI (ui/) reads: ui/public/data/bundle.json.

Reads the raw recordings, the feature cache and the CSVs that run_baseline.py,
run_analysis.py, run_dtw.py and run_tuning.py write (and run_corruption.py, if its
results exist). It fits only the PCA-5 / LDA-5 / Kernel PCA-5 front-ends once per
grouped fold, so it runs on one CPU core in well under a minute.

Bundle contents:
  recordings   36 recordings with running speed, slip, pole-pass frequency, CV fold
  signals      a 3,072-sample excerpt (gUserX/Y/Z, DC removed) per recording, int16
  spectra      whole-recording Welch PSD (8192-sample segments) per recording, dB, int16
  embedding    every window projected to 5-D by each front-end, fitted per grouped fold
  results      metrics, confusions, separability, analysis, DTW and tuning tables
  multi-seed   model grid, corruption sweeps, OOD splits and Subspace Alignment as mean +- std
               over every seed found (results/<name>[_seed<k>]/ or results/multiseed/);
               meta.grid_seeds says how many seeds the model grid averages (1 = seed 0 only)

Usage: python scripts/export_ui_data.py [--seed 0]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import base64  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motorpr.analysis import (BPFI_ORDER, BPFO_ORDER, load_feature_cache,  # noqa: E402
                              recording_spectrum, sideband_levels)
from motorpr.data import CLASSES, FS, STEP, WINDOW, load_all  # noqa: E402
from motorpr.evaluation import (fisher_criterion, front_ends, make_pipeline,  # noqa: E402
                                recording_folds, transfer_fisher)
import aggregate_seeds  # noqa: E402  (scripts/ is on sys.path when run as a script)

RESULTS = ROOT / "results"
OUT = ROOT / "ui" / "public" / "data" / "bundle.json"

CLASS_NAMES = {
    "H": "Healthy",
    "B1": "Insufficient lubrication",
    "B2": "Severely insufficient lubrication",
    "B3": "Cracked outer ring",
    "V": "Voltage imbalance",
    "R": "Broken rotor bar",
}
USER_CH = [3, 4, 5]          # gUserX, gUserY, gUserZ
EXCERPT_START = 45_056       # samples (~7.5 min into each 15-min run)
EXCERPT_LEN = 3 * WINDOW     # 3,072 samples = 30.7 s
PROJECTIONS = ("pca5", "lda5", "kpca5")


def b64_int16(a: np.ndarray) -> tuple[str, float]:
    """Quantise to int16 by max |a| and base64-encode (little-endian, row-major)."""
    a = np.asarray(a, dtype=np.float64)
    scale = float(np.abs(a).max()) or 1.0
    q = np.round(a / scale * 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode(), scale


def b64_uint8(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype=np.uint8).tobytes()).decode()


def read_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return json.loads(pd.read_csv(path).to_json(orient="records"))


def recordings_and_signals(recs, folds_grouped, meta) -> tuple[list, dict, dict]:
    """Per recording: metadata, the signal excerpt and the whole-recording spectrum."""
    fold_of = {}
    for k, (_, te) in enumerate(folds_grouped):
        for rid in meta.rec_id.iloc[te].unique():
            fold_of[rid] = k
    conds = sorted({(r.load, r.supply_hz) for r in recs})
    n_win = meta.groupby("rec_id").size()

    rows, signals, spectra = [], {}, {}
    df = FS / 8192
    for r in recs:
        f, p = recording_spectrum(r.signal)
        sb = sideband_levels(f, p, r.supply_hz)
        rows.append(dict(id=r.rec_id, label=r.label, load=r.load, supply=r.supply_hz,
                         n_windows=int(n_win[r.rec_id]), f1=sb["f1"], slip=sb["slip"],
                         f_pole=sb["f_pole"], fold=fold_of[r.rec_id],
                         cond=conds.index((r.load, r.supply_hz))))
        x = r.signal[EXCERPT_START:EXCERPT_START + EXCERPT_LEN, USER_CH]
        x = x - x.mean(axis=0)
        b, s = b64_int16(x.T)                                      # planar (3, n)
        signals[r.rec_id] = dict(b64=b, scale=s)
        db = 10 * np.log10(p + 1e-30)
        q = np.round(db * 100).clip(-32768, 32767).astype("<i2")   # 0.01 dB steps
        spectra[r.rec_id] = base64.b64encode(q.tobytes()).decode()
    return rows, dict(start=EXCERPT_START, n=EXCERPT_LEN,
                      channels=["gUserX", "gUserY", "gUserZ"], data=signals), \
        dict(df=df, n_bins=int(len(f)), data=spectra)


def embeddings(F: pd.DataFrame, meta: pd.DataFrame, folds, seed: int) -> dict:
    """Fit scaler + projection on each grouped training fold; project every window."""
    X, y = F.to_numpy(), meta.label.to_numpy()
    fronts = front_ends(seed=seed)
    out = {}
    fisher = []
    for name in PROJECTIONS:
        per_fold = []
        for k, (tr, te) in enumerate(folds):
            pipe = make_pipeline(fronts[name], []).fit(X[tr], y[tr])
            Z = pipe.transform(X)
            scale = np.abs(Z).max(axis=0)
            q = np.round(Z / scale * 32767).astype("<i2")
            per_fold.append(dict(scale=scale.tolist(),
                                 b64=base64.b64encode(q.tobytes()).decode()))
            fisher.append(dict(proj=name, fold=k,
                               train=fisher_criterion(Z[tr], y[tr]),
                               heldout=transfer_fisher(Z[tr], y[tr], Z[te], y[te])))
            print(f"  {name} fold {k}: J train {fisher[-1]['train']:.1f}, "
                  f"held-out {fisher[-1]['heldout']:.1f}", flush=True)
        out[name] = per_fold
    rec_ids = sorted(meta.rec_id.unique())
    return dict(n=len(meta), dims=5,
                label=b64_uint8(meta.label.map(CLASSES.index)),
                rec=b64_uint8(meta.rec_id.map(rec_ids.index)),
                rec_ids=rec_ids,
                win=base64.b64encode(meta.win_idx.to_numpy().astype("<u2").tobytes()).decode(),
                proj=out, fisher=fisher)


def records(df: pd.DataFrame | None) -> list[dict] | None:
    return None if df is None else json.loads(df.to_json(orient="records"))   # NaN -> null


def multiseed(name: str, table: str) -> pd.DataFrame | None:
    """Mean and std over seeds of results/<name>[_seed<k>]/<table>.csv, with n_seeds.

    Aggregated here from the per-seed folders, like scripts/aggregate_seeds.py does; if
    results/multiseed/<name>__<table>.csv covers more seeds (e.g. copied from the cluster run),
    that file is used instead. None when neither exists.
    """
    _, _, keys, values = next(t for t in aggregate_seeds.TABLES if t[:2] == (name, table))
    frames = [pd.read_csv(d / f"{table}.csv").assign(seed=k)
              for k, d in aggregate_seeds.seed_dirs(name).items() if (d / f"{table}.csv").exists()]
    local = None
    if frames:
        df = pd.concat(frames, ignore_index=True)
        cols = [c for c in values if c in df.columns]
        local = df.groupby(keys, sort=False)[cols].agg(["mean", "std"])
        local.columns = [f"{c}_{stat}" for c, stat in local.columns]
        local["n_seeds"] = df.groupby(keys, sort=False)["seed"].nunique()
        local = local.reset_index()
    path = RESULTS / "multiseed" / f"{name}__{table}.csv"
    shared = pd.read_csv(path) if path.exists() else None
    found = [d for d in (local, shared) if d is not None]
    return max(found, key=lambda d: d.n_seeds.max()) if found else None


def model_grid() -> tuple[list[dict] | None, int]:
    """Grouped / condition / naive macro-F1 per model: the multi-seed mean when at least two
    seeds exist (std over seeds), otherwise seed 0 (std over folds)."""
    ms = multiseed("baseline", "metrics")
    if ms is not None and ms.n_seeds.min() >= 2:
        grid = ms.rename(columns={"macro_f1_mean": "macro_f1", "macro_f1_std": "macro_f1_std",
                                  "acc_mean": "acc", "acc_std": "acc_std"})
        return records(grid[["split", "front", "clf", "acc", "acc_std", "macro_f1",
                             "macro_f1_std", "n_seeds"]]), int(ms.n_seeds.min())
    return read_csv(RESULTS / "baseline" / "metrics.csv"), 1


def fisher_curves() -> list[dict]:
    """Fisher J vs k. Held-out J is the 5-seed mean +- std when make_report_figures.py has
    written analysis/fisher_vs_k_seeds.csv; in-sample J is always seed 0."""
    seed0 = pd.concat([pd.read_csv(p) for p in (RESULTS / "baseline" / "fisher_vs_k.csv",
                                                 RESULTS / "analysis" / "kpca_fisher_vs_k.csv")
                       if p.exists()], ignore_index=True)
    seeds = RESULTS / "analysis" / "fisher_vs_k_seeds.csv"
    if seeds.exists():
        s = pd.read_csv(seeds).groupby(["proj", "k"]).fisher_heldout.agg(
            fisher_heldout="mean", fisher_heldout_std="std", n_seeds="count").reset_index()
        seed0 = seed0.drop(columns=["fisher_heldout"]).merge(s, on=["proj", "k"], how="right")
    return records(seed0)


def confusions(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k].astype(int).tolist() for k in d.files}


def top_per_group(rows: list[dict] | None, key, value: str, n: int) -> dict:
    if rows is None:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for g, d in df.groupby(key):
        g = "-".join(g) if isinstance(g, tuple) else g
        out[g] = json.loads(d.nlargest(n, value).to_json(orient="records"))
    return out


def feature_families(path: Path) -> list[dict] | None:
    """Best held-out J per (pair, feature family), then the median over the 15 pairs."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    best = df.groupby(["class_a", "class_b", "feat_type"]).j_heldout.max().reset_index()
    fam = best.groupby("feat_type").j_heldout.agg(["median", "max"]).reset_index()
    top = best.loc[best.groupby("feat_type").j_heldout.idxmax()]
    fam = fam.merge(top.rename(columns={"j_heldout": "max"}), on=["feat_type", "max"])
    fam["best_pair"] = fam.class_a + "-" + fam.class_b
    return json.loads(fam[["feat_type", "median", "max", "best_pair"]]
                      .sort_values("median", ascending=False).to_json(orient="records"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta, F = load_feature_cache()
    folds = recording_folds(meta, "grouped", args.seed)
    print(f"{len(meta)} windows x {F.shape[1]} features")

    print("recordings, excerpts and spectra ...", flush=True)
    recs = load_all()
    rec_rows, signals, spectra = recordings_and_signals(recs, folds, meta)

    print("front-end embeddings per fold ...", flush=True)
    emb = embeddings(F, meta, folds, args.seed)

    an, base = RESULTS / "analysis", RESULTS / "baseline"
    pair_fisher = read_csv(an / "pair_fisher.csv")
    grid, grid_seeds = model_grid()
    print(f"model grid: {'seed 0' if grid_seeds == 1 else f'mean of {grid_seeds} seeds'}")
    separability = read_csv(base / "separability.csv")
    sep_ms = multiseed("baseline", "separability")
    if separability and sep_ms is not None and sep_ms.n_seeds.min() >= 2:
        held = dict(zip(sep_ms.front, sep_ms.fisher_heldout_mean))
        for r in separability:                   # held-out J over seeds; in-sample J stays seed 0
            r["fisher_heldout"] = held.get(r["front"], r["fisher_heldout"])
    bundle = dict(
        meta=dict(classes=CLASSES, class_names=CLASS_NAMES, fs=FS, window=WINDOW, step=STEP,
                  n_windows=len(meta), n_features=int(F.shape[1]), n_recordings=len(recs),
                  seed=args.seed, grid_seeds=grid_seeds, bpfo_order=BPFO_ORDER,
                  bpfi_order=BPFI_ORDER,
                  generated=dt.date.today().isoformat(),
                  citation=json.loads((ROOT / "data" / "raw" / "SOURCE.json").read_text())),
        recordings=rec_rows,
        signals=signals,
        spectra=spectra,
        peaks=read_csv(an / "peaks.csv"),
        embedding=emb,
        metrics=grid,
        per_class=read_csv(base / "per_class.csv"),
        confusion=confusions(base / "confusion.npz"),
        separability=separability,
        fisher_vs_k=fisher_curves(),
        pairs=read_csv(an / "pair_separability.csv"),
        pair_top=top_per_group(pair_fisher, ["class_a", "class_b"], "j_heldout", 5),
        class_vs_rest=top_per_group(read_csv(an / "class_vs_rest.csv"), "cls", "j_min", 6),
        feature_families=feature_families(an / "pair_fisher.csv"),
        sidebands=read_csv(an / "sidebands.csv"),
        sideband_ranks=read_csv(an / "sideband_ranks.csv"),
        regions=read_csv(an / "regions.csv"),
        dtw=dict(summary=read_csv(RESULTS / "dtw" / "summary.csv"),
                 folds=read_csv(RESULTS / "dtw" / "metrics.csv"),
                 confusion=confusions(RESULTS / "dtw" / "confusion.npz")),
        tuning=dict(summary=read_csv(RESULTS / "tuning" / "summary.csv"),
                    folds=read_csv(RESULTS / "tuning" / "nested_folds.csv")),
        corruption={c: records(multiseed(f"corruption_{c}", "metrics")) for c in ("both", "test")},
        ood=dict(metrics=records(multiseed("ood", "metrics")),
                 per_class=records(multiseed("ood", "per_class"))),
        sa=dict(metrics=records(multiseed("subspace_alignment", "metrics")),
                fisher=records(multiseed("subspace_alignment", "fisher"))),
    )
    missing = [k for k, v in bundle.items() if v is None]
    if missing:
        print("not found (run the matching script first):", ", ".join(missing))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
