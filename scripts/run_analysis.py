"""Separability analysis (ROADMAP item 2). Needs the feature cache from run_baseline.py.

Stages:
  fisher    per-feature, per-class-pair held-out Fisher scores (2a)
  sidebands rotor-bar sidebands: does R differ from the other classes? (2b; needs data/raw
            and pair_fisher.csv from the fisher stage)
  aliasing  B3 and V: aliased energy or low-frequency secondary effects? (2c; needs data/raw)

Outputs (results/analysis/):
  pair_fisher_folds.csv   every (fold, pair, feature) score
  pair_fisher.csv         per (pair, feature): median / worst / mean held-out J over folds,
                          and the in-sample J on the training folds for comparison
  class_vs_rest.csv       per class and feature: weakest and mean held-out J over its 5 pairs
  feature_types.csv       best channel per (pair, feature type)
  pair_separability.csv   held-out J per pair along its 2-class LDA direction (126 features)
  sidebands.csv           per recording: f1, slip, pole-pass f_p, sideband levels at f1 +- f_p
                          (0.012 Hz Welch spectrum), and aliased shaft harmonics on them
  sideband_ranks.csv      R's rank among the 6 classes at each condition, per measure
  r_vs_b2.csv             held-out J of every feature for the R-B2 pair
  regions.csv             held-out J per spectral region (sub-synchronous, shaft harmonics,
                          aliased shaft / line harmonics, other), total level and slip,
                          for every pair involving B3 or V
  regions_line_alias_30hz.csv  the 2x-line alias at 30 Hz supply only (the one speed where
                          it lands on its own bin)
  peaks.csv               strongest spectral peaks of every recording, the known source each
                          matches (in-band or aliased), and its level over H at that condition
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
from motorpr.data import load_all, windows_with_meta  # noqa: E402
from motorpr.evaluation import recording_folds  # noqa: E402
from motorpr.features import SIDEBAND  # noqa: E402
from motorpr.runlog import write_runlog  # noqa: E402

OUT = ROOT / "results" / "analysis"
STAGES = ("fisher", "sidebands", "aliasing")
EXCESS_DB = 10.0     # a peak counts as class-specific when it is this far above H


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


def stage_sidebands(meta: pd.DataFrame, F: pd.DataFrame) -> None:
    rows = []
    for r in load_all():
        f, p = an.recording_spectrum(r.signal)
        s = an.sideband_levels(f, p, r.supply_hz)
        rows.append(dict(rec_id=r.rec_id, label=r.label, load=r.load, supply_hz=r.supply_hz,
                         **s, in_feature_band=SIDEBAND[0] <= s["f_pole"] <= SIDEBAND[1],
                         on_aliased_harmonic=an.sideband_alias_overlap(s["f1"], s["f_pole"])))
    sb = pd.DataFrame(rows)
    sb.to_csv(OUT / "sidebands.csv", index=False)

    # the hand-written features, averaged over the 6 channels, then per-recording medians
    agg = pd.DataFrame({
        "feat_sideband": F.filter(like="_sideband").mean(axis=1),
        "feat_e1x": F.filter(like="_e1x").mean(axis=1),
        "feat_slip": F["gX_slip"],                      # identical on every channel
    })
    rec = an.recording_medians(agg, meta, list(agg.columns)).merge(
        sb[["rec_id", "slip", "snr_db", "sbr_db", "resolved"]], on="rec_id")
    ranks = []
    for col in ("snr_db", "sbr_db", "slip", "feat_sideband", "feat_e1x", "feat_slip"):
        use = rec[rec.resolved] if col in ("snr_db", "sbr_db") else rec
        ranks.append(an.condition_ranks(use, col, "R").assign(measure=col))
    ranks = pd.concat(ranks, ignore_index=True)
    ranks.to_csv(OUT / "sideband_ranks.csv", index=False)

    pf = pd.read_csv(OUT / "pair_fisher.csv")
    rb2 = pf[(pf.class_a == "B2") & (pf.class_b == "R")].sort_values("j_heldout", ascending=False)
    rb2.to_csv(OUT / "r_vs_b2.csv", index=False)

    for load, d in sb.groupby("load"):
        print(f"[sidebands] load={load}: f_p {d.f_pole.min():.2f}-{d.f_pole.max():.2f} Hz; "
              f"inside the feature's {SIDEBAND[0]}-{SIDEBAND[1]} Hz band: "
              f"{int(d.in_feature_band.sum())}/{len(d)}; resolved: {int(d.resolved.sum())}/{len(d)}")
    print("Sidebands sitting on an aliased shaft harmonic:",
          sb[sb.on_aliased_harmonic != ""].groupby("supply_hz").on_aliased_harmonic
          .agg(lambda s: f"{len(s)} recs ({s.iloc[0]})").to_dict())
    print("\nR's rank among 6 classes per condition (1 = largest):")
    print(ranks.pivot_table(index="measure", columns=["load", "supply_hz"], values="rank")
          .to_string())
    print("\nTop R-B2 features (median held-out J):")
    print(rb2.head(8)[["feature", "j_heldout", "j_heldout_min", "j_train"]].round(2)
          .to_string(index=False))
    print("R-B2 sideband features:",
          ", ".join(f"{r.feature} {r.j_heldout:.2f}" for r in rb2[rb2.feat_type == "sideband"]
                    .itertuples()))


def stage_aliasing(meta: pd.DataFrame, F: pd.DataFrame, seed: int) -> None:
    recs = load_all()

    # window level: held-out J of each spectral region, total level and slip
    W, meta_w = windows_with_meta(recs)
    assert (meta_w[["rec_id", "win_idx"]].to_numpy() == meta[["rec_id", "win_idx"]].to_numpy()).all()
    R = an.region_energies(W, meta_w["supply_hz"].to_numpy())
    del W
    R["mech_slip"] = F["gX_slip"].to_numpy()
    folds = recording_folds(meta, "grouped", seed)
    s = an.summarise_pair_fisher(an.pair_fisher_table(R, meta, folds))
    s = s[s.class_a.isin(["B3", "V"]) | s.class_b.isin(["B3", "V"])].copy()
    s["pair"] = s.class_a + "-" + s.class_b
    s.to_csv(OUT / "regions.csv", index=False)
    print("[aliasing] median held-out J per measure, pairs involving B3 or V:")
    print(s.pivot(index="feature", columns="pair", values="j_heldout").round(2).to_string())
    print("Median log10 energy share at aliased shaft harmonics, and log10 total level:")
    print(pd.concat([meta[["label", "load"]], R], axis=1).groupby(["label", "load"])
          [["region_shaft_alias", "region_total"]].median().unstack().round(2).to_string())

    # 2x line (the V signature) folds to DC at 50 Hz and onto ~f1 at 40 Hz; only at 30 Hz
    # does it land on a distinct bin (40 Hz). Score it there alone: 2 usable folds.
    m30 = (meta.supply_hz == 30).to_numpy()
    sub = meta[m30].reset_index(drop=True)
    f30 = [(tr, te) for tr, te in recording_folds(sub, "grouped", seed)
           if len(te) and sub.label.iloc[te].nunique() == len(an.CLASSES)]
    s30 = an.summarise_pair_fisher(an.pair_fisher_table(
        R.loc[m30, ["region_line_alias"]].reset_index(drop=True), sub, f30))
    s30 = s30[s30.class_a.isin(["B3", "V"]) | s30.class_b.isin(["B3", "V"])]
    s30.assign(supply_hz=30, n_folds=len(f30)).to_csv(OUT / "regions_line_alias_30hz.csv",
                                                       index=False)
    print(f"\n2x-line alias (40 Hz) at 30 Hz supply only, {len(f30)} folds: " + ", ".join(
        f"{r.class_a}-{r.class_b} {r.j_heldout:.2f}" for r in s30.itertuples()))

    # recording level: strongest peaks, their likely source, and their level over H
    spec = {r.rec_id: an.recording_spectrum(r.signal) for r in recs}
    rows = []
    for r in recs:
        f, p = spec[r.rec_id]
        f1 = an.refine_running_speed(f, p, r.supply_hz)
        src = an.candidate_sources(f1, r.supply_hz)
        p_h = spec[f"H_{r.load}_{r.supply_hz}Hz"][1]
        for q in an.spectral_peaks(f, p).itertuples():
            label = an.explain_frequency(q.freq, src)
            rows.append(dict(rec_id=r.rec_id, label=r.label, load=r.load, supply_hz=r.supply_hz,
                             f1=f1, freq=q.freq, order=q.freq / f1, level_db=q.level_db,
                             prominence_db=q.prominence_db, source=label,
                             category=an.peak_category(label),
                             over_h_db=an.excess_over_reference(f, p, p_h, q.freq)))
    peaks = pd.DataFrame(rows)
    peaks.to_csv(OUT / "peaks.csv", index=False)
    spec_pk = peaks[(peaks.label.isin(["B3", "V"])) & (peaks.over_h_db >= EXCESS_DB)]
    print(f"\nPeaks of B3 / V standing >= {EXCESS_DB:.0f} dB over H at the same condition, "
          "by likely source:")
    print(spec_pk.groupby(["label", "category"]).size().unstack(fill_value=0).to_string())
    print("\nOver-H level of every peak, median per recording (dB):")
    print(peaks[peaks.label.isin(["B3", "V"])].groupby(["label", "load", "supply_hz"])
          .over_h_db.median().unstack().round(1).to_string())


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
    if "sidebands" in args.stages:
        stage_sidebands(meta, F)
    if "aliasing" in args.stages:
        stage_aliasing(meta, F, args.seed)
    write_runlog(OUT, **vars(args), split="grouped", n_windows=len(F))


if __name__ == "__main__":
    main()
