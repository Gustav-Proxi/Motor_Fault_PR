import { motion } from "framer-motion";
import { scaleLinear } from "d3-scale";
import { useState } from "react";
import { CLASSES, CONDITIONS, featFamily, featLabel, type Cls } from "../lib/data";
import { useData, useWidth } from "../lib/hooks";
import { inkOn, useTheme } from "../lib/theme";
import { ClassMark, HBar, Section, Seg, TableView, Tip } from "../components/ui";

type Row = Record<string, any>;

function PairMatrix({ mode, sel, onSel }: { mode: "heldout" | "train"; sel: string; onSel: (k: string) => void }) {
  const d = useData();
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(420);
  const [hover, setHover] = useState<{ x: number; y: number; r: Row } | null>(null);
  const L = 34;
  const cell = Math.min(64, (w - L - 4) / 6);
  const get = (a: Cls, b: Cls) => d.pairs.find((p) => (p.class_a === a && p.class_b === b) || (p.class_a === b && p.class_b === a))!;
  const key = (r: Row) => `${r.class_a}-${r.class_b}`;
  const vals = d.pairs.map((p) => (mode === "heldout" ? p.j_heldout : p.j_train));
  const color = scaleLinear().domain([Math.min(...vals), Math.max(...vals)]).range([0.08, 1]);
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={L + cell * 6 + 4} height={L + cell * 6 + 4} className="chart" role="img" aria-label="Fisher J for every class pair">
        {CLASSES.map((c, i) => (
          <g key={c}>
            <text x={L + i * cell + cell / 2} y={L - 10} textAnchor="middle" className="lbl-strong">{c}</text>
            <text x={L - 10} y={L + i * cell + cell / 2 + 4} textAnchor="end" className="lbl-strong">{c}</text>
          </g>
        ))}
        {CLASSES.map((a, i) => CLASSES.map((b, j) => {
          if (j >= i) return j === i ? (
            <rect key={a + b} x={L + j * cell + 2} y={L + i * cell + 2} width={cell - 4} height={cell - 4} rx={8} fill="none" stroke="var(--hair2)" />
          ) : null;
          const r = get(a, b);
          const v = mode === "heldout" ? r.j_heldout : r.j_train;
          const fill = t.seq(color(v));
          const on = sel === key(r);
          return (
            <g key={a + b} style={{ cursor: "pointer" }} onClick={() => onSel(key(r))}
              onPointerEnter={() => setHover({ x: L + j * cell + cell, y: L + i * cell, r })} onPointerLeave={() => setHover(null)}>
              <motion.rect x={L + j * cell + 1} y={L + i * cell + 1} width={cell - 2} height={cell - 2} rx={8}
                initial={false} animate={{ fill }} transition={{ duration: 0.6, delay: (i + j) * 0.03 }}
                stroke={on ? t.accent : "none"} strokeWidth={on ? 3 : 0} />
              <text x={L + j * cell + cell / 2} y={L + i * cell + cell / 2 + 4} textAnchor="middle"
                style={{ fill: inkOn(fill), fontSize: cell > 50 ? 13 : 11, fontWeight: 600 }}>
                {mode === "heldout" ? v.toFixed(1) : Math.round(v)}
              </text>
            </g>
          );
        }))}
      </svg>
      {hover && (
        <Tip x={hover.x} y={hover.y} w={w}>
          <b>{hover.r.class_a} vs {hover.r.class_b}</b>
          <div className="num"><span className="k">held-out J (median)</span> {hover.r.j_heldout.toFixed(2)}</div>
          <div className="num"><span className="k">worst fold</span> {hover.r.j_heldout_min.toFixed(2)}</div>
          <div className="num"><span className="k">in-sample J</span> {hover.r.j_train.toFixed(0)}</div>
        </Tip>
      )}
    </div>
  );
}

function FeatureBars({ rows, value, label, max }: { rows: Row[]; value: string; label: (r: Row) => string; max?: number }) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(380);
  const L = Math.min(190, w * 0.48);
  const top = max ?? Math.max(...rows.map((r) => r[value]));
  const x = scaleLinear().domain([0, top]).range([L, w - 44]);
  return (
    <div ref={ref}>
      <svg width={w} height={rows.length * 26 + 6} className="chart" role="img" aria-label="Feature ranking">
        {rows.map((r, i) => (
          <motion.g key={label(r) + i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}>
            <text x={L - 8} y={i * 26 + 17} textAnchor="end" className="lbl">{label(r)}</text>
            <HBar x0={x(0)} x1={x(r[value])} y={i * 26 + 6} h={16} fill={t.series[0]} />
            <text x={x(r[value]) + 6} y={i * 26 + 18} className="lbl-strong">{r[value].toFixed(2)}</text>
          </motion.g>
        ))}
      </svg>
    </div>
  );
}

function SidebandRanks() {
  const d = useData();
  const { t } = useTheme();
  const measures: [string, string][] = [["snr_db", "sideband vs floor"], ["sbr_db", "sideband vs carrier"], ["feat_sideband", "sideband feature"], ["slip", "slip"]];
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="data" style={{ minWidth: 520 }}>
        <thead>
          <tr><th>R's rank of 6 by</th>{CONDITIONS.map((c) => <th key={`${c.load}${c.supply}`} style={{ textAlign: "center" }}>{c.supply} Hz {c.load ? "loaded" : "unloaded"}</th>)}</tr>
        </thead>
        <tbody>
          {measures.map(([m, lab]) => (
            <tr key={m}>
              <td>{lab}</td>
              {CONDITIONS.map((c) => {
                const r = d.sideband_ranks.find((x) => x.measure === m && x.load === c.load && x.supply_hz === c.supply);
                const rank = r?.rank as number | undefined;
                const fill = rank ? t.seq(1 - (rank - 1) / 6) : "transparent";
                return (
                  <td key={`${c.load}${c.supply}`} style={{ textAlign: "center" }}>
                    {rank ? <span style={{ display: "inline-block", minWidth: 30, padding: "3px 0", borderRadius: 7, background: fill, color: inkOn(fill), fontWeight: 650 }}>{rank}</span>
                      : <span className="muted" title="sidebands not resolved (f_p below 0.3 Hz)">n/r</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="sub" style={{ marginTop: 8 }}>1 = strongest of the six classes at that condition. n/r: unloaded, f_p is too small to separate the sidebands from the carrier.</p>
    </div>
  );
}

export function Separability() {
  const d = useData();
  const { t } = useTheme();
  const [mode, setMode] = useState<"heldout" | "train">("heldout");
  const [pair, setPair] = useState("H-B1");
  const [cls, setCls] = useState<Cls>("B1");
  const top = d.pair_top[pair] ?? d.pair_top[pair.split("-").reverse().join("-")] ?? [];
  const cvr = d.class_vs_rest[cls] ?? [];
  const fams = (d.feature_families ?? []).slice(0, 8);

  return (
    <Section id="separability" eyebrow="Separability analysis" tags={["2a", "2b", "2c", "R2"]}
      title="What actually tells the classes apart"
      lede={<>For every class pair we score separability <b>on held-out recordings</b>: class means come from the training fold and
        the spread from unseen windows. In-sample, LDA separates every pair (J in the hundreds). Held-out, J falls to 0.4–7.4.
        B1 separates from everything. B2, B3, V and R form a confusable block.</>}>
      <div className="split">
        <div className="card">
          <div className="controls" style={{ justifyContent: "space-between" }}>
            <h3>Pair separability along the 2-class LDA direction</h3>
            <Seg value={mode} onChange={setMode} options={[{ value: "heldout", label: "Held-out" }, { value: "train", label: "In-sample" }]} />
          </div>
          <div style={{ display: "flex", gap: 22, flexWrap: "wrap", alignItems: "flex-start" }}>
            <div style={{ flex: "1 1 320px", minWidth: 0 }}>
              <PairMatrix mode={mode} sel={pair} onSel={setPair} />
              <p className="sub">Median over 6 grouped folds. {t.mode === "dark" ? "Lighter" : "Darker"} = more separable{mode === "train" ? " (note the scale: hundreds)" : ""}. Click a cell.</p>
            </div>
            <div style={{ flex: "1 1 260px", minWidth: 0 }}>
              <span className="ctl-label">Top features for {pair.replace("-", " vs ")}</span>
              <FeatureBars rows={top} value="j_heldout" label={(r) => featLabel(r.feature)} />
              <p className="sub">Held-out J of each single feature.</p>
            </div>
          </div>
          <TableView cols={["pair", "held-out J", "worst fold", "in-sample J"]} rows={d.pairs.map((p) => [`${p.class_a}-${p.class_b}`, p.j_heldout, p.j_heldout_min, p.j_train])} />
        </div>
        <div className="stack">
          <div className="card">
            <h3>One class against all five others</h3>
            <p className="sub">Features ranked by their weakest pair (the J against the hardest other class).</p>
            <Seg value={cls} onChange={setCls} options={CLASSES.map((c) => ({ value: c, label: <><ClassMark c={c} size={10} />{c}</> }))} />
            <div style={{ marginTop: 10 }}>
              <FeatureBars rows={cvr} value="j_min" label={(r) => featLabel(r.feature)} max={1} />
            </div>
            <p className="sub" style={{ color: "var(--ink2)" }}>No single feature separates any class from all five others: the best weakest-pair J is below 1. Separation always needs combinations.</p>
          </div>
          <div className="card">
            <h3>Feature families</h3>
            <p className="sub">Best held-out J per pair and family, median over the 15 pairs.</p>
            <FeatureBars rows={fams} value="median" label={(r) => featFamily(r.feat_type)} />
          </div>
        </div>
      </div>
      <div className="grid2" style={{ marginTop: 18 }}>
        <div className="card">
          <h3>B1: lubrication loss shows up as impulsiveness</h3>
          <p className="sub" style={{ color: "var(--ink2)", fontSize: 14.5 }}>
            B1 is the only class with held-out J above 2 against all five others. Against H the strongest features are kurtosis, peak-to-peak
            and impulse factor: insufficient lubrication makes the vibration impulsive, and time-domain statistics capture that without
            any defect frequency.
          </p>
          <FeatureBars rows={(d.pair_top["H-B1"] ?? []).slice(0, 4)} value="j_heldout" label={(r) => featLabel(r.feature)} />
        </div>
        <div className="card">
          <h3>R: its rotor-bar sidebands never stand out</h3>
          <p className="sub" style={{ color: "var(--ink2)", fontSize: 14.5 }}>
            Measured directly at f1 ± f_p on a 0.012 Hz spectrum, R's sidebands are never the strongest at their condition. Its one
            consistent signature is slip under load.
          </p>
          <SidebandRanks />
        </div>
      </div>
    </Section>
  );
}
