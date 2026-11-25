import { scaleBand, scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import { CLASSES, CLF_LABEL, CONDITIONS, FRONTS, FRONT_LABEL } from "../lib/data";
import { useData, useWidth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { AxisX, AxisY, HBar, Section, Seg, TableView, Tip, VBar } from "../components/ui";

type Split = "cross_speed" | "cross_load";
type Row = Record<string, any>;
const SPLIT_LABEL: Record<Split, string> = { cross_speed: "New speed", cross_load: "New load" };

/** Which operating conditions are trained on and which are tested, per split. */
function ConditionMap({ split }: { split: Split }) {
  const { t } = useTheme();
  const test = (c: { load: number; supply: number }) => (split === "cross_speed" ? c.supply === 40 : c.load === 1);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(6, minmax(0, 1fr))", gap: 6, margin: "4px 0 14px" }}>
      {CONDITIONS.map((c) => {
        const on = test(c);
        return (
          <div key={`${c.load}${c.supply}`} style={{
            borderRadius: 10, padding: "8px 4px", textAlign: "center", font: "600 11.5px var(--mono)",
            border: `1px solid ${on ? t.accent : "var(--hair2)"}`, background: on ? "var(--accent-wash)" : "transparent",
            transition: "all .35s",
          }}>
            {c.supply} Hz<br /><span style={{ fontWeight: 400, color: "var(--muted)" }}>{c.load ? "loaded" : "unloaded"}</span>
            <div style={{ marginTop: 4, color: on ? "var(--ink)" : "var(--muted)" }}>{on ? "test" : "train"}</div>
          </div>
        );
      })}
    </div>
  );
}

function OodBars({ split, rows }: { split: Split; rows: Row[] }) {
  const d = useData();
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(420);
  const [hover, setHover] = useState<{ x: number; y: number; r: Row; g: number } | null>(null);
  const best = (front: string) => rows.filter((r) => r.split === split && r.front === front)
    .reduce((a, b) => (b.macro_f1_mean > a.macro_f1_mean ? b : a));
  const grouped = (front: string) => Math.max(...d.metrics.filter((m) => m.split === "grouped" && m.front === front).map((m) => m.macro_f1));
  const L = Math.min(110, w * 0.3);
  const y = scaleBand().domain(FRONTS).range([6, FRONTS.length * 34 + 6]).padding(0.32);
  const x = scaleLinear().domain([0, 0.8]).range([L, w - 52]);
  const h = FRONTS.length * 34 + 44;
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={w} height={h} className="chart" role="img" aria-label={`Best macro-F1 per front-end, ${SPLIT_LABEL[split]}`}>
        <line x1={x(1 / 6)} x2={x(1 / 6)} y1={2} y2={h - 34} stroke={t.muted} />
        {FRONTS.map((f) => {
          const r = best(f);
          const g = grouped(f);
          const yy = y(f) ?? 0;
          return (
            <g key={f} onPointerEnter={() => setHover({ x: x(r.macro_f1_mean), y: yy, r, g })} onPointerLeave={() => setHover(null)}>
              <rect x={0} y={yy - 4} width={w} height={y.bandwidth() + 8} fill="transparent" />
              <text x={L - 8} y={yy + y.bandwidth() / 2 + 4} textAnchor="end" className="lbl">{FRONT_LABEL[f]}</text>
              <HBar x0={x(0)} x1={x(r.macro_f1_mean)} y={yy} h={y.bandwidth()} fill={split === "cross_speed" ? t.series[0] : t.series[1]} />
              {r.macro_f1_std > 0 && <line x1={x(r.macro_f1_mean - r.macro_f1_std)} x2={x(r.macro_f1_mean + r.macro_f1_std)} y1={yy + y.bandwidth() / 2} y2={yy + y.bandwidth() / 2} stroke={t.ink2} />}
              <line x1={x(g)} x2={x(g)} y1={yy - 3} y2={yy + y.bandwidth() + 3} stroke={t.ink} strokeWidth={2} />
              <text x={x(Math.max(r.macro_f1_mean, g)) + 8} y={yy + y.bandwidth() / 2 + 4} className="lbl-strong">{r.macro_f1_mean.toFixed(2)}</text>
            </g>
          );
        })}
        <AxisX scale={x} y={h - 34} ticks={4} label="macro-F1, best classifier" />
      </svg>
      {hover && (
        <Tip x={hover.x} y={hover.y} w={w}>
          <b>{FRONT_LABEL[hover.r.front]} + {CLF_LABEL[hover.r.clf]}</b>
          <div className="num"><span className="k">{SPLIT_LABEL[split].toLowerCase()}</span> {hover.r.macro_f1_mean.toFixed(3)}{hover.r.macro_f1_std > 0 ? ` ± ${hover.r.macro_f1_std.toFixed(3)}` : ""}</div>
          <div className="num"><span className="k">grouped CV (tick)</span> {hover.g.toFixed(3)}</div>
        </Tip>
      )}
    </div>
  );
}

function RecallStrip({ rows, split }: { rows: Row[]; split: Split }) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(420);
  const sub = rows.filter((r) => r.split === split && r.front === "raw" && r.clf === "rf");
  const bw = w - 34 - 44;
  return (
    <div ref={ref}>
      <svg width={w} height={CLASSES.length * 22 + 4} className="chart" role="img" aria-label="Per-class recall of raw features + random forest">
        {CLASSES.map((c, i) => {
          const r = sub.find((q) => q.cls === c);
          const v = r ? r.recall_mean : 0;
          return (
            <g key={c}>
              <text x={26} y={i * 22 + 15} textAnchor="end" className="lbl">{c}</text>
              <rect x={32} y={i * 22 + 4} width={bw} height={14} rx={3} fill={t.grid} opacity={0.5} />
              <HBar x0={32} x1={32 + bw * v} y={i * 22 + 4} h={14} fill={t.cls[c]} />
              <text x={32 + bw + 6} y={i * 22 + 15} className="lbl">{v.toFixed(2)}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function SaChart({ rows, target, stat }: { rows: Row[]; target: "speed" | "load1"; stat: "best" | "mean" }) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(460);
  const methods: [string, string][] = [["none", "no adaptation"], ["zscore", "per-domain z-score"], ["sa", "Subspace Alignment"]];
  const data = useMemo(() => {
    const sel = rows.filter((r) => (target === "load1" ? r.target === "load1" : r.target.startsWith("speed")));
    const out: Record<string, number> = {};
    for (const k of [5, 10, 20]) for (const [m] of methods) {
      // mean over the speed targets per classifier, then best or mean over classifiers
      const byClf = new Map<string, number[]>();
      for (const r of sel.filter((q) => q.k === k && q.method === m)) byClf.set(r.clf, [...(byClf.get(r.clf) ?? []), r.macro_f1_mean]);
      const per = [...byClf.values()].map((v) => v.reduce((a, b) => a + b, 0) / v.length);
      out[`${k}|${m}`] = per.length ? (stat === "best" ? Math.max(...per) : per.reduce((a, b) => a + b, 0) / per.length) : 0;
    }
    return out;
  }, [rows, target, stat]); // eslint-disable-line react-hooks/exhaustive-deps
  const h = 240;
  const x0 = scaleBand().domain(["5", "10", "20"]).range([44, w - 8]).padding(0.22);
  const x1 = scaleBand().domain(methods.map((m) => m[0])).range([0, x0.bandwidth()]).padding(0.12);
  const y = scaleLinear().domain([0, 0.7]).range([h - 34, 14]);
  return (
    <div ref={ref}>
      <svg width={w} height={h} className="chart" role="img" aria-label="Macro-F1 with and without adaptation">
        <AxisY scale={y} x={44} x2={w - 8} ticks={7} label="macro-F1" />
        <line x1={44} x2={w - 8} y1={y(1 / 6)} y2={y(1 / 6)} stroke={t.muted} />
        {[5, 10, 20].map((k) => methods.map(([m], i) => {
          const v = data[`${k}|${m}`];
          const xx = (x0(String(k)) ?? 0) + (x1(m) ?? 0);
          const bw = Math.min(24, x1.bandwidth());
          const off = (x1.bandwidth() - bw) / 2;
          return (
            <g key={`${k}${m}`}>
              <VBar x={xx + off} w={bw} y0={y(0)} y1={y(v)} fill={t.series[i]} />
              <text x={xx + x1.bandwidth() / 2} y={y(v) - 5} textAnchor="middle" className="lbl">{v.toFixed(2)}</text>
            </g>
          );
        }))}
        {[5, 10, 20].map((k) => (
          <text key={k} x={(x0(String(k)) ?? 0) + x0.bandwidth() / 2} y={h - 14} textAnchor="middle" className="lbl">k = {k}</text>
        ))}
      </svg>
      <div className="legend" style={{ marginTop: 4 }}>
        {methods.map(([m, lab], i) => <span key={m} className="item"><i style={{ width: 12, height: 12, borderRadius: 3, background: t.series[i], display: "inline-block" }} />{lab}</span>)}
      </div>
    </div>
  );
}

export function Domain() {
  const d = useData();
  const [split, setSplit] = useState<Split>("cross_load");
  const [target, setTarget] = useState<"speed" | "load1">("load1");
  const [stat, setStat] = useState<"best" | "mean">("best");
  const ood = d.ood.metrics, oodCls = d.ood.per_class, sa = d.sa.metrics;
  const nSeeds = Math.max(...(ood ?? sa ?? [{ n_seeds: 1 }]).map((r) => r.n_seeds ?? 1));
  const seedsTxt = nSeeds > 1 ? `mean over ${nSeeds} seeds` : "seed 0";

  return (
    <Section id="domain" eyebrow="Unseen conditions" tags={["3a", "3b"]}
      title="A new speed is easy. A new load is not."
      lede={<>Grouped cross-validation mixes speed and load. Two fixed splits separate them: train on 30 and 50 Hz and test on
        40 Hz, or train unloaded and test loaded. Loading lowers the shaft speed and raises the slip, so the whole harmonic comb
        moves, and the PCA front-ends collapse. <b>Subspace Alignment</b> and plain per-domain z-scoring use the new condition's
        unlabelled windows to recover.</>}>
      <div className="grid2">
        <div className="card">
          <div className="controls" style={{ justifyContent: "space-between" }}>
            <h3>Out-of-distribution splits</h3>
            <Seg value={split} onChange={setSplit} options={[{ value: "cross_speed", label: "New speed (40 Hz)" }, { value: "cross_load", label: "New load" }]} />
          </div>
          <ConditionMap split={split} />
          {ood ? (
            <>
              <p className="sub">Best classifier per front-end ({seedsTxt}, ± std). The black tick is the same front-end under grouped CV ({d.meta.grid_seeds > 1 ? `${d.meta.grid_seeds} seeds` : "seed 0"}).</p>
              <OodBars split={split} rows={ood} />
              {oodCls && (
                <>
                  <span className="ctl-label">Recall per class, raw features + random forest</span>
                  <RecallStrip rows={oodCls} split={split} />
                </>
              )}
              <p className="sub" style={{ color: "var(--ink2)", marginTop: 8 }}>
                {split === "cross_load"
                  ? "B1 survives the load change, while R collapses: its one consistent cue, the slip under load, never appears in training."
                  : "The test speed lies between the two training speeds at both loads, so this is interpolation, and it beats grouped CV."}
              </p>
              <TableView cols={["split", "front-end", "classifier", "macro-F1", "± std", "seeds"]}
                rows={ood.map((r) => [r.split, FRONT_LABEL[r.front], CLF_LABEL[r.clf], r.macro_f1_mean, r.macro_f1_std ?? 0, r.n_seeds])} />
            </>
          ) : <Missing what="scripts/run_ood.py" />}
        </div>
        <div className="card">
          <div className="controls" style={{ justifyContent: "space-between" }}>
            <h3>Adapting to the new condition</h3>
            <Seg value={target} onChange={setTarget} options={[{ value: "speed", label: "Unseen speed (mean of 3)" }, { value: "load1", label: "Unseen load" }]} />
          </div>
          {sa ? (
            <>
              <div className="controls">
                <Seg value={stat} onChange={setStat} options={[{ value: "best", label: "Best classifier" }, { value: "mean", label: "Mean of 6 classifiers" }]} />
                <span className="ctl-label">{seedsTxt}</span>
              </div>
              <SaChart rows={sa} target={target} stat={stat} />
              <p className="sub" style={{ color: "var(--ink2)", marginTop: 10 }}>
                Both adaptations use the target's unlabelled windows. Most of the gain comes from z-scoring each condition with its own
                statistics; the alignment step adds little on top, and in either direction depending on k. The condition shift is mostly a
                change of location and scale per feature.
              </p>
            </>
          ) : <Missing what="scripts/run_sa.py" />}
        </div>
      </div>
    </Section>
  );
}

function Missing({ what }: { what: string }) {
  return (
    <div className="callout">
      <b>Not in this bundle.</b> Run <span className="mono">{what} --seed s</span> for the seeds you want, then
      <span className="mono"> scripts/export_ui_data.py</span>.
    </div>
  );
}
