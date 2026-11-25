import { motion } from "framer-motion";
import { scaleLinear } from "d3-scale";
import { useEffect, useMemo, useState } from "react";
import { CLASSES, CONDITIONS, FRONT_LABEL, CLF_LABEL, recId } from "../lib/data";
import { mulberry32 } from "../lib/dsp";
import { useData, useInView, useReducedMotion, useWidth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { AxisX, ClassMark, Counter, Section, Seg, TableView, Tip } from "../components/ui";

type Split = "naive" | "grouped" | "condition";
const TICKS = 24; // window cells drawn per recording tile

function Windowing() {
  const d = useData();
  const { t } = useTheme();
  const reduced = useReducedMotion();
  const [ref, w] = useWidth<HTMLDivElement>(800);
  const [vref, inView] = useInView<HTMLDivElement>();
  const [k, setK] = useState(0);
  useEffect(() => {
    if (!inView || reduced) return;
    const id = setInterval(() => setK((v) => (v + 1) % 5), 1500);
    return () => clearInterval(id);
  }, [inView, reduced]);
  const x = d.signal.get("B2_1_40Hz")![0];
  const n = x.length;
  const h = 120;
  const sx = (i: number) => (i / (n - 1)) * w;
  const path = useMemo(() => {
    let m = 0;
    for (const v of x) m = Math.max(m, Math.abs(v));
    let s = "";
    for (let i = 0; i < n; i += 2) s += (i ? "L" : "M") + sx(i).toFixed(1) + " " + (60 - (x[i] / m) * 44).toFixed(1);
    return s;
  }, [x, w]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div ref={vref}>
      <div ref={ref}>
        <svg width={w} height={h + 26} className="chart" role="img" aria-label="Sliding 1024-sample windows with 50 percent overlap">
          <motion.rect y={8} height={104} rx={8} fill={t.accent} opacity={0.1} initial={false}
            animate={{ x: sx(Math.max(0, k - 1) * 512), width: k ? sx(1024) : 0 }} transition={{ duration: 0.6 }} />
          <path d={path} fill="none" stroke={t.cls.B2} strokeWidth={1} opacity={0.9} />
          <motion.rect y={6} height={108} rx={8} fill="none" stroke={t.accent} strokeWidth={2} initial={false}
            animate={{ x: sx(k * 512), width: sx(1024) }} transition={{ type: "spring", stiffness: 90, damping: 18 }} />
          <motion.text y={h + 18} className="lbl-strong" initial={false} animate={{ x: sx(k * 512) + 6 }} transition={{ type: "spring", stiffness: 90, damping: 18 }}>
            window {k + 1} · samples {k * 512}–{k * 512 + 1023}
          </motion.text>
        </svg>
      </div>
      <p className="sub" style={{ marginTop: 4 }}>1,024 samples (10.24 s) per window and a 512-sample step: each window shares half its samples with the next (shaded). That gives about 174 windows per recording.</p>
    </div>
  );
}

function SplitGrid({ split, fold }: { split: Split; fold: number }) {
  const d = useData();
  const { t } = useTheme();
  // naive split: a fixed pseudo-random fold per window cell
  const naiveFold = useMemo(() => {
    const r = mulberry32(7);
    return Array.from({ length: 36 * TICKS }, () => Math.floor(r() * 6));
  }, []);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "34px repeat(6, minmax(0,1fr))", gap: 5, fontSize: 11 }}>
      <div />
      {CONDITIONS.map((c) => <div key={`${c.load}${c.supply}`} className="mono muted" style={{ textAlign: "center", fontSize: 10.5 }}>{c.supply} Hz {c.load ? "L" : "U"}</div>)}
      {CLASSES.map((cls, ci) => (
        <div key={cls} style={{ display: "contents" }}>
          <div className="mono" style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}><ClassMark c={cls} size={10} />{cls}</div>
          {CONDITIONS.map((c, k) => {
            const rec = d.recById.get(recId(cls, c.load, c.supply))!;
            const wholeTest = split === "grouped" ? rec.fold === fold : split === "condition" ? rec.cond === fold : false;
            return (
              <div key={k} style={{
                display: "grid", gridTemplateColumns: `repeat(${TICKS / 2}, 1fr)`, gap: 1.5, padding: 4, borderRadius: 7,
                border: `1px solid ${wholeTest ? t.accent : "var(--hair)"}`,
                boxShadow: wholeTest ? `0 0 14px ${t.accent}55` : "none", transition: "border-color .4s, box-shadow .4s",
              }}>
                {Array.from({ length: TICKS }, (_, i) => {
                  const test = split === "naive" ? naiveFold[(ci * 6 + k) * TICKS + i] === fold : wholeTest;
                  return <div key={i} style={{ height: 6, borderRadius: 1.5, background: test ? t.accent : t.mode === "dark" ? "#34363b" : "#d7d6cf", transition: `background .45s ${(i % 6) * 0.02}s` }} />;
                })}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function ScoreStrips({ split }: { split: Split }) {
  const d = useData();
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(500);
  const [hover, setHover] = useState<{ x: number; y: number; s: string; f: string; c: string; v: number } | null>(null);
  const x = scaleLinear().domain([0, 1]).range([92, w - 14]);
  const rows: Split[] = ["naive", "condition", "grouped"];
  const h = 48 + rows.length * 44;
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={w} height={h} className="chart" role="img" aria-label="Macro-F1 of all 30 models under each split">
        <line x1={x(1 / 6)} x2={x(1 / 6)} y1={8} y2={h - 30} stroke={t.muted} strokeWidth={1} />
        <text x={x(1 / 6) + 4} y={14}>chance</text>
        {rows.map((s, i) => {
          const ms = d.metrics.filter((m) => m.split === s);
          const best = ms.reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
          const yy = 34 + i * 44;
          return (
            <g key={s} opacity={s === split ? 1 : 0.45} style={{ transition: "opacity .4s" }}>
              <text x={84} y={yy + 4} textAnchor="end" className={s === split ? "lbl-strong" : "lbl"}>{s}</text>
              <line x1={x(0)} x2={x(1)} y1={yy} y2={yy} stroke={t.grid} />
              {ms.map((m) => (
                <circle key={m.front + m.clf} cx={x(m.macro_f1)} cy={yy} r={m === best ? 6 : 4.5}
                  fill={m === best ? t.accent : t.ink2} opacity={m === best ? 1 : 0.55} stroke={t.surface} strokeWidth={1.5}
                  onPointerEnter={() => setHover({ x: x(m.macro_f1), y: yy, s, f: m.front, c: m.clf, v: m.macro_f1 })}
                  onPointerLeave={() => setHover(null)} />
              ))}
              <text x={Math.min(x(best.macro_f1), w - 40)} y={yy - 10} textAnchor="middle" className="lbl-strong">{best.macro_f1.toFixed(2)}</text>
            </g>
          );
        })}
        <AxisX scale={x} y={h - 34} ticks={5} label="macro-F1" />
      </svg>
      {hover && (
        <Tip x={hover.x} y={hover.y} w={w}>
          <b>{FRONT_LABEL[hover.f]} + {CLF_LABEL[hover.c]}</b>
          <div className="num"><span className="k">{hover.s} split</span> {hover.v.toFixed(3)}</div>
        </Tip>
      )}
    </div>
  );
}

export function Validation() {
  const d = useData();
  const reduced = useReducedMotion();
  const [split, setSplit] = useState<Split>("grouped");
  const [fold, setFold] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [vref, inView] = useInView<HTMLDivElement>();
  useEffect(() => {
    if (!playing || !inView || reduced) return;
    const id = setInterval(() => setFold((f) => (f + 1) % 6), 1800);
    return () => clearInterval(id);
  }, [playing, inView, reduced]);
  const ms = d.metrics.filter((m) => m.split === split);
  const best = ms.reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
  const lo = Math.min(...ms.map((m) => m.macro_f1));
  const gbest = d.metrics.filter((m) => m.split === "grouped").reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
  const seeds = (d.meta.grid_seeds > 1 ? `mean of ${d.meta.grid_seeds} seeds` : "seed 0");
  const text: Record<Split, string> = {
    naive: "Windows are shuffled at random. Every recording lands in both train and test, and neighbouring windows share half their samples, so a model can simply recognise the recording. Near-perfect, and meaningless.",
    grouped: "Each fold holds out one whole recording per class, so the test recordings are never seen in training. This is the protocol behind every number on this page.",
    condition: "Each fold holds out one whole speed and load condition, for every class at once. No recording at that operating point is seen in training.",
  };

  return (
    <Section id="validation" eyebrow="Validation" tags={["protocol"]}
      title="The 1.00 that isn't: why recordings are held out whole"
      lede={<>Overlapping windows from one recording are almost copies of each other. A random split therefore leaks, and every
        model looks perfect. Holding out <b>whole recordings</b> is the honest test, and it puts the best model at{" "}
        {gbest.macro_f1.toFixed(2)} macro-F1 ({seeds}) against a chance level of 0.17.</>}>
      <div className="card" style={{ marginBottom: 18 }}>
        <h3>Cutting a recording into windows</h3>
        <Windowing />
      </div>
      <div className="split" ref={vref}>
        <div className="card">
          <div className="controls" style={{ justifyContent: "space-between" }}>
            <Seg value={split} onChange={(s) => { setSplit(s); setFold(0); }} options={[
              { value: "naive", label: "Naive random" }, { value: "grouped", label: "Grouped by recording" }, { value: "condition", label: "Leave condition out" }]} />
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <span className="ctl-label">Fold {fold + 1} / 6</span>
              <button className="btn ghost" onClick={() => setPlaying(!playing)}>{playing ? "❚❚" : "▶"}</button>
            </div>
          </div>
          <SplitGrid split={split} fold={fold} />
          <div className="legend" style={{ marginTop: 12 }}>
            <span className="item"><i style={{ width: 14, height: 6, borderRadius: 2, background: "var(--accent)", display: "inline-block" }} />test windows</span>
            <span className="item"><i style={{ width: 14, height: 6, borderRadius: 2, background: "var(--axis)", display: "inline-block" }} />training windows</span>
          </div>
          <p className="sub" style={{ marginTop: 10, color: "var(--ink2)", fontSize: 14 }}>{text[split]}</p>
        </div>
        <div className="card">
          <span className="ctl-label">Best macro-F1, {split} split</span>
          <div style={{ fontSize: 64, fontWeight: 760, letterSpacing: "-0.04em", lineHeight: 1.05, margin: "6px 0 2px" }}>
            <Counter value={best.macro_f1} />
          </div>
          <p className="sub">{FRONT_LABEL[best.front]} + {CLF_LABEL[best.clf]} · {seeds} · all 30 models range {lo.toFixed(2)}–{best.macro_f1.toFixed(2)}</p>
          <ScoreStrips split={split} />
          <TableView cols={["split", "front-end", "classifier", "macro-F1", "± std"]}
            rows={d.metrics.map((m) => [m.split, FRONT_LABEL[m.front], CLF_LABEL[m.clf], m.macro_f1, m.macro_f1_std])} />
        </div>
      </div>
    </Section>
  );
}
