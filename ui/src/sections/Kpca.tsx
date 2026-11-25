import { motion } from "framer-motion";
import { scaleLinear, scaleLog } from "d3-scale";
import { useState } from "react";
import { CLF_LABEL, FRONT_LABEL } from "../lib/data";
import { useData, useWidth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { AxisX, AxisY, Section, Seg, TableView, Tip } from "../components/ui";

// framer-motion line-draw; typed loosely because its SVG pathLength types reject numbers here
const DRAW = { initial: { pathLength: 0 }, whileInView: { pathLength: 1 } } as Record<string, unknown>;

function FisherVsK({ col }: { col: "fisher_heldout" | "fisher_train" }) {
  const d = useData();
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(420);
  const h = 250;
  const projs = ["pca", "lda", "kpca"];
  const names: Record<string, string> = { pca: "PCA", lda: "LDA", kpca: "Kernel PCA" };
  const rows = d.fisher_vs_k;
  const vals = rows.map((r) => r[col] as number);
  const x = scaleLinear().domain([1, 5]).range([46, w - 92]);
  const y = col === "fisher_train"
    ? scaleLog().domain([0.1, Math.max(...vals) * 1.3]).range([h - 30, 14])
    : scaleLinear().domain([0, Math.max(...vals) * 1.1]).range([h - 30, 14]).nice();
  return (
    <div ref={ref}>
      <svg width={w} height={h} className="chart" role="img" aria-label="Fisher J against the number of projected dimensions">
        <AxisY scale={y} x={46} x2={w - 92} ticks={col === "fisher_train" ? 4 : 5} format={(v) => (v >= 1 ? `${+v.toFixed(0)}` : `${v}`)} label={col === "fisher_train" ? "J in-sample (log)" : "J held-out"} />
        {projs.map((p, i) => {
          const s = rows.filter((r) => r.proj === p).sort((a, b) => a.k - b.k);
          if (!s.length) return null;
          const path = s.map((r, k) => `${k ? "L" : "M"}${x(r.k)} ${y(Math.max(0.1, r[col]))}`).join("");
          const last = s[s.length - 1];
          return (
            <g key={p}>
              <motion.path d={path} fill="none" stroke={t.series[i]} strokeWidth={2} {...DRAW} viewport={{ once: true }} transition={{ duration: 1.1, delay: i * 0.15 }} />
              {s.map((r) => <circle key={r.k} cx={x(r.k)} cy={y(Math.max(0.1, r[col]))} r={4} fill={t.series[i]} stroke={t.surface} strokeWidth={2} />)}
              <text x={x(5) + 10} y={y(Math.max(0.1, last[col])) + 4 + (p === "lda" && col === "fisher_heldout" ? -7 : p === "pca" && col === "fisher_heldout" ? 7 : 0)} className="lbl">{names[p]} {last[col].toFixed(1)}</text>
            </g>
          );
        })}
        <AxisX scale={x} y={h - 30} ticks={5} format={(v) => `${v}`} label="projected dimensions k" />
      </svg>
    </div>
  );
}

function KernelWidth() {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(420);
  const h = 210;
  const gammas: [number, string][] = [[0.002, "γ = 0.002 (wide)"], [1 / 126, "γ = 1/126 (default)"], [0.03, "γ = 0.03 (narrow)"]];
  const x = scaleLinear().domain([0, 30]).range([40, w - 16]);
  const y = scaleLinear().domain([0, 1]).range([h - 30, 12]);
  return (
    <div ref={ref}>
      <svg width={w} height={h} className="chart" role="img" aria-label="RBF kernel similarity against distance for three kernel widths">
        <AxisY scale={y} x={40} x2={w - 16} ticks={4} label="kernel value k(x, x′)" />
        <rect x={x(13)} width={x(19) - x(13)} y={12} height={h - 42} fill={t.accent} opacity={0.1} />
        <text x={x(16)} y={26} textAnchor="middle" className="lbl">typical pair ≈ 16</text>
        {gammas.map(([g, lab], i) => {
          let p = "";
          for (let dd = 0; dd <= 30; dd += 0.25) p += `${dd ? "L" : "M"}${x(dd)} ${y(Math.exp(-g * dd * dd))}`;
          const lx = 6 + i * 4;
          return (
            <g key={g}>
              <path d={p} fill="none" stroke={t.series[i]} strokeWidth={2} />
              <text x={x(lx) + 6} y={y(Math.exp(-g * lx * lx)) - 6} className="lbl">{lab}</text>
            </g>
          );
        })}
        <AxisX scale={x} y={h - 30} ticks={6} label="distance between standardised windows" />
      </svg>
    </div>
  );
}

/** Gain from the rounded scores, as in the report table: "+0.13", "−0.02", "0.00". */
function gainText(tuned: number, def: number) {
  const g = Math.round((Math.round(tuned * 100) - Math.round(def * 100))) / 100;
  return g === 0 ? "0.00" : `${g > 0 ? "+" : "−"}${Math.abs(g).toFixed(2)}`;
}

function Dumbbell() {
  const d = useData();
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(560);
  const [hover, setHover] = useState<{ x: number; y: number; r: Record<string, any> } | null>(null);
  const rows = d.tuning.summary ?? [];
  const L = Math.min(210, w * 0.42);
  const rh = 24;
  const x = scaleLinear().domain([0.1, 0.55]).range([L, w - 56]);
  const groupGap = 14;
  let yy = 18;
  const placed = rows.map((r, i) => {
    if (i > 0 && rows[i - 1].front !== r.front) yy += groupGap;
    const o = { r, y: yy };
    yy += rh;
    return o;
  });
  const h = yy + 34;
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={w} height={h} className="chart" role="img" aria-label="Macro-F1 with fixed defaults and after nested tuning">
        {x.ticks(5).map((v) => <line key={v} x1={x(v)} x2={x(v)} y1={8} y2={h - 30} stroke={t.grid} />)}
        <line x1={x(1 / 6)} x2={x(1 / 6)} y1={8} y2={h - 30} stroke={t.muted} />
        {placed.map(({ r, y }, i) => {
          const big = Math.abs(r.gain) >= 0.05;
          return (
            <g key={r.front + r.clf} className="row-in" style={{ animationDelay: `${i * 30}ms` }}
              onPointerEnter={() => setHover({ x: x(r.f1_tuned), y, r })} onPointerLeave={() => setHover(null)}>
              <rect x={0} y={y - rh / 2} width={w} height={rh} fill="transparent" />
              <text x={L - 10} y={y + 4} textAnchor="end" className="lbl">{FRONT_LABEL[r.front]} · {CLF_LABEL[r.clf]}</text>
              <line x1={x(r.f1_default)} x2={x(r.f1_tuned)} y1={y} y2={y} stroke={t.series[0]} strokeWidth={2} opacity={0.6} />
              <circle cx={x(r.f1_default)} cy={y} r={4.5} fill={t.surface} stroke={t.ink2} strokeWidth={2} />
              <circle cx={x(r.f1_tuned)} cy={y} r={5} fill={t.series[0]} stroke={t.surface} strokeWidth={2} />
              <text x={w - 50} y={y + 4} className={big ? "lbl-strong" : "lbl"}>{gainText(r.f1_tuned, r.f1_default)}</text>
            </g>
          );
        })}
        <AxisX scale={x} y={h - 30} ticks={5} label="macro-F1 (mean of 6 outer folds)" />
      </svg>
      {hover && (
        <Tip x={hover.x} y={hover.y} w={w}>
          <b>{FRONT_LABEL[hover.r.front]} + {CLF_LABEL[hover.r.clf]}</b>
          <div className="num"><span className="k">default</span> {hover.r.f1_default.toFixed(3)}</div>
          <div className="num"><span className="k">tuned</span> {hover.r.f1_tuned.toFixed(3)}</div>
          <div><span className="k">most chosen</span> {hover.r.top_params} ({hover.r.top_params_folds}/6 folds)</div>
        </Tip>
      )}
    </div>
  );
}

export function Kpca() {
  const d = useData();
  const [col, setCol] = useState<"fisher_heldout" | "fisher_train">("fisher_heldout");
  return (
    <Section id="kpca" eyebrow="Kernel PCA & tuning" tags={["5a", "5b", "R4"]}
      title="A non-linear projection, and a fair hyperparameter search"
      lede={<>We wrote RBF Kernel PCA by hand: centred kernel, top-k eigenvectors, fitted on 1,500 training windows per fold.
        Then every classifier was tuned with nested, recording-grouped CV. On clean data neither the non-linearity nor the tuning
        moves the needle. The limit is recording-to-recording shift, not model capacity.</>}>
      <div className="grid2">
        <div className="card">
          <div className="controls" style={{ justifyContent: "space-between" }}>
            <h3>Fisher J against k</h3>
            <Seg value={col} onChange={setCol} options={[{ value: "fisher_heldout", label: "Held-out" }, { value: "fisher_train", label: "In-sample" }]} />
          </div>
          <FisherVsK col={col} />
          <p className="sub" style={{ color: "var(--ink2)" }}>Held-out, Kernel PCA tracks linear PCA up to k = 4 and falls behind at k = 5. In-sample, LDA looks hundreds of times better, an illusion that disappears on unseen recordings.</p>
        </div>
        <div className="card">
          <h3>Why a wider kernel wins</h3>
          <p className="sub">RBF similarity exp(−γ·d²). In 126 standardised dimensions, two windows sit about 16 apart.</p>
          <KernelWidth />
          <p className="sub" style={{ color: "var(--ink2)" }}>At γ = 0.03 almost every pair of windows looks unrelated. Nested tuning most often picks γ = 0.002, which keeps the kernel close to linear, and so Kernel PCA moves toward plain PCA.</p>
        </div>
      </div>
      <div className="card" style={{ marginTop: 18 }}>
        <h3>Nested grouped tuning: default ○ → tuned ●</h3>
        <p className="sub">Inner 5-fold grouped grid search inside each of the 6 outer folds; both columns train on every second window. Hover a row for the selected setting.</p>
        <Dumbbell />
        <p className="sub" style={{ color: "var(--ink2)" }}>On raw and LDA-5 features the gain lies between −0.03 and +0.04, inside the fold-to-fold spread of 0.10–0.17. The largest gain is Kernel PCA-5 with an RBF SVM (0.22 → 0.35), and even tuned, Kernel PCA stays below the linear front-ends.</p>
        <TableView cols={["front-end", "classifier", "default", "tuned", "most chosen", "folds"]}
          rows={(d.tuning.summary ?? []).map((r) => [FRONT_LABEL[r.front], CLF_LABEL[r.clf], r.f1_default, r.f1_tuned, r.top_params, r.top_params_folds])} />
      </div>
    </Section>
  );
}
