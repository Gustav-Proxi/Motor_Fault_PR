import { motion } from "framer-motion";
import { useState } from "react";
import { CLASSES, CLFS, CLF_LABEL, FRONTS, FRONT_LABEL } from "../lib/data";
import { useData, useWidth } from "../lib/hooks";
import { inkOn, useTheme } from "../lib/theme";
import { Counter, HBar, Section, Seg, TableView } from "../components/ui";

type Split = "grouped" | "condition" | "naive";

function Grid({ split, sel, onSel }: { split: Split; sel: string; onSel: (k: string) => void }) {
  const d = useData();
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(700);
  const L = Math.min(118, w * 0.2);
  const top = 52;
  const cw = (w - L) / CLFS.length;
  const ch = 50;
  const ms = d.metrics.filter((m) => m.split === split);
  const best = ms.reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
  return (
    <div ref={ref}>
      <svg width={w} height={top + ch * FRONTS.length + 6} className="chart" role="img" aria-label={`Macro-F1 of every front-end and classifier, ${split} split`}>
        {CLFS.map((c, j) => (
          <text key={c} x={L + j * cw + cw / 2} y={top - 26} textAnchor="middle" className="lbl">
            {CLF_LABEL[c].split(" ").map((wd, k) => <tspan key={k} x={L + j * cw + cw / 2} dy={k ? 14 : 0}>{wd}</tspan>)}
          </text>
        ))}
        {FRONTS.map((f, i) => (
          <g key={f}>
            <text x={L - 10} y={top + i * ch + ch / 2 + 4} textAnchor="end" className="lbl-strong">{FRONT_LABEL[f]}</text>
            {CLFS.map((c, j) => {
              const m = ms.find((x) => x.front === f && x.clf === c)!;
              const fill = t.seq(m.macro_f1);
              const key = `${split}|${f}|${c}`;
              const isBest = m === best;
              return (
                <g key={c} style={{ cursor: "pointer" }} onClick={() => onSel(`${f}|${c}`)}>
                  <motion.rect x={L + j * cw + 2} y={top + i * ch + 2} width={cw - 4} height={ch - 4} rx={9}
                    initial={false} animate={{ fill }} transition={{ duration: 0.7, delay: (i * 6 + j) * 0.012 }}
                    stroke={sel === `${f}|${c}` ? t.accent : isBest ? t.ink : "none"} strokeWidth={sel === `${f}|${c}` ? 3 : isBest ? 1.5 : 0} />
                  <text x={L + j * cw + cw / 2} y={top + i * ch + ch / 2 + 5} textAnchor="middle" key={key}
                    style={{ fill: inkOn(fill), fontSize: 14, fontWeight: isBest ? 750 : 560 }}>{m.macro_f1.toFixed(2)}</text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}

function Confusion({ cm }: { cm: number[][] }) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(320);
  const L = 30;
  const cell = Math.min(46, (w - L) / 6);
  return (
    <div ref={ref}>
      <svg width={L + cell * 6} height={L + cell * 6 + 18} className="chart" role="img" aria-label="Row-normalised confusion matrix">
        {CLASSES.map((c, i) => (
          <g key={c}>
            <text x={L + i * cell + cell / 2} y={L - 8} textAnchor="middle" className="lbl-strong">{c}</text>
            <text x={L - 8} y={L + i * cell + cell / 2 + 4} textAnchor="end" className="lbl-strong">{c}</text>
          </g>
        ))}
        {cm.map((row, i) => {
          const s = row.reduce((a, b) => a + b, 0) || 1;
          return row.map((v, j) => {
            const p = v / s;
            const fill = t.seq(p);
            return (
              <g key={`${i}${j}`}>
                <motion.rect x={L + j * cell + 1} y={L + i * cell + 1} width={cell - 2} height={cell - 2} rx={5} initial={false} animate={{ fill }} transition={{ duration: 0.5 }}
                  stroke={i === j ? t.ink2 : "none"} strokeWidth={i === j ? 1 : 0} />
                <text x={L + j * cell + cell / 2} y={L + i * cell + cell / 2 + 4} textAnchor="middle" style={{ fill: inkOn(fill), fontSize: 11 }}>{p >= 0.005 ? p.toFixed(2).replace(/^0/, "") : ""}</text>
              </g>
            );
          });
        })}
        <text x={L} y={L + cell * 6 + 14}>rows: true class · columns: predicted</text>
      </svg>
    </div>
  );
}

export function Models() {
  const d = useData();
  const { t } = useTheme();
  const [split, setSplit] = useState<Split>("grouped");
  const gbest = d.metrics.filter((x) => x.split === "grouped").reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
  const [sel, setSel] = useState(`${gbest.front}|${gbest.clf}`);
  const multi = d.meta.grid_seeds > 1;
  const [f, c] = sel.split("|");
  const cm = d.confusion[`${split}|${sel}`];
  const m = d.metrics.find((x) => x.split === split && x.front === f && x.clf === c)!;
  const recall = cm.map((row, i) => row[i] / (row.reduce((a, b) => a + b, 0) || 1));
  const [ref, w] = useWidth<HTMLDivElement>(300);
  const bw = w - 80;

  return (
    <Section id="models" eyebrow="Model arena" tags={["baseline"]}
      title="Five front-ends × six classifiers"
      lede={<>Every model is a single pipeline (scaling, projection, classifier) fitted on the training fold only. Switch to the
        <b> naive</b> split and every cell turns near-perfect. That is the leakage. On unseen recordings{" "}
        {FRONT_LABEL[gbest.front]} with a {CLF_LABEL[gbest.clf].replace(/^(Linear|Random)\b/, (s) => s.toLowerCase())} is best, and Kernel PCA is worst.</>}>
      <div className="split">
        <div className="card">
          <div className="controls" style={{ justifyContent: "space-between" }}>
            <h3>Macro-F1, {multi ? `mean of ${d.meta.grid_seeds} seeds × 6 folds` : "mean of 6 folds (seed 0)"}</h3>
            <Seg value={split} onChange={setSplit} options={[{ value: "grouped", label: "Grouped" }, { value: "condition", label: "Leave condition out" }, { value: "naive", label: "Naive" }]} />
          </div>
          <Grid split={split} sel={sel} onSel={setSel} />
          <p className="sub">Cell shade = macro-F1 on a 0–1 scale (chance 0.17). Outlined: best in this split. Click a cell for its confusion matrix.</p>
          <TableView cols={["front-end", "classifier", "macro-F1", "± std", "accuracy"]}
            rows={d.metrics.filter((x) => x.split === split).map((x) => [FRONT_LABEL[x.front], CLF_LABEL[x.clf], x.macro_f1, x.macro_f1_std, x.acc])} />
        </div>
        <div className="card">
          <h3>{FRONT_LABEL[f]} + {CLF_LABEL[c]}</h3>
          <p className="sub">{split} split · macro-F1 <b style={{ color: "var(--ink)" }}><Counter value={m.macro_f1} duration={0.6} /></b> ± {m.macro_f1_std.toFixed(2)} across {multi ? "seeds" : "folds"}{multi ? " · confusion and recall below: seed 0" : ""}</p>
          <Confusion cm={cm} />
          <div ref={ref} style={{ marginTop: 12 }}>
            <span className="ctl-label">Recall per class</span>
            <svg width={w} height={CLASSES.length * 22 + 6} className="chart" role="img" aria-label="Recall per class" style={{ marginTop: 6 }}>
              {CLASSES.map((cl, i) => (
                <g key={cl}>
                  <text x={24} y={i * 22 + 15} textAnchor="end" className="lbl">{cl}</text>
                  <rect x={32} y={i * 22 + 4} width={bw} height={14} rx={3} fill={t.grid} opacity={0.5} />
                  <motion.g initial={false} animate={{ scaleX: 1 }}>
                    <HBar x0={32} x1={32 + bw * recall[i]} y={i * 22 + 4} h={14} fill={t.cls[cl]} />
                  </motion.g>
                  <text x={32 + bw + 6} y={i * 22 + 15} className="lbl">{recall[i].toFixed(2)}</text>
                </g>
              ))}
            </svg>
          </div>
          <p className="sub" style={{ color: "var(--ink2)", marginTop: 8 }}>
            B1 is recognised best. H, B3 and V blur together, and R is confused with B2 in both directions, which is where the
            separability analysis comes in.
          </p>
        </div>
      </div>
    </Section>
  );
}
