import { scaleBand, scaleLinear } from "d3-scale";
import { useEffect, useMemo, useRef, useState } from "react";
import { CLASSES, CONDITIONS, recId, type Cls } from "../lib/data";
import { dtw, envelope } from "../lib/dsp";
import { setupCanvas, useData, useFrame, useInView, useReducedMotion, useWidth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { AxisX, ClassMark, HBar, Section, Seg } from "../components/ui";

interface Pick { cls: Cls; cond: number; win: number }

function Picker({ label, v, onChange }: { label: string; v: Pick; onChange: (p: Pick) => void }) {
  return (
    <div className="controls" style={{ marginBottom: 8 }}>
      <span className="ctl-label" style={{ width: 70 }}>{label}</span>
      <Seg value={v.cls} onChange={(cls) => onChange({ ...v, cls })} options={CLASSES.map((c) => ({ value: c, label: <><ClassMark c={c} size={9} />{c}</> }))} />
      <select value={v.cond} onChange={(e) => onChange({ ...v, cond: Number(e.target.value) })} aria-label={`${label} condition`}>
        {CONDITIONS.map((c, i) => <option key={i} value={i}>{c.supply} Hz · {c.load ? "loaded" : "unloaded"}</option>)}
      </select>
      <select value={v.win} onChange={(e) => onChange({ ...v, win: Number(e.target.value) })} aria-label={`${label} window`}>
        {[0, 1, 2, 3, 4].map((w) => <option key={w} value={w}>window {w + 1}</option>)}
      </select>
    </div>
  );
}

function CostMatrix({ a, b, band, ca, cb }: { a: Float32Array[]; b: Float32Array[]; band: number; ca: string; cb: string }) {
  const { t } = useTheme();
  const reduced = useReducedMotion();
  const [ref, w] = useWidth<HTMLDivElement>(560);
  const [vref, inView] = useInView<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const res = useMemo(() => dtw(a, b, band), [a, b, band]);
  const anim = useRef(0);
  useEffect(() => { anim.current = reduced ? 3 : 0; }, [res, reduced]);
  const size = Math.min(w, 560);
  const E = 46; // envelope strip width
  const N = size - E - 4;
  const T = res.T;
  const cs = N / T;
  const meanEnv = (e: Float32Array[]) => e[0].map((_, i) => (e[0][i] + e[1][i] + e[2][i]) / 3);
  const ea = useMemo(() => meanEnv(a), [a]);
  const eb = useMemo(() => meanEnv(b), [b]);

  const drawn = useRef<{ res: unknown; key: string }>({ res: null, key: "" });
  useFrame((dt) => {
    const c = canvas.current;
    if (!c) return;
    // once the animation has finished, redraw only when something changed (saves CPU)
    const key = `${size}|${t.mode}|${ca}|${cb}`;
    if (anim.current >= 3 && drawn.current.res === res && drawn.current.key === key) return;
    drawn.current = { res, key };
    if (anim.current < 3) anim.current = Math.min(3, anim.current + dt * 1.25);
    const g = setupCanvas(c, size, size);
    g.clearRect(0, 0, size, size);
    const p = anim.current; // 0..2 sweep, 2..3 path
    const sweep = Math.min(1, p / 2) * (2 * T);
    let cmax = 0, dmax = 0;
    for (let i = 0; i < T * T; i++) { cmax = Math.max(cmax, res.cost[i]); if (Number.isFinite(res.D[i])) dmax = Math.max(dmax, res.D[i]); }
    for (let i = 0; i < T; i++)
      for (let j = 0; j < T; j++) {
        const inBand = Math.abs(i - j) <= band;
        const done = i + j <= sweep;
        let col: string;
        if (!inBand) col = t.mode === "dark" ? "#16171a" : "#efeee9";
        else if (done) col = t.seq(0.15 + 0.85 * (res.D[i * T + j] / dmax));
        else col = t.seq(0.05 + 0.45 * (res.cost[i * T + j] / cmax));
        g.fillStyle = col;
        g.fillRect(E + 4 + j * cs, E + 4 + i * cs, Math.ceil(cs), Math.ceil(cs));
      }
    // sweep front
    if (p < 2) {
      g.strokeStyle = t.accent; g.lineWidth = 2; g.globalAlpha = 0.8;
      g.beginPath(); g.moveTo(E + 4 + Math.min(sweep, T) * cs, E + 4 + Math.max(0, sweep - T) * cs); g.lineTo(E + 4 + Math.max(0, sweep - T) * cs, E + 4 + Math.min(sweep, T) * cs); g.stroke();
      g.globalAlpha = 1;
    }
    // warping path
    if (p > 2) {
      const k = Math.floor(Math.min(1, p - 2) * res.path.length);
      g.strokeStyle = t.accent; g.lineWidth = 3; g.lineJoin = "round"; g.shadowColor = t.accent; g.shadowBlur = 12;
      g.beginPath();
      res.path.slice(0, Math.max(1, k)).forEach(([i, j], n) => { const x = E + 4 + (j + 0.5) * cs, y = E + 4 + (i + 0.5) * cs; n ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke(); g.shadowBlur = 0;
      // diagonal reference
      g.strokeStyle = t.ink2; g.lineWidth = 1; g.globalAlpha = 0.5; g.beginPath(); g.moveTo(E + 4, E + 4); g.lineTo(E + 4 + N, E + 4 + N); g.stroke(); g.globalAlpha = 1;
    }
    // envelopes: reference along the top, query down the left
    const lo = Math.min(...ea, ...eb), hi = Math.max(...ea, ...eb);
    g.strokeStyle = cb; g.lineWidth = 1.8; g.beginPath();
    eb.forEach((v, j) => { const x = E + 4 + (j + 0.5) * cs, y = E - 4 - ((v - lo) / (hi - lo)) * (E - 10); j ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
    g.strokeStyle = ca; g.beginPath();
    ea.forEach((v, i) => { const y = E + 4 + (i + 0.5) * cs, x = E - 4 - ((v - lo) / (hi - lo)) * (E - 10); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  }, inView);

  return (
    <div ref={vref}>
      <div ref={ref}>
        <canvas ref={canvas} role="img" aria-label={`DTW cost matrix with a band of ${band} frames; distance ${res.dist.toFixed(1)}`} />
      </div>
      <p className="sub" style={{ marginTop: 6 }}>Rows: query frames, columns: reference frames (64 × 0.16 s). The DP fills the band one anti-diagonal at a time, then the optimal alignment is traced back. Grey cells lie outside the Sakoe–Chiba band.</p>
    </div>
  );
}

function ResultBars() {
  const d = useData();
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(380);
  const rows = (d.dtw.summary ?? []).filter((r) => r.k === 1);
  const label: Record<string, string> = { dtw: "DTW on envelopes", nowarp: "no warping (band 0)", features: "126 hand-written features" };
  const order = ["dtw", "nowarp", "features"];
  const y = scaleBand().domain(order).range([8, order.length * 34 + 8]).padding(0.3);
  const x = scaleLinear().domain([0, 0.5]).range([Math.min(170, w * 0.45), w - 50]);
  return (
    <div ref={ref}>
      <svg width={w} height={order.length * 34 + 56} className="chart" role="img" aria-label="Macro-F1 of 1-NN with each distance">
        <line x1={x(1 / 6)} x2={x(1 / 6)} y1={4} y2={order.length * 34 + 10} stroke={t.muted} />
        {order.map((k) => {
          const r = rows.find((q) => q.distance === k);
          if (!r) return null;
          return (
            <g key={k}>
              <text x={x(0) - 8} y={(y(k) ?? 0) + y.bandwidth() / 2 + 4} textAnchor="end" className="lbl">{label[k]}</text>
              <HBar x0={x(0)} x1={x(r.macro_f1)} y={y(k) ?? 0} h={y.bandwidth()} fill={k === "features" ? t.series[1] : t.series[0]} />
              <line x1={x(r.macro_f1 - r.macro_f1_std)} x2={x(r.macro_f1 + r.macro_f1_std)} y1={(y(k) ?? 0) + y.bandwidth() / 2} y2={(y(k) ?? 0) + y.bandwidth() / 2} stroke={t.ink2} strokeWidth={1} />
              <text x={x(r.macro_f1 + r.macro_f1_std) + 6} y={(y(k) ?? 0) + y.bandwidth() / 2 + 4} className="lbl-strong">{r.macro_f1.toFixed(2)}</text>
            </g>
          );
        })}
        <AxisX scale={x} y={order.length * 34 + 14} ticks={5} label="macro-F1, 1-NN (± fold std)" />
      </svg>
    </div>
  );
}

export function Dtw() {
  const d = useData();
  const { t } = useTheme();
  const [qa, setQa] = useState<Pick>({ cls: "R", cond: 0, win: 0 });
  const [qb, setQb] = useState<Pick>({ cls: "R", cond: 1, win: 2 });
  const [band, setBand] = useState(8);
  const env = (p: Pick) => {
    const c = CONDITIONS[p.cond];
    return envelope(d.signal.get(recId(p.cls, c.load, c.supply))!, p.win * 512);
  };
  const a = useMemo(() => env(qa), [qa]); // eslint-disable-line react-hooks/exhaustive-deps
  const b = useMemo(() => env(qb), [qb]); // eslint-disable-line react-hooks/exhaustive-deps
  const dB = useMemo(() => dtw(a, b, band).dist, [a, b, band]);
  const d0 = useMemo(() => dtw(a, b, 0).dist, [a, b]);
  const s = (d.dtw.summary ?? []);
  const rec = (k: string, c: string) => s.find((r) => r.distance === k && r.k === 1)?.[`recall_${c}`] as number;

  return (
    <Section id="dtw" eyebrow="DTW studio" tags={["4a", "4b", "R3"]}
      title="Does elastic alignment help on steady-state runs?"
      lede={<>The proposal planned DTW for load- and speed-change segments, but every recording is a steady 15-minute run. So we use DTW
        as the distance of a k-NN classifier between window <b>envelopes</b>: the log RMS of 16-sample frames on the three gravity-compensated
        axes, a 64 × 3 sequence. Pick two windows and watch the alignment being computed.</>}>
      <div className="split">
        <div className="card">
          <Picker label="Query" v={qa} onChange={setQa} />
          <Picker label="Reference" v={qb} onChange={setQb} />
          <div className="controls">
            <span className="ctl-label">Band ± frames</span>
            <input type="range" min={0} max={24} value={band} onChange={(e) => setBand(Number(e.target.value))} aria-label="Sakoe-Chiba band" />
            <span className="mono num">{band}</span>
          </div>
          <CostMatrix a={a} b={b} band={band} ca={t.cls[qa.cls]} cb={t.cls[qb.cls]} />
        </div>
        <div className="stack">
          <div className="card">
            <h3>This pair</h3>
            <dl className="kv" style={{ marginTop: 8 }}>
              <dt>DTW distance, band {band}</dt><dd>{dB.toFixed(2)}</dd>
              <dt>No warping, band 0</dt><dd>{d0.toFixed(2)}</dd>
              <dt>Warping saves</dt><dd>{(100 * (1 - dB / d0)).toFixed(1)}%</dd>
            </dl>
            <p className="sub" style={{ marginTop: 10, color: "var(--ink2)" }}>Two windows of a steady run start at arbitrary points in the shaft cycle, so there's no common event to align. The path mostly follows random fluctuations.</p>
          </div>
          <div className="card">
            <h3>Grouped-CV result</h3>
            <p className="sub">About 450 reference windows, 1,584 queries.</p>
            <ResultBars />
            <p className="sub" style={{ color: "var(--ink2)" }}>Warping adds nothing measurable: 0.28 against 0.26 without it, well inside the fold spread. The envelope is complementary, though:</p>
            <dl className="kv">
              <dt style={{ display: "flex", gap: 6, alignItems: "center" }}><ClassMark c="R" />R recall</dt><dd>DTW {rec("dtw", "R")?.toFixed(2)} · features {rec("features", "R")?.toFixed(2)}</dd>
              <dt style={{ display: "flex", gap: 6, alignItems: "center" }}><ClassMark c="B1" />B1 recall</dt><dd>DTW {rec("dtw", "B1")?.toFixed(2)} · features {rec("features", "B1")?.toFixed(2)}</dd>
            </dl>
          </div>
        </div>
      </div>
    </Section>
  );
}
