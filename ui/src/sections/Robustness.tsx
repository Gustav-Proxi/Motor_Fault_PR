import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import { FRONT_LABEL } from "../lib/data";
import { corrupt, spectrumDb, windowStats, type Mode } from "../lib/dsp";
import { useData, useWidth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { RecMatrix } from "../components/RecMatrix";
import { SpectrumChart } from "../components/Spectrum";
import { AxisX, AxisY, Section, Seg, TableView } from "../components/ui";

const WIN = 1024;
const START = 1024; // second window of the excerpt

function Wave({ clean, dirty, missing }: { clean: Float32Array; dirty: Float32Array; missing?: Uint8Array }) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(700);
  const h = 190;
  const m = useMemo(() => {
    let v = 0;
    for (let i = START; i < START + WIN; i++) v = Math.max(v, Math.abs(clean[i]), Math.abs(dirty[i]));
    return v;
  }, [clean, dirty]);
  const X = (i: number) => (i / (WIN - 1)) * w;
  const Y = (v: number) => h / 2 - (v / m) * (h / 2 - 8);
  const path = (x: Float32Array) => {
    let s = "";
    for (let i = 0; i < WIN; i++) s += (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(x[START + i]).toFixed(1);
    return s;
  };
  const gaps: [number, number][] = [];
  if (missing) {
    let s = -1;
    for (let i = 0; i <= WIN; i++) {
      const g = i < WIN && missing[START + i];
      if (g && s < 0) s = i;
      if (!g && s >= 0) { gaps.push([s, i]); s = -1; }
    }
  }
  return (
    <div ref={ref}>
      <svg width={w} height={h} className="chart" role="img" aria-label="Clean and corrupted waveform of one window">
        <line x1={0} x2={w} y1={h / 2} y2={h / 2} stroke={t.grid} />
        {gaps.map(([a, b]) => <rect key={a} x={X(a)} width={Math.max(1, X(b) - X(a))} y={0} height={h} fill={t.series[1]} opacity={0.12} />)}
        <path d={path(clean)} fill="none" stroke={t.series[0]} strokeWidth={1.2} opacity={0.85} />
        <path d={path(dirty)} fill="none" stroke={t.series[1]} strokeWidth={1.2} opacity={0.9} />
      </svg>
    </div>
  );
}

function SweepCurves({ rows, mode }: { rows: Record<string, any>[]; mode: Mode }) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(360);
  const h = 220;
  const fronts = ["raw", "pca5", "lda5", "kpca5"].filter((f) => rows.some((r) => r.front === f));
  const levels = [...new Set(rows.filter((r) => r.mode === mode).map((r) => r.level as number))].sort((a, b) => a - b);
  const x = scaleLinear().domain([0, Math.max(0.5, ...levels)]).range([40, w - 60]);
  const y = scaleLinear().domain([0, 1]).range([h - 30, 12]);
  const series = fronts.map((f) => levels.map((l) => Math.max(...rows.filter((r) => r.mode === mode && r.front === f && r.level === l).map((r) => r.macro_f1))));
  const colors = [t.series[0], t.series[1], t.series[2], t.cls.B3];
  return (
    <div ref={ref}>
      <svg width={w} height={h} className="chart" role="img" aria-label={`Best macro-F1 per front-end under ${mode}`}>
        <AxisY scale={y} x={40} x2={w - 60} ticks={4} label="macro-F1" />
        <line x1={40} x2={w - 60} y1={y(1 / 6)} y2={y(1 / 6)} stroke={t.muted} />
        {series.map((s, i) => (
          <g key={fronts[i]}>
            <path d={s.map((v, k) => `${k ? "L" : "M"}${x(levels[k])} ${y(v)}`).join("")} fill="none" stroke={colors[i]} strokeWidth={2} />
            {s.map((v, k) => <circle key={k} cx={x(levels[k])} cy={y(v)} r={4} fill={colors[i]} stroke={t.surface} strokeWidth={2} />)}
            <text x={x(levels[levels.length - 1]) + 8} y={y(s[s.length - 1]) + 4} className="lbl">{FRONT_LABEL[fronts[i]]}</text>
          </g>
        ))}
        <AxisX scale={x} y={h - 30} ticks={5} label={`${mode} level`} />
      </svg>
    </div>
  );
}

export function Robustness() {
  const d = useData();
  const { t } = useTheme();
  const [id, setId] = useState("V_1_30Hz");
  const [mode, setMode] = useState<Mode>("noise");
  const [level, setLevel] = useState(0.3);
  const [seed, setSeed] = useState(3);
  const [which, setWhich] = useState<"both" | "test">("both");
  const chans = d.signal.get(id)!;
  const out = useMemo(() => corrupt(chans, mode, level, seed), [chans, mode, level, seed]);
  const s0 = windowStats(chans, START), s1 = windowStats(out.chans, START);
  const spec0 = useMemo(() => spectrumDb(chans, START, WIN), [chans]);
  const spec1 = useMemo(() => spectrumDb(out.chans, START, WIN), [out]);
  const sweep = d.corruption[which];
  const readout = mode === "noise" ? `SNR ${(10 * Math.log10(1 / Math.max(level, 1e-9))).toFixed(1)} dB`
    : mode === "dropout" ? `${(level * 100).toFixed(0)}% of samples dropped, then linearly interpolated`
    : `axis gains drift ${out.gain ? out.gain.map(([a, b]) => `${a.toFixed(2)}→${b.toFixed(2)}`).join(", ") : "1.00"}`;
  const feats: [string, keyof typeof s0][] = [["RMS (g)", "rms"], ["Kurtosis", "kurtosis"], ["Crest factor", "crest"], ["Impulse factor", "impulse"], ["Spectral centroid (Hz)", "centroid"]];

  return (
    <Section id="robustness" eyebrow="Robustness lab" tags={["1a", "1b", "1c"]}
      title="Break the sensor and see what survives"
      lede={<>The robustness sweeps corrupt each <b>whole raw recording</b> before features are computed: white noise relative to
        the vibration, BLE-style packet dropout with linear interpolation, and per-axis gain drift. Here the same three corruptions run
        live, in your browser, on one real window.</>}>
      <div className="split-r">
        <div className="card">
          <h3>Recording</h3>
          <p className="sub">Pick any of the 36.</p>
          <RecMatrix value={id} onChange={setId} compact />
          <div className="controls" style={{ marginTop: 16 }}>
            <Seg value={mode} onChange={setMode} options={[{ value: "noise", label: "Noise" }, { value: "dropout", label: "Dropout" }, { value: "gain", label: "Gain drift" }]} />
          </div>
          <div className="controls">
            <span className="ctl-label">Level</span>
            <input type="range" min={0} max={0.5} step={0.05} value={level} onChange={(e) => setLevel(Number(e.target.value))} aria-label="Corruption level" />
            <span className="mono num">{level.toFixed(2)}</span>
            <button className="btn ghost" onClick={() => setSeed((s) => s + 1)}>New draw</button>
          </div>
          <p className="sub" style={{ color: "var(--ink2)" }}>{readout}</p>
          <table className="data">
            <thead><tr><th>Window feature (X axis)</th><th>clean</th><th>corrupted</th><th>change</th></tr></thead>
            <tbody>
              {feats.map(([lab, k]) => (
                <tr key={k}><td>{lab}</td><td>{s0[k].toPrecision(3)}</td><td>{s1[k].toPrecision(3)}</td>
                  <td>{s0[k] ? `${((s1[k] / s0[k] - 1) * 100).toFixed(0)}%` : "–"}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="stack">
          <div className="card">
            <h3>Window 2 of the excerpt, gUserX</h3>
            <Wave clean={chans[0]} dirty={out.chans[0]} missing={out.missing} />
            <div className="legend" style={{ marginTop: 6 }}>
              <span className="item"><i style={{ width: 16, height: 2, background: t.series[0], display: "inline-block" }} />clean</span>
              <span className="item"><i style={{ width: 16, height: 2, background: t.series[1], display: "inline-block" }} />corrupted</span>
              {mode === "dropout" && <span className="item"><i style={{ width: 12, height: 10, background: t.series[1], opacity: 0.25, display: "inline-block" }} />dropped and interpolated</span>}
            </div>
            <div style={{ marginTop: 12 }}>
              <SpectrumChart range={[0.3, 50]} height={220} ariaLabel="Spectrum of the clean and the corrupted window"
                series={[{ db: spec0, df: 100 / WIN, color: t.series[0], label: "clean" }, { db: spec1, df: 100 / WIN, color: t.series[1], label: "corrupted" }]} />
            </div>
          </div>
          <div className="card">
            <div className="controls" style={{ justifyContent: "space-between" }}>
              <h3>Sweep results</h3>
              <Seg value={which} onChange={setWhich} options={[{ value: "both", label: "Train + test corrupted" }, { value: "test", label: "Clean train, corrupted test" }]} />
            </div>
            {sweep ? (
              <>
                <p className="sub">Best classifier per front-end, grouped CV. Horizontal line: chance.</p>
                <div className="grid3">
                  {(["noise", "dropout", "gain"] as Mode[]).map((m) => <div key={m}><span className="ctl-label">{m}</span><SweepCurves rows={sweep} mode={m} /></div>)}
                </div>
                <TableView cols={["mode", "level", "front-end", "classifier", "macro-F1"]} rows={sweep.map((r) => [r.mode, r.level, r.front, r.clf, r.macro_f1])} />
              </>
            ) : (
              <div className="callout">
                <b>The sweep results aren't in this bundle.</b> They come from <span className="mono">scripts/run_corruption.py --corrupt {which}</span>{" "}
                (tasks 1a–1c, multi-seed). Run it, then <span className="mono">scripts/export_ui_data.py</span>, and the curves appear here.
                <div style={{ marginTop: 8, color: "var(--ink2)" }}>
                  Preliminary single-seed findings (ROADMAP, 2026-10-10):{" "}
                  {which === "both"
                    ? "with train and test both corrupted, noise and dropout barely matter even at 50%. Gain drift hurts, and LDA most of all."
                    : "with clean training and corrupted test data, 10% noise or dropout already drops LDA to about 0.2, and at 50% it falls below chance. Gain drift barely matters, and raw features degrade least."}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}
