import { useMemo, useRef, useState } from "react";
import { prettyId, recId, type Cls } from "../lib/data";
import { alias, candidateSources, explainFrequency, spectrumDb } from "../lib/dsp";
import { setupCanvas, useData, useFrame, useInView, useReducedMotion, useWidth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { RecMatrix } from "../components/RecMatrix";
import { SpectrumChart, type SpecMarker } from "../components/Spectrum";
import { ClassMark, Section, Seg } from "../components/ui";

const WIN = 1024;

/** "alias:4x f1 / 2x line" -> "4× f1′" (′ = folded down from above 50 Hz, as on the live spectrum) */
function peakLabel(src: string | null) {
  if (!src) return "unexplained";
  const first = src.split(" / ")[0].replace(/(\d)x /g, "$1× ");
  return first.startsWith("alias:") ? `${first.slice(6)}′` : first;
}

/** Live oscilloscope (3 lanes) + rolling 1024-sample spectrum, drawn on canvas each frame. */
function Scope({ id, speed, playing }: { id: string; speed: number; playing: boolean }) {
  const d = useData();
  const { t } = useTheme();
  const reduced = useReducedMotion();
  const rec = d.recById.get(id)!;
  const chans = d.signal.get(id)!;
  const [wrapRef, w] = useWidth<HTMLDivElement>(800);
  const [vref, inView] = useInView<HTMLDivElement>();
  const scope = useRef<HTMLCanvasElement>(null);
  const spec = useRef<HTMLCanvasElement>(null);
  const st = useRef({ pos: WIN, lastSpec: -1e9, db: null as Float32Array | null, smooth: null as Float32Array | null });
  const amp = useMemo(() => {
    let m = 0;
    for (const c of chans) for (const v of c) m = Math.max(m, Math.abs(v));
    return m;
  }, [chans]);
  const narrow = w < 700;
  const scopeW = narrow ? w : Math.floor(w * 0.58);
  const specW = narrow ? w : w - scopeW - 18;
  const H = 250;

  useFrame((dt) => {
    const s = st.current;
    const n = chans[0].length;
    if (playing && !reduced) s.pos += dt * 100 * speed;
    if (s.pos >= n) s.pos = WIN;
    const end = Math.floor(s.pos);
    const start = end - WIN;
    const cc = t.cls[rec.label];

    // ---------- scope
    const sc = scope.current;
    if (sc) {
      const g = setupCanvas(sc, scopeW, H);
      g.clearRect(0, 0, scopeW, H);
      const lane = (H - 24) / 3;
      g.font = "600 11px ui-monospace, Menlo, monospace";
      for (let c = 0; c < 3; c++) {
        const y0 = 6 + lane * c + lane / 2;
        g.strokeStyle = t.grid; g.lineWidth = 1;
        g.beginPath(); g.moveTo(26, y0); g.lineTo(scopeW, y0); g.stroke();
        g.fillStyle = t.muted; g.fillText(["X", "Y", "Z"][c], 6, y0 + 4);
        g.strokeStyle = cc; g.lineWidth = 1.4; g.lineJoin = "round";
        g.beginPath();
        const x = chans[c];
        for (let i = 0; i < WIN; i++) {
          const X = 26 + (i / (WIN - 1)) * (scopeW - 26);
          const Y = y0 - (x[start + i] / amp) * (lane * 0.48);
          i ? g.lineTo(X, Y) : g.moveTo(X, Y);
        }
        g.stroke();
      }
      // fading head glow
      const grd = g.createLinearGradient(scopeW - 60, 0, scopeW, 0);
      grd.addColorStop(0, "rgba(0,0,0,0)"); grd.addColorStop(1, t.accent + "33");
      g.fillStyle = grd; g.fillRect(scopeW - 60, 0, 60, H - 18);
      g.fillStyle = t.muted; g.font = "11px ui-monospace, Menlo, monospace";
      g.fillText(`10.24 s window · ±${(amp * 1000).toFixed(0)} mg full scale · t = ${((d.signalStart + end) / 100).toFixed(1)} s`, 26, H - 4);
    }

    // ---------- rolling spectrum (recomputed every 0.15 s of signal)
    if (end - s.lastSpec >= 15 || !s.db) {
      s.db = spectrumDb(chans, start, WIN);
      s.lastSpec = end;
    }
    if (!s.smooth || s.smooth.length !== s.db.length) s.smooth = Float32Array.from(s.db);
    for (let k = 0; k < s.db.length; k++) s.smooth[k] += (s.db[k] - s.smooth[k]) * Math.min(1, dt * 8);
    const sp = spec.current;
    if (sp) {
      const g = setupCanvas(sp, specW, H);
      g.clearRect(0, 0, specW, H);
      const L = 34, B = H - 22, T = 18;
      let hi = -Infinity;
      for (let k = 2; k < s.smooth.length; k++) hi = Math.max(hi, s.smooth[k]);
      hi = Math.ceil((hi + 3) / 10) * 10;
      const lo = hi - 60;
      const X = (f: number) => L + (f / 50) * (specW - L - 6);
      const Y = (v: number) => T + ((hi - Math.max(lo, v)) / (hi - lo)) * (B - T);
      g.font = "11px system-ui, sans-serif"; g.fillStyle = t.muted; g.textAlign = "right";
      for (let v = lo; v <= hi; v += 20) { g.strokeStyle = t.grid; g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(specW, Y(v)); g.stroke(); g.fillText(`${v}`, L - 5, Y(v) + 4); }
      g.textAlign = "center";
      for (let f = 0; f <= 50; f += 10) g.fillText(`${f}`, X(f), B + 15);
      // shaft-harmonic positions, folded
      for (let k = 1; k <= 6; k++) {
        const fa = alias(k * rec.f1);
        g.strokeStyle = k * rec.f1 > 50 ? t.accent : t.ink2; g.globalAlpha = 0.5; g.setLineDash([]);
        g.beginPath(); g.moveTo(X(fa), T); g.lineTo(X(fa), B); g.stroke(); g.globalAlpha = 1;
        g.fillStyle = k * rec.f1 > 50 ? t.accent : t.ink2; g.fillText(k * rec.f1 > 50 ? `${k}×′` : `${k}×`, X(fa), T - 4);
      }
      g.beginPath();
      for (let k = 1; k < s.smooth.length; k++) { const xx = X(k * 100 / WIN), yy = Y(s.smooth[k]); k > 1 ? g.lineTo(xx, yy) : g.moveTo(xx, yy); }
      g.strokeStyle = cc; g.lineWidth = 1.6; g.stroke();
      g.lineTo(X(50), B); g.lineTo(X(100 / WIN), B); g.closePath();
      g.fillStyle = cc + "1a"; g.fill();
      g.fillStyle = t.muted; g.textAlign = "left"; g.fillText("dB", 4, T - 4);
    }
  }, inView);

  return (
    <div ref={vref}>
      <div ref={wrapRef} style={{ display: "flex", gap: 18, flexDirection: narrow ? "column" : "row" }}>
        <canvas ref={scope} role="img" aria-label={`Live three-axis vibration of ${id}`} />
        <canvas ref={spec} role="img" aria-label={`Live spectrum of the current window of ${id}`} />
      </div>
      <p className="sub" style={{ marginTop: 8 }}>
        The spectrum is the Hann periodogram of the scope's current window, summed over the three axes. Ticks mark shaft harmonics k × f1. Those marked
        <span style={{ color: "var(--accent)" }}> k×′</span> are folded down from above 50 Hz.
      </p>
    </div>
  );
}

export function Explorer() {
  const d = useData();
  const { t } = useTheme();
  const [id, setId] = useState("R_1_50Hz");
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [zoom, setZoom] = useState<"full" | "side">("full");
  const rec = d.recById.get(id)!;
  const ref = recId("H", rec.load, rec.supply);
  const sources = useMemo(() => candidateSources(rec.f1, rec.supply, d.meta.bpfo_order, d.meta.bpfi_order), [rec, d]);
  const peaks = useMemo(() => d.peaks.filter((p) => p.rec_id === id).sort((a, b) => b.prominence_db - a.prominence_db).slice(0, 5), [d, id]);
  const markers: SpecMarker[] = zoom === "side"
    ? [
        { f: rec.f1, label: "f1", strong: true },
        { f: rec.f1 - rec.f_pole, label: "f1−fp" },
        { f: rec.f1 + rec.f_pole, label: "f1+fp" },
      ]
    : peaks.map((p) => ({ f: p.freq, label: peakLabel(p.source) }));
  const range: [number, number] = zoom === "side" ? [Math.max(0.2, rec.f1 - 12), rec.f1 + 12] : [0.3, 50];
  const series = [
    ...(rec.label !== "H" ? [{ db: d.spectrum.get(ref)!, df: d.specDf, color: t.muted, label: `H (same condition)`, width: 1.2 }] : []),
    { db: d.spectrum.get(id)!, df: d.specDf, color: t.cls[rec.label], label: prettyId(id), width: 1.6, wash: true },
  ];

  return (
    <Section id="explorer" eyebrow="Signal explorer" tags={["data", "2b", "2c"]}
      title="Listen to all 36 recordings"
      lede={<>Every tile is one 15-minute recording. Its sparkline is the vibration envelope of a 30-second excerpt. The scope plays
        that excerpt in real time on the gravity-compensated axes, and the spectrum underneath covers the <b>whole</b> recording at
        0.012 Hz resolution, with the strongest peaks labelled by the shaft, line or bearing frequency they match.</>}>
      <div className="split-r">
        <div className="card">
          <h3>Recordings</h3>
          <p className="sub">Rows are classes, columns are operating conditions.</p>
          <RecMatrix value={id} onChange={setId} />
          <dl className="kv" style={{ marginTop: 16 }}>
            <dt>Class</dt><dd style={{ display: "flex", gap: 6, alignItems: "center" }}><ClassMark c={rec.label as Cls} />{d.meta.class_names[rec.label]}</dd>
            <dt>Running speed f1</dt><dd>{rec.f1.toFixed(3)} Hz</dd>
            <dt>Slip</dt><dd>{(rec.slip * 100).toFixed(2)}%</dd>
            <dt>Pole-pass f_p</dt><dd>{rec.f_pole.toFixed(2)} Hz</dd>
            <dt>Windows</dt><dd>{rec.n_windows}</dd>
            <dt>CV fold</dt><dd>held out in grouped fold {rec.fold + 1}</dd>
          </dl>
        </div>
        <div className="stack">
          <div className="card">
            <div className="controls" style={{ justifyContent: "space-between" }}>
              <h3>{prettyId(id)} <span className="muted" style={{ fontWeight: 400 }}>live</span></h3>
              <div style={{ display: "flex", gap: 10 }}>
                <button className="btn ghost" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play"}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
                <Seg value={speed} onChange={setSpeed} options={[{ value: 1, label: "1×" }, { value: 4, label: "4×" }]} />
              </div>
            </div>
            <Scope id={id} speed={speed} playing={playing} />
          </div>
          <div className="card">
            <div className="controls" style={{ justifyContent: "space-between" }}>
              <h3>Whole-recording spectrum</h3>
              <Seg value={zoom} onChange={setZoom} options={[{ value: "full", label: "0–50 Hz" }, { value: "side", label: "Around f1 (sidebands)" }]} />
            </div>
            <SpectrumChart series={series} range={range} markers={markers} height={280}
              bands={zoom === "side" ? [{ f0: rec.f1 - 3, f1: rec.f1 - 0.5, label: "feature band" }, { f0: rec.f1 + 0.5, f1: rec.f1 + 3, label: "feature band" }] : undefined}
              explain={(f) => explainFrequency(f, sources, 0.25)}
              ariaLabel={`Welch spectrum of ${id}`} />
            <div className="legend" style={{ marginTop: 6 }}>
              {series.map((s) => <span key={s.label} className="item"><i style={{ width: 16, height: 2, background: s.color, display: "inline-block" }} />{s.label}</span>)}
            </div>
            {zoom === "side" && (
              <p className="sub" style={{ marginTop: 8, color: "var(--ink2)" }}>
                The sideband feature integrates 0.5–3 Hz either side of f1 (shaded), but the measured pole-pass spacing here is{" "}
                {rec.f_pole.toFixed(2)} Hz. Across all 36 recordings the true sidebands fall inside that band in none.
              </p>
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}
