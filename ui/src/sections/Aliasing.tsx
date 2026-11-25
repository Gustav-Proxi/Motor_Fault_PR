import { animate } from "framer-motion";
import { scaleLinear } from "d3-scale";
import { useEffect, useMemo, useRef, useState } from "react";
import { recId } from "../lib/data";
import { alias } from "../lib/dsp";
import { useData, useWidth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { SpectrumChart } from "../components/Spectrum";
import { AxisX, AxisY, Section, Seg, Tip } from "../components/ui";

type Family = "shaft" | "line" | "bearing" | "sideband";
interface Pt { name: string; fam: Family; f: number }

function FoldChart({ pts, p, families }: { pts: Pt[]; p: number; families: Record<Family, boolean> }) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(700);
  const h = 340;
  const m = { l: 46, r: 14, t: 20, b: 40 };
  const x = scaleLinear().domain([0, 200]).range([m.l, w - m.r]);
  const y = scaleLinear().domain([0, 60]).range([h - m.b, m.t]);
  const yv = (f: number) => (1 - p) * f + p * alias(f);
  const [hover, setHover] = useState<Pt | null>(null);
  const curve = useMemo(() => {
    let d = "";
    for (let f = 0; f <= 200; f += 0.5) {
      const v = Math.min(62, yv(f));
      d += (d ? "L" : "M") + x(f).toFixed(1) + " " + y(v).toFixed(1);
    }
    return d;
  }, [p, w]); // eslint-disable-line react-hooks/exhaustive-deps
  const famColor: Record<Family, string> = { shaft: t.series[0], line: t.series[1], bearing: t.series[2], sideband: t.ink };
  const shown = pts.filter((q) => families[q.fam]);
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={w} height={h} className="chart" role="img" aria-label="True frequency against the frequency a 100 Hz sensor reports">
        <defs>
          <clipPath id="foldclip"><rect x={m.l} y={m.t - 4} width={w - m.l - m.r} height={h - m.t - m.b + 4} /></clipPath>
        </defs>
        <AxisY scale={y} x={m.l} x2={w - m.r} ticks={6} label="apparent Hz" />
        <rect x={m.l} width={w - m.l - m.r} y={y(60)} height={y(50) - y(60)} fill={t.accent} opacity={0.08} />
        <line x1={m.l} x2={w - m.r} y1={y(50)} y2={y(50)} stroke={t.accent} strokeWidth={1.5} />
        <text x={w - m.r} y={y(50) - 6} textAnchor="end" className="lbl-strong">Nyquist 50 Hz</text>
        {[50, 100, 150].map((f) => <line key={f} x1={x(f)} x2={x(f)} y1={m.t} y2={h - m.b} stroke={t.grid} />)}
        <g clipPath="url(#foldclip)">
          <path d={curve} fill="none" stroke={t.accent} strokeWidth={2.5} strokeLinejoin="round" />
          {shown.map((q) => {
            const yy = Math.min(62, yv(q.f));
            const c = famColor[q.fam];
            return (
              <g key={q.name} onPointerEnter={() => setHover(q)} onPointerLeave={() => setHover(null)} style={{ cursor: "default" }}>
                <line x1={x(q.f)} x2={x(q.f)} y1={y(yy)} y2={h - m.b} stroke={c} strokeWidth={1} opacity={0.35} />
                {p > 0.5 && q.f > 50 && <line x1={m.l} x2={x(q.f)} y1={y(yy)} y2={y(yy)} stroke={c} strokeWidth={1} opacity={0.25 * p} />}
                <circle cx={x(q.f)} cy={y(yy)} r={hover === q ? 7 : 5} fill={c} stroke={t.surface} strokeWidth={2} />
                <circle cx={x(q.f)} cy={y(yy)} r={12} fill="transparent" />
              </g>
            );
          })}
        </g>
        <AxisX scale={x} y={h - m.b} ticks={10} label="true frequency (Hz)" />
      </svg>
      {hover && (
        <Tip x={x(hover.f)} y={y(Math.min(62, yv(hover.f)))} w={w}>
          <b>{hover.name}</b>
          <div className="num"><span className="k">true</span> {hover.f.toFixed(2)} Hz</div>
          <div className="num"><span className="k">phone reports</span> {alias(hover.f).toFixed(2)} Hz</div>
        </Tip>
      )}
    </div>
  );
}

export function Aliasing() {
  const d = useData();
  const { t } = useTheme();
  const [cond, setCond] = useState({ load: 1, supply: 50 });
  const [folded, setFolded] = useState(true);
  const [p, setP] = useState(1);
  const [fam, setFam] = useState<Record<Family, boolean>>({ shaft: true, line: true, bearing: false, sideband: true });
  const pRef = useRef(1);
  useEffect(() => {
    const ctl = animate(pRef.current, folded ? 1 : 0, { duration: 1.4, ease: [0.65, 0, 0.35, 1], onUpdate: (v) => { pRef.current = v; setP(v); } });
    return () => ctl.stop();
  }, [folded]);

  const rec = d.recById.get(recId("R", cond.load, cond.supply))!;
  const { f1, f_pole: fp, supply } = rec;
  const pts: Pt[] = [
    ...Array.from({ length: 8 }, (_, k) => ({ name: `${k + 1}× f1`, fam: "shaft" as Family, f: (k + 1) * f1 })),
    ...[1, 2, 3, 4].map((m) => ({ name: `${m}× line`, fam: "line" as Family, f: m * supply })),
    { name: "BPFO", fam: "bearing", f: d.meta.bpfo_order * f1 },
    { name: "BPFI", fam: "bearing", f: d.meta.bpfi_order * f1 },
    { name: "f1 − f_p", fam: "sideband", f: f1 - fp },
    { name: "f1 + f_p", fam: "sideband", f: f1 + fp },
  ];
  // nearest aliased shaft harmonic to each sideband
  const near = (f: number) => {
    let best = { k: 0, d: Infinity };
    for (let k = 2; k <= 8; k++) {
      if (k * f1 <= 50) continue;
      const dd = Math.abs(alias(k * f1) - f);
      if (dd < best.d) best = { k, d: dd };
    }
    return best;
  };
  const up = near(f1 + fp), lo = near(f1 - fp);
  const famLabel: Record<Family, string> = { shaft: "Shaft k × f1", line: "Line m × supply", bearing: "BPFO / BPFI", sideband: "Rotor-bar sidebands" };
  const famColor: Record<Family, string> = { shaft: t.series[0], line: t.series[1], bearing: t.series[2], sideband: t.ink };

  const markers = [
    { f: f1, label: "f1", strong: true },
    { f: f1 - fp, label: "f1−fp", strong: true },
    { f: f1 + fp, label: "f1+fp", strong: true },
    ...[2, 3, 4, 5, 6].filter((k) => k * f1 > 50).map((k) => ({ f: alias(k * f1), label: `${k}×′` })),
  ];

  return (
    <Section id="aliasing" eyebrow="Aliasing lab" tags={["2b", "2c"]}
      title="Above 50 Hz, the spectrum folds like paper"
      lede={<>A 100 Hz sensor cannot represent a tone above 50 Hz. Instead it reports the folded frequency |f − 100·round(f/100)|.
        Toggle the fold and watch harmonics, line frequencies and bearing tones collapse into the 0–50 Hz band, sometimes right on top
        of each other.</>}>
      <div className="split">
        <div className="card">
          <div className="controls">
            <Seg value={folded ? 1 : 0} onChange={(v) => setFolded(v === 1)} options={[{ value: 0, label: "Unfolded (ideal sensor)" }, { value: 1, label: "Folded (100 Hz phone)" }]} />
            <Seg value={cond.supply} onChange={(s) => setCond({ ...cond, supply: s })} options={[30, 40, 50].map((s) => ({ value: s, label: `${s} Hz` }))} />
            <Seg value={cond.load} onChange={(l) => setCond({ ...cond, load: l })} options={[{ value: 0, label: "Unloaded" }, { value: 1, label: "Loaded" }]} />
          </div>
          <FoldChart pts={pts} p={p} families={fam} />
          <div className="legend" style={{ marginTop: 8 }}>
            {(Object.keys(famLabel) as Family[]).map((k) => (
              <button key={k} aria-pressed={fam[k]} onClick={() => setFam({ ...fam, [k]: !fam[k] })}>
                <svg width="12" height="12" aria-hidden><circle cx="6" cy="6" r="5" fill={famColor[k]} /></svg>{famLabel[k]}
              </button>
            ))}
          </div>
          <p className="sub" style={{ marginTop: 8 }}>Frequencies use the measured running speed of the R recording at this condition (f1 = {f1.toFixed(2)} Hz, f_p = {fp.toFixed(2)} Hz).</p>
        </div>
        <div className="stack">
          <div className="card">
            <h3>The 50 Hz coincidence</h3>
            <p className="sub">A 4-pole motor runs at f1 = (supply/2)(1 − s), and broken bars modulate it at f_p = 2·s·supply.</p>
            <div className="callout" style={{ fontFamily: "var(--mono)", fontSize: 13.5, lineHeight: 1.7 }}>
              f1 + f_p = 2·supply − 3·f1<br />
              at 50 Hz: <b>100 − 3·f1 = alias(3·f1)</b><br />
              and f1 − f_p = <b>alias(5·f1)</b>, for any slip
            </div>
            <dl className="kv" style={{ marginTop: 14 }}>
              <dt>f1 + f_p</dt><dd>{(f1 + fp).toFixed(3)} Hz · nearest folded harmonic {up.k}× at Δ {up.d.toFixed(3)} Hz</dd>
              <dt>f1 − f_p</dt><dd>{(f1 - fp).toFixed(3)} Hz · nearest folded harmonic {lo.k}× at Δ {lo.d.toFixed(3)} Hz</dd>
            </dl>
            <p className="sub" style={{ marginTop: 12, color: "var(--ink2)" }}>
              {supply === 50
                ? "Exact overlap. At 50 Hz supply a 100 Hz sensor cannot tell rotor-bar sidebands from running-speed harmonics."
                : "Not exact at this supply frequency. The match only becomes close when the slip is tiny."}
            </p>
          </div>
          <div className="card">
            <h3>R at {supply} Hz, {cond.load ? "loaded" : "unloaded"}: around f1</h3>
            <SpectrumChart
              series={[{ db: d.spectrum.get(rec.id)!, df: d.specDf, color: t.cls.R, label: "R", wash: true }]}
              range={[Math.max(0.3, f1 - 12), Math.min(50, f1 + 12)]}
              markers={markers}
              height={220}
              ariaLabel="Spectrum of the R recording around the running speed"
            />
          </div>
        </div>
      </div>
    </Section>
  );
}
