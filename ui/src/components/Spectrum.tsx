import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import { useWidth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { AxisX, AxisY, Tip } from "./ui";

export interface SpecSeries { db: Float32Array; df: number; color: string; label: string; width?: number; wash?: boolean }
export interface SpecMarker { f: number; label: string; strong?: boolean }

/** Two staggered label rows; strong markers first, a label that would overlap is dropped (row -1). */
function placeLabels(ms: SpecMarker[], x: (f: number) => number) {
  const order = [...ms.keys()].sort((a, b) => Number(!!ms[b].strong) - Number(!!ms[a].strong) || ms[a].f - ms[b].f);
  const spans: [number, number][][] = [[], []];
  const rows = new Array(ms.length).fill(-1);
  for (const i of order) {
    const half = ms[i].label.length * 3.3 + 4;
    const a = x(ms[i].f) - half, b = x(ms[i].f) + half;
    for (const r of [0, 1]) {
      if (spans[r].every(([s, e]) => b < s || a > e)) { spans[r].push([a, b]); rows[i] = r; break; }
    }
  }
  return ms.map((mk, i) => ({ mk, row: rows[i] }));
}

/** Spectrum line chart (dB vs Hz) with hover crosshair. */
export function SpectrumChart(props: {
  series: SpecSeries[];
  range: [number, number];
  height?: number;
  markers?: SpecMarker[];
  bands?: { f0: number; f1: number; label: string }[];
  explain?: (f: number) => string;
  ariaLabel: string;
}) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(700);
  const h = props.height ?? 260;
  const m = { l: 44, r: 12, t: props.markers?.length ? 36 : 18, b: 34 };
  const [hover, setHover] = useState<{ x: number; f: number } | null>(null);
  const [f0, f1] = props.range;

  const { paths, y } = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    for (const s of props.series) {
      const i0 = Math.max(1, Math.floor(f0 / s.df)), i1 = Math.min(s.db.length - 1, Math.ceil(f1 / s.df));
      for (let i = i0; i <= i1; i++) { lo = Math.min(lo, s.db[i]); hi = Math.max(hi, s.db[i]); }
    }
    const yy = scaleLinear().domain([Math.max(lo, hi - 75), hi + 4]).range([h - m.b, m.t]).nice();
    const xx = scaleLinear().domain([f0, f1]).range([m.l, w - m.r]);
    const ps = props.series.map((s) => {
      const i0 = Math.max(1, Math.floor(f0 / s.df)), i1 = Math.min(s.db.length - 1, Math.ceil(f1 / s.df));
      const n = i1 - i0 + 1;
      // max-pool to ~2 points per pixel so peaks survive downsampling
      const per = Math.max(1, Math.floor(n / ((w - m.l - m.r) * 2)));
      let d = "";
      for (let i = i0; i <= i1; i += per) {
        let v = -Infinity;
        for (let k = i; k < Math.min(i + per, i1 + 1); k++) v = Math.max(v, s.db[k]);
        const X = xx(i * s.df), Y = yy(Math.max(v, yy.domain()[0]));
        d += (d ? "L" : "M") + X.toFixed(1) + " " + Y.toFixed(1);
      }
      return d;
    });
    return { paths: ps, y: yy };
  }, [props.series, f0, f1, w, h, m.b, m.l, m.r, m.t]);
  const x = scaleLinear().domain([f0, f1]).range([m.l, w - m.r]);
  const base = h - m.b;

  const hv = hover && props.series.map((s) => ({ s, v: s.db[Math.round(hover.f / s.df)] }));
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={w} height={h} className="chart" role="img" aria-label={props.ariaLabel}
        onPointerMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const px = e.clientX - r.left;
          if (px < m.l || px > w - m.r) return setHover(null);
          setHover({ x: px, f: x.invert(px) });
        }}
        onPointerLeave={() => setHover(null)}>
        <AxisY scale={y} x={m.l} x2={w - m.r} ticks={5} />
        <text x={m.l - 6} y={h - 6} textAnchor="end">dB</text>
        {props.bands?.map((b) => (
          <g key={b.label + b.f0}>
            <rect x={x(b.f0)} width={Math.max(0, x(b.f1) - x(b.f0))} y={m.t} height={base - m.t} fill={t.accent} opacity={0.1} />
            <text x={(x(b.f0) + x(b.f1)) / 2} y={m.t + 12} textAnchor="middle">{b.label}</text>
          </g>
        ))}
        {props.series.map((s, i) => (
          <g key={s.label}>
            {s.wash && <path d={`${paths[i]}L${x(Math.min(f1, (s.db.length - 1) * s.df))} ${base}L${x(f0)} ${base}Z`} fill={s.color} opacity={0.1} />}
            <path d={paths[i]} fill="none" stroke={s.color} strokeWidth={s.width ?? 1.6} strokeLinejoin="round" />
          </g>
        ))}
        {placeLabels(props.markers?.filter((mk) => mk.f >= f0 && mk.f <= f1) ?? [], x).map(({ mk, row }, i) => (
          <g key={mk.label + i}>
            <line x1={x(mk.f)} x2={x(mk.f)} y1={m.t + 4} y2={base} stroke={mk.strong ? t.ink : t.muted} strokeWidth={1} opacity={mk.strong ? 0.8 : 0.5} />
            {row >= 0 && <text x={x(mk.f)} y={m.t - 4 - row * 12} textAnchor="middle" className={mk.strong ? "lbl-strong" : undefined}>{mk.label}</text>}
          </g>
        ))}
        <AxisX scale={x} y={base} ticks={8} label="Hz" />
        {hover && <line x1={hover.x} x2={hover.x} y1={m.t} y2={base} stroke={t.ink2} strokeWidth={1} />}
        {hv && hv.map(({ s, v }) => v !== undefined && (
          <circle key={s.label} cx={hover!.x} cy={y(v)} r={4} fill={s.color} stroke={t.surface} strokeWidth={2} />
        ))}
      </svg>
      {hover && hv && (
        <Tip x={hover.x} y={40} w={w}>
          <b className="num">{hover.f.toFixed(2)} Hz</b>
          {hv.map(({ s, v }) => (
            <div key={s.label} className="num"><span className="k">{s.label}</span> {v?.toFixed(1)} dB</div>
          ))}
          {props.explain && props.explain(hover.f) && <div className="k">≈ {props.explain(hover.f)}</div>}
        </Tip>
      )}
    </div>
  );
}
