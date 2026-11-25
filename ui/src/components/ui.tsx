import { animate, motion, useInView as useMotionInView } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { ScaleLinear, ScaleLogarithmic, ScaleBand } from "d3-scale";
import { CLASSES, type Cls } from "../lib/data";
import { CLASS_SHAPE, useTheme } from "../lib/theme";
import { useReducedMotion } from "../lib/hooks";

export function Section(props: { id: string; tags: string[]; eyebrow: string; title: ReactNode; lede?: ReactNode; children: ReactNode }) {
  return (
    <section id={props.id} className="section">
      <div className="wrap">
        <motion.div
          initial={{ opacity: 0, y: 26 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.7, ease: [0.2, 0.7, 0.2, 1] }}
        >
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <span className="eyebrow"><span className="dot" />{props.eyebrow}</span>
            {props.tags.map((t) => <span key={t} className="tag">{t}</span>)}
          </div>
          <h2>{props.title}</h2>
          {props.lede && <p className="lede">{props.lede}</p>}
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.8, delay: 0.1, ease: [0.2, 0.7, 0.2, 1] }}
        >
          {props.children}
        </motion.div>
      </div>
    </section>
  );
}

export function Seg<T extends string | number>(props: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
      {props.label && <span className="ctl-label">{props.label}</span>}
      <div className="seg" role="group" aria-label={props.label}>
        {props.options.map((o) => (
          <button key={String(o.value)} aria-pressed={o.value === props.value} onClick={() => props.onChange(o.value)}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** SVG marker in the class's color and shape. */
export function Shape({ shape, color, size = 12, ring }: { shape: number; color: string; size?: number; ring?: string }) {
  const s = size;
  const h = s / 2;
  const stroke = ring ? { stroke: ring, strokeWidth: 2, paintOrder: "stroke" as const } : {};
  let el: ReactNode;
  switch (shape) {
    case 1: el = <rect x={s * 0.14} y={s * 0.14} width={s * 0.72} height={s * 0.72} rx={1.5} fill={color} {...stroke} />; break;
    case 2: el = <path d={`M${h} ${s * 0.08} L${s * 0.94} ${s * 0.88} L${s * 0.06} ${s * 0.88} Z`} fill={color} {...stroke} />; break;
    case 3: el = <path d={`M${h} ${s * 0.04} L${s * 0.96} ${h} L${h} ${s * 0.96} L${s * 0.04} ${h} Z`} fill={color} {...stroke} />; break;
    case 4: el = <path d={`M${s * 0.36} ${s * 0.06}h${s * 0.28}v${s * 0.3}h${s * 0.3}v${s * 0.28}h-${s * 0.3}v${s * 0.3}h-${s * 0.28}v-${s * 0.3}h-${s * 0.3}v-${s * 0.28}h${s * 0.3}z`} fill={color} {...stroke} />; break;
    case 5: el = <circle cx={h} cy={h} r={s * 0.34} fill="none" stroke={color} strokeWidth={s * 0.2} />; break;
    default: el = <circle cx={h} cy={h} r={s * 0.42} fill={color} {...stroke} />;
  }
  return <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} style={{ flex: "none", overflow: "visible" }} aria-hidden>{el}</svg>;
}

export function ClassMark({ c, size = 12 }: { c: Cls; size?: number }) {
  const { t } = useTheme();
  return <Shape shape={CLASS_SHAPE[c]} color={t.cls[c]} size={size} />;
}

export function ClassLegend(props: { visible?: Record<Cls, boolean>; onToggle?: (c: Cls) => void; names?: Record<Cls, string>; only?: Cls[] }) {
  const list = props.only ?? [...CLASSES];
  return (
    <div className="legend" role="list">
      {list.map((c) => {
        const body = (<><ClassMark c={c} /> <b style={{ color: "var(--ink)", fontWeight: 650 }}>{c}</b>{props.names && <span>{props.names[c]}</span>}</>);
        return props.onToggle ? (
          <button key={c} role="listitem" aria-pressed={props.visible ? props.visible[c] : true} onClick={() => props.onToggle!(c)} title={`Show or hide ${c}`}>{body}</button>
        ) : (
          <span key={c} role="listitem" className="item">{body}</span>
        );
      })}
    </div>
  );
}

export function Counter({ value, digits = 2, duration = 1.1 }: { value: number; digits?: number; duration?: number }) {
  // Renders the final value by default; counts up from 0 the first time it scrolls into view.
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useMotionInView(ref, { once: true });
  const reduced = useReducedMotion();
  const shown = useRef<number | null>(null);
  const fmtN = (v: number) => v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const initial = useRef(fmtN(value)).current;
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!inView || reduced) {
      el.textContent = fmtN(value);
      if (reduced) shown.current = value;
      return;
    }
    const ctl = animate(shown.current ?? 0, value, {
      duration,
      ease: [0.2, 0.7, 0.2, 1],
      onUpdate: (v) => { el.textContent = fmtN(v); shown.current = v; },
    });
    return () => ctl.stop();
  }, [value, inView, reduced, digits, duration]); // eslint-disable-line react-hooks/exhaustive-deps
  return <span ref={ref}>{initial}</span>;
}

export function Stat({ label, value, note, digits }: { label: string; value: number | string; note?: string; digits?: number }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value">{typeof value === "number" ? <Counter value={value} digits={digits ?? 0} /> : value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export function TableView({ cols, rows, label = "Show the numbers" }: { cols: string[]; rows: (string | number)[][]; label?: string }) {
  return (
    <details className="tv">
      <summary>{label}</summary>
      <div className="tv-scroll">
        <table className="data">
          <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>{r.map((v, j) => <td key={j}>{typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/** Tooltip positioned inside a relative container; flips to stay inside it. */
export function Tip({ x, y, w, children }: { x: number; y: number; w: number; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [tw, setTw] = useState(180);
  useEffect(() => { if (ref.current) setTw(ref.current.offsetWidth); });
  const left = x + 14 + tw > w ? x - 14 - tw : x + 14;
  return <div ref={ref} className="tip" style={{ left: Math.max(0, left), top: Math.max(0, y - 10) }}>{children}</div>;
}

// ---------------------------------------------------------------- SVG axes

type Lin = ScaleLinear<number, number> | ScaleLogarithmic<number, number>;

export function AxisX({ scale, y, ticks = 6, format, label, x0, x1 }: { scale: Lin; y: number; ticks?: number; format?: (v: number) => string; label?: string; x0?: number; x1?: number }) {
  const tv = scale.ticks(ticks);
  const [a, b] = scale.range();
  return (
    <g>
      <line x1={x0 ?? a} x2={x1 ?? b} y1={y} y2={y} stroke="var(--axis)" />
      {tv.map((v) => (
        <text key={v} x={scale(v)} y={y + 15} textAnchor="middle">{format ? format(v) : String(v)}</text>
      ))}
      {label && <text x={b} y={y + 29} textAnchor="end">{label}</text>}
    </g>
  );
}
export function AxisY({ scale, x, x2, ticks = 5, format, label }: { scale: Lin; x: number; x2: number; ticks?: number; format?: (v: number) => string; label?: string }) {
  const tv = scale.ticks(ticks);
  const [b] = scale.range();
  const top = Math.min(...scale.range());
  return (
    <g>
      {tv.map((v) => (
        <g key={v}>
          <line x1={x} x2={x2} y1={scale(v)} y2={scale(v)} stroke="var(--grid)" />
          <text x={x - 6} y={scale(v) + 4} textAnchor="end">{format ? format(v) : String(v)}</text>
        </g>
      ))}
      {label && <text x={x} y={top - 10} textAnchor="start">{label}</text>}
      <line x1={x} x2={x} y1={top} y2={b} stroke="transparent" />
    </g>
  );
}
export function BandAxisY({ scale, x, format }: { scale: ScaleBand<string>; x: number; format?: (v: string) => string }) {
  return (
    <g>
      {scale.domain().map((d) => (
        <text key={d} className="lbl" x={x} y={(scale(d) ?? 0) + scale.bandwidth() / 2 + 4} textAnchor="end">{format ? format(d) : d}</text>
      ))}
    </g>
  );
}

/** Horizontal bar with a 4px rounded data end and a square baseline end. */
export function HBar({ x0, x1, y, h, fill }: { x0: number; x1: number; y: number; h: number; fill: string }) {
  const w = Math.max(0, x1 - x0);
  const r = Math.min(4, w, h / 2);
  const d = `M${x0} ${y} H${x0 + w - r} Q${x0 + w} ${y} ${x0 + w} ${y + r} V${y + h - r} Q${x0 + w} ${y + h} ${x0 + w - r} ${y + h} H${x0} Z`;
  return <path d={d} fill={fill} />;
}
/** Vertical column, rounded top. */
export function VBar({ x, w, y0, y1, fill }: { x: number; w: number; y0: number; y1: number; fill: string }) {
  const h = Math.max(0, y0 - y1);
  const r = Math.min(4, h, w / 2);
  const d = `M${x} ${y0} V${y1 + r} Q${x} ${y1} ${x + r} ${y1} H${x + w - r} Q${x + w} ${y1} ${x + w} ${y1 + r} V${y0} Z`;
  return <path d={d} fill={fill} />;
}
