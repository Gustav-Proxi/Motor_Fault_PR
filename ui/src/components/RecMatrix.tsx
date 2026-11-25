import { useMemo } from "react";
import { CLASSES, CONDITIONS, recId, type Cls } from "../lib/data";
import { useData } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { ClassMark } from "./ui";

/** 6 classes x 6 conditions picker; each tile carries a sparkline of its vibration envelope. */
export function RecMatrix({ value, onChange, compact }: { value: string; onChange: (id: string) => void; compact?: boolean }) {
  const d = useData();
  const { t } = useTheme();
  const env = useMemo(() => {
    const out = new Map<string, number[]>();
    let gmax = 0;
    for (const [id, ch] of d.signal) {
      const e: number[] = [];
      const step = 96;
      for (let s = 0; s + step <= ch[0].length; s += step) {
        let acc = 0;
        for (const c of ch) for (let i = s; i < s + step; i++) acc += c[i] * c[i];
        e.push(Math.sqrt(acc / (3 * step)));
      }
      gmax = Math.max(gmax, ...e);
      out.set(id, e);
    }
    for (const [id, e] of out) out.set(id, e.map((v) => v / gmax));
    return out;
  }, [d]);

  return (
    <div className="recgrid" role="grid" aria-label="Pick a recording">
      <div />
      {CONDITIONS.map((c) => (
        <div key={`${c.load}${c.supply}`} className="hd">{c.supply} Hz<br />{c.load ? "load" : "no load"}</div>
      ))}
      {CLASSES.map((cls: Cls) => (
        <div key={cls} style={{ display: "contents" }}>
          <div className="rl"><ClassMark c={cls} size={10} />{cls}</div>
          {CONDITIONS.map((c) => {
            const id = recId(cls, c.load, c.supply);
            const e = env.get(id) ?? [];
            const pts = e.map((v, i) => `${(i / (e.length - 1)) * 100},${30 - v * 26}`).join(" ");
            return (
              <button key={id} className="rec-tile" aria-pressed={value === id} onClick={() => onChange(id)}
                title={`${id}`} style={{ height: compact ? 28 : 34 }} aria-label={`Recording ${id}`}>
                <svg viewBox="0 0 100 32" preserveAspectRatio="none">
                  <polygon points={`0,32 ${pts} 100,32`} fill={t.cls[cls]} opacity={0.22} />
                  <polyline points={pts} fill="none" stroke={t.cls[cls]} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
                </svg>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}
