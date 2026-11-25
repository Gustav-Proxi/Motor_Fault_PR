import { useEffect, useMemo, useRef, useState } from "react";
import { CLASSES, type Cls } from "../lib/data";
import { setupCanvas, useData, useFrame, useReducedMotion, useWidth } from "../lib/hooks";
import { CLASS_SHAPE, useTheme } from "../lib/theme";

export type View = "all" | "test" | "testonly";

/** Draw one class marker centred at (x, y) with half-size r (same shapes as the 3-D view). */
function mark(g: CanvasRenderingContext2D, shape: number, x: number, y: number, r: number) {
  g.beginPath();
  switch (shape) {
    case 1: g.rect(x - r * 0.8, y - r * 0.8, r * 1.6, r * 1.6); break;
    case 2: g.moveTo(x, y - r); g.lineTo(x + r * 0.95, y + r * 0.75); g.lineTo(x - r * 0.95, y + r * 0.75); g.closePath(); break;
    case 3: g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y); g.closePath(); break;
    case 4: {
      const a = r * 0.32;
      g.moveTo(x - a, y - r); g.lineTo(x + a, y - r); g.lineTo(x + a, y - a); g.lineTo(x + r, y - a); g.lineTo(x + r, y + a);
      g.lineTo(x + a, y + a); g.lineTo(x + a, y + r); g.lineTo(x - a, y + r); g.lineTo(x - a, y + a); g.lineTo(x - r, y + a);
      g.lineTo(x - r, y - a); g.lineTo(x - a, y - a); g.closePath(); break;
    }
    case 5: g.arc(x, y, r * 0.8, 0, Math.PI * 2); break;
    default: g.arc(x, y, r * 0.85, 0, Math.PI * 2);
  }
}

/**
 * 2-D stand-in for the 3-D feature space when WebGL is unavailable: the first two projected
 * dimensions, the same emphasis, k-NN rings and neighbour lines, and a morph between projections.
 */
export function Scatter2D(props: {
  target: Float32Array; fold: number; view: View; visible: Record<Cls, boolean>; flags: Uint8Array | null;
  probe: number | null; nbrs: number[]; height: number;
  onPick: (i: number | null) => void; onHover: (i: number | null, x: number, y: number) => void;
}) {
  const d = useData();
  const { t } = useTheme();
  const reduced = useReducedMotion();
  const [wrap, w] = useWidth<HTMLDivElement>(700);
  const canvas = useRef<HTMLCanvasElement>(null);
  const n = d.emb.n;
  const st = useRef<{ from: Float32Array | null; cur: Float32Array | null; u: number }>({ from: null, cur: null, u: 1 });
  const [animating, setAnimating] = useState(false);
  const h = props.height;

  // screen mapping from the robust extent of the target layout
  const map = useMemo(() => {
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < n; i += 3) { xs.push(props.target[i * 3]); ys.push(props.target[i * 3 + 1]); }
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const q = (a: number[], p: number) => a[Math.floor(p * (a.length - 1))];
    const [x0, x1, y0, y1] = [q(xs, 0.01), q(xs, 0.99), q(ys, 0.01), q(ys, 0.99)];
    const s = Math.min((w - 40) / (x1 - x0 || 1), (h - 90) / (y1 - y0 || 1));
    return { s, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
  }, [props.target, n, w, h]);
  const sx = (v: number) => w / 2 + (v - map.cx) * map.s;
  const sy = (v: number) => (h - 40) / 2 - (v - map.cy) * map.s;

  useEffect(() => {
    const s = st.current;
    s.from = s.cur ? s.cur.slice() : props.target.slice();
    s.cur = s.from.slice();
    s.u = reduced ? 1 : 0;
    setAnimating(true);
  }, [props.target, reduced]);

  const draw = () => {
    const c = canvas.current;
    const pos = st.current.cur;
    if (!c || !pos) return;
    const g = setupCanvas(c, w, h);
    g.clearRect(0, 0, w, h);
    g.strokeStyle = t.grid;
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(sx(0), 0); g.lineTo(sx(0), h - 40); g.moveTo(0, sy(0)); g.lineTo(w, sy(0)); g.stroke();
    const order: number[] = [];
    for (let i = 0; i < n; i++) if (d.emb.recFold[i] !== props.fold) order.push(i);
    for (let i = 0; i < n; i++) if (d.emb.recFold[i] === props.fold) order.push(i);
    for (const i of order) {
      const cls = CLASSES[d.emb.label[i]];
      if (!props.visible[cls]) continue;
      const test = d.emb.recFold[i] === props.fold;
      let a = 1, r = 3;
      if (props.view === "all") { a = test ? 0.95 : 0.6; r = test ? 3.4 : 2.4; }
      if (props.view === "test") { a = test ? 1 : 0.14; r = test ? 3.8 : 2; }
      if (props.view === "testonly" && !test) continue;
      if (props.probe === i) { r = 7; a = 1; }
      const x = sx(pos[i * 3]), y = sy(pos[i * 3 + 1]);
      g.globalAlpha = a;
      g.fillStyle = t.cls[cls];
      mark(g, CLASS_SHAPE[cls], x, y, r);
      if (CLASS_SHAPE[cls] === 5) { g.strokeStyle = t.cls[cls]; g.lineWidth = r * 0.45; g.stroke(); } else g.fill();
      if (props.flags && test && props.flags[i] === 2) {
        g.strokeStyle = t.bad; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, r + 3, 0, Math.PI * 2); g.stroke();
      }
    }
    g.globalAlpha = 1;
    if (props.probe !== null) {
      const px = sx(pos[props.probe * 3]), py = sy(pos[props.probe * 3 + 1]);
      for (const j of props.nbrs) {
        g.strokeStyle = t.cls[CLASSES[d.emb.label[j]]]; g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(px, py); g.lineTo(sx(pos[j * 3]), sy(pos[j * 3 + 1])); g.stroke();
      }
    }
    g.fillStyle = t.muted; g.font = "12px ui-monospace, Menlo, monospace"; g.textAlign = "right";
    g.fillText(w < 520 ? "2-D view · dims 1–2" : "2-D view (WebGL unavailable) · dimensions 1 and 2", w - 12, h - 52);
    g.textAlign = "left";
  };

  useFrame((dt) => {
    const s = st.current;
    if (!s.from || !s.cur) return;
    s.u = Math.min(1, s.u + dt / 1.2);
    const e = s.u < 0.5 ? 4 * s.u ** 3 : 1 - (-2 * s.u + 2) ** 3 / 2;
    for (let i = 0; i < n * 3; i++) s.cur[i] = s.from[i] + (props.target[i] - s.from[i]) * e;
    draw();
    if (s.u >= 1) setAnimating(false);
  }, animating);
  // static redraw when emphasis, theme or size change
  useEffect(() => { if (!animating) draw(); });

  const nearest = (ex: number, ey: number) => {
    const pos = st.current.cur;
    if (!pos) return null;
    let best = -1, bd = 100; // within 10 px
    for (let i = 0; i < n; i++) {
      const cls = CLASSES[d.emb.label[i]];
      if (!props.visible[cls] || (props.view === "testonly" && d.emb.recFold[i] !== props.fold)) continue;
      const dx = sx(pos[i * 3]) - ex, dy = sy(pos[i * 3 + 1]) - ey;
      const dd = dx * dx + dy * dy;
      if (dd < bd) { bd = dd; best = i; }
    }
    return best < 0 ? null : best;
  };

  return (
    <div ref={wrap} style={{ position: "absolute", inset: 0 }}>
      <canvas ref={canvas} role="img" aria-label="2-D scatter of the projected windows"
        onPointerMove={(e) => { const i = nearest(e.nativeEvent.offsetX, e.nativeEvent.offsetY); props.onHover(i, e.nativeEvent.offsetX, e.nativeEvent.offsetY); }}
        onPointerLeave={() => props.onHover(null, 0, 0)}
        onClick={(e) => props.onPick(nearest(e.nativeEvent.offsetX, e.nativeEvent.offsetY))} />
    </div>
  );
}
