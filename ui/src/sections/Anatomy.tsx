import { motion } from "framer-motion";
import { scaleLinear } from "d3-scale";
import { useMemo, useRef, useState } from "react";
import { CLASSES, recId, type Cls } from "../lib/data";
import { alias, candidateSources, explainFrequency } from "../lib/dsp";
import { setupCanvas, useData, useFrame, useInView, useReducedMotion, useWidth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { AxisX, ClassMark, Section, Seg } from "../components/ui";

const STROBE = 24; // shown rotation = true rotation / 24
const N_BALLS = 9;
const FTF = 0.38; // cage speed / shaft speed (6204-type bearing)

interface Marker { name: string; f: number; band?: [number, number] }

function signatures(c: Cls, f1: number, supply: number, fp: number, bpfo: number): { note: string; m: Marker[] } {
  switch (c) {
    case "H": return { note: "No fault: only the running-speed harmonics.", m: [] };
    case "B1":
    case "B2": return { note: "Lubrication loss has no defect line. It shows up as impulsive, broadband vibration, which time-domain statistics (kurtosis, crest, impulse factor) pick up at any bandwidth.", m: [{ name: "broadband", f: 0, band: [0, 160] }] };
    case "B3": return { note: "A cracked outer ring rings at the ball-pass frequency BPFO ≈ 3.05 × f1. It sits above Nyquist at every speed, so the phone sees it folded or not at all.", m: [{ name: "BPFO", f: bpfo * f1 }, { name: "2×BPFO", f: 2 * bpfo * f1 }] };
    case "V": return { note: "Voltage imbalance shows at twice the line frequency (60/80/100 Hz). Every one of those is above 50 Hz.", m: [{ name: "2× line", f: 2 * supply }] };
    case "R": return { note: "Broken bars add sidebands at f1 ± f_p, with f_p = 2·slip·f_supply. These lie well below Nyquist, but see the aliasing lab for the catch at 50 Hz.", m: [{ name: "f1 − f_p", f: f1 - fp }, { name: "f1 + f_p", f: f1 + fp }] };
  }
}

function MachineCanvas({ cls, f1, supply, slip }: { cls: Cls; f1: number; supply: number; slip: number }) {
  const { t } = useTheme();
  const reduced = useReducedMotion();
  const [wrap, w] = useWidth<HTMLDivElement>(900);
  const [vref, inView] = useInView<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const st = useRef({ rotor: 0, field: 0, cage: 0, time: 0, impacts: [] as { a: number; age: number }[], lastHit: -1, sparks: [] as { x: number; y: number; vx: number; vy: number; age: number }[] });
  const h = w < 640 ? 560 : 330;

  useFrame((dt) => {
    const c = canvas.current;
    if (!c) return;
    const s = st.current;
    const sp = reduced ? 0 : 1;
    s.time += dt * sp;
    s.rotor += dt * sp * 2 * Math.PI * (f1 / STROBE);
    s.field += dt * sp * 2 * Math.PI * (supply / 2 / STROBE);
    s.cage += dt * sp * 2 * Math.PI * ((FTF * f1) / STROBE);
    const g = setupCanvas(c, w, h);
    g.clearRect(0, 0, w, h);
    const stacked = w < 640;
    const R = stacked ? Math.min(w * 0.36, 120) : Math.min(h * 0.4, w * 0.18);
    const A = stacked ? { x: w / 2, y: h * 0.25 } : { x: w * 0.27, y: h * 0.47 };
    const B = stacked ? { x: w / 2, y: h * 0.74 } : { x: w * 0.73, y: h * 0.47 };
    const cc = t.cls[cls];
    const steel = t.mode === "dark" ? "#59616c" : "#9aa3ad";
    const steelD = t.mode === "dark" ? "#2c3239" : "#c5cbd1";
    const copper = "#b87333";

    // ------------------------------------------------ rotor end view
    g.save();
    g.translate(A.x, A.y);
    // stator
    g.beginPath(); g.arc(0, 0, R, 0, Math.PI * 2); g.fillStyle = steelD; g.fill();
    g.beginPath(); g.arc(0, 0, R * 0.7, 0, Math.PI * 2); g.fillStyle = t.surface; g.fill();
    for (let k = 0; k < 36; k++) {
      const a = (k / 36) * Math.PI * 2;
      const phase = Math.floor(k / 2) % 3; // A B C A B C ...
      const dim = cls === "V" && phase === 2;
      g.save(); g.rotate(a);
      g.fillStyle = dim ? cc : copper;
      g.globalAlpha = dim ? 0.55 + 0.35 * Math.sin(s.time * 6) : 0.9;
      g.fillRect(R * 0.72, -R * 0.035, R * 0.16, R * 0.07);
      g.restore();
    }
    g.globalAlpha = 1;
    // rotating field lobes in the air gap (synchronous speed)
    const ampl = cls === "V" ? 0.75 + 0.25 * Math.sin(2 * s.field) : 1; // imbalance -> elliptical field
    for (const off of [0, Math.PI]) {
      const a = s.field + off;
      const gx = Math.cos(a) * R * 0.69, gy = Math.sin(a) * R * 0.69;
      const grd = g.createRadialGradient(gx, gy, 0, gx, gy, R * 0.32 * ampl);
      grd.addColorStop(0, t.accent + "aa"); grd.addColorStop(1, t.accent + "00");
      g.fillStyle = grd; g.beginPath(); g.arc(gx, gy, R * 0.32 * ampl, 0, Math.PI * 2); g.fill();
    }
    // rotor + bars
    g.rotate(s.rotor);
    g.beginPath(); g.arc(0, 0, R * 0.66, 0, Math.PI * 2); g.fillStyle = steel; g.fill();
    for (let k = 0; k < 28; k++) {
      const a = (k / 28) * Math.PI * 2;
      const x = Math.cos(a) * R * 0.57, y = Math.sin(a) * R * 0.57;
      g.beginPath(); g.arc(x, y, R * 0.045, 0, Math.PI * 2);
      if (cls === "R" && (k === 0 || k === 1)) {
        g.fillStyle = cc; g.fill();
        g.strokeStyle = t.surface; g.lineWidth = 2;
        g.beginPath(); g.moveTo(x - R * 0.05, y - R * 0.05); g.lineTo(x + R * 0.05, y + R * 0.05); g.stroke();
        const grd = g.createRadialGradient(x, y, 0, x, y, R * 0.16);
        grd.addColorStop(0, cc + "88"); grd.addColorStop(1, cc + "00");
        g.fillStyle = grd; g.beginPath(); g.arc(x, y, R * 0.16, 0, Math.PI * 2); g.fill();
      } else { g.fillStyle = t.mode === "dark" ? "#c9ced6" : "#eef0f2"; g.fill(); }
    }
    g.beginPath(); g.arc(0, 0, R * 0.16, 0, Math.PI * 2); g.fillStyle = steelD; g.fill();
    g.fillStyle = t.accent; g.fillRect(R * 0.08, -R * 0.03, R * 0.1, R * 0.06);
    g.restore();

    // ------------------------------------------------ bearing
    g.save();
    g.translate(B.x, B.y);
    const pitch = R * 0.68, ball = R * 0.15;
    g.beginPath(); g.arc(0, 0, R, 0, Math.PI * 2); g.arc(0, 0, R * 0.86, 0, Math.PI * 2, true); g.fillStyle = steel; g.fill("evenodd");
    // lubricant film
    if (cls !== "B2") {
      g.beginPath(); g.arc(0, 0, R * 0.85, 0, Math.PI * 2); g.arc(0, 0, R * 0.51, 0, Math.PI * 2, true);
      g.fillStyle = cls === "B1" ? "rgba(232,194,106,0.10)" : "rgba(232,194,106,0.26)"; g.fill("evenodd");
    } else {
      const heat = 0.35 + 0.25 * Math.sin(s.time * 3.1);
      const grd = g.createRadialGradient(0, 0, R * 0.4, 0, 0, R * 1.05);
      grd.addColorStop(0, cc + "00"); grd.addColorStop(0.7, cc + Math.round(heat * 120).toString(16).padStart(2, "0")); grd.addColorStop(1, cc + "00");
      g.fillStyle = grd; g.beginPath(); g.arc(0, 0, R * 1.05, 0, Math.PI * 2); g.fill();
    }
    // crack on the outer race (B3)
    const crackA = -Math.PI / 2;
    if (cls === "B3") {
      g.strokeStyle = cc; g.lineWidth = 2.4; g.beginPath();
      const pts = [[0.86, -0.02], [0.9, 0.025], [0.93, -0.015], [0.97, 0.02], [1.0, 0]];
      pts.forEach(([r, da], i) => { const x = Math.cos(crackA + da) * R * r, y = Math.sin(crackA + da) * R * r; i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke();
    }
    // inner ring turns with the shaft
    g.save(); g.rotate(s.rotor);
    g.beginPath(); g.arc(0, 0, R * 0.5, 0, Math.PI * 2); g.arc(0, 0, R * 0.36, 0, Math.PI * 2, true); g.fillStyle = steel; g.fill("evenodd");
    g.fillStyle = t.accent; g.fillRect(R * 0.38, -R * 0.02, R * 0.1, R * 0.04);
    g.restore();
    // balls in the cage
    for (let k = 0; k < N_BALLS; k++) {
      const a = s.cage + (k / N_BALLS) * Math.PI * 2;
      const x = Math.cos(a) * pitch, y = Math.sin(a) * pitch;
      const grd = g.createRadialGradient(x - ball * 0.35, y - ball * 0.35, ball * 0.1, x, y, ball);
      grd.addColorStop(0, t.mode === "dark" ? "#f1f3f5" : "#ffffff"); grd.addColorStop(1, t.mode === "dark" ? "#6a737d" : "#9aa3ad");
      g.fillStyle = grd; g.beginPath(); g.arc(x, y, ball, 0, Math.PI * 2); g.fill();
      if (cls === "B3") {
        const d = Math.atan2(Math.sin(a - crackA), Math.cos(a - crackA));
        if (Math.abs(d) < 0.05 && s.lastHit !== k) { s.impacts.push({ a: crackA, age: 0 }); s.lastHit = k; }
      }
      if (cls === "B2" && Math.random() < dt * 1.6) {
        s.sparks.push({ x: x * 1.1, y: y * 1.1, vx: (Math.random() - 0.5) * 60, vy: (Math.random() - 0.5) * 60, age: 0 });
      }
    }
    for (const im of s.impacts) {
      im.age += dt;
      const x = Math.cos(im.a) * R * 0.93, y = Math.sin(im.a) * R * 0.93;
      g.strokeStyle = cc; g.globalAlpha = Math.max(0, 1 - im.age / 0.9); g.lineWidth = 2;
      g.beginPath(); g.arc(x, y, 5 + im.age * 48, 0, Math.PI * 2); g.stroke();
    }
    g.globalAlpha = 1;
    s.impacts = s.impacts.filter((i) => i.age < 0.9);
    for (const p of s.sparks) {
      p.age += dt; p.x += p.vx * dt; p.y += p.vy * dt;
      g.fillStyle = cc; g.globalAlpha = Math.max(0, 1 - p.age / 0.6);
      g.beginPath(); g.arc(p.x, p.y, 2, 0, Math.PI * 2); g.fill();
    }
    g.globalAlpha = 1;
    s.sparks = s.sparks.filter((p) => p.age < 0.6);
    g.restore();

    // captions
    g.fillStyle = t.muted; g.font = "12px ui-monospace, Menlo, monospace"; g.textAlign = "center";
    g.fillText(`rotor · end view · field ${(supply / 2).toFixed(0)} Hz, rotor ${f1.toFixed(2)} Hz`, A.x, A.y + R + 22);
    g.fillText(`bearing · cage at ${(FTF * f1).toFixed(1)} Hz · slip ${(slip * 100).toFixed(1)}%`, B.x, B.y + R + 22);
  }, inView);

  return (
    <div ref={vref}>
      <div ref={wrap}>
        <canvas ref={canvas} role="img" aria-label={`Animated cross-section of the motor and bearing for class ${cls}`} />
      </div>
    </div>
  );
}

function Ruler({ cls, f1, supply, fp, phone }: { cls: Cls; f1: number; supply: number; fp: number; phone: boolean }) {
  const d = useData();
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(900);
  const h = 170;
  const x = scaleLinear().domain([0, 160]).range([24, w - 16]);
  const sig = signatures(cls, f1, supply, fp, d.meta.bpfo_order);
  const shaft: Marker[] = [1, 2, 3, 4, 5, 6].map((k) => ({ name: `${k}×f1`, f: k * f1 })).filter((m) => m.f < 160);
  const pos = (f: number) => x(phone ? alias(f) : f);
  const yAxis = 120;
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={w} height={h} className="chart" role="img" aria-label="Frequency ruler with the Nyquist limit">
        <defs>
          <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="7" stroke="var(--grid)" strokeWidth="3" />
          </pattern>
          <linearGradient id="wall" x1="0" x2="1">
            <stop offset="0" stopColor={t.accent} stopOpacity="0.55" />
            <stop offset="1" stopColor={t.accent} stopOpacity="0" />
          </linearGradient>
        </defs>
        <motion.rect y={20} height={yAxis - 20} fill="url(#hatch)" initial={false}
          animate={{ x: x(50), width: phone ? 0 : x(160) - x(50), opacity: phone ? 0 : 1 }} transition={{ duration: 0.8 }} />
        <rect x={x(50)} y={14} width={18} height={yAxis - 14} fill="url(#wall)" />
        <line x1={x(50)} x2={x(50)} y1={14} y2={yAxis} stroke={t.accent} strokeWidth={2} />
        <text x={x(50) + 6} y={26} className="lbl-strong">Nyquist 50 Hz</text>
        {!phone && <text x={(x(50) + x(160)) / 2 + 30} y={58} textAnchor="middle" className="lbl">above 50 Hz: lost or folded by a 100 Hz phone</text>}
        {sig.m.filter((m) => m.band).map((m) => (
          <rect key={m.name} x={x(0)} width={x(phone ? 50 : 160) - x(0)} y={yAxis - 34} height={22} rx={4} fill={t.cls[cls]} opacity={0.16} />
        ))}
        {shaft.map((m) => (
          <motion.g key={m.name} initial={false} animate={{ x: pos(m.f) }} transition={{ type: "spring", stiffness: 70, damping: 16 }}>
            <line y1={yAxis - 30} y2={yAxis} stroke={t.ink2} strokeWidth={1.5} opacity={0.55} />
            <text y={yAxis - 34} textAnchor="middle">{m.name}</text>
          </motion.g>
        ))}
        {sig.m.filter((m) => !m.band).map((m, i) => (
          <motion.g key={m.name} initial={false} animate={{ x: pos(m.f) }} transition={{ type: "spring", stiffness: 60, damping: 14, delay: i * 0.08 }}>
            <line y1={40 + i * 14} y2={yAxis} stroke={t.cls[cls]} strokeWidth={2.5} />
            <circle cy={40 + i * 14} r={5} fill={t.cls[cls]} stroke={t.surface} strokeWidth={2} />
            <text y={34 + i * 14} textAnchor="middle" className="lbl-strong">{m.name}</text>
          </motion.g>
        ))}
        <AxisX scale={x} y={yAxis} ticks={8} format={(v) => `${v}`} label={phone ? "apparent frequency on the phone (Hz)" : "true frequency (Hz)"} />
      </svg>
      <p className="sub" style={{ marginTop: 6, color: "var(--ink2)", fontSize: 14 }}>{sig.note}</p>
    </div>
  );
}

export function Anatomy() {
  const d = useData();
  const [cls, setCls] = useState<Cls>("B3");
  const [cond, setCond] = useState({ load: 1, supply: 50 });
  const [phone, setPhone] = useState(false);
  const rec = d.recById.get(recId(cls, cond.load, cond.supply))!;
  const sig = signatures(cls, rec.f1, rec.supply, rec.f_pole, d.meta.bpfo_order);
  const sources = useMemo(() => candidateSources(rec.f1, rec.supply, d.meta.bpfo_order, d.meta.bpfi_order), [rec, d]);
  const lines = sig.m.filter((m) => !m.band);

  return (
    <Section id="anatomy" eyebrow="The machine" tags={["dataset"]}
      title="Six health states, one squirrel-cage motor"
      lede={<>An iPhone 15 Pro Max lies on the motor and logs six IMU channels at 100 Hz. Each class is one 15-minute steady run per
        speed and load. Pick a fault to see where it lives in the machine and where its textbook signature sits on the frequency axis.
        Then flip to <b>what the phone sees</b>.</>}>
      <div className="card">
        <div className="controls">
          <Seg value={cls} onChange={setCls} options={CLASSES.map((c) => ({ value: c, label: <><ClassMark c={c} size={10} />{c}</> }))} />
          <Seg value={cond.supply} onChange={(s) => setCond({ ...cond, supply: s })} options={[30, 40, 50].map((s) => ({ value: s, label: `${s} Hz` }))} />
          <Seg value={cond.load} onChange={(l) => setCond({ ...cond, load: l })} options={[{ value: 0, label: "Unloaded" }, { value: 1, label: "Loaded" }]} />
        </div>
        <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}><ClassMark c={cls} size={14} /> {cls} · {d.meta.class_names[cls]}</h3>
        <MachineCanvas cls={cls} f1={rec.f1} supply={rec.supply} slip={rec.slip} />
        <div className="controls" style={{ marginTop: 6, justifyContent: "space-between" }}>
          <span className="ctl-label">Signature on the frequency axis</span>
          <Seg value={phone ? 1 : 0} onChange={(v) => setPhone(v === 1)} options={[{ value: 0, label: "True spectrum" }, { value: 1, label: "What the phone sees" }]} />
        </div>
        <Ruler cls={cls} f1={rec.f1} supply={rec.supply} fp={rec.f_pole} phone={phone} />
        {lines.length > 0 && (
          <div className="kv" style={{ marginTop: 10 }}>
            {lines.map((m) => {
              const ap = alias(m.f);
              const others = sources.filter((s) => !(s.name === m.name || (m.name === "2× line" && s.name === "2× line")));
              const hit = explainFrequency(ap, others, 0.3);
              return (
                <div key={m.name} style={{ display: "contents" }}>
                  <dt>{m.name}</dt>
                  <dd>{m.f.toFixed(2)} Hz {m.f > 50 ? <>→ appears at <b>{ap.toFixed(2)} Hz</b></> : "(in band)"}{hit ? <> · lands on {hit}</> : ""}</dd>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Section>
  );
}
