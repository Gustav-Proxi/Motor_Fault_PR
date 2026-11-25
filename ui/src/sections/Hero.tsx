import { Canvas, useFrame as useR3fFrame, useThree } from "@react-three/fiber";
import { motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { CLASSES, prettyId, recId, type Cls } from "../lib/data";
import { CLF_LABEL, FRONT_LABEL } from "../lib/data";
import { useData, useFrame, useInView, useReducedMotion } from "../lib/hooks";
import { hasWebGL } from "../lib/webgl";
import { GLBoundary } from "../components/GLBoundary";
import { useTheme } from "../lib/theme";
import { ClassMark, Seg, Stat } from "../components/ui";

/** Shared playback state the 3D scene reads every frame (no React re-render per sample). */
interface Player { chans: Float32Array[]; pos: number; gain: number; f1: number; accent: string; ink: string; bg: string; cls: string }

/** Phone screen: live 3-axis trace of the last 2.5 s, painted on a 512 x 256 canvas. */
function paintScreen(g: CanvasRenderingContext2D, p: Player, i: number) {
  const n = p.chans[0].length;
  g.fillStyle = p.bg;
  g.fillRect(0, 0, 512, 256);
  g.strokeStyle = "rgba(255,255,255,0.07)";
  g.lineWidth = 1;
  for (let y = 32; y < 256; y += 48) { g.beginPath(); g.moveTo(0, y); g.lineTo(512, y); g.stroke(); }
  const span = 250;
  for (let c = 0; c < 3; c++) {
    const yc = 70 + c * 62;
    g.strokeStyle = c === 0 ? p.accent : c === 1 ? p.ink : "rgba(200,200,200,0.55)";
    g.lineWidth = 2.2;
    g.beginPath();
    for (let k = 0; k < span; k++) {
      const s = p.chans[c][(i - span + k + n) % n];
      const x = (k / (span - 1)) * 512;
      const y = yc - s * p.gain * 900;
      k ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.stroke();
  }
  g.fillStyle = p.ink;
  g.font = "600 22px ui-monospace, Menlo, monospace";
  g.fillText(`${p.cls}  ·  100 Hz`, 16, 30);
  g.fillStyle = p.accent;
  g.beginPath();
  g.arc(490, 22, 7, 0, Math.PI * 2);
  g.fill();
}

/** 2-D stand-in for the 3-D motor when WebGL is unavailable: same recording, same shake. */
function MotorFallback({ player, active }: { player: React.MutableRefObject<Player>; active: boolean }) {
  const { t } = useTheme();
  const body = useRef<SVGGElement>(null);
  const mark = useRef<SVGCircleElement>(null);
  const screen = useRef<HTMLCanvasElement>(null);
  const angle = useRef(0);
  useFrame((dt) => {
    const p = player.current;
    const n = p.chans[0].length;
    p.pos = (p.pos + dt * 100) % n;
    const i = Math.floor(p.pos);
    const k = p.gain * 170; // 3-D units -> SVG px
    body.current?.setAttribute("transform", `translate(${(p.chans[0][i] * k).toFixed(2)} ${(-p.chans[2][i] * k).toFixed(2)})`);
    angle.current += dt * 2 * Math.PI * (p.f1 / 24);
    const a = angle.current;
    if (mark.current) {
      mark.current.setAttribute("cy", (222 + 30 * Math.sin(a)).toFixed(1));
      mark.current.setAttribute("cx", (548 + 7 * Math.cos(a)).toFixed(1));
      mark.current.setAttribute("opacity", Math.cos(a) > 0 ? "1" : "0.25");
    }
    const c = screen.current;
    if (c) paintScreen(c.getContext("2d")!, p, i);
  }, active);
  const metal = t.mode === "dark" ? "#6b7889" : "#8793a3";
  const dark = t.mode === "dark" ? "#3a424d" : "#4b5562";
  return (
    <div style={{ position: "absolute", inset: 0, background: t.mode === "dark" ? "#0e0f12" : "#e9e9e5" }}>
      <svg viewBox="0 0 640 400" width="100%" height="100%" role="img" aria-label="Motor vibrating with the selected recording (2-D view)">
        <defs>
          <linearGradient id="hf-metal" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor={metal} /><stop offset="0.55" stopColor={dark} /><stop offset="1" stopColor="#20262e" />
          </linearGradient>
        </defs>
        <ellipse cx="300" cy="332" rx="230" ry="16" fill="#000" opacity="0.25" />
        <rect x="110" y="304" width="400" height="14" rx="3" fill="#16191d" />
        <g ref={body}>
          <rect x="170" y="262" width="60" height="44" rx="4" fill={dark} />
          <rect x="370" y="262" width="60" height="44" rx="4" fill={dark} />
          <rect x="100" y="160" width="52" height="112" rx="18" fill={dark} />
          {Array.from({ length: 4 }, (_, j) => <line key={j} x1={108 + j * 11} x2={108 + j * 11} y1="172" y2="260" stroke="#b9c0c8" strokeWidth="2" opacity="0.5" />)}
          <rect x="140" y="150" width="320" height="130" rx="26" fill="url(#hf-metal)" />
          {Array.from({ length: 21 }, (_, j) => <line key={j} x1={160 + j * 14} x2={160 + j * 14} y1="156" y2="274" stroke="#20262e" strokeWidth="3" opacity="0.55" />)}
          <rect x="455" y="166" width="24" height="98" rx="8" fill={dark} />
          <rect x="190" y="134" width="180" height="16" rx="6" fill="#0d0f12" />
          <rect x="196" y="137" width="168" height="6" rx="3" fill={t.accent} opacity="0.7" />
        </g>
        <rect x="479" y="214" width="62" height="16" rx="3" fill="#b9c0c8" />
        <ellipse cx="548" cy="222" rx="11" ry="36" fill={dark} />
        <circle ref={mark} cx="555" cy="222" r="4.5" fill={t.accent} />
      </svg>
      <canvas ref={screen} width={512} height={256}
        style={{ position: "absolute", right: 18, top: 18, width: "38%", borderRadius: 12, border: "4px solid #0d0f12", boxShadow: "0 8px 30px rgba(0,0,0,.35)" }}
        aria-label="Phone screen: live trace" />
      <span className="pill" style={{ position: "absolute", left: 14, top: 14, font: "12px var(--mono)", color: "var(--ink2)" }}>2-D view (WebGL unavailable)</span>
    </div>
  );
}

function Controls() {
  const { camera, gl, size } = useThree();
  // pull the camera back on portrait screens so the whole motor stays in frame
  useEffect(() => { camera.position.setLength(size.width / size.height < 0.9 ? 9.8 : 6.6); }, [camera, size]);
  const ctl = useMemo(() => {
    const c = new OrbitControls(camera, gl.domElement);
    c.enableZoom = false;
    c.enablePan = false;
    c.enableDamping = true;
    c.autoRotate = true;
    c.autoRotateSpeed = 0.55;
    c.minPolarAngle = 0.6;
    c.maxPolarAngle = 1.45;
    c.target.set(0.15, 0.25, 0);
    return c;
  }, [camera, gl]);
  useEffect(() => () => ctl.dispose(), [ctl]);
  useR3fFrame(() => ctl.update());
  return null;
}

function Motor({ player }: { player: React.MutableRefObject<Player> }) {
  const body = useRef<THREE.Group>(null);
  const shaft = useRef<THREE.Group>(null);
  const screenCanvas = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = 512;
    c.height = 256;
    return c;
  }, []);
  const screenTex = useMemo(() => {
    const t = new THREE.CanvasTexture(screenCanvas);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [screenCanvas]);

  const fins = useMemo(() => Array.from({ length: 30 }, (_, i) => (i / 30) * Math.PI * 2), []);
  const phoneGeo = useMemo(() => new RoundedBoxGeometry(1.55, 0.07, 0.76, 4, 0.08), []);
  const metal = useMemo(() => new THREE.MeshStandardMaterial({ color: "#6b7889", metalness: 0.45, roughness: 0.42 }), []);
  const dark = useMemo(() => new THREE.MeshStandardMaterial({ color: "#3a424d", metalness: 0.4, roughness: 0.55 }), []);
  const steel = useMemo(() => new THREE.MeshStandardMaterial({ color: "#b9c0c8", metalness: 0.95, roughness: 0.22 }), []);

  useR3fFrame((_, dt) => {
    const p = player.current;
    const n = p.chans[0].length;
    p.pos = (p.pos + dt * 100) % n; // real time: 100 samples per second
    const i = Math.floor(p.pos);
    if (body.current) {
      // the motor literally shakes with the recording (amplified for visibility)
      body.current.position.set(p.chans[0][i] * p.gain, 0.0 + p.chans[2][i] * p.gain, p.chans[1][i] * p.gain);
      body.current.rotation.x = p.chans[1][i] * p.gain * 0.6;
    }
    // stroboscopic shaft: true speed / 24 so the rotation can be followed by eye
    if (shaft.current) shaft.current.rotation.x -= dt * 2 * Math.PI * (p.f1 / 24);

    paintScreen(screenCanvas.getContext("2d")!, p, i);
    screenTex.needsUpdate = true;
  });

  return (
    <group>
      <group ref={body}>
        {/* housing with cooling fins */}
        <mesh rotation={[0, 0, Math.PI / 2]} material={metal}>
          <cylinderGeometry args={[0.95, 0.95, 2.3, 64]} />
        </mesh>
        {fins.map((a) => (
          <mesh key={a} position={[0, Math.cos(a) * 1.0, Math.sin(a) * 1.0]} rotation={[a, 0, 0]} material={metal}>
            <boxGeometry args={[2.1, 0.12, 0.035]} />
          </mesh>
        ))}
        {/* end bells */}
        <mesh position={[1.22, 0, 0]} rotation={[0, 0, Math.PI / 2]} material={dark}>
          <cylinderGeometry args={[0.9, 0.95, 0.2, 64]} />
        </mesh>
        <mesh position={[-1.42, 0, 0]} rotation={[0, 0, Math.PI / 2]} material={dark}>
          <cylinderGeometry args={[0.98, 0.98, 0.55, 64, 1, true]} />
        </mesh>
        {[0, 1, 2, 3].map((k) => (
          <mesh key={k} position={[-1.69, 0, 0]} rotation={[0, Math.PI / 2, 0]} material={steel}>
            <torusGeometry args={[0.22 + k * 0.2, 0.018, 8, 48]} />
          </mesh>
        ))}
        {/* terminal box */}
        <mesh position={[0.35, 0.25, 1.02]} material={dark}>
          <boxGeometry args={[0.62, 0.5, 0.3]} />
        </mesh>
        {/* feet + base */}
        {[-0.75, 0.75].map((x) => (
          <mesh key={x} position={[x, -0.95, 0]} material={dark}>
            <boxGeometry args={[0.42, 0.38, 1.6]} />
          </mesh>
        ))}
        {/* the phone on top, screen live */}
        <group position={[0.0, 1.14, 0]} rotation={[0, 0, 0]}>
          <mesh geometry={phoneGeo}>
            <meshStandardMaterial color="#0d0f12" metalness={0.4} roughness={0.25} />
          </mesh>
          <mesh position={[0, 0.037, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[1.44, 0.66]} />
            <meshBasicMaterial map={screenTex} toneMapped={false} />
          </mesh>
        </group>
      </group>
      {/* shaft + coupling (not shaken, so the strobe reads cleanly) */}
      <group ref={shaft} position={[1.32, 0, 0]}>
        <mesh position={[0.4, 0, 0]} rotation={[0, 0, Math.PI / 2]} material={steel}>
          <cylinderGeometry args={[0.09, 0.09, 0.8, 24]} />
        </mesh>
        <mesh position={[0.82, 0, 0]} rotation={[0, 0, Math.PI / 2]} material={dark}>
          <cylinderGeometry args={[0.3, 0.3, 0.16, 48]} />
        </mesh>
        <mesh position={[0.91, 0.2, 0]}>
          <boxGeometry args={[0.02, 0.12, 0.06]} />
          <meshBasicMaterial color={player.current.accent} toneMapped={false} />
        </mesh>
      </group>
      {/* base plate */}
      <mesh position={[0, -1.18, 0]}>
        <boxGeometry args={[3.6, 0.08, 2.0]} />
        <meshStandardMaterial color="#16191d" metalness={0.3} roughness={0.8} />
      </mesh>
    </group>
  );
}

function Floor({ color }: { color: string }) {
  const grid = useMemo(() => {
    const g = new THREE.GridHelper(16, 32, color, color);
    (g.material as THREE.Material).transparent = true;
    (g.material as THREE.Material).opacity = 0.18;
    g.position.y = -1.23;
    return g;
  }, [color]);
  return <primitive object={grid} />;
}

export function Hero() {
  const d = useData();
  const { t } = useTheme();
  const reduced = useReducedMotion();
  const [cls, setCls] = useState<Cls>("B2");
  const [cond, setCond] = useState({ load: 1, supply: 50 });
  const [ref, inView] = useInView<HTMLDivElement>("0px");
  const id = recId(cls, cond.load, cond.supply);
  const rec = d.recById.get(id)!;

  // one gain for every recording, so the shake keeps the true relative amplitudes
  const gain = useMemo(() => {
    let m = 0;
    for (const ch of d.signal.values()) for (const c of ch) for (let i = 0; i < c.length; i += 7) m = Math.max(m, Math.abs(c[i]));
    return 0.05 / m;
  }, [d]);
  const player = useRef<Player>({ chans: d.signal.get(id)!, pos: 0, gain, f1: rec.f1, accent: t.accent, ink: t.ink, bg: "#07080a", cls: id });
  player.current.chans = d.signal.get(id)!;
  player.current.f1 = rec.f1;
  player.current.cls = prettyId(id);
  player.current.accent = t.accent;
  player.current.ink = "#f0efec";

  const best = d.metrics.filter((m) => m.split === "grouped").reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
  const naive = Math.max(...d.metrics.filter((m) => m.split === "naive").map((m) => m.macro_f1));

  return (
    <header className="hero" id="top">
      <div className="wrap">
        <div className="hero-grid">
          <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, ease: [0.2, 0.7, 0.2, 1] }}>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <span className="eyebrow"><span className="dot" />CSE 455/555 · Fall 2026 · Final project</span>
            </div>
            <h1 style={{ marginTop: 18 }}>
              Which motor faults <span className="grad">survive a smartphone?</span>
            </h1>
            <p className="lede">
              A phone samples vibration at <b>100 Hz</b>, so nothing above <b>50 Hz</b> survives. Higher frequencies are lost or folded
              down. We test which of six induction-motor health states stay separable after that cut, which projection keeps them
              apart, and how it all holds up on recordings no model has seen. Everything is classical: hand-written features, no
              pre-trained models.
            </p>
            <div className="controls" style={{ marginTop: 10 }}>
              <Seg label="Shake the motor with" value={cls} onChange={setCls}
                options={CLASSES.map((c) => ({ value: c, label: <><ClassMark c={c} size={10} />{c}</> }))} />
            </div>
            <div className="controls">
              <Seg label="Supply" value={cond.supply} onChange={(s) => setCond({ ...cond, supply: s })}
                options={[30, 40, 50].map((s) => ({ value: s, label: `${s} Hz` }))} />
              <Seg value={cond.load} onChange={(l) => setCond({ ...cond, load: l })}
                options={[{ value: 0, label: "Unloaded" }, { value: 1, label: "Loaded" }]} />
            </div>
          </motion.div>
          <motion.div ref={ref} className="hero-canvas card" style={{ padding: 0 }}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1.1, delay: 0.15 }}>
            {hasWebGL() ? (
            <GLBoundary fallback={<MotorFallback player={player} active={inView && !reduced} />}>
            <Canvas
              frameloop={inView && !reduced ? "always" : "demand"}
              resize={{ offsetSize: true }}
              dpr={[1, 1.6]}
              camera={{ position: [4.3, 2.3, 4.4], fov: 38 }}
              gl={{ antialias: true, powerPreference: "low-power" }}
              aria-label={`3D motor vibrating with recording ${id}`}
            >
              <color attach="background" args={[t.mode === "dark" ? "#0e0f12" : "#e9e9e5"]} />
              <fog attach="fog" args={[t.mode === "dark" ? "#0e0f12" : "#e9e9e5", 8, 16]} />
              <hemisphereLight args={["#e6e8ff", "#30303a", 1.4]} />
              <ambientLight intensity={0.35} />
              <directionalLight position={[4, 6, 3]} intensity={2.8} />
              <directionalLight position={[-4, 2, 4]} intensity={0.9} color="#c9d4ff" />
              <pointLight position={[-3, 1.5, -2.5]} intensity={18} color={t.accent} distance={9} />
              <pointLight position={[3, -0.5, 3]} intensity={8} color="#3987e5" distance={8} />
              <Motor player={player} />
              <Floor color={t.mode === "dark" ? "#8f8d86" : "#52514e"} />
              <Controls />
            </Canvas>
            </GLBoundary>
            ) : <MotorFallback player={player} active={inView && !reduced} />}
            <div className="hud">
              <span className="pill">{prettyId(id)} · {d.meta.class_names[cls]}</span>
              <span className="pill">f1 = {rec.f1.toFixed(2)} Hz · slip {(rec.slip * 100).toFixed(1)}% · shaft strobe ÷24</span>
            </div>
          </motion.div>
        </div>
        <div className="stats" style={{ marginTop: 34 }}>
          <Stat label="Recordings" value={d.meta.n_recordings} note="6 classes × 3 speeds × 2 loads" />
          <Stat label="Labelled windows" value={d.meta.n_windows} note="10.24 s each, 50% overlap" />
          <Stat label="Hand-written features" value={d.meta.n_features} note="21 per channel × 6" />
          <Stat label="Best macro-F1, unseen recordings" value={best.macro_f1} digits={2}
            note={`${FRONT_LABEL[best.front]} + ${CLF_LABEL[best.clf]} · ${d.meta.grid_seeds > 1 ? `mean of ${d.meta.grid_seeds} seeds` : "seed 0"} · chance 0.17`} />
          <Stat label="Naive random split" value={naive} digits={2} note="leaks overlapping windows" />
        </div>
      </div>
    </header>
  );
}
