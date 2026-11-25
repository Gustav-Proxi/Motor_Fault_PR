import { Canvas, useFrame as useR3fFrame, useThree } from "@react-three/fiber";
import { motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { CLASSES, prettyId, recId, type Cls } from "../lib/data";
import { useData, useInView, useReducedMotion } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { ClassMark, Seg, Stat } from "../components/ui";

/** Shared playback state the 3D scene reads every frame (no React re-render per sample). */
interface Player { chans: Float32Array[]; pos: number; gain: number; f1: number; accent: string; ink: string; bg: string; cls: string }

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

    // phone screen: live 3-axis trace of the last 2.5 s
    const g = screenCanvas.getContext("2d")!;
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
          <Stat label="Best macro-F1, unseen recordings" value={best.macro_f1} digits={2} note="LDA-5 + random forest · chance 0.17" />
          <Stat label="Naive random split" value={naive} digits={2} note="leaks overlapping windows" />
        </div>
      </div>
    </header>
  );
}
