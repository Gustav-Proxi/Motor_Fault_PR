import { Canvas, useFrame as useR3fFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { scaleLog } from "d3-scale";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CLASSES, prettyId, type Cls, type Proj } from "../lib/data";
import { macroF1, nearest, vote } from "../lib/dsp";
import { fmt, useData, useInView, useReducedMotion, useWidth } from "../lib/hooks";
import { CLASS_SHAPE, useTheme } from "../lib/theme";
import { ClassLegend, ClassMark, Counter, HBar, Section, Seg } from "../components/ui";

const PROJ_LABEL: Record<Proj, string> = { pca5: "PCA-5", lda5: "LDA-5", kpca5: "Kernel PCA-5" };
type View = "all" | "test" | "testonly";

const vert = /* glsl */ `
attribute vec3 aColor; attribute float aShape; attribute float aSize; attribute float aAlpha; attribute float aFlag;
varying vec3 vColor; varying float vShape; varying float vAlpha; varying float vFlag;
uniform float uPx;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = aSize * uPx * (62.0 / -mv.z) * (aFlag > 1.5 ? 1.5 : 1.0);
  vColor = aColor; vShape = aShape; vAlpha = aAlpha; vFlag = aFlag;
}`;
const frag = /* glsl */ `
varying vec3 vColor; varying float vShape; varying float vAlpha; varying float vFlag;
uniform vec3 uRing; uniform vec3 uBad;
float sdBox(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }
float sdTri(vec2 p, float r) {
  const float k = 1.7320508;
  p.x = abs(p.x) - r; p.y = p.y + r / k;
  if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
  p.x -= clamp(p.x, -2.0 * r, 0.0);
  return -length(p) * sign(p.y);
}
float shape(vec2 p, float s) {
  if (s < 0.5) return length(p) - 0.62;
  if (s < 1.5) return sdBox(p, vec2(0.5));
  if (s < 2.5) return sdTri(p + vec2(0.0, 0.12), 0.72);
  if (s < 3.5) return (abs(p.x) + abs(p.y) - 0.72) * 0.7071;
  if (s < 4.5) return min(sdBox(p, vec2(0.66, 0.2)), sdBox(p, vec2(0.2, 0.66)));
  return abs(length(p) - 0.5) - 0.16;
}
void main() {
  if (vAlpha < 0.01) discard;
  vec2 p = vec2(gl_PointCoord.x * 2.0 - 1.0, 1.0 - gl_PointCoord.y * 2.0);
  float sc = vFlag > 1.5 ? 1.5 : 1.0;
  p *= sc;
  float d = shape(p, vShape);
  float aa = fwidth(d) * 1.2;
  float ring = 0.16;
  vec3 col = vColor;
  float a = 1.0 - smoothstep(-aa, aa, d - ring);
  if (d > 0.0) col = uRing;                       // surface ring separates overlapping marks
  if (vFlag > 1.5) {                              // misclassified: outer ring
    float r = abs(length(p) - 1.18) - 0.1;
    float ra = 1.0 - smoothstep(-aa, aa, r);
    if (ra > a) { col = uBad; a = ra; }
  }
  if (a < 0.02) discard;
  gl_FragColor = vec4(col, a * vAlpha);
}`;

/** hex -> sRGB components in 0..1 (no colour-management conversion). */
const srgb = (h: string): [number, number, number] => {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};

interface Cloud {
  pos: Float32Array; // current display positions (n*3)
  from: Float32Array;
  to: Float32Array;
  t: number;
  delay: Float32Array;
}

function useDisplay(proj: Proj, fold: number) {
  const d = useData();
  return useMemo(() => {
    const Z = d.emb.Z[proj][fold];
    const n = d.emb.n;
    const c = [0, 0, 0];
    let nt = 0;
    for (let i = 0; i < n; i++) if (d.emb.recFold[i] !== fold) { for (let k = 0; k < 3; k++) c[k] += Z[i * 5 + k]; nt++; }
    for (let k = 0; k < 3; k++) c[k] /= nt;
    const r: number[] = [];
    for (let i = 0; i < n; i += 3) if (d.emb.recFold[i] !== fold) r.push(Math.hypot(Z[i * 5] - c[0], Z[i * 5 + 1] - c[1], Z[i * 5 + 2] - c[2]));
    r.sort((a, b) => a - b);
    const s = 2.3 / (r[Math.floor(r.length * 0.95)] || 1);
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) out[i * 3 + k] = (Z[i * 5 + k] - c[k]) * s;
    return out;
  }, [d, proj, fold]);
}

function Points(props: {
  target: Float32Array; fold: number; view: View; visible: Record<Cls, boolean>; flags: Uint8Array | null;
  probe: number | null; onPick: (i: number) => void; onHover: (i: number | null, x: number, y: number) => void;
}) {
  const d = useData();
  const { t } = useTheme();
  const reduced = useReducedMotion();
  const n = d.emb.n;
  const geo = useMemo(() => new THREE.BufferGeometry(), []);
  const cloud = useRef<Cloud | null>(null);
  const mat = useMemo(() => new THREE.ShaderMaterial({
    vertexShader: vert, fragmentShader: frag, transparent: true, depthWrite: false,
    uniforms: { uPx: { value: Math.min(2, window.devicePixelRatio || 1) }, uRing: { value: new THREE.Color() }, uBad: { value: new THREE.Color() } },
  }), []);

  if (!cloud.current) {
    const delay = new Float32Array(n);
    for (let i = 0; i < n; i++) delay[i] = Math.random() * 0.35;
    const start = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) start[i] = (Math.random() - 0.5) * 0.4;
    cloud.current = { pos: start, from: start.slice(), to: props.target, t: 0, delay };
    geo.setAttribute("position", new THREE.BufferAttribute(start, 3));
    geo.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute("aShape", new THREE.BufferAttribute(new Float32Array(n), 1));
    geo.setAttribute("aSize", new THREE.BufferAttribute(new Float32Array(n), 1));
    geo.setAttribute("aAlpha", new THREE.BufferAttribute(new Float32Array(n), 1));
    geo.setAttribute("aFlag", new THREE.BufferAttribute(new Float32Array(n), 1));
    geo.computeBoundingSphere();
  }
  // new target -> morph from wherever the points are now
  useEffect(() => {
    const c = cloud.current!;
    c.from = c.pos.slice();
    c.to = props.target;
    c.t = reduced ? 1 : 0;
  }, [props.target, reduced]);

  // colours / shapes / emphasis
  useEffect(() => {
    const col = geo.getAttribute("aColor") as THREE.BufferAttribute;
    const shp = geo.getAttribute("aShape") as THREE.BufferAttribute;
    const size = geo.getAttribute("aSize") as THREE.BufferAttribute;
    const alpha = geo.getAttribute("aAlpha") as THREE.BufferAttribute;
    const flag = geo.getAttribute("aFlag") as THREE.BufferAttribute;
    const cc = CLASSES.map((c) => srgb(t.cls[c]));
    for (let i = 0; i < n; i++) {
      const ci = d.emb.label[i];
      const cls = CLASSES[ci];
      const test = d.emb.recFold[i] === props.fold;
      col.setXYZ(i, cc[ci][0], cc[ci][1], cc[ci][2]);
      shp.setX(i, CLASS_SHAPE[cls]);
      const on = props.visible[cls];
      let a = 1, s = 1;
      if (props.view === "all") { a = test ? 0.95 : 0.75; s = test ? 1.15 : 0.9; }
      if (props.view === "test") { a = test ? 1 : 0.16; s = test ? 1.35 : 0.75; }
      if (props.view === "testonly") { a = test ? 1 : 0; s = test ? 1.35 : 0; }
      if (props.probe === i) { s = 2.6; a = 1; }
      alpha.setX(i, on ? a : 0);
      size.setX(i, on ? s : 0);
      flag.setX(i, props.flags && test ? props.flags[i] : 0);
    }
    for (const a of [col, shp, size, alpha, flag]) a.needsUpdate = true;
    mat.uniforms.uRing.value.setRGB(...srgb(t.mode === "dark" ? "#0e0f12" : "#f6f6f3"));
    mat.uniforms.uBad.value.setRGB(...srgb(t.bad));
  }, [d, t, n, props.fold, props.view, props.visible, props.flags, props.probe, geo, mat]);

  useR3fFrame((_, dt) => {
    const c = cloud.current!;
    if (c.t >= 1.4) return;
    c.t = Math.min(1.4, c.t + dt / 1.3);
    for (let i = 0; i < n; i++) {
      let u = Math.min(1, Math.max(0, (c.t - c.delay[i]) / 1.0));
      u = u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
      for (let k = 0; k < 3; k++) c.pos[i * 3 + k] = c.from[i * 3 + k] + (c.to[i * 3 + k] - c.from[i * 3 + k]) * u;
    }
    (geo.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    if (c.t >= 1.4) geo.computeBoundingSphere();
  });

  const pickable = (e: ThreeEvent<PointerEvent | MouseEvent>) => {
    const i = e.index;
    if (i === undefined) return null;
    const a = (geo.getAttribute("aAlpha") as THREE.BufferAttribute).getX(i);
    return a > 0.1 ? i : null;
  };
  return (
    <points geometry={geo} material={mat}
      onClick={(e) => { e.stopPropagation(); const i = pickable(e); if (i !== null) props.onPick(i); }}
      onPointerMove={(e) => { const i = pickable(e); props.onHover(i, e.nativeEvent.offsetX, e.nativeEvent.offsetY); }}
      onPointerOut={() => props.onHover(null, 0, 0)} />
  );
}

function Links({ target, probe, nbrs }: { target: Float32Array; probe: number; nbrs: number[] }) {
  const d = useData();
  const { t } = useTheme();
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(nbrs.length * 6);
    const col = new Float32Array(nbrs.length * 6);
    nbrs.forEach((j, k) => {
      for (let a = 0; a < 3; a++) { pos[k * 6 + a] = target[probe * 3 + a]; pos[k * 6 + 3 + a] = target[j * 3 + a]; }
      const c = new THREE.Color(t.cls[CLASSES[d.emb.label[j]]]);
      col.set([c.r, c.g, c.b, c.r, c.g, c.b], k * 6);
    });
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return g;
  }, [target, probe, nbrs, d, t]);
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial vertexColors transparent opacity={0.9} />
    </lineSegments>
  );
}

function Rig({ auto }: { auto: boolean }) {
  const { camera, gl } = useThree();
  const ctl = useMemo(() => {
    const c = new OrbitControls(camera, gl.domElement);
    c.enableDamping = true;
    c.autoRotateSpeed = 0.8;
    c.enablePan = false;
    c.minDistance = 3;
    c.maxDistance = 14;
    return c;
  }, [camera, gl]);
  ctl.autoRotate = auto;
  useEffect(() => () => ctl.dispose(), [ctl]);
  useR3fFrame(() => ctl.update());
  return null;
}

function Axes({ color }: { color: string }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-3, 0, 0, 3, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, -3, 0, 0, 3]), 3));
    return g;
  }, []);
  return <lineSegments geometry={geo}><lineBasicMaterial color={color} transparent opacity={0.35} /></lineSegments>;
}

function FisherBars({ train, heldout }: { train: number; heldout: number }) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(300);
  const x = scaleLog().domain([0.5, 1000]).range([84, w - 46]).clamp(true);
  const rows: [string, number, string][] = [["training fold", train, t.series[0]], ["held-out", heldout, t.series[1]]];
  return (
    <div ref={ref}>
      <svg width={w} height={70} className="chart" role="img" aria-label={`Fisher J: training ${fmt(train, 1)}, held-out ${fmt(heldout, 1)}`}>
        {rows.map(([lab, v, c], i) => (
          <g key={lab}>
            <text x={78} y={20 + i * 30 + 4} textAnchor="end" className="lbl">{lab}</text>
            <HBar x0={x(0.5)} x1={x(Math.max(0.5, v))} y={10 + i * 30} h={18} fill={c} />
            <text x={x(Math.max(0.5, v)) + 6} y={20 + i * 30 + 4} className="lbl-strong">{v.toFixed(1)}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export function FeatureSpace() {
  const d = useData();
  const { t } = useTheme();
  const reduced = useReducedMotion();
  const [proj, setProj] = useState<Proj>("lda5");
  const [fold, setFold] = useState(0);
  const [view, setView] = useState<View>("test");
  const [visible, setVisible] = useState<Record<Cls, boolean>>({ H: true, B1: true, B2: true, B3: true, V: true, R: true });
  const [auto, setAuto] = useState(true);
  const [probe, setProbe] = useState<number | null>(null);
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const [knn, setKnn] = useState<{ flags: Uint8Array; done: number; total: number; f1: number; recall: number[]; correct: number } | null>(null);
  const [ref, inView] = useInView<HTMLDivElement>("0px");
  const target = useDisplay(proj, fold);
  const fisher = d.emb.fisher.find((f) => f.proj === proj && f.fold === fold)!;
  const Z = d.emb.Z[proj][fold];
  const train = useMemo(() => {
    const idx: number[] = [];
    for (let i = 0; i < d.emb.n; i++) if (d.emb.recFold[i] !== fold) idx.push(i);
    return Int32Array.from(idx);
  }, [d, fold]);
  const nbrs = useMemo(() => (probe === null ? [] : nearest(Z, 5, train.filter((j) => j !== probe), probe, 5)), [Z, train, probe]);
  const runId = useRef(0);

  useEffect(() => { setKnn(null); setProbe(null); runId.current++; }, [proj, fold]);

  const runKnn = () => {
    const id = ++runId.current;
    const test: number[] = [];
    for (let i = 0; i < d.emb.n; i++) if (d.emb.recFold[i] === fold) test.push(i);
    const flags = new Uint8Array(d.emb.n);
    const yT: number[] = [], yP: number[] = [];
    let k = 0;
    const per = reduced ? test.length : 14;
    const step = () => {
      if (id !== runId.current) return;
      for (let s = 0; s < per && k < test.length; s++, k++) {
        const q = test[k];
        const p = vote(nearest(Z, 5, train, q, 5).map((j) => d.emb.label[j]), CLASSES);
        flags[q] = p === d.emb.label[q] ? 1 : 2;
        yT.push(d.emb.label[q]); yP.push(p);
      }
      const { f1, recall } = macroF1(yT, yP);
      setKnn({ flags: flags.slice(), done: k, total: test.length, f1, recall, correct: yT.filter((v, i) => v === yP[i]).length });
      if (k < test.length) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  const probeInfo = probe !== null && (() => {
    const rec = d.emb.recIds[d.emb.rec[probe]];
    const pred = vote(nbrs.map((j) => d.emb.label[j]), CLASSES);
    return { rec, win: d.emb.win[probe], truth: CLASSES[d.emb.label[probe]], pred: CLASSES[pred], test: d.emb.recFold[probe] === fold };
  })();
  const baseline = d.metrics.find((m) => m.split === "grouped" && m.front === proj && m.clf === "knn_euc")!;

  return (
    <Section id="space" eyebrow="Feature space" tags={["5a", "projections"]}
      title="6,264 windows, three ways to squeeze 126 features into 5"
      lede={<>Each point is one 10-second window, projected by a front-end fitted on the <b>training recordings</b> of one fold.
        The first three of its five dimensions are drawn. Switch projection and the cloud morphs. Turn on <b>held-out</b> to see
        where unseen recordings land. LDA pulls the training windows into tight class clusters, but the held-out windows land
        elsewhere.</>}>
      <div className="split">
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div className="controls" style={{ padding: "16px 16px 0" }}>
            <Seg value={proj} onChange={setProj} options={(["pca5", "lda5", "kpca5"] as Proj[]).map((p) => ({ value: p, label: PROJ_LABEL[p] }))} />
            <Seg label="Fold" value={fold} onChange={setFold} options={[0, 1, 2, 3, 4, 5].map((f) => ({ value: f, label: f + 1 }))} />
          </div>
          <div className="controls" style={{ padding: "0 16px" }}>
            <Seg value={view} onChange={setView} options={[{ value: "all", label: "All windows" }, { value: "test", label: "Highlight held-out" }, { value: "testonly", label: "Held-out only" }]} />
            <button className="btn ghost" onClick={() => setAuto(!auto)}>{auto ? "Stop rotation" : "Rotate"}</button>
          </div>
          <div ref={ref} style={{ height: 520, position: "relative", cursor: hover ? "pointer" : "grab" }}>
            <Canvas frameloop={inView ? "always" : "demand"} resize={{ offsetSize: true }} dpr={[1, 1.6]} camera={{ position: [4.4, 2.9, 4.8], fov: 42 }}
              raycaster={{ params: { Points: { threshold: 0.07 } } } as never} gl={{ antialias: true, powerPreference: "low-power" }}
              onPointerMissed={() => setProbe(null)} aria-label={`3-D scatter of windows projected by ${PROJ_LABEL[proj]}, fold ${fold + 1}`}>
              <color attach="background" args={[t.mode === "dark" ? "#0e0f12" : "#f6f6f3"]} />
              <Axes color={t.muted} />
              <Points target={target} fold={fold} view={view} visible={visible} flags={knn?.flags ?? null} probe={probe}
                onPick={setProbe} onHover={(i, x, y) => setHover((p) => (i === null ? null : p && p.i === i ? p : { i, x, y }))} />
              {probe !== null && nbrs.length > 0 && <Links target={target} probe={probe} nbrs={nbrs} />}
              <Rig auto={auto && !reduced} />
            </Canvas>
            {hover && (
              <div className="tip" style={{ left: Math.min(hover.x + 14, 9999), top: hover.y + 10 }}>
                <b style={{ display: "flex", gap: 6, alignItems: "center" }}><ClassMark c={CLASSES[d.emb.label[hover.i]]} />{prettyId(d.emb.recIds[d.emb.rec[hover.i]])}</b>
                <div className="k">window {d.emb.win[hover.i]} · {d.emb.recFold[hover.i] === fold ? "held-out" : "training"}</div>
                <div className="k">click to show its 5 nearest training windows</div>
              </div>
            )}
            <div style={{ position: "absolute", left: 16, bottom: 14, right: 16 }}>
              <ClassLegend visible={visible} onToggle={(c) => setVisible({ ...visible, [c]: !visible[c] })} />
            </div>
          </div>
        </div>
        <div className="stack">
          <div className="card">
            <h3>Fisher J, fold {fold + 1}</h3>
            <p className="sub">Between-class over within-class scatter in the 5-D space (log scale). Held-out J uses the spread of unseen windows around the training class means.</p>
            <FisherBars train={fisher.train} heldout={fisher.heldout} />
            <p className="sub" style={{ marginTop: 6, color: "var(--ink2)" }}>
              {proj === "lda5"
                ? `LDA maximises J on the training fold (${fisher.train.toFixed(0)}), and held-out it falls to ${fisher.heldout.toFixed(1)}. It mostly learns which recording a window came from.`
                : `${PROJ_LABEL[proj]} is unsupervised, so its training J is no better than its held-out J.`}
            </p>
          </div>
          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
              <h3>Live 5-NN on the held-out recordings</h3>
              <button className="btn" onClick={runKnn}>▶ Classify fold {fold + 1}</button>
            </div>
            <p className="sub">Runs in your browser: Euclidean 5-NN in the 5-D {PROJ_LABEL[proj]} space, the same model as the baseline's k-NN on this front-end.</p>
            {knn ? (
              <>
                <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
                  <span style={{ fontSize: 44, fontWeight: 740, letterSpacing: "-0.03em" }}><Counter value={knn.f1} duration={0.3} /></span>
                  <span className="muted">macro-F1 · {knn.done}/{knn.total} windows</span>
                </div>
                <div style={{ display: "flex", gap: 10, margin: "4px 0 10px", flexWrap: "wrap" }}>
                  <span className="badge"><span style={{ color: t.good }}>✓</span> {knn.correct} correct</span>
                  <span className="badge"><span style={{ color: t.bad }}>✗</span> {knn.done - knn.correct} wrong (ringed in the plot)</span>
                </div>
                <RecallBars recall={knn.recall} />
                <p className="sub" style={{ marginTop: 6 }}>Baseline mean over all 6 folds for this model: {baseline.macro_f1.toFixed(2)} ± {baseline.macro_f1_std.toFixed(2)}.</p>
              </>
            ) : <p className="sub">Each held-out window lights up as it is classified.</p>}
          </div>
          {probeInfo && (
            <div className="card">
              <h3>Selected window</h3>
              <dl className="kv" style={{ marginTop: 8 }}>
                <dt>Recording</dt><dd>{prettyId(probeInfo.rec)} · window {probeInfo.win}</dd>
                <dt>Role</dt><dd>{probeInfo.test ? "held-out" : "training"} in fold {fold + 1}</dd>
                <dt>True class</dt><dd style={{ display: "flex", gap: 6, alignItems: "center" }}><ClassMark c={probeInfo.truth} />{probeInfo.truth}</dd>
                <dt>5-NN vote</dt><dd style={{ display: "flex", gap: 6, alignItems: "center" }}><ClassMark c={probeInfo.pred} />{probeInfo.pred} {probeInfo.pred === probeInfo.truth ? <span style={{ color: t.good }}>✓</span> : <span style={{ color: t.bad }}>✗</span>}</dd>
              </dl>
              <div style={{ marginTop: 10, display: "grid", gap: 4 }}>
                {nbrs.map((j) => (
                  <div key={j} className="mono" style={{ fontSize: 12.5, display: "flex", gap: 6, alignItems: "center" }}>
                    <ClassMark c={CLASSES[d.emb.label[j]]} size={10} />{prettyId(d.emb.recIds[d.emb.rec[j]])} <span className="muted">w{d.emb.win[j]}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}

function RecallBars({ recall }: { recall: number[] }) {
  const { t } = useTheme();
  const [ref, w] = useWidth<HTMLDivElement>(300);
  const bw = w - 40 - 44;
  return (
    <div ref={ref}>
      <svg width={w} height={CLASSES.length * 22 + 4} className="chart" role="img" aria-label="Recall per class">
        {CLASSES.map((c, i) => (
          <g key={c}>
            <text x={30} y={i * 22 + 15} textAnchor="end" className="lbl">{c}</text>
            <rect x={36} y={i * 22 + 4} width={bw} height={14} rx={3} fill={t.grid} opacity={0.5} />
            <HBar x0={36} x1={36 + bw * recall[i]} y={i * 22 + 4} h={14} fill={t.cls[c]} />
            <text x={36 + bw + 6} y={i * 22 + 15} className="lbl">{recall[i].toFixed(2)}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
