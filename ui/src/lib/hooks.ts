import { createContext, useContext, useEffect, useRef, useState, type RefObject } from "react";
import type { AppData } from "./data";

export const DataCtx = createContext<AppData | null>(null);
export function useData(): AppData {
  const d = useContext(DataCtx);
  if (!d) throw new Error("data not loaded");
  return d;
}

/** True while the element is on screen (animations pause otherwise, to spare the CPU/GPU). */
export function useInView<T extends Element>(margin = "120px"): [RefObject<T>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { rootMargin: margin });
    io.observe(el);
    return () => io.disconnect();
  }, [margin]);
  return [ref, inView];
}

/** requestAnimationFrame loop with dt in seconds; runs only while `active`. */
export function useFrame(cb: (dt: number, t: number) => void, active: boolean) {
  const cbRef = useRef(cb);
  cbRef.current = cb;
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const t0 = last;
    if (!active) {
      // paint one still frame so an off-screen canvas is never blank
      raf = requestAnimationFrame((now) => cbRef.current(0, (now - t0) / 1000));
      return () => cancelAnimationFrame(raf);
    }
    const loop = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      cbRef.current(dt, (now - t0) / 1000);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [active]);
}

export function useReducedMotion() {
  const [r, setR] = useState(() => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);
  useEffect(() => {
    const m = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!m) return;
    const on = () => setR(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return r;
}

/** Element width via ResizeObserver. */
export function useWidth<T extends Element>(initial = 600): [RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/** Canvas sized to its CSS box at devicePixelRatio (capped at 2). */
export function setupCanvas(c: HTMLCanvasElement, w: number, h: number) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
  }
  const ctx = c.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");
