import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Cls } from "./data";

export type Mode = "dark" | "light";

// Categorical slots 1-6 of the dataviz reference palette, validated (adjacent pairs) for
// both modes; classes keep the same slot everywhere. Light-mode aqua/yellow/magenta are
// below 3:1 on the surface, so every chart that uses them is directly labelled.
const CLASS_HEX: Record<Mode, Record<Cls, string>> = {
  dark: { H: "#3987e5", B1: "#d95926", B2: "#199e70", B3: "#c98500", V: "#d55181", R: "#008300" },
  light: { H: "#2a78d6", B1: "#eb6834", B2: "#1baf7a", B3: "#eda100", V: "#e87ba4", R: "#008300" },
};
// Method comparisons (never on the same chart as classes): first three slots, which
// validate all-pairs in both modes.
const SERIES_HEX: Record<Mode, string[]> = {
  dark: ["#3987e5", "#d95926", "#199e70"],
  light: ["#2a78d6", "#eb6834", "#1baf7a"],
};
// Sequential blue ramp 100..700. On dark, low values recede toward the dark surface.
const BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
  "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

export interface Tokens {
  mode: Mode;
  cls: Record<Cls, string>;
  series: string[];
  page: string;
  surface: string;
  raised: string;
  ink: string;
  ink2: string;
  muted: string;
  grid: string;
  axis: string;
  accent: string;
  good: string;
  bad: string;
  seq: (t: number) => string;
}

function hexToRgb(h: string) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function lerpHex(a: string, b: string, t: number) {
  const A = hexToRgb(a);
  const B = hexToRgb(b);
  const c = A.map((v, i) => Math.round(v + (B[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function ramp(stops: string[]) {
  return (t: number) => {
    const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    return lerpHex(stops[i], stops[i + 1], x - i);
  };
}
/** Relative luminance, to pick white or near-black text inside a filled cell. */
export function inkOn(color: string) {
  const m = color.match(/\d+/g);
  const [r, g, b] = (color.startsWith("#") ? hexToRgb(color) : (m ?? []).map(Number)).map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.32 ? "#0b0b0b" : "#ffffff";
}

export const TOKENS: Record<Mode, Tokens> = {
  dark: {
    mode: "dark",
    cls: CLASS_HEX.dark,
    series: SERIES_HEX.dark,
    page: "#0b0c0e",
    surface: "#141518",
    raised: "#1c1e22",
    ink: "#f0efec",
    ink2: "#c3c2b7",
    muted: "#8f8d86",
    grid: "#2a2b2e",
    axis: "#3a3b3e",
    accent: "#9085e9",
    good: "#0ca30c",
    bad: "#d03b3b",
    seq: ramp([...BLUE].reverse()),
  },
  light: {
    mode: "light",
    cls: CLASS_HEX.light,
    series: SERIES_HEX.light,
    page: "#f4f4f1",
    surface: "#fcfcfb",
    raised: "#ffffff",
    ink: "#0b0b0b",
    ink2: "#52514e",
    muted: "#6f6d68",
    grid: "#e1e0d9",
    axis: "#c3c2b7",
    accent: "#6250d6",
    good: "#0ca30c",
    bad: "#d03b3b",
    seq: ramp(BLUE),
  },
};

const Ctx = createContext<{ t: Tokens; toggle: () => void }>({ t: TOKENS.dark, toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>(() => {
    try {
      const s = localStorage.getItem("motorpr-theme");
      if (s === "light" || s === "dark") return s;
    } catch {
      /* storage unavailable */
    }
    return "dark";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = mode;
    try {
      localStorage.setItem("motorpr-theme", mode);
    } catch {
      /* storage unavailable */
    }
  }, [mode]);
  const value = useMemo(
    () => ({ t: TOKENS[mode], toggle: () => setMode((m) => (m === "dark" ? "light" : "dark")) }),
    [mode],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
export const useTheme = () => useContext(Ctx);

/** Marker shape per class: the secondary channel for scatter plots (identity never by color alone). */
export const CLASS_SHAPE: Record<Cls, number> = { H: 0, B1: 1, B2: 2, B3: 3, V: 4, R: 5 };
export const SHAPE_NAME = ["circle", "square", "triangle", "diamond", "plus", "ring"];
