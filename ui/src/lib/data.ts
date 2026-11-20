// Types and decoding for public/data/bundle.json (written by scripts/export_ui_data.py).

export const CLASSES = ["H", "B1", "B2", "B3", "V", "R"] as const;
export type Cls = (typeof CLASSES)[number];
export const PROJ = ["pca5", "lda5", "kpca5"] as const;
export type Proj = (typeof PROJ)[number];

export const FRONT_LABEL: Record<string, string> = {
  raw: "Raw 126-D",
  "pca0.95": "PCA 95% var",
  pca5: "PCA-5",
  lda5: "LDA-5",
  kpca5: "Kernel PCA-5",
};
export const CLF_LABEL: Record<string, string> = {
  knn_euc: "k-NN Euclidean",
  knn_mah: "k-NN Mahalanobis",
  svm_lin: "Linear SVM",
  svm_rbf: "RBF SVM",
  rf: "Random forest",
  gnb: "Gaussian NB",
};
export const FRONTS = ["raw", "pca0.95", "pca5", "lda5", "kpca5"];
export const CLFS = ["knn_euc", "knn_mah", "svm_lin", "svm_rbf", "rf", "gnb"];

export interface Recording {
  id: string;
  label: Cls;
  load: number;
  supply: number;
  n_windows: number;
  f1: number;
  slip: number;
  f_pole: number;
  fold: number;
  cond: number;
}
export interface MetricRow {
  split: string;
  front: string;
  clf: string;
  acc: number;
  acc_std: number;
  macro_f1: number;
  macro_f1_std: number;
}
export interface Peak {
  rec_id: string;
  label: Cls;
  freq: number;
  order: number;
  level_db: number;
  prominence_db: number;
  source: string | null;
  category: string;
  over_h_db: number | null;
}
type Row = Record<string, any>;

interface RawBundle {
  meta: {
    classes: Cls[];
    class_names: Record<Cls, string>;
    fs: number;
    window: number;
    step: number;
    n_windows: number;
    n_features: number;
    n_recordings: number;
    seed: number;
    bpfo_order: number;
    bpfi_order: number;
    generated: string;
    citation: { doi: string; licence: string; citation: string };
  };
  recordings: Recording[];
  signals: { start: number; n: number; channels: string[]; data: Record<string, { b64: string; scale: number }> };
  spectra: { df: number; n_bins: number; data: Record<string, string> };
  peaks: Peak[];
  embedding: {
    n: number;
    dims: number;
    label: string;
    rec: string;
    rec_ids: string[];
    win: string;
    proj: Record<Proj, { scale: number[]; b64: string }[]>;
    fisher: { proj: Proj; fold: number; train: number; heldout: number }[];
  };
  metrics: MetricRow[];
  per_class: Row[];
  confusion: Record<string, number[][]>;
  separability: Row[];
  fisher_vs_k: Row[];
  pairs: Row[];
  pair_top: Record<string, Row[]>;
  class_vs_rest: Record<string, Row[]>;
  feature_families: Row[];
  sidebands: Row[];
  sideband_ranks: Row[];
  regions: Row[];
  dtw: { summary: Row[]; folds: Row[]; confusion: Record<string, number[][]> | null };
  tuning: { summary: Row[]; folds: Row[] };
  corruption: { both: Row[] | null; test: Row[] | null };
}

export interface AppData extends Omit<RawBundle, "signals" | "spectra" | "embedding"> {
  recById: Map<string, Recording>;
  /** 3 channels (gUserX/Y/Z) in g, DC removed. */
  signal: Map<string, Float32Array[]>;
  signalStart: number;
  /** whole-recording Welch PSD in dB, bin spacing specDf. */
  spectrum: Map<string, Float32Array>;
  specDf: number;
  emb: {
    n: number;
    label: Uint8Array; // class index
    rec: Uint8Array; // index into recIds
    win: Uint16Array;
    recIds: string[];
    recFold: Uint8Array; // grouped fold of each window's recording
    /** per projection, per fold: n x 5 row-major */
    Z: Record<Proj, Float32Array[]>;
    fisher: RawBundle["embedding"]["fisher"];
  };
}

function bytes(b64: string): Uint8Array {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}
function int16(b64: string): Int16Array {
  const b = bytes(b64);
  return new Int16Array(b.buffer, b.byteOffset, b.byteLength / 2);
}

export function decode(raw: RawBundle): AppData {
  const signal = new Map<string, Float32Array[]>();
  for (const [id, { b64, scale }] of Object.entries(raw.signals.data)) {
    const q = int16(b64);
    const n = raw.signals.n;
    const chans: Float32Array[] = [];
    for (let c = 0; c < 3; c++) {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = (q[c * n + i] / 32767) * scale;
      chans.push(a);
    }
    signal.set(id, chans);
  }
  const spectrum = new Map<string, Float32Array>();
  for (const [id, b64] of Object.entries(raw.spectra.data)) {
    const q = int16(b64);
    const a = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) a[i] = q[i] / 100;
    spectrum.set(id, a);
  }
  const e = raw.embedding;
  const label = bytes(e.label);
  const rec = bytes(e.rec);
  const winB = bytes(e.win);
  const win = new Uint16Array(winB.buffer, winB.byteOffset, winB.byteLength / 2);
  const recById = new Map(raw.recordings.map((r) => [r.id, r]));
  const recFold = new Uint8Array(e.n);
  for (let i = 0; i < e.n; i++) recFold[i] = recById.get(e.rec_ids[rec[i]])!.fold;
  const Z = {} as Record<Proj, Float32Array[]>;
  for (const p of PROJ) {
    Z[p] = e.proj[p].map(({ scale, b64 }) => {
      const q = int16(b64);
      const a = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) a[i] = (q[i] / 32767) * scale[i % 5];
      return a;
    });
  }
  const { signals: _s, spectra: _p, embedding: _e, ...rest } = raw;
  return {
    ...rest,
    recById,
    signal,
    signalStart: raw.signals.start,
    spectrum,
    specDf: raw.spectra.df,
    emb: { n: e.n, label, rec, win, recIds: e.rec_ids, recFold, Z, fisher: e.fisher },
  };
}

/** Fetch the bundle with download progress (0..1). */
export async function loadBundle(onProgress: (p: number) => void): Promise<AppData> {
  const res = await fetch("data/bundle.json");
  if (!res.ok) throw new Error(`data/bundle.json: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 3.2e6;
  if (!res.body) return decode(await res.json());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(0.98, got / total));
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) {
    buf.set(c, o);
    o += c.length;
  }
  const raw = JSON.parse(new TextDecoder().decode(buf)) as RawBundle;
  onProgress(1);
  return decode(raw);
}

export const COND_LABEL = (load: number, supply: number) => `${supply} Hz · ${load ? "loaded" : "unloaded"}`;
export const recId = (c: Cls, load: number, supply: number) => `${c}_${load}_${supply}Hz`;
export const CONDITIONS: { load: number; supply: number }[] = [
  { load: 0, supply: 30 },
  { load: 0, supply: 40 },
  { load: 0, supply: 50 },
  { load: 1, supply: 30 },
  { load: 1, supply: 40 },
  { load: 1, supply: 50 },
];

/** "gUserX_kurt" -> "gUserX · kurtosis" */
const FEAT_NAMES: Record<string, string> = {
  rms: "RMS", var: "variance", skew: "skewness", kurt: "kurtosis", crest: "crest factor",
  p2p: "peak-to-peak", shape: "shape factor", impulse: "impulse factor", e1x: "1× energy",
  e2x: "2× energy", sideband: "sideband energy", band_lo: "band < 0.5 f1",
  band_1x: "band ≈ 1×", band_2x: "band ≈ 2×", band_hi: "band > 2.5 f1",
  centroid: "spectral centroid", spread: "spectral spread", entropy: "spectral entropy",
  sk_max: "spectral kurtosis max", sk_mean: "spectral kurtosis mean", slip: "slip",
};
export const featFamily = (t: string) => FEAT_NAMES[t] ?? t;
export function featLabel(name: string) {
  const i = name.indexOf("_");
  const ch = name.slice(0, i);
  return `${ch} · ${featFamily(name.slice(i + 1))}`;
}

/** "B2_1_50Hz" -> "B2 · 50 Hz · loaded" */
export function prettyId(id: string) {
  const [c, l, s] = id.split("_");
  return `${c} · ${s.replace("Hz", " Hz")} · ${l === "1" ? "loaded" : "unloaded"}`;
}
