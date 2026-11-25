// Signal processing in the browser: FFT, periodogram, aliasing, the three corruptions
// (ports of src/motorpr/corruption.py), DTW and k-NN.

export const FS = 100;

/** In-place radix-2 complex FFT. re/im length must be a power of two. */
export function fft(re: Float64Array, im: Float64Array) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const ai = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k + len / 2] = re[i + k] - ar;
        im[i + k + len / 2] = im[i + k] - ai;
        re[i + k] += ar;
        im[i + k] += ai;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

const hannCache = new Map<number, Float64Array>();
function hann(n: number) {
  let w = hannCache.get(n);
  if (!w) {
    w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
    hannCache.set(n, w);
  }
  return w;
}

/** One-sided Hann periodogram (power) of a zero-mean segment, length a power of two. */
export function periodogram(x: ArrayLike<number>, start = 0, n = 1024): Float64Array {
  const w = hann(n);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[start + i];
  mean /= n;
  for (let i = 0; i < n; i++) re[i] = (x[start + i] - mean) * w[i];
  fft(re, im);
  let wss = 0;
  for (let i = 0; i < n; i++) wss += w[i] * w[i];
  const out = new Float64Array(n / 2 + 1);
  for (let k = 0; k <= n / 2; k++) {
    const p = (re[k] * re[k] + im[k] * im[k]) / (FS * wss);
    out[k] = k === 0 || k === n / 2 ? p : 2 * p;
  }
  return out;
}

/** Periodogram summed over channels, in dB. */
export function spectrumDb(chans: ArrayLike<number>[], start = 0, n = 1024): Float32Array {
  const acc = new Float64Array(n / 2 + 1);
  for (const c of chans) {
    const p = periodogram(c, start, n);
    for (let k = 0; k < acc.length; k++) acc[k] += p[k];
  }
  const out = new Float32Array(acc.length);
  for (let k = 0; k < acc.length; k++) out[k] = 10 * Math.log10(acc[k] + 1e-20);
  return out;
}

/** Apparent frequency of a tone after sampling at fs (folds into [0, fs/2]). */
export const alias = (f: number, fs = FS) => Math.abs(f - fs * Math.round(f / fs));

export interface Source {
  name: string;
  kind: "shaft" | "line" | "bearing";
  trueHz: number;
  apparentHz: number;
  aliased: boolean;
}
/** Port of analysis.candidate_sources. */
export function candidateSources(f1: number, supply: number, bpfo = 3.05, bpfi = 4.95): Source[] {
  const rows: [string, Source["kind"], number][] = [];
  for (let k = 1; k <= 8; k++) rows.push([`${k}× f1`, "shaft", k * f1]);
  for (let m = 1; m <= 4; m++) rows.push([`${m}× line`, "line", m * supply]);
  rows.push(["BPFO", "bearing", bpfo * f1], ["2×BPFO", "bearing", 2 * bpfo * f1]);
  rows.push(["BPFI", "bearing", bpfi * f1], ["2×BPFI", "bearing", 2 * bpfi * f1]);
  return rows.map(([name, kind, f]) => ({ name, kind, trueHz: f, apparentHz: alias(f), aliased: f > FS / 2 }));
}
export function explainFrequency(f: number, sources: Source[], tol = 0.25) {
  return sources
    .filter((s) => Math.abs(s.apparentHz - f) <= tol)
    .map((s) => (s.aliased ? `alias of ${s.name}` : s.name))
    .join(" / ");
}

// ------------------------------------------------------------- seeded RNG + corruption

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r: () => number) {
  const u = Math.max(r(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}
const std = (x: Float32Array) => {
  let m = 0;
  for (const v of x) m += v;
  m /= x.length;
  let s = 0;
  for (const v of x) s += (v - m) ** 2;
  return Math.sqrt(s / x.length);
};

export type Mode = "noise" | "dropout" | "gain";
export interface Corrupted {
  chans: Float32Array[];
  missing?: Uint8Array;
  gain?: [number, number][]; // per axis gain at start and end
}
/** Same definitions as corruption.py; level in [0, 0.5]. */
export function corrupt(chans: Float32Array[], mode: Mode, level: number, seed = 1): Corrupted {
  const r = mulberry32(seed);
  const n = chans[0].length;
  if (level <= 0) return { chans };
  if (mode === "noise") {
    return {
      chans: chans.map((c) => {
        const s = Math.sqrt(level) * std(c);
        const o = new Float32Array(n);
        for (let i = 0; i < n; i++) o[i] = c[i] + gauss(r) * s;
        return o;
      }),
    };
  }
  if (mode === "dropout") {
    const missing = new Uint8Array(n);
    const target = Math.floor(level * n);
    let count = 0;
    while (count < target) {
      const g = 5 + Math.floor(r() * 46);
      const s = Math.floor(r() * (n - g));
      for (let i = s; i < s + g; i++) if (!missing[i]) (missing[i] = 1), count++;
    }
    const out = chans.map((c) => {
      const o = new Float32Array(n);
      let last = -1;
      for (let i = 0; i <= n; i++) {
        if (i < n && missing[i]) continue;
        // fill the gap (last, i) by linear interpolation
        for (let j = last + 1; j < i; j++) {
          const a = last >= 0 ? c[last] : c[i];
          const b = i < n ? c[i] : c[last];
          const t = last >= 0 && i < n ? (j - last) / (i - last) : 0;
          o[j] = a + (b - a) * t;
        }
        if (i < n) o[i] = c[i];
        last = i;
      }
      return o;
    });
    return { chans: out, missing };
  }
  const gain: [number, number][] = [];
  const out = chans.map((c) => {
    const a = r() * 2 - 1;
    const b = r() * 2 - 1;
    gain.push([1 + level * a, 1 + level * b]);
    const o = new Float32Array(n);
    for (let i = 0; i < n; i++) o[i] = c[i] * (1 + level * (a + ((b - a) * i) / (n - 1)));
    return o;
  });
  return { chans: out, gain };
}

// ------------------------------------------------------------- a few window features

export function windowStats(chans: Float32Array[], start: number, n = 1024) {
  // channel 0 (gUserX), DC removed per window as in data.make_windows
  const x = chans[0];
  let m = 0;
  for (let i = 0; i < n; i++) m += x[start + i];
  m /= n;
  let s2 = 0, s4 = 0, peak = 0, sabs = 0;
  for (let i = 0; i < n; i++) {
    const v = x[start + i] - m;
    s2 += v * v;
    s4 += v ** 4;
    peak = Math.max(peak, Math.abs(v));
    sabs += Math.abs(v);
  }
  const rms = Math.sqrt(s2 / n);
  const p = periodogram(x, start, n);
  let tot = 0, cen = 0;
  for (let k = 1; k < p.length; k++) {
    tot += p[k];
    cen += p[k] * ((k * FS) / n);
  }
  return {
    rms,
    kurtosis: s4 / n / (s2 / n) ** 2,
    crest: peak / rms,
    impulse: peak / (sabs / n),
    centroid: cen / tot,
  };
}

// ------------------------------------------------------------- DTW

/** log10 frame-RMS envelope of one window: (T = n/frame) x 3, as dtw.envelope. */
export function envelope(chans: Float32Array[], start: number, n = 1024, frame = 16): Float32Array[] {
  const T = Math.floor(n / frame);
  return chans.map((c) => {
    let m = 0;
    for (let i = 0; i < n; i++) m += c[start + i];
    m /= n;
    const e = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      let s = 0;
      for (let i = 0; i < frame; i++) s += (c[start + t * frame + i] - m) ** 2;
      e[t] = Math.log10(Math.sqrt(s / frame) + 1e-12);
    }
    return e;
  });
}

export interface DtwResult {
  T: number;
  cost: Float32Array; // T x T local cost
  D: Float32Array; // T x T accumulated (Infinity outside the band)
  path: [number, number][];
  dist: number;
}
/** Multivariate DTW with a Sakoe-Chiba band (same recursion as dtw.dtw_to_many). */
export function dtw(a: Float32Array[], b: Float32Array[], band: number): DtwResult {
  const T = a[0].length;
  const cost = new Float32Array(T * T);
  for (let i = 0; i < T; i++)
    for (let j = 0; j < T; j++) {
      let s = 0;
      for (let c = 0; c < a.length; c++) s += (a[c][i] - b[c][j]) ** 2;
      cost[i * T + j] = Math.sqrt(s);
    }
  const D = new Float32Array(T * T).fill(Infinity);
  const at = (i: number, j: number) => (i < 0 || j < 0 ? (i === -1 && j === -1 ? 0 : Infinity) : D[i * T + j]);
  for (let i = 0; i < T; i++)
    for (let j = Math.max(0, i - band); j <= Math.min(T - 1, i + band); j++)
      D[i * T + j] = cost[i * T + j] + Math.min(at(i - 1, j), at(i, j - 1), at(i - 1, j - 1));
  const path: [number, number][] = [];
  let i = T - 1;
  let j = T - 1;
  path.push([i, j]);
  while (i > 0 || j > 0) {
    const opts: [number, number, number][] = [
      [i - 1, j - 1, at(i - 1, j - 1)],
      [i - 1, j, at(i - 1, j)],
      [i, j - 1, at(i, j - 1)],
    ];
    opts.sort((x, y) => x[2] - y[2]);
    [i, j] = [opts[0][0], opts[0][1]];
    path.push([i, j]);
  }
  path.reverse();
  return { T, cost, D, path, dist: D[T * T - 1] };
}

// ------------------------------------------------------------- k-NN (sklearn-compatible votes)

/** Indices of the k nearest rows of Z (n x d) among `train` to row q. */
export function nearest(Z: Float32Array, d: number, train: Int32Array, q: number, k = 5): number[] {
  const best: [number, number][] = [];
  for (const t of train) {
    let s = 0;
    for (let c = 0; c < d; c++) s += (Z[q * d + c] - Z[t * d + c]) ** 2;
    if (best.length < k || s < best[best.length - 1][0]) {
      best.push([s, t]);
      best.sort((x, y) => x[0] - y[0]);
      if (best.length > k) best.pop();
    }
  }
  return best.map((b) => b[1]);
}
/** Majority vote; ties go to the class sklearn would pick (alphabetically first label). */
export function vote(labels: number[], names: readonly string[]): number {
  const count = new Map<number, number>();
  for (const l of labels) count.set(l, (count.get(l) ?? 0) + 1);
  const top = Math.max(...count.values());
  return [...count.entries()]
    .filter(([, c]) => c === top)
    .map(([l]) => l)
    .sort((a, b) => (names[a] < names[b] ? -1 : 1))[0];
}

export function macroF1(yTrue: number[], yPred: number[], nCls = 6) {
  let f = 0;
  const recall: number[] = [];
  for (let c = 0; c < nCls; c++) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if (yPred[i] === c && yTrue[i] === c) tp++;
      else if (yPred[i] === c) fp++;
      else if (yTrue[i] === c) fn++;
    }
    f += tp ? (2 * tp) / (2 * tp + fp + fn) : 0;
    recall.push(tp + fn ? tp / (tp + fn) : 0);
  }
  return { f1: f / nCls, recall };
}
