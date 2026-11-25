import { motion, useScroll, useSpring } from "framer-motion";
import { useEffect, useState } from "react";
import { loadBundle, type AppData } from "./lib/data";
import { DataCtx } from "./lib/hooks";
import { useTheme } from "./lib/theme";
import { Hero } from "./sections/Hero";
import { Anatomy } from "./sections/Anatomy";
import { Explorer } from "./sections/Explorer";
import { Aliasing } from "./sections/Aliasing";
import { Validation } from "./sections/Validation";
import { FeatureSpace } from "./sections/FeatureSpace";
import { Separability } from "./sections/Separability";
import { Models } from "./sections/Models";
import { Robustness } from "./sections/Robustness";
import { Dtw } from "./sections/Dtw";
import { Kpca } from "./sections/Kpca";
import { Findings } from "./sections/Findings";

const NAV: [string, string][] = [
  ["anatomy", "Machine"], ["explorer", "Signals"], ["aliasing", "Aliasing"], ["validation", "Validation"],
  ["space", "Feature space"], ["separability", "Separability"], ["models", "Models"], ["robustness", "Robustness"],
  ["dtw", "DTW"], ["kpca", "Kernel PCA"], ["findings", "Findings"],
];

function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--surface)" stroke="var(--hair2)" />
      <path d="M4 16h4l3-8 4 16 4-12 3 4h6" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function TopBar() {
  const { t, toggle } = useTheme();
  const [active, setActive] = useState("");
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 30 });
  useEffect(() => {
    const io = new IntersectionObserver(
      (es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)),
      { rootMargin: "-45% 0px -50% 0px" },
    );
    NAV.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <>
      <div className="topbar">
        <a className="brand" href="#top"><Logo /> Motor·PR</a>
        <nav className="nav" aria-label="Sections">
          {NAV.map(([id, label]) => <a key={id} href={`#${id}`} className={active === id ? "on" : ""}>{label}</a>)}
        </nav>
        <button className="icon-btn" onClick={toggle} aria-label={`Switch to ${t.mode === "dark" ? "light" : "dark"} mode`} title="Theme">
          {t.mode === "dark" ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
          )}
        </button>
      </div>
      <motion.div className="progress" style={{ scaleX, width: "100%" }} />
    </>
  );
}

function Footer({ d }: { d: AppData }) {
  return (
    <footer>
      <div className="wrap" style={{ display: "grid", gap: 10 }}>
        <div style={{ color: "var(--ink)", fontWeight: 650 }}>Bandwidth-Limited Pattern Recognition</div>
        <div>CSE 455/555 Applied Pattern Recognition, Fall 2026 · Vaishak G Kumar and Sanika Vilas Najan · Track Option 2: dimensionality reduction vs. noise and out-of-distribution data.</div>
        <div>Data: {d.meta.citation.citation} Data DOI <a href={`https://doi.org/${d.meta.citation.doi}`}>{d.meta.citation.doi}</a> ({d.meta.citation.licence}).</div>
        <div>Numbers are read from the project's results files (seed {d.meta.seed}, exported {d.meta.generated}). Live demos (k-NN, corruption, DTW, spectra) run in the browser on the exported windows.</div>
      </div>
    </footer>
  );
}

export function App() {
  const [data, setData] = useState<AppData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [p, setP] = useState(0);
  useEffect(() => { loadBundle(setP).then(setData).catch((e) => setErr(String(e))); }, []);
  // deep links (#dtw, ...) once the sections exist
  useEffect(() => {
    if (!data || !location.hash) return;
    requestAnimationFrame(() => document.getElementById(location.hash.slice(1))?.scrollIntoView({ behavior: "instant" as ScrollBehavior }));
  }, [data]);

  if (err) {
    return (
      <div className="card err">
        <h3>Data bundle not found</h3>
        <p className="sub" style={{ marginTop: 8 }}>{err}</p>
        <p>From the repository root, generate it with <code>python scripts/export_ui_data.py</code>. It needs the feature cache and the results folders, so run <code>run_baseline.py</code>, <code>run_analysis.py</code>, <code>run_dtw.py</code> and <code>run_tuning.py</code> first. Then reload.</p>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="loading">
        <div>
          <Logo />
          <div className="mono muted" style={{ marginTop: 12, fontSize: 13 }}>loading 36 recordings · {Math.round(p * 100)}%</div>
          <div className="bar"><i style={{ width: `${p * 100}%` }} /></div>
        </div>
      </div>
    );
  }
  return (
    <DataCtx.Provider value={data}>
      <TopBar />
      <main>
        <Hero />
        <Anatomy />
        <Explorer />
        <Aliasing />
        <Validation />
        <FeatureSpace />
        <Separability />
        <Models />
        <Robustness />
        <Dtw />
        <Kpca />
        <Findings />
      </main>
      <Footer d={data} />
    </DataCtx.Provider>
  );
}
