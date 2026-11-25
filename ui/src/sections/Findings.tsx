import { motion } from "framer-motion";
import { useData } from "../lib/hooks";
import { Section } from "../components/ui";

export function Findings() {
  const d = useData();
  const g = d.metrics.filter((m) => m.split === "grouped");
  const best = g.reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
  const naive = Math.max(...d.metrics.filter((m) => m.split === "naive").map((m) => m.macro_f1));
  const lda = d.separability.find((s) => s.front === "lda5")!;
  const dtw1 = d.dtw.summary?.find((r) => r.distance === "dtw" && r.k === 1);
  const now1 = d.dtw.summary?.find((r) => r.distance === "nowarp" && r.k === 1);
  const b1 = d.pairs.filter((p) => p.class_a === "B1" || p.class_b === "B1").map((p) => p.j_heldout);
  const cards = [
    { big: `${naive.toFixed(2)} → ${best.macro_f1.toFixed(2)}`, h: "Leakage is the first result", p: "A random split over overlapping windows scores near-perfect. Holding out whole recordings, the best model (LDA-5 + random forest) reaches 0.49 macro-F1, about three times chance." },
    { big: `${Math.round(lda.fisher_train)} → ${lda.fisher_heldout.toFixed(0)}`, h: "LDA learns the recording", p: "LDA's Fisher J on its own training fold is huge. On unseen recordings it falls to the level of PCA. Most of what it separates is recording identity." },
    { big: `J ${Math.min(...b1).toFixed(1)}–${Math.max(...b1).toFixed(1)}`, h: "Lubrication loss survives", p: "B1 separates from all five other classes on held-out recordings, through impulsiveness (kurtosis, peak-to-peak, impulse factor), with no defect frequency needed." },
    { big: "f1 + f_p = alias(3f1)", h: "Rotor bars hide in the fold", p: "At 50 Hz supply the rotor-bar sidebands coincide exactly with aliased running-speed harmonics, and R's sidebands are never the strongest at their condition. That is the R–B2 confusion." },
    { big: dtw1 && now1 ? `${dtw1.macro_f1.toFixed(2)} vs ${now1.macro_f1.toFixed(2)}` : "DTW", h: "Warping adds nothing", p: "On steady-state windows DTW k-NN scores the same as no warping. The envelope does see R that the features miss, but alignment isn't the reason." },
    { big: "±0.04", h: "Tuning isn't the bottleneck", p: "Nested grouped tuning moves the linear front-ends by −0.03 to +0.04, and Kernel PCA stays below them. The limit is the shift between recordings, since each class is recorded once per condition." },
  ];
  return (
    <Section id="findings" eyebrow="What we found" tags={["summary"]}
      title="The boundary, not the accuracy"
      lede={<>A 100 Hz sensor keeps some fault structure and erases the rest. These are the results that carry the report. All are
        single-seed (seed 0) until the multi-seed rerun lands.</>}>
      <div className="grid3">
        {cards.map((c, i) => (
          <motion.div key={c.h} className="card finding" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }} transition={{ duration: 0.6, delay: i * 0.08 }}>
            <div className="big num">{c.big}</div>
            <h3>{c.h}</h3>
            <p>{c.p}</p>
          </motion.div>
        ))}
      </div>
    </Section>
  );
}
