# Presentation

| File | What it is |
|---|---|
| `Motor_Fault_Slides.html` | 11-slide deck in a single self-contained HTML file. It works offline. |
| `tools/deck_template.html`, `tools/build_deck.py` | The deck's source, and the script that inlines the report figures into it. |

All numbers come from the final 5-seed run (`results/multiseed/`), the same as the report. The separability, tuning and DTW numbers are from seed 0, as in the report.

## Presenting

Open `Motor_Fault_Slides.html` in Chrome. The keys are:

- ← → (or space): change slide
- **N**: speaker notes
- **F**: fullscreen
- **D**: switch between light and dark

`#5` at the end of the URL opens slide 5. To get a PDF, print from Chrome: each slide is one 16:9 page.

The slides are:

1. Title
2. The problem: the 50 Hz Nyquist limit
3. Dataset
4. Method and protocol
5. Leakage: 1.00 → 0.47
6. Model grid and the LDA illusion
7. What survives at 100 Hz
8. Sensor corruption
9. Unseen speed and load
10. Kernel PCA, tuning and DTW
11. Conclusions and demo

The live demo is the UI in `ui/`. Run `cd ui && npm run dev`, then open http://localhost:5173.

## Rebuilding the deck

Edit `tools/deck_template.html`. Then render the three report figures and inline them:

```bash
mkdir -p /tmp/deckfigs
for f in fig_projection fig_corruption fig_hyper; do
  pdftoppm -r 220 -png -singlefile report/figures/$f.pdf /tmp/deckfigs/$f
done
python presentation/tools/build_deck.py /tmp/deckfigs presentation/Motor_Fault_Slides.html
```
