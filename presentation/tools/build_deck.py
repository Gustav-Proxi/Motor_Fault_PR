"""Build the self-contained slide deck: inline the report figures into deck_template.html.

usage: python build_deck.py <figure-png-dir> <out.html>
The PNGs are the report figures rendered at 220 dpi, e.g.
  pdftoppm -r 220 -png -singlefile report/figures/fig_projection.pdf <dir>/fig_projection
(likewise fig_corruption and fig_hyper).
"""
import base64, sys
from pathlib import Path

figs, out = Path(sys.argv[1]), Path(sys.argv[2])
html = (Path(__file__).parent / "deck_template.html").read_text()
for key, name in [("PROJ", "fig_projection"), ("CORR", "fig_corruption"), ("HYPER", "fig_hyper")]:
    b = base64.b64encode((figs / f"{name}.png").read_bytes()).decode()
    html = html.replace(f"%%{key}%%", f"data:image/png;base64,{b}")
assert "%%" not in html, "unfilled placeholder"
out.write_text(html)
print(out, round(out.stat().st_size / 1e6, 2), "MB")
