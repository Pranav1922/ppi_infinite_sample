"""Landing page — hero, headline results, and what each page contains."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from components.loaders import available
from components.ui import caveat, footer, hero, kpis, section

hero(
    "PPI Finite-Sample Lab",
    "Understanding confidence intervals when human labels are scarce.",
    eyebrow="Finite-sample prediction-powered inference",
    chips=("PPI++", "LC-PPI", "Finite-N", "Real Data"),
)

kpis([
    {"label": "PPI++ coverage · n = 10", "value": "0.878",
     "sub": "nominal 95 % · deficit 0.072", "tone": "warn"},
    {"label": "PPI++ coverage · n = 4", "value": "0.681",
     "sub": "nominal 95 % · deficit 0.269", "tone": "warn"},
    {"label": "Corrected interval · n = 4…50", "value": "0.9495–0.9505",
     "sub": "every gold-sample size tested", "tone": "good"},
    {"label": "Width cost · ρ = 0, n = 10", "value": "1.081×",
     "sub": "0.471× where the predictor is good", "tone": "accent"},
])

section("The question",
        "Prediction-powered inference buys statistical power by pairing a few gold labels with a "
        "large pool of model predictions. Its guarantee holds asymptotically in the number of gold "
        "labels — and gold labels are exactly what is scarce.")

st.markdown("""
This lab reports what the interval **actually does** at four to fifty gold labels: where it fails,
why it fails, and what repairs it. Every number shown is read from a saved result file; the pages
name the file they read.
""")

section("Where to start")
c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(border=True):
        st.markdown("#### Research")
        st.markdown("""
- **Overview** — the methods, the finite-`N` problem, the headline result
- **Method Comparison** — every method side by side on a configuration you choose
- **Finite-N Explorer** — quantiles, the proved bound, and the boundary verdict
- **Coverage Boundary** — the flagship view: where coverage falls below nominal
""")
with c2:
    with st.container(border=True):
        st.markdown("#### Application & background")
        st.markdown("""
- **Galaxy Zoo** — 16,743 real galaxy classifications, scope stated plainly
- **Methodology** — how the experiments were run and what the status labels mean
- **Reproducibility** — commands, seeds, provenance, and what this dashboard may not do
""")

caveat()

missing = [f for f in ("A_reproduction.csv", "E_finite_N.csv", "gate2_boundary.csv",
                       "I_real_data.csv") if not available(f)]
if missing:
    st.warning("Saved results not found: " + ", ".join(missing)
               + ". Run `make reproduce` in the repository first.")
footer()
