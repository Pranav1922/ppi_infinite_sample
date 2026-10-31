"""Methodology — how the experiments were run, and what the status labels mean."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from components.ui import caveat, footer, kpis, page_head, section

page_head("Methodology",
          "How the numbers on these pages were produced, and how strongly each is claimed.")

section("Design")
st.markdown("""
Every simulation is Monte-Carlo over `R` replications under a bivariate Gaussian model. Within a
cell the data are generated **once** and the identical arrays are passed to every method, so all
comparisons are paired — a difference between two methods carries no Monte-Carlo noise of its own.
Methods receive arrays, never a random generator, which enforces that structurally rather than by
convention.

Seeds derive from a single base entropy through a collision-free spawn key built from **sorted**
sweep indices, so adding a value to a sweep does not change any existing cell's draws. The ground
truth is carried on the data container but projected away before any method sees it.
""")
kpis([
    {"label": "Monte-Carlo s.e. near 0.95", "value": "0.0005", "sub": "at R = 200,000"},
    {"label": "gold-sample sizes", "value": "4 – 100"},
    {"label": "preregistered boundary cells", "value": "417", "sub": "Gate 2"},
    {"label": "bound violations", "value": "0", "sub": "across those cells", "tone": "good"},
])

section("What the status labels mean",
        "The research package labels every statement by the strength of its evidence. The same "
        "vocabulary is used on these pages.")
st.markdown("""
| label | meaning |
|---|---|
| **PROVED** | a theorem with a written proof, checked in the manuscript |
| **DERIVED** | follows from a calculation, but the remainder is not rigorously controlled |
| **NUMERICALLY VERIFIED** | measured in simulation; not a proof |
| **PRIOR ART** | established elsewhere; cited, not claimed |
| **DISPROVED** | shown false by an explicit counterexample |
| **OPEN** | not settled either way |
""")

section("What is claimed, and what is not")
c1, c2 = st.columns(2, gap="medium")
with c1, st.container(border=True):
    st.markdown("##### Claimed here")
    st.markdown("""
- an explicit, non-asymptotic bound on the gap between the deployed interval and its classical
  idealisation
- the identification of the finite-pool failure mechanism, and the region where it applies
- the systematic measurement programme behind both
""")
with c2, st.container(border=True):
    st.markdown("##### Not claimed — prior art")
    st.markdown("""
- the PPI / GREG equivalence (Mozer, 2026)
- Behrens–Fisher conservativeness (Mickey & Brown, 1966)
- the `t_min` critical value (Hsu, 1938)
- its optimality within that class (Wang & Liu, 2026)
- Welch–Satterthwaite degrees of freedom
""")

section("Known limitations")
st.markdown("""
- The theory is Gaussian. Outside that model there are measurements, not theorems.
- Global conservativeness of the deployed interval is **disproved**; for the safeguard it is
  **open** — no violation was seen in the cells tested, which is not a proof.
- The bound's constant is not uniform in predictor quality: it grows without bound as `ρ² → 1`.
- Severe right-skew degrades the corrected and classical intervals together.
""")
caveat()
footer()
