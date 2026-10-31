"""Reproducibility — provenance, commands, and what this dashboard may not do."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from components.loaders import RESULTS, load_results, provenance, rel
from components.ui import badge, footer, kpis, page_head, section

page_head("Reproducibility",
          "Where these numbers came from, and the guarantees this view operates under.")

p = provenance()
kpis([
    {"label": "working tree commit", "value": p["working_tree_commit"]},
    {"label": "commit recorded in results", "value": p["results_commit"],
     "sub": "the code that produced them", "tone": "accent"},
    {"label": "result files available", "value": f"{len(sorted(RESULTS.glob('*.csv')))}"},
])

section("The read-only guarantee",
        "This dashboard is a presentation layer. It reads; it never writes.")
st.markdown(f"""
{badge('live')} &nbsp; Nothing on these pages modifies a result CSV, a config, a seed, a figure or
a manuscript file. Two tests enforce it rather than assert it:

1. a **static scan** rejecting write APIs anywhere under `dashboard/` — `to_csv`, `write_text`,
   `np.save`, `open(..., 'w')`, `unlink`, and others;
2. a **hash test** that SHA-256s every artefact under `results/`, `data/` and `docs/` before and
   after rendering all pages headlessly, and fails if a single byte changes.

Paths are displayed repo-relative, so no filesystem path is exposed.
""", unsafe_allow_html=True)

section("Reproducing the results")
st.code("""make reproduce   # regenerate every results/*.csv
make verify      # V1-V11 verification checks
make test        # the test suite
make manuscript  # rebuild the manuscript from its sources

streamlit run dashboard/app.py""", language="bash")

section("Saved artefacts this dashboard reads")
rows = []
for name in sorted(x.name for x in RESULTS.glob("*.csv")):
    try:
        df = load_results(name)
        rows.append({"file": f"results/{name}", "rows": f"{len(df):,}",
                     "methods": str(df.method.nunique()) if "method" in df.columns else "—"})
    except Exception:
        continue
if rows:
    st.dataframe(pd.DataFrame(rows).set_index("file"), width='stretch', height=340)

section("Two things this view is careful about")
c1, c2 = st.columns(2, gap="medium")
with c1, st.container(border=True):
    st.markdown("##### Nominal is not empirical")
    st.markdown("Every interval shown is *constructed* at a nominal 95 % level. Whether it "
                "*attains* 95 % is what the research measures, and often it does not.")
with c2, st.container(border=True):
    st.markdown("##### Saved versus live")
    st.markdown("Saved research results name the file they came from. Numbers computed on the fly "
                "— because the saved experiments store averages, not individual intervals — are "
                "marked as live illustrations and enter no table in the paper.")
footer()
