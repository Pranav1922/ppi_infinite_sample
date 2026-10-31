"""PPI Finite-Sample Lab — dashboard entry point and navigation.

    streamlit run dashboard/app.py

A **presentation layer only**. It reads saved artefacts under `results/` and `data/` and writes
nothing: no result CSV, config, seed, figure or manuscript file is modified by anything here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from components.ui import CSS, sidebar_footer  # noqa: E402

PAGES = Path(__file__).resolve().parent / "pages"


def build() -> "st.navigation":
    return st.navigation({
        "RESEARCH": [
            st.Page(PAGES / "home.py", title="Home", icon=":material/home:", default=True),
            st.Page(PAGES / "overview.py", title="Overview", icon=":material/menu_book:"),
            st.Page(PAGES / "method_comparison.py", title="Method Comparison", icon=":material/compare_arrows:"),
            st.Page(PAGES / "finite_n_explorer.py", title="Finite-N Explorer", icon=":material/tune:"),
            st.Page(PAGES / "coverage_boundary.py", title="Coverage Boundary", icon=":material/insights:"),
        ],
        "APPLICATION": [
            st.Page(PAGES / "galaxy_zoo.py", title="Galaxy Zoo", icon=":material/auto_awesome:"),
        ],
        "ABOUT": [
            st.Page(PAGES / "methodology.py", title="Methodology", icon=":material/science:"),
            st.Page(PAGES / "reproducibility.py", title="Reproducibility", icon=":material/verified:"),
        ],
    })


def main() -> None:
    st.set_page_config(page_title="PPI Finite-Sample Lab", layout="wide", page_icon=":material/analytics:",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    nav = build()
    sidebar_footer()
    nav.run()


if __name__ == "__main__":
    main()
