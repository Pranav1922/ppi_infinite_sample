"""Smoke tests for the read-only research dashboard.

These check that the dashboard *loads* and *reads*, and — most importantly — that running it
cannot modify a research artefact.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
sys.path.insert(0, str(DASH))

pytest.importorskip("streamlit")
pytest.importorskip("altair")

PAGES = sorted((DASH / "pages").glob("*.py"))


# --------------------------------------------------------------------------- structure

def test_expected_layout_exists():
    assert (DASH / "app.py").is_file()
    assert (DASH / "README.md").is_file()
    assert (DASH / "components" / "loaders.py").is_file()
    assert (DASH / "components" / "ui.py").is_file()
    names = {p.name for p in PAGES}
    assert names == {"home.py", "overview.py", "method_comparison.py", "finite_n_explorer.py",
                     "coverage_boundary.py", "galaxy_zoo.py", "methodology.py",
                     "reproducibility.py"}, sorted(names)


def test_navigation_groups_match_the_specified_sections():
    import app
    src = (DASH / "app.py").read_text()
    for group in ("RESEARCH", "APPLICATION", "ABOUT"):
        assert f'"{group}"' in src, group
    for page in ("home.py", "overview.py", "method_comparison.py", "finite_n_explorer.py",
                 "coverage_boundary.py", "galaxy_zoo.py", "methodology.py",
                 "reproducibility.py"):
        assert page in src, page
    assert callable(app.build)


def test_design_system_exposes_the_status_badges():
    from components.ui import BADGE_KINDS, badge
    assert set(BADGE_KINDS) >= {"empirical", "theory", "boundary", "realdata", "live"}
    for kind in BADGE_KINDS:
        html = badge(kind)
        assert html.startswith("<span class='badge'") and "</span>" in html


def test_boundary_diagram_renders_both_regimes():
    from components.ui import boundary_diagram
    import streamlit as st
    captured = []
    real = st.markdown
    st.markdown = lambda body, **kw: captured.append(body)
    try:
        boundary_diagram(50, 25)      # boundary active
        boundary_diagram(10, 25)      # standard regime
    finally:
        st.markdown = real
    assert len(captured) == 2
    assert "finite-pool risk" in captured[0] and "standard quantile regime" in captured[0]
    assert "your N = 25" in captured[0]


def test_every_module_parses_and_app_imports():
    for f in [DASH / "app.py", *PAGES, DASH / "components" / "loaders.py",
              DASH / "components" / "ui.py"]:
        ast.parse(f.read_text()), f.name
    import app                                     # noqa: F401  - entry point imports cleanly
    assert callable(app.main)


# --------------------------------------------------------------------------- read-only guarantee

#: Method names that are *never* in-memory operations — a call to one of these writes.
BANNED_UNQUALIFIED = {
    "to_csv", "to_parquet", "to_json", "to_excel", "savefig", "unlink",
    "write_text", "write_bytes", "writelines",
}
#: Ambiguous names (``DataFrame.rename`` is in-memory, ``os.rename`` is not), banned only when
#: qualified by a filesystem module.
BANNED_QUALIFIED = {
    ("os", "remove"), ("os", "rename"), ("os", "replace"), ("os", "unlink"), ("os", "rmdir"),
    ("os", "makedirs"), ("os", "mkdir"), ("shutil", "rmtree"), ("shutil", "copy"),
    ("shutil", "move"), ("np", "save"), ("np", "savez"), ("np", "savez_compressed"),
    ("np", "savetxt"), ("numpy", "save"), ("numpy", "savez"), ("numpy", "savetxt"),
}


def _qualifier(node: ast.Attribute) -> str | None:
    return node.value.id if isinstance(node.value, ast.Name) else None


def test_no_module_calls_a_write_api():
    """Detected on the parsed syntax tree, not in the source text.

    Text matching gave false positives the moment a page *documented* the read-only guarantee by
    naming the APIs it avoids, and it could not tell ``DataFrame.rename`` (in memory) from
    ``os.rename`` (on disk). The AST distinguishes both: prose is invisible to it, and an
    ambiguous name is judged by its qualifier. The hash test below is the backstop that catches a
    write by any route at all.
    """
    for f in [DASH / "app.py", *PAGES, *(DASH / "components").glob("*.py")]:
        for node in ast.walk(ast.parse(f.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute):
                assert fn.attr not in BANNED_UNQUALIFIED, \
                    f"{f.name}:{node.lineno} calls .{fn.attr}()"
                q = _qualifier(fn)
                assert (q, fn.attr) not in BANNED_QUALIFIED, \
                    f"{f.name}:{node.lineno} calls {q}.{fn.attr}()"
            if (isinstance(fn, ast.Name) and fn.id == "open") or \
                    (isinstance(fn, ast.Attribute) and fn.attr == "open"):
                modes = [a for a in node.args[1:]] + [k.value for k in node.keywords
                                                      if k.arg == "mode"]
                for arg in modes:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        assert not set(arg.value) & set("wax+"), \
                            f"{f.name}:{node.lineno} opens for writing"


def test_running_the_loaders_leaves_every_artefact_byte_identical():
    from components.loaders import load_galaxies, load_results, provenance

    watched = sorted((ROOT / "results").glob("*.csv")) + [ROOT / "data" / "galaxies.npz"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    for name in ("A_reproduction.csv", "C_sample_size.csv", "E_finite_N.csv",
                 "gate2_boundary.csv", "gate2_quantiles.csv", "I_real_data.csv"):
        assert not load_results(name).empty
    load_galaxies()
    provenance()
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    assert before == after, "the dashboard modified a saved artefact"


def test_no_absolute_paths_are_exposed():
    from components.loaders import ROOT as LOADER_ROOT, rel
    assert rel(LOADER_ROOT / "results" / "E_finite_N.csv") == "results/E_finite_N.csv"
    assert not rel(LOADER_ROOT / "results" / "E_finite_N.csv").startswith("/")


# --------------------------------------------------------------------------- data access

def test_loading_saved_results():
    from components.loaders import load_results
    df = load_results("gate2_boundary.csv")
    for col in ("n", "N", "rho", "coverage_actual", "bound_B", "limit_pred_rho_to_1"):
        assert col in df.columns
    assert len(df) > 400


def test_method_selection_pivot():
    from components.loaders import METHOD_LABELS, load_results, pivot_metrics
    df = load_results("A_reproduction.csv")
    tbl = pivot_metrics(df, ["coverage", "width_mean"], n=10, rho=0.9)
    assert not tbl.empty and "coverage" in tbl.columns
    assert {"B1", "B3", "B9"} <= set(tbl.index)
    assert all(m in METHOD_LABELS for m in ("B1", "B3", "B9", "B9-tmin"))
    assert pivot_metrics(df, ["coverage"], n=10**6).empty      # empty selection is handled


@pytest.mark.parametrize("n,N,expect_active", [(50, 25, True), (50, 49, False),
                                               (10, 25, False), (20, 5, True)])
def test_finite_N_selection_boundary_logic(n, N, expect_active):
    """The page's boundary rule and the quantile it implies."""
    from scipy import stats
    assert bool(N < n - 1) is expect_active
    df_used = min(n - 2, N - 1)
    q_std = stats.t.ppf(0.975, n - 2)
    q_safe = stats.t.ppf(0.975, df_used)
    assert bool(q_safe > q_std) is expect_active
    if not expect_active:
        assert q_safe == q_std


def test_finite_N_selection_bound_evaluates():
    sys.path.insert(0, str(ROOT / "src"))
    from ppi_finite.theory_gate1 import coverage_bound
    b = coverage_bound(50, 0.9, 25.0, 0.05)
    assert 0.0 < b.total <= 1.0 and np.isfinite(b.C)


def test_galaxies_loading():
    from components.loaders import load_galaxies
    Y, Yhat = load_galaxies()
    assert Y.shape == Yhat.shape == (16743,)
    assert np.isfinite(Y).all() and np.isfinite(Yhat).all()
    assert 0.0 <= float(Y.mean()) <= 1.0


def test_galaxies_results_are_labelled_as_containment_not_coverage():
    """The scope caveat must be present verbatim on the Galaxies page."""
    src = (DASH / "pages" / "galaxy_zoo.py").read_text()
    from components.ui import GALAXIES_SCOPE
    assert "not superpopulation coverage" in GALAXIES_SCOPE
    assert "GALAXIES_SCOPE" in src
    assert "containment fraction" in src


def test_research_pages_carry_the_nominal_versus_empirical_caveat():
    """The ABOUT pages state it in prose; the research pages must render the callout."""
    for f in PAGES:
        if f.name == "reproducibility.py":
            assert "Nominal is not empirical" in f.read_text()
            continue
        assert "caveat(" in f.read_text(), f"{f.name} does not show the coverage caveat"


@pytest.mark.slow
def test_every_page_renders_without_error_and_writes_nothing():
    """Render the router and all pages headlessly, then prove no artefact changed."""
    from streamlit.testing.v1 import AppTest

    watched = (sorted((ROOT / "results").glob("*.csv"))
               + [ROOT / "data" / "galaxies.npz"]
               + sorted((ROOT / "docs").glob("*.md"))
               + sorted((ROOT / "configs").glob("*.toml"))
               + sorted((ROOT / "src").rglob("*.py")))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}

    for script in [DASH / "app.py", *PAGES]:
        at = AppTest.from_file(str(script), default_timeout=120).run()
        assert not at.exception, f"{script.name}: {at.exception}"
        assert len(at.markdown) > 0, f"{script.name} rendered nothing"
        assert not at.error, f"{script.name} rendered an error box"
        assert not at.warning, f"{script.name}: {[w.value for w in at.warning]}"

    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    assert before == after, "rendering the dashboard modified a saved artefact"


def test_no_declared_font_size_is_unreadably_small():
    """Guards against the redesign shrinking labels below ~11.5px."""
    import re
    from components import ui
    sizes = [float(x) for x in re.findall(r"font-size:\s*([\d.]+)rem", ui.CSS)]
    assert sizes, "no font sizes found in the design system"
    assert min(sizes) >= 0.72, f"font size {min(sizes)}rem is too small"


def test_charts_declare_an_explicit_height():
    """A chart without a height can collapse; style_chart always sets one."""
    import inspect
    from components import ui
    assert "height: int = 380" in inspect.getsource(ui.style_chart)
    for f in PAGES:
        src = f.read_text()
        assert src.count("st.altair_chart(") == src.count("style_chart("), \
            f"{f.name}: a chart bypasses the themed wrapper"


def test_scientific_wording_is_unchanged_by_the_redesign():
    """The redesign may restyle anything except the claims."""
    from components.ui import COVERAGE_CAVEAT, GALAXIES_SCOPE
    assert "constructed</i> at the nominal" in COVERAGE_CAVEAT
    assert "not superpopulation coverage" in GALAXIES_SCOPE
    assert "containment under repeated subsampling from one" in GALAXIES_SCOPE
    assert "it is not the superpopulation coverage the simulations measure" in GALAXIES_SCOPE
