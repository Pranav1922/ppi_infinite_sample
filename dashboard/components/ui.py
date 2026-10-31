"""Design system for the research dashboard: theme, layout primitives, standing caveats.

Presentation only. Nothing here computes, reads or writes a research quantity — the scientific
wording it carries (`COVERAGE_CAVEAT`, `GALAXIES_SCOPE`) is fixed text quoted from the manuscript's
own qualifications and must not be edited for style.
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import streamlit as st

DASHBOARD = Path(__file__).resolve().parent.parent
if str(DASHBOARD) not in sys.path:                       # so `pages/*.py` can import components
    sys.path.insert(0, str(DASHBOARD))

from components.loaders import provenance  # noqa: E402

# --------------------------------------------------------------------------- palette

INK        = "#e6edf3"
MUTED      = "#8b949e"
FAINT      = "#6e7681"
SURFACE    = "#161b22"
SURFACE_2  = "#1c232c"
BORDER     = "#262d36"
ACCENT     = "#58a6ff"
ACCENT_DIM = "#1f6feb"
DANGER     = "#f85149"
OK         = "#3fb950"
WARN       = "#d29922"

BADGE_KINDS = {
    "empirical":  ("Empirical result",        ACCENT,  "rgba(88,166,255,.12)"),
    "theory":     ("Theoretical bound",       "#a371f7", "rgba(163,113,247,.12)"),
    "boundary":   ("Boundary active",         DANGER,  "rgba(248,81,73,.12)"),
    "safe":       ("Standard regime",         OK,      "rgba(63,185,80,.12)"),
    "realdata":   ("Real-data demonstration", WARN,    "rgba(210,153,34,.12)"),
    "live":       ("Live illustration",       "#39c5cf", "rgba(57,197,207,.12)"),
}

CSS = f"""
<style>
  .block-container {{ max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }}
  html, body, [class*="css"] {{ font-family: Georgia, 'Iowan Old Style', 'Times New Roman', serif; }}

  h1, h2, h3, h4 {{ font-family: Georgia, 'Iowan Old Style', serif; color: {INK};
                    letter-spacing: -0.015em; }}
  h1 {{ font-size: 2.0rem; line-height: 1.15; margin-bottom: .3rem; }}
  h2 {{ font-size: 1.34rem; margin-top: 2.2rem; }}
  h3 {{ font-size: 1.08rem; margin-top: 1.4rem; }}
  .stMarkdown p, .stMarkdown li {{ font-size: 1.0rem; line-height: 1.62; color: {INK}; }}
  .stMarkdown p {{ max-width: 78ch; }}

  code, .mono {{ font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: .88em;
                 background: {SURFACE_2}; padding: .08em .34em; border-radius: 4px;
                 border: 1px solid {BORDER}; }}

  /* ---- hero ---- */
  .hero {{ border: 1px solid {BORDER}; border-radius: 14px; background:
           linear-gradient(180deg, {SURFACE} 0%, #12171e 100%);
           padding: 2.1rem 2.2rem 1.7rem; margin-bottom: 1.6rem; }}
  .hero .eyebrow {{ font-family: ui-monospace, Menlo, monospace; font-size: .72rem;
                    letter-spacing: .16em; text-transform: uppercase; color: {ACCENT};
                    margin-bottom: .55rem; }}
  .hero h1 {{ font-size: 2.5rem; margin: 0 0 .45rem; }}
  .hero .sub {{ font-size: 1.09rem; color: {MUTED}; margin: 0 0 1.15rem; max-width: 62ch; }}

  /* ---- badges ---- */
  .badge {{ display: inline-block; font-family: ui-monospace, Menlo, monospace;
            font-size: .735rem; letter-spacing: .05em; padding: .28rem .66rem;
            border-radius: 999px; margin: 0 .38rem .4rem 0; border: 1px solid; }}

  /* ---- kpi ---- */
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
           gap: .85rem; margin: .3rem 0 .4rem; }}
  .kpi {{ border: 1px solid {BORDER}; border-radius: 11px; background: {SURFACE};
          padding: 1.0rem 1.1rem; }}
  .kpi .lab {{ font-family: ui-monospace, Menlo, monospace; font-size: .725rem;
               letter-spacing: .09em; text-transform: uppercase; color: {MUTED};
               margin-bottom: .5rem; line-height: 1.35; }}
  .kpi .val {{ font-size: 1.72rem; line-height: 1; color: {INK}; }}
  .kpi .sub {{ font-size: .8rem; color: {FAINT}; margin-top: .42rem; }}
  .kpi.warn .val {{ color: {DANGER}; }}
  .kpi.good .val {{ color: {OK}; }}
  .kpi.accent .val {{ color: {ACCENT}; }}

  /* ---- section rule ---- */
  .sect {{ margin: 2.4rem 0 .2rem; border-top: 1px solid {BORDER}; padding-top: 1.15rem; }}
  .sect .t {{ font-size: 1.3rem; color: {INK}; }}
  .sect .d {{ font-size: .93rem; color: {MUTED}; margin-top: .22rem; max-width: 74ch; }}

  /* ---- callouts ---- */
  .note {{ border-radius: 9px; padding: .85rem 1.05rem; margin: .9rem 0; font-size: .93rem;
           line-height: 1.55; border: 1px solid; }}
  .note.caveat {{ border-color: rgba(210,153,34,.4); background: rgba(210,153,34,.07); }}
  .note.live   {{ border-color: rgba(57,197,207,.4); background: rgba(57,197,207,.07); }}
  .note.scope  {{ border-color: rgba(248,81,73,.38); background: rgba(248,81,73,.07); }}

  .srcnote {{ font-family: ui-monospace, Menlo, monospace; font-size: .72rem; color: {FAINT};
              margin-top: .45rem; }}

  /* ---- boundary diagram ---- */
  .bd {{ border: 1px solid {BORDER}; border-radius: 11px; background: {SURFACE};
         padding: 1.35rem 1.5rem 1.1rem; margin: .5rem 0 .3rem; }}
  .bd .track {{ position: relative; height: 8px; border-radius: 4px; margin: 2.5rem 0 .5rem;
                background: linear-gradient(90deg, rgba(248,81,73,.55) 0%,
                            rgba(248,81,73,.55) 50%, rgba(63,185,80,.5) 50%,
                            rgba(63,185,80,.5) 100%); }}
  .bd .tick {{ position: absolute; top: -34px; transform: translateX(-50%); text-align: center; }}
  .bd .tick .cap {{ font-family: ui-monospace, Menlo, monospace; font-size: .72rem;
                    color: {MUTED}; white-space: nowrap; }}
  .bd .tick .pin {{ width: 2px; height: 24px; background: {BORDER}; margin: .3rem auto 0; }}
  .bd .marker .cap {{ color: {INK}; }}
  .bd .zone {{ display: flex; justify-content: space-between; font-family: ui-monospace, Menlo,
               monospace; font-size: .73rem; color: {FAINT}; margin-top: .55rem; }}

  /* ---- tables & sidebar ---- */
  [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 9px; }}
  [data-testid="stDataFrame"] * {{ font-size: .875rem !important; }}
  section[data-testid="stSidebar"] {{ background: #0a0e13; border-right: 1px solid {BORDER}; }}
  section[data-testid="stSidebar"] .stMarkdown p {{ font-size: .86rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
</style>
"""


# --------------------------------------------------------------------------- primitives

def boot(title: str) -> None:
    """Page config + theme. Call once, first, on every page."""
    st.set_page_config(page_title=f"{title} · PPI Finite-Sample Lab", layout="wide",
                       page_icon=":material/analytics:", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


def badge(kind: str, text: str | None = None) -> str:
    """Inline status badge. Returns HTML so it can be composed into other blocks."""
    label, colour, bg = BADGE_KINDS[kind]
    return (f"<span class='badge' style='color:{colour};border-color:{colour}55;"
            f"background:{bg}'>{text or label}</span>")


def badges(*kinds: str) -> None:
    st.markdown("".join(badge(k) for k in kinds), unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "", chips: tuple[str, ...] = ()) -> None:
    chip_html = "".join(
        f"<span class='badge' style='color:{ACCENT};border-color:{ACCENT}44;"
        f"background:rgba(88,166,255,.10)'>{c}</span>" for c in chips)
    st.markdown(
        f"<div class='hero'>"
        f"{f'<div class=eyebrow>{eyebrow}</div>' if eyebrow else ''}"
        f"<h1>{title}</h1><p class='sub'>{subtitle}</p>{chip_html}</div>",
        unsafe_allow_html=True)


def page_head(title: str, subtitle: str, *badge_kinds: str) -> None:
    """Standard page title block."""
    st.markdown(f"<h1>{title}</h1><p class='sub' style='color:{MUTED};font-size:1.02rem;"
                f"max-width:74ch;margin-top:-.2rem'>{subtitle}</p>", unsafe_allow_html=True)
    if badge_kinds:
        badges(*badge_kinds)


def kpis(items: list[dict]) -> None:
    """A responsive row of KPI cards. Each item: {label, value, sub?, tone?}."""
    cards = []
    for item in items:
        tone = item.get("tone", "")
        sub = item.get("sub")
        sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
        cards.append(
            f"<div class='kpi {tone}'>"
            f"<div class='lab'>{item['label']}</div>"
            f"<div class='val'>{item['value']}</div>"
            f"{sub_html}</div>")
    st.markdown("<div class='kpis'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def section(title: str, description: str = "") -> None:
    st.markdown(f"<div class='sect'><div class='t'>{title}</div>"
                f"{f'<div class=d>{description}</div>' if description else ''}</div>",
                unsafe_allow_html=True)


def source_note(*paths: str) -> None:
    st.markdown(f"<div class='srcnote'>read from {' · '.join(paths)}</div>",
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- fixed scientific text
# The wording below is quoted from the manuscript's own qualifications. Style may change; the
# claims may not.

COVERAGE_CAVEAT = (
    "<b>Nominal is not empirical.</b> Every interval here is <i>constructed</i> at the nominal "
    "95 % level. Whether it <i>attains</i> 95 % is exactly what this research measures — and often "
    "it does not. Read the coverage column, never the nominal level, as the empirical statement.")

GALAXIES_SCOPE = (
    "<b>This is a real-data demonstration, not superpopulation coverage.</b> "
    "The Galaxies data are a <i>fully observed population</i>. Gold/unlabelled splits are drawn "
    "repeatedly from it, and what is reported is the fraction of those splits whose interval "
    "contains the population value. That is <b>containment under repeated subsampling from one "
    "fixed finite population</b> — it is not the superpopulation coverage the simulations measure, "
    "the two are not the same, and nothing on this page is evidence for the theory.")


def caveat(text: str = COVERAGE_CAVEAT, kind: str = "caveat") -> None:
    st.markdown(f"<div class='note {kind}'>{text}</div>", unsafe_allow_html=True)


def live_note(text: str) -> None:
    st.markdown(
        f"<div class='note live'>{badge('live')}<br><b>Computed live for illustration.</b> {text} "
        "It is not a saved research result and does not enter any table in the paper.</div>",
        unsafe_allow_html=True)


def boundary_diagram(n: int, N: int) -> None:
    """A horizontal N-axis showing which side of `N = n − 1` the configuration sits on."""
    edge = n - 1
    active = N < edge
    span = max(edge * 2.0, float(N) * 1.15, 4.0)
    pos = min(max(float(N) / span * 100.0, 2.0), 98.0)
    edge_pos = min(max(edge / span * 100.0, 2.0), 98.0)
    colour = DANGER if active else OK
    st.markdown(f"""
<div class='bd'>
  <div class='track' style='background:linear-gradient(90deg,
       rgba(248,81,73,.5) 0%, rgba(248,81,73,.5) {edge_pos}%,
       rgba(63,185,80,.45) {edge_pos}%, rgba(63,185,80,.45) 100%)'>
    <div class='tick' style='left:{edge_pos}%'>
      <div class='cap'>N = n − 1 = {edge}</div><div class='pin'></div></div>
    <div class='tick marker' style='left:{pos}%;top:14px'>
      <div class='pin' style='background:{colour};height:18px;margin:0 auto .3rem'></div>
      <div class='cap' style='color:{colour}'>your N = {N}</div></div>
  </div>
  <div class='zone'><span>N &lt; n − 1 · finite-pool risk</span>
                    <span>N ≥ n − 1 · standard quantile regime</span></div>
</div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- charts

def style_chart(chart: alt.Chart, height: int = 380) -> alt.Chart:
    """Apply the dashboard's chart theme. Presentation only — no data is transformed."""
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=BORDER, strokeWidth=1)
            .configure_axis(grid=True, gridColor="#1e2530", gridOpacity=0.9,
                            domainColor=BORDER, tickColor=BORDER,
                            labelColor=MUTED, titleColor=MUTED,
                            labelFontSize=11, titleFontSize=12,
                            labelFont="ui-monospace, Menlo, monospace",
                            titleFont="Georgia, serif")
            .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=11,
                              titleFontSize=11, symbolStrokeWidth=2,
                              labelFont="ui-monospace, Menlo, monospace")
            .configure_title(color=INK, fontSize=13, font="Georgia, serif", anchor="start"))


# --------------------------------------------------------------------------- chrome

def sidebar_footer() -> None:
    p = provenance()
    st.sidebar.markdown(
        f"<div class='srcnote' style='margin-top:1.6rem;border-top:1px solid {BORDER};"
        f"padding-top:.8rem'>read-only view · writes nothing<br>"
        f"tree <code>{p['working_tree_commit']}</code><br>"
        f"results <code>{p['results_commit']}</code></div>", unsafe_allow_html=True)


def footer() -> None:
    p = provenance()
    st.markdown(
        f"<div class='srcnote' style='margin-top:3rem;border-top:1px solid {BORDER};"
        f"padding-top:1rem'>Read-only presentation layer — no result, config, seed or manuscript "
        f"file is written. Working tree <code>{p['working_tree_commit']}</code> · results "
        f"generated at commit <code>{p['results_commit']}</code></div>", unsafe_allow_html=True)


# Back-compat for the existing page modules and tests.
def page(title: str, subtitle: str = "") -> None:
    boot(title)
    page_head(title, subtitle)


def sidebar_nav() -> None:
    sidebar_footer()
