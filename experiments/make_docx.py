r"""Typeset docs/MANUSCRIPT.md as a journal manuscript: docs/MANUSCRIPT.docx (+ .pdf).

Formatting only. The scientific text, numbers, tables and references come verbatim from
docs/MANUSCRIPT.md; display equations come from experiments/docxbuild/equations.py, whose LaTeX is
keyed to the exact source text it replaces so a change to the mathematics cannot be typeset from a
stale transcription. The nine main figures are placed at their discussion points in the body and the
seven supplementary figures are kept together in section 12.

    python3 experiments/make_docx.py           # writes .docx, then .pdf via LibreOffice
"""
from __future__ import annotations

import functools
import pathlib
import re
import subprocess
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(pathlib.Path(__file__).parent / "docxbuild"))
from equations import (EQUATIONS, MATH_BLOCKS, is_inline_math,  # noqa: E402
                       is_math_variable, needs_equation, unicode_to_latex)
from mdblocks import inline_runs, parse    # noqa: E402
from omml import check_roundtrip, latex_to_omml  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "MANUSCRIPT.md"
DOCX = ROOT / "docs" / "MANUSCRIPT.docx"

# STIX Two Text is the Times-compatible serif used in scientific publishing. Chosen
# over Times New Roman because it composes combining marks over Greek: Times renders
# the tuning coefficient "λ̂" as a struck-through lambda, which misreads the notation.
BODY_FONT = "STIX Two Text"
MATH_FONT = "STIX Two Math"
# Courier New is present on macOS, Windows and Linux alike; monospace spans are
# path/command text and therefore ASCII, so its narrow coverage is not a constraint.
MONO_FONT = "Courier New"
BODY_PT = 10.0
REF_PT = 9.0      # the reference list sets a step down, as journals set it
MARGIN_CM = 2.2
TEXT_W_CM = 21.0 - 2 * MARGIN_CM       # A4 (21cm) less the side margins

# Main figures are placed immediately before these headings.
FIGURE_ANCHORS = {
    "7.2 What the deficit is made of": [1, 2],
    "7.5 Finite pool size": [3],
    "8. Robustness": [4],
    "9. Real-Data Demonstrations": [5],
    "9.2 MT-bench: an LLM-annotation population": [6],
}

FIGURE_ANCHORS_FLAT = [n for v in FIGURE_ANCHORS.values() for n in v]


# --------------------------------------------------------------------------- low-level helpers

def _set(el, tag: str, **attrs):
    child = OxmlElement(tag)
    for k, v in attrs.items():
        child.set(qn(f"w:{k}"), v)
    el.append(child)
    return child


SUPPLEMENTAL = pathlib.Path("/System/Library/Fonts/Supplemental")
# Ordered fallbacks for the handful of operators STIX Two Text does not carry. STIX Two Math is its
# companion face, so the substitution is invisible; Times New Roman catches the precomposed Y-macron
# that neither STIX face has.
FALLBACK_FONTS = [
    (BODY_FONT, SUPPLEMENTAL / "STIXTwoText.ttf"),
    ("STIX Two Math", SUPPLEMENTAL / "STIXTwoMath.otf"),
    ("Times New Roman", SUPPLEMENTAL / "Times New Roman.ttf"),
]


@functools.lru_cache(maxsize=1)
def coverage() -> list[tuple[str, frozenset]]:
    """(font name, covered codepoints) for each fallback, or [] if the fonts cannot be read.

    Declaring the font explicitly for every character stops LibreOffice from substituting one
    silently. Silent substitution still *renders* correctly, but it emits the substituted glyph
    out of order in the PDF text layer, which scrambles copy-paste and extraction.
    """
    out = []
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return out
    for name, path in FALLBACK_FONTS:
        if not path.exists():
            continue
        try:
            f = TTFont(str(path), fontNumber=0)
            out.append((name, frozenset(c for t in f["cmap"].tables for c in t.cmap)))
        except Exception:
            continue
    return out


def font_for(ch: str) -> str:
    for name, chars in coverage():
        if ord(ch) in chars:
            return name
    return BODY_FONT


def font_segments(text: str):
    """Split text into maximal runs that one font can render."""
    if not coverage():
        yield text, BODY_FONT
        return
    buf, cur = "", None
    for ch in text:
        f = BODY_FONT if ch.isspace() or ord(ch) < 128 else font_for(ch)
        if cur is None or f == cur:
            buf, cur = buf + ch, f
        else:
            yield buf, cur
            buf, cur = ch, f
    if buf:
        yield buf, cur


def style_run(run, *, bold=False, italic=False, code=False, size=BODY_PT, font=None):
    run.bold = bold
    run.italic = italic
    f = run.font
    f.size = Pt(size - 0.75 if code else size)
    f.name = font or (MONO_FONT if code else BODY_FONT)
    # east-asian + complex-script names, or Word falls back for non-latin glyphs
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), f.name)
    return run


MATH_CHARS = set("αβγδΔεζηθΘλΛμνρσΣτφΦχψωΩ̂̄²³⁴⁵⁶⁷⁸⁹≤≥≠≈±√∈⊥∞Σ∏∫←→⁻⁺·×÷̃")
PATHY = re.compile(r"\.(py|csv|md|npz|toml|json|txt)\b|^(make|pytest|python3?|git|pip)\s|"
                   r"^(docs|src|results|experiments|figures|data|tests)/|^[a-z_]+\(\)$")


def is_pathlike(span: str) -> bool:
    """True for a file, command or code identifier; False for inline mathematics.

    A bare "/" is not evidence of a path -- the manuscript writes division that way ("σ²/n"), so a
    span carrying any mathematical character is mathematics regardless of its slashes.
    """
    span = span.strip()
    if MATH_CHARS & set(span):
        return False
    return bool(PATHY.search(span))


#: Function and operator names are set upright, not italic: they are names, not variables.
_UPRIGHT = ("min", "max", "exp", "log", "lim", "Var", "Cov", "Corr", "sd", "RSS", "MSE")


def _letter_runs(text: str):
    """Split into maximal runs of letters (italic, they are the variables) and non-letters."""
    upright = set()
    for name in _UPRIGHT:
        for m in re.finditer(re.escape(name), text):
            upright.update(range(m.start(), m.end()))
    buf, cur = "", None
    for i, ch in enumerate(text):
        ital = ch.isalpha() and i not in upright
        if cur is None or ital == cur:
            buf, cur = buf + ch, ital
        else:
            yield buf, cur
            buf, cur = ch, ital
    if buf:
        yield buf, cur


def add_runs(par, text: str, *, size=BODY_PT, base_bold=False, base_italic=False):
    """Backticked spans render as monospace only when they are paths or commands.

    The manuscript writes its inline mathematics in backticks too, and a monospace face composes
    combining marks badly -- it turns the estimator's hat into a caret. Those spans get the body
    serif instead, which renders the mathematics correctly.
    """
    for chunk, b, i, c in inline_runs(text):
        if not chunk:
            continue
        # A backticked span that is mathematics becomes a native Word equation, so the reader
        # never sees `t_{n-2}` or `sigma_f²` where a real subscript is meant. Paths, DOIs,
        # numbers and identifiers stay in the monospace face.
        if c and is_math_variable(chunk):
            style_run(par.add_run(chunk), bold=b or base_bold, italic=True, size=size)
            continue
        if c and is_inline_math(chunk):
            if needs_equation(chunk):
                latex = unicode_to_latex(chunk)
                check_roundtrip(latex)
                # inline maths is set one point below the prose it sits in. Word grows a line
                # to the full height of an inline equation object, and a script or accent on a
                # full-size glyph overtops the line box; a point smaller reads the same and
                # keeps math-bearing lines at prose height.
                par._p.append(parse_xml(latex_to_omml(
                    latex, display=False, font=MATH_FONT,
                    size_pt=(size or BODY_PT) - 1.0)))
                continue
            # plain notation: set as text so it keeps the prose line height. Letters are the
            # variables and are italic; digits, operators and delimiters stay upright, which is
            # the mathematical convention and what an equation object would have produced.
            for seg, font in font_segments(chunk):
                for part, ital in _letter_runs(seg):
                    style_run(par.add_run(part), bold=b or base_bold, italic=ital,
                              size=size, font=font)
            continue
        mono = c and is_pathlike(chunk)
        if mono:
            style_run(par.add_run(chunk), bold=b or base_bold, italic=i or base_italic,
                      code=True, size=size)
            continue
        for seg, font in font_segments(chunk):
            style_run(par.add_run(seg), bold=b or base_bold, italic=i or base_italic,
                      size=size, font=font)
    return par


def page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for instr in (" PAGE ", " NUMPAGES "):
        if instr == " NUMPAGES ":
            style_run(p.add_run(" of "), size=9)
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), instr)
        r = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "18")
        rpr.append(sz)
        r.append(rpr)
        t = OxmlElement("w:t")
        t.text = "1"
        r.append(t)
        fld.append(r)
        p._p.append(fld)


# --------------------------------------------------------------------------- block emitters

def emit_equation(doc, latex: str, number: str | None):
    check_roundtrip(latex)      # fail the build rather than typeset a truncated equation
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.keep_together = True
    pf.tab_stops.add_tab_stop(Cm(TEXT_W_CM / 2), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Cm(TEXT_W_CM), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t")
    p._p.append(parse_xml(latex_to_omml(latex, display=False)))
    if number:
        style_run(p.add_run(f"\t({number})"), size=BODY_PT)


def emit_math_block(doc, rows):
    """A run of labelled mathematical lines: leading text, a native Word equation, trailing text.

    Used for the algorithm of section 5 and the pair of quantile rules, both of which are sequences
    of formulae rather than one numbered equation. Every formula becomes an OMML object, so none is
    left as a monospace string like `t_{n-2}` or `sqrt(V)`.
    """
    for n, (lead, latex, trail) in enumerate(rows):
        check_roundtrip(latex)
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.9)
        pf.space_before = Pt(1)
        pf.space_after = Pt(1)
        pf.line_spacing = 1.0
        pf.keep_together = True
        # An algorithm broken mid-step is unreadable, so the numbered steps are held together;
        # a labelled variant (a row whose lead is a phrase, not a step number) may start a page.
        pf.keep_with_next = n < len(rows) - 1 and len(rows[n + 1][0]) <= 4
        pf.tab_stops.add_tab_stop(Cm(TEXT_W_CM - 0.5), WD_TAB_ALIGNMENT.RIGHT)
        if lead:
            style_run(p.add_run(lead + " "), size=BODY_PT)
        p._p.append(parse_xml(latex_to_omml(latex, display=False)))
        if trail:
            style_run(p.add_run("\t" + trail), size=BODY_PT - 0.5, italic=True)


def emit_listing(doc, text: str):
    """A fenced block that is not an equation (the algorithm of section 5)."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.8)
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.keep_together = True
    for n, line in enumerate(text.split("\n")):
        if n:
            p.add_run().add_break()
        style_run(p.add_run(line), code=True, size=BODY_PT)
    shade = OxmlElement("w:shd")
    shade.set(qn("w:val"), "clear")
    shade.set(qn("w:fill"), "F5F5F5")
    p._p.get_or_add_pPr().append(shade)


def emit_table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    size = 8.0 if len(header) <= 8 else 7.0

    for cell, text in zip(t.rows[0].cells, header):
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(cell.paragraphs[0], text, size=size, base_bold=True)
        shade = OxmlElement("w:shd")
        shade.set(qn("w:val"), "clear")
        shade.set(qn("w:fill"), "EDEDED")
        cell._tc.get_or_add_tcPr().append(shade)

    for row in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(cell.paragraphs[0], text, size=size)

    keep_together = len(rows) <= 12   # taller than this and Word should be free to break it
    for n, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr()
        _set(trpr, "w:cantSplit")                      # no row split across a page break
        if n == 0:
            _set(trpr, "w:tblHeader")                  # repeat header on each page
        if keep_together and n < len(t.rows) - 1:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.space_before = Pt(1)
                par.paragraph_format.space_after = Pt(1)
                par.paragraph_format.line_spacing = 1.0

    tblpr = t._tbl.tblPr
    margins = OxmlElement("w:tblCellMar")
    for side, w in (("left", "60"), ("right", "60"), ("top", "20"), ("bottom", "20")):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), w)
        m.set(qn("w:type"), "dxa")
        margins.append(m)
    tblpr.append(margins)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def emit_figure(doc, number: str, path: str, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    # Placed at its own printed size (pixels / dpi), so the journal renderings' 8 pt lettering
    # prints at 8 pt; capped just inside the text width.
    from PIL import Image
    with Image.open(ROOT / path) as im:
        natural_cm = im.width / im.info.get("dpi", (600, 600))[0] * 2.54
    p.add_run().add_picture(str(ROOT / path), width=Cm(min(natural_cm, TEXT_W_CM - 0.4)))

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.space_after = Pt(7)
    cap.paragraph_format.left_indent = Cm(0.5)
    cap.paragraph_format.right_indent = Cm(0.5)
    cap.paragraph_format.line_spacing = 1.0
    cap.paragraph_format.keep_together = True
    # Statistics and Computing: "Fig." and the number in bold, no punctuation after the number
    style_run(cap.add_run(f"Fig. {number}"), bold=True, size=8.5)
    cap.add_run(" ")
    add_runs(cap, caption, size=8.5)


def emit_heading(doc, text: str, level: int):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(10 if level == 1 else 6)
    pf.space_after = Pt(4)
    pf.keep_with_next = True
    pf.keep_together = True
    add_runs(p, text, size=13 if level == 1 else 11.5, base_bold=True)
    return p


# --------------------------------------------------------------------------- assembly

def collect_main_figures(blocks) -> dict[int, tuple[str, str]]:
    """Pull the nine main-figure caption pairs out of section 12 so they can be relocated."""
    figs: dict[int, tuple[str, str]] = {}
    inside = False
    for n, b in enumerate(blocks):
        if b[0] == "h3" and b[1] == "Main figures":
            inside = True
            continue
        if inside and b[0] == "h2":
            break
        if inside and b[0] == "para" and b[1].startswith("**Figure "):
            num = b[1].split("**")[1].replace("Figure ", "").strip()
            path = b[1].split("`")[1]
            caption = blocks[n + 1][1] if blocks[n + 1][0] == "para" else ""
            figs[int(num)] = (path, caption)
    if len(figs) != len(FIGURE_ANCHORS_FLAT):
        raise SystemExit(f"expected {len(FIGURE_ANCHORS_FLAT)} main figures in the figure section, found {len(figs)}")
    return figs


def build() -> Document:
    blocks = parse(SRC.read_text())
    main_figs = collect_main_figures(blocks)
    used_eq_keys: set[str] = set()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left", "right", "top", "bottom"):
        setattr(sec, f"{side}_margin", Cm(MARGIN_CM))
    page_number_footer(sec)

    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(BODY_PT)
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.widow_control = True

    # skip the relocated main-figure listing in section 12
    skip_from = next(n for n, b in enumerate(blocks)
                     if b[0] == "h2" and b[1] == "Figure manifest")
    skip_to = next(n for n in range(skip_from + 1, len(blocks)) if blocks[n][0] == "h2")
    skipped = set(range(skip_from, skip_to))

    in_refs = False
    title_block = 0        # paragraphs between the title and the first section: author, then place
    consumed: set[int] = set()

    for n, b in enumerate(blocks):
        if n in skipped or n in consumed:
            continue
        kind = b[0]

        if kind in ("h1", "h2", "h3"):
            level = int(kind[1])
            for fnum in FIGURE_ANCHORS.get(b[1], []):
                path, caption = main_figs[fnum]
                emit_figure(doc, str(fnum), path, caption)
            if level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(14)
                style_run(p.add_run(b[1]), bold=True, size=16)
                title_block = 1
                continue
            title_block = 0
            in_refs = b[1] == "References"
            h = emit_heading(doc, b[1], level - 1)
            if b[1].startswith("S. Supplementary Material"):
                # the supplement starts a fresh page: the main manuscript ends at the references.
                # Carried by the heading itself, so no empty paragraph can strand a blank page.
                h.paragraph_format.page_break_before = True
            continue

        if kind == "fence":
            key = b[1]
            if key in MATH_BLOCKS:
                emit_math_block(doc, MATH_BLOCKS[key])
                continue
            if key in EQUATIONS:
                used_eq_keys.add(key)
                for latex, number in EQUATIONS[key]:
                    emit_equation(doc, latex, number)
            else:
                emit_listing(doc, b[1])
            continue

        if kind == "table":
            emit_table(doc, b[1], b[2])
            continue

        if kind == "quote":
            emit_quote(doc, b[1], used_eq_keys)
            continue

        if kind == "li":
            if b[2]:
                # the marker is written out, not left to Word's List Number style: that style
                # carries one counter for the whole document, so a later list continues the
                # earlier one instead of restarting at 1.
                p = doc.add_paragraph()
                pf = p.paragraph_format
                pf.left_indent = Cm(1.3 + 0.5 * (b[3] // 2))
                pf.first_line_indent = Cm(-0.5)
                style_run(p.add_run(f"{b[4]} "), bold=False)
            else:
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.left_indent = Cm(0.8 + 0.5 * (b[3] // 2))
            p.paragraph_format.space_after = Pt(3)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_runs(p, b[1])
            continue

        if kind == "rule":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            pbdr = OxmlElement("w:pBdr")
            bot = OxmlElement("w:bottom")
            for k, v in (("val", "single"), ("sz", "4"), ("space", "1"), ("color", "BBBBBB")):
                bot.set(qn(f"w:{k}"), v)
            pbdr.append(bot)
            p._p.get_or_add_pPr().append(pbdr)
            continue

        if kind == "para" and title_block:
            # 1 = author, 2 = affiliation: centred under the title. Anything after that is the
            # build-provenance note, which sets badly centred -- its monospace paths break the line
            # awkwardly -- so it runs as an indented justified note instead.
            p = doc.add_paragraph()
            pf = p.paragraph_format
            if title_block <= 2:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pf.space_after = Pt(4 if title_block == 1 else 16)
                add_runs(p, b[1], size=12.0 if title_block == 1 else 10.5)
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pf.left_indent = pf.right_indent = Cm(1.2)
                pf.space_after = Pt(14)
                pf.line_spacing = 1.0
                add_runs(p, b[1], size=8.5)
            title_block += 1
            continue

        if kind == "para" and b[1].startswith("**Figure S") and "`" in b[1]:
            num = b[1].split("**")[1].replace("Figure ", "").strip()
            path = b[1].split("`")[1]
            caption = blocks[n + 1][1] if blocks[n + 1][0] == "para" else ""
            if caption:
                consumed.add(n + 1)
            emit_figure(doc, num, path, caption)
            continue

        if kind == "para" and in_refs:
            emit_reference(doc, b[1])
            continue

        if kind == "para":
            p = doc.add_paragraph()
            # ragged right where a long unbreakable URL would stretch a justified line into gaps
            ragged = b[1].startswith("*Source:") or "https://" in b[1]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ragged else WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.keep_together = b[1].startswith(("**Table", "*Source:"))
            if b[1].startswith("**Algorithm"):
                p.paragraph_format.keep_with_next = True   # the title stays with the algorithm
            if b[1].startswith("**Table"):
                # a lead-in belongs with its table; a paragraph that merely *discusses* the table
                # must not be chained to whatever follows -- that strands a third of a page when
                # the next block is a figure.
                p.paragraph_format.keep_with_next = (
                    n + 1 < len(blocks) and blocks[n + 1][0] in ("table", "para"))
                p.paragraph_format.space_before = Pt(9)
            add_runs(p, b[1], size=9 if b[1].startswith("*Source:") else BODY_PT)
            continue

    missing = set(EQUATIONS) - used_eq_keys
    if missing:
        raise SystemExit(
            "equations.py has key(s) that no longer appear in MANUSCRIPT.md — the LaTeX is stale:\n"
            + "\n".join(f"  {k[:70]}..." for k in missing))
    return doc


def emit_quote(doc, inner_blocks, used_eq_keys: set[str]):
    """Theorem / limitation boxes: indented, ruled on the left."""
    for b in inner_blocks:
        if b[0] == "fence" and b[1] in MATH_BLOCKS:
            emit_math_block(doc, MATH_BLOCKS[b[1]])
            continue
        if b[0] == "fence" and b[1] in EQUATIONS:
            used_eq_keys.add(b[1])
            for latex, number in EQUATIONS[b[1]]:
                emit_equation(doc, latex, number)
            continue
        if b[0] == "fence":
            emit_listing(doc, b[1])
            continue
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Cm(0.8)
        pf.right_indent = Cm(0.5)
        pf.space_after = Pt(4)
        pf.keep_together = True
        add_runs(p, b[1])
        pbdr = OxmlElement("w:pBdr")
        left = OxmlElement("w:left")
        for k, v in (("val", "single"), ("sz", "12"), ("space", "10"), ("color", "999999")):
            left.set(qn(f"w:{k}"), v)
        pbdr.append(left)
        p._p.get_or_add_pPr().append(pbdr)


def emit_reference(doc, ref: str):
    """One author–year reference entry: hanging indent, one step down from the body size."""
    p = doc.add_paragraph()
    # ragged right: entries end in long unbreakable DOI and arXiv strings, which justification
    # would stretch into wide word gaps
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.left_indent = Cm(0.9)
    pf.first_line_indent = Cm(-0.9)
    pf.space_after = Pt(2)
    pf.line_spacing = 1.0
    pf.keep_together = True
    add_runs(p, ref, size=REF_PT)


def stamp_page_count(docx_path: pathlib.Path, pdf_path: pathlib.Path) -> int:
    """Record the real page count in the DOCX's own properties.

    python-docx does not paginate, so it writes `Pages: 1, Words: 0` into `docProps/app.xml`.
    Anything reading those properties therefore reports a page count that has nothing to do with
    the document -- which is exactly how a stale "24 pages" can be quoted against a 31-page file.
    LibreOffice has just paginated the DOCX to produce the PDF, so its page count is authoritative;
    we write it back so the DOCX describes itself truthfully.
    """
    import re
    import shutil
    import zipfile

    try:
        import fitz
        pages = fitz.open(str(pdf_path)).page_count
    except Exception:
        return 0

    tmp = docx_path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(docx_path) as zin, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/app.xml":
                app = data.decode("utf8")
                if "<Pages>" in app:
                    app = re.sub(r"<Pages>\d+</Pages>", f"<Pages>{pages}</Pages>", app)
                else:
                    app = app.replace("</Properties>", f"<Pages>{pages}</Pages></Properties>")
                data = app.encode("utf8")
            zout.writestr(item, data)
    shutil.move(tmp, docx_path)
    return pages


def main() -> int:
    doc = build()
    doc.save(DOCX)
    print(f"wrote {DOCX.relative_to(ROOT)}")
    r = subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(DOCX.parent), str(DOCX)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr, file=sys.stderr)
        return 1
    pages = stamp_page_count(DOCX, DOCX.with_suffix(".pdf"))
    print(f"wrote docs/MANUSCRIPT.pdf ({pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
