"""Every displayed equation in the DOCX must be a native Word equation (OMML), never text.

Guards the property that a reader can click any formula in Microsoft Word and enter Equation
Editor: no formula may survive as a plain run, a monospace listing, an image, or a raw LaTeX
string such as `t_{n-2}` or `\\sqrt{...}`.
"""
from __future__ import annotations

import html
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCX = ROOT / "docs" / "MANUSCRIPT.docx"
sys.path.insert(0, str(ROOT / "experiments" / "docxbuild"))

from equations import (EQUATIONS, INLINE_MATH, MATH_BLOCKS,  # noqa: E402
                       is_inline_math, is_math_variable, needs_equation,
                       unicode_to_latex)
from mdblocks import inline_runs, parse  # noqa: E402
from omml import check_roundtrip, latex_to_omml  # noqa: E402

pytestmark = pytest.mark.skipif(not DOCX.exists(), reason="run `make manuscript-docx` first")


def _xml() -> str:
    return zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf8")


def _inline_spans():
    """Every backticked span in the manuscript, with its occurrence count."""
    import collections
    out = collections.Counter()

    def walk(bs):
        for b in bs:
            if b[0] in ("para", "li", "h1", "h2", "h3"):
                yield b[1]
            elif b[0] == "quote":
                yield from walk(b[1])
            elif b[0] == "table":
                yield from (b[1] or []) + [c for r in b[2] for c in r]
    for text in walk(parse((ROOT / "docs" / "MANUSCRIPT.md").read_text())):
        for chunk, _e, _s, code in inline_runs(str(text) or ""):
            if code:
                out[chunk] += 1
    return out


def _fenced_blocks():
    def walk(bs):
        for b in bs:
            if b[0] == "fence":
                yield b[1]
            elif b[0] == "quote":
                yield from walk(b[1])
    return list(walk(parse((ROOT / "docs" / "MANUSCRIPT.md").read_text())))


# --------------------------------------------------------------------------- coverage

def test_every_displayed_block_is_mapped_to_omml():
    """No fenced block may fall through to the plain-text listing renderer."""
    unmapped = [b for b in _fenced_blocks() if b not in EQUATIONS and b not in MATH_BLOCKS]
    assert not unmapped, "displayed blocks left as plain text:\n" + "\n".join(
        b[:90] for b in unmapped)


def test_expected_number_of_omml_objects_in_the_docx():
    """One OMML object per displayed equation line, plus one per inline maths occurrence."""
    spans = _inline_spans()
    expected = (sum(len(v) for v in EQUATIONS.values())
                + sum(len(v) for v in MATH_BLOCKS.values())
                + sum(n for s, n in spans.items()
                      if is_inline_math(s) and needs_equation(s)))
    assert _xml().count("<m:oMath") == expected


def test_every_inline_maths_span_converts_and_round_trips():
    for span in _inline_spans():
        if is_inline_math(span):
            check_roundtrip(unicode_to_latex(span))


def test_variables_are_italic_prose_and_identifiers_stay_code():
    """A lone variable is notation, not an equation; a path or a DOI is neither."""
    assert is_math_variable("n") and not is_inline_math("n")
    for ident in ("experiments/make_tables.py", "10.1214/aos/1176343594", "PARTIAL", "127"):
        assert not is_inline_math(ident) and not is_math_variable(ident), ident


def test_docx_uses_real_math_structures_not_flat_runs():
    """Fractions, radicals, bars, hats and scripts must be present as OMML elements."""
    xml = _xml()
    for tag, why in [("<m:f>", "fractions"), ("<m:rad>", "radicals"), ("<m:bar>", "bars"),
                     ("<m:acc>", "hats"), ("<m:sSub>", "subscripts"),
                     ("<m:sSup>", "superscripts"), ("<m:sSubSup>", "sub+superscripts")]:
        assert xml.count(tag) > 0, f"no {why} ({tag}) in the DOCX"


def test_no_equation_is_an_image():
    """Equations must be OMML, not pictures. The only images are the manuscript's figures."""
    sys.path.insert(0, str(ROOT / "experiments"))
    from make_manuscript import MAIN_FIGURES, SUPP_FIGURES  # noqa: PLC0415

    n_figures = len(MAIN_FIGURES) + len(SUPP_FIGURES)
    z = zipfile.ZipFile(DOCX)
    media = [n for n in z.namelist() if n.startswith("word/media/")]
    assert len(media) == n_figures, (
        f"expected exactly the {n_figures} figures, found {len(media)}: {media}")


# --------------------------------------------------------------------------- no LaTeX left

def test_no_raw_latex_survives_as_document_text():
    """The reader must never see \\frac, \\sqrt, \\hat, or a braced subscript as literal text."""
    xml = _xml()
    body = html.unescape(" ".join(re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", xml, re.S)))
    for bad in ("\\frac", "\\sqrt", "\\hat", "\\bar", "\\alpha", "\\sigma", "sqrt(",
                "fbar_n", "fbar_N", "sigma_hat_f"):
        assert bad not in body, f"raw LaTeX/ASCII maths {bad!r} left as document text"
    assert not re.search(r"[A-Za-zχ²]_\{", body), "a braced subscript survives as literal text"


def test_every_mapped_equation_round_trips():
    """No character of any formula may be lost in the LaTeX -> OMML conversion."""
    n = 0
    for span in _inline_spans():
        if is_inline_math(span):
            check_roundtrip(unicode_to_latex(span)); n += 1
    for rows in EQUATIONS.values():
        for latex, _ in rows:
            check_roundtrip(latex); n += 1
    for rows in MATH_BLOCKS.values():
        for _, latex, _ in rows:
            check_roundtrip(latex); n += 1
    for latex in INLINE_MATH.values():
        check_roundtrip(latex); n += 1
    assert n >= 200, f"only {n} formulae checked"


@pytest.mark.parametrize("src,latex", sorted(INLINE_MATH.items()))
def test_inline_math_preserves_every_symbol(src, latex):
    """The rendered OMML must carry the same symbols as the source span."""
    frag = latex_to_omml(latex, display=False)
    rendered = "".join(re.findall(r"<m:t[^>]*>(.*?)</m:t>", frag))
    rendered = html.unescape(rendered).replace(" ", "")
    for ch in src:
        if ch.isalnum() and ch not in "{}_^":
            base = {"₁": "1", "₂": "2", "²": "2", "−": "-"}.get(ch, ch)
            # An accent is drawn as OMML *structure* -- <m:bar>, <m:acc> -- not as a character,
            # so a precomposed letter such as "Ȳ" comes back as a bare "Y". Compare the base
            # letter too, exactly as check_manuscript_pdf.glyphs() does for the PDF text layer.
            stripped = "".join(c for c in unicodedata.normalize("NFD", ch)
                               if not unicodedata.combining(c))
            assert base in rendered or ch in rendered or (stripped and stripped in rendered), \
                f"{ch!r} lost from {src!r}"
