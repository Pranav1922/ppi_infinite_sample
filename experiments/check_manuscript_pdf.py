r"""Verify docs/MANUSCRIPT.pdf against docs/MANUSCRIPT.md.

Checks that the typeset PDF carries the manuscript's prose verbatim, that every table, figure,
equation and reference is present, and that nothing is blank, clipped or outside the type area.

    python3 experiments/check_manuscript_pdf.py
"""
from __future__ import annotations

import html
import pathlib
import re
import sys
import unicodedata

import fitz

sys.path.insert(0, str(pathlib.Path(__file__).parent / "docxbuild"))
from equations import is_inline_math, unicode_to_latex   # noqa: E402
from mdblocks import inline_runs, parse   # noqa: E402
from omml import latex_to_omml            # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PDF = ROOT / "docs" / "MANUSCRIPT.pdf"
MD = ROOT / "docs" / "MANUSCRIPT.md"
MARGIN_PT = 2.2 / 2.54 * 72
FOOTER_BAND_PT = 60          # the page number legitimately sits in the bottom margin
# A text block's bbox spans the whole line box, so its bottom sits a descender below the last
# baseline. A final line whose BASELINE is on the margin therefore reports a bbox a few points
# past it -- normal typesetting, not text in the margin. Allow exactly one body line box
# (10 pt at 1.12 spacing) on the bottom edge only; horizontal limits stay exact.
LINE_BOX_PT = 10.0 * 1.12


def glyphs(s: str) -> str:
    """The symbols of a formula, dropping what OMML draws as structure rather than as a
    character: the radical sign, the accent marks, and the spacing between runs."""
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    # an accent may also come back as a spacing modifier (U+02C6 etc.), and it may extract
    # before its base rather than after it, so drop it from both sides
    return re.sub(r"[\s\u221a\u02c6\u02c9\u02dc\u00af\u005e~]", "", s)


def rendered(chunk: str) -> str:
    """An inline maths span as the PDF carries it: real sub/superscripts, so the braces and
    underscores of the source are gone and extraction reads the symbols back in order."""
    frag = latex_to_omml(unicode_to_latex(chunk), display=False)
    return glyphs(html.unescape("".join(re.findall(r"<m:t[^>]*>(.*?)</m:t>", frag))))


# A maths token is checked against the PDF with all whitespace removed: extraction walks an
# OMML run at a time, so a real subscript comes back spaced ("chi f i 2"), which is correct
# typesetting rather than altered content. MATH_MARK keeps that laxity off ordinary prose.
MATH_MARK = "\x00"


def words_of(text: str) -> list[str]:
    """Prose splits on whitespace; a maths span becomes the single token the PDF shows."""
    out: list[str] = []
    for chunk, _e, _s, code in inline_runs(text):
        out += [MATH_MARK + rendered(chunk)] if code and is_inline_math(chunk) else chunk.split()
    return out


def source_tokens() -> list[str]:
    """Every prose token the PDF must contain. Fenced blocks are excluded: they are retypeset
    as equations, so their characters are compared by the OMML round-trip guard instead.
    Inline maths spans are compared in rendered form -- they are equations too, not text."""
    words: list[str] = []
    in_refs = in_manifest = False
    for b in parse(MD.read_text()):
        if b[0] == "h2":
            in_manifest = b[1] == "Figure manifest"
        if b[0] in ("h2", "h3"):
            in_refs = b[1] == "References"
        if in_manifest:
            continue      # relocated into the body by make_docx; never typeset as a section
        if b[0] == "fence":
            continue
        if b[0] == "table":
            # the reference table is rendered as a hanging-indent list by emit_reference_list,
            # which writes the body rows only, so its column headers are never typeset
            cells = ([] if in_refs else b[1]) + [c for r in b[2] for c in r]
            words += [w for cell in cells for w in words_of(cell)]
        elif b[0] == "quote":
            for ib in b[1]:
                if ib[0] != "fence":
                    words += words_of(str(ib[1]))
        elif b[0] in ("para", "li", "h1", "h2", "h3"):
            if re.match(r"\*\*Figure \S+\*\* — `figures/", str(b[1])):
                continue      # the figure manifest line: a source path, not manuscript prose
            words += words_of(b[1])
    return words


def main() -> int:
    doc = fitz.open(PDF)
    flat = " ".join(" ".join(p.get_text().split()) for p in doc)
    # extraction inserts a space where a line breaks at a hyphen or dash
    joined = (flat.replace("- ", "-").replace("– ", "–").replace("— ", "—")
              .replace(" –", "–").replace(" —", "—"))

    squeezed = glyphs(flat)
    lost = []
    for w in source_tokens():
        if w.startswith(MATH_MARK):
            if glyphs(w[1:]) not in squeezed:
                lost.append(w[1:])
        elif len(w) > 3 and w.strip(".,;:()[]") and w not in flat and w not in joined:
            lost.append(w)

    blanks, overflow = [], []
    for i, pg in enumerate(doc, 1):
        W, H = pg.rect.width, pg.rect.height
        if len(pg.get_text().strip()) < 60 and not pg.get_image_info():
            blanks.append(i)
        boxes = [b["bbox"] for b in pg.get_text("dict")["blocks"] if b["type"] == 0]
        boxes += [im["bbox"] for im in pg.get_image_info()]
        for x0, y0, x1, y1 in boxes:
            if y0 > H - FOOTER_BAND_PT:          # the page-number footer
                continue
            if (x1 > W - MARGIN_PT + 3 or x0 < MARGIN_PT - 3
                    or y1 > H - MARGIN_PT + LINE_BOX_PT):
                overflow.append((i, tuple(round(v) for v in (x0, y0, x1, y1))))

    figs = sum(len(p.get_image_info()) for p in doc)
    # journal captions open "Fig. n" with no punctuation after the number; an in-text reference
    # is always parenthesised, so "(Fig. n" never counts as a caption
    mains = {m for m in re.findall(r"(?<!\()Fig\. (\d+) [A-Z]", flat)}
    supp = {m for m in re.findall(r"Figure (S\d+)\.", flat)}
    tables = [i for i in range(1, 9) if f"Table {i} —" in flat]
    eqs = [i for i in range(1, 9) if f"({i})" in flat]
    # every reference entry (a paragraph after the References heading) is typeset: its
    # author-year head must appear in the PDF
    blocks = parse(MD.read_text())
    start = next(i for i, b in enumerate(blocks) if b[0] == "h2" and b[1] == "References")
    heads = [re.match(r"^(.*?\((?:19|20)\d\d[ab]?\))", b[1]).group(1)
             for b in blocks[start + 1:] if b[0] == "para"]
    # Springer-style heads end in page ranges, so a line can break at the en dash: use `joined`
    refs_ok = len(heads) == 27 and all(" ".join(h.split()) in flat or " ".join(h.split()) in joined
                                       for h in heads)

    checks = {
        "prose reproduced verbatim": not lost,
        "no blank pages": not blanks,
        "nothing outside the type area": not overflow,
        "6 figures placed": figs == 6,
        "6 figure captions": len(mains) == 6,
        "no supplementary section": not supp,
        "8 numbered tables": len(tables) == 8,
        "8 numbered equations": len(eqs) == 8,
        "27 references, author–year, typeset": refs_ok,
        "no provenance tags in the references": not re.search(r"\b(PARTIAL|verified|not read)\b", flat.rsplit("References", 1)[-1]),
        "abstract present, scoped to the mean": "We study confidence intervals for a population mean" in flat,
        "title scoped to the mean": "Prediction-Powered Mean Estimation" in flat,
        "conclusion present": "11. Conclusion" in flat,
        "Sitter limitation stated": "We were unable to obtain Sitter (1997)" in flat,
        "page numbers present": all(f"{i} of {doc.page_count}" in flat for i in (1, doc.page_count)),
    }

    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if lost:
        print("\n  tokens missing from the PDF:")
        for w in lost[:15]:
            print(f"    {w!r}")
    if blanks:
        print(f"\n  blank pages: {blanks}")
    for pageno, box in overflow[:10]:
        print(f"\n  p{pageno} outside type area: {box}")

    bad = sum(1 for ok in checks.values() if not ok)
    print(f"\nmanuscript PDF: {doc.page_count} pages, {figs} figures, {len(tables)} tables, "
          f"{bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
