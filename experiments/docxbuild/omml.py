r"""Minimal LaTeX-subset -> OMML (Office Math) converter.

Covers only what docs/MANUSCRIPT.md's display equations use: \frac, \sqrt, sub/superscripts,
\bar, \hat, \text, and a symbol table. Deliberately small: every equation in the manuscript is
hand-written LaTeX in equations.py and checked against the Markdown source, so the converter never
has to guess at ambiguous input.
"""
from __future__ import annotations

import re
from collections import Counter
from xml.sax.saxutils import escape, unescape

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "Theta": "Θ", "lambda": "λ",
    "Lambda": "Λ", "mu": "μ", "nu": "ν", "rho": "ρ", "sigma": "σ", "Sigma": "Σ", "tau": "τ",
    "phi": "φ", "Phi": "Φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Omega": "Ω",
    "times": "×", "cdot": "·", "pm": "±", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "approx": "≈", "sim": "~", "in": "∈", "perp": "⊥", "to": "→",
    "infty": "∞", "partial": "∂", "sum": "Σ", "prod": "Π", "int": "∫", "mid": "|",
    "left": "", "right": "", "!": "", ",": " ", ";": " ",
    "quad": " ", "qquad": "  ", " ": " ", "colon": ":",
    "{": "{", "}": "}", "%": "%", "&": "&", "#": "#", "$": "$", "_": "_",
}
ACCENTS = {"bar": "̄", "hat": "̂", "tilde": "̃", "vec": "⃗", "dot": "̇"}


def _tokenize(s: str) -> list[str]:
    return [t for t in re.findall(r"\\[A-Za-z]+|\\.|[{}^_]|[^\\{}^_]", s) if t != ""]


def _parse(tokens: list[str], i: int = 0, stop: str | None = None) -> tuple[list, int]:
    """Parse into a node list until `stop` token (or end). Returns (nodes, next_index)."""
    out: list = []
    while i < len(tokens):
        t = tokens[i]
        if t == "}" or (stop is not None and t == stop):
            if stop is None:
                raise ValueError("unbalanced '}' in LaTeX")
            return out, i + 1
        if t in ("^", "_"):
            base = out.pop() if out else ("grp", [])
            arg, i = _parse_one(tokens, i + 1)
            kind = "sup" if t == "^" else "sub"
            # x_a^b and x^b_a must collapse to one sSubSup, not nested scripts
            if base[0] == "sub" and kind == "sup":
                out.append(("subsup", base[1], base[2], arg))
            elif base[0] == "sup" and kind == "sub":
                out.append(("subsup", base[1], arg, base[2]))
            else:
                out.append((kind, base, arg))
            continue
        node, i = _parse_one(tokens, i)
        out.append(node)
    return out, i


def _parse_one(tokens: list[str], i: int) -> tuple[tuple, int]:
    """Parse exactly one unit -- a group, a command with its arguments, or a single character."""
    while i < len(tokens) and tokens[i] == " ":
        i += 1
    if i >= len(tokens):
        raise ValueError("missing argument")
    t = tokens[i]
    if t == "{":
        grp, j = _parse(tokens, i + 1, "}")
        return ("grp", grp), j
    if t.startswith("\\"):
        cmd = t[1:]
        if cmd == "frac":
            num, i = _parse_one(tokens, i + 1)
            den, i = _parse_one(tokens, i)
            return ("frac", num, den), i
        if cmd == "sqrt":
            body, i = _parse_one(tokens, i + 1)
            return ("rad", body), i
        if cmd == "bar":
            body, i = _parse_one(tokens, i + 1)
            return ("bar", body), i
        if cmd in ACCENTS:
            body, i = _parse_one(tokens, i + 1)
            return ("acc", ACCENTS[cmd], body), i
        if cmd in ("text", "mathrm", "operatorname"):
            body, i = _read_raw(tokens, i + 1)
            return ("text", body), i
        if cmd in ("quad", "qquad"):
            # a spacing command goes in a normal-text run: LibreOffice's math layout drops white
            # space inside an ordinary math run, which closed up "b·δ, θ̂" in Eq. (1)
            return ("text", SYMBOLS[cmd]), i + 1
        if cmd in SYMBOLS:
            return ("run", SYMBOLS[cmd]), i + 1
        raise ValueError(f"unsupported LaTeX command: \\{cmd}")
    return ("run", t), i + 1


def _read_raw(tokens: list[str], i: int) -> tuple[str, int]:
    """Read a braced group as literal text (for \\text{...})."""
    if tokens[i] != "{":
        return tokens[i], i + 1
    depth, buf, i = 1, [], i + 1
    while i < len(tokens) and depth:
        if tokens[i] == "{":
            depth += 1
        elif tokens[i] == "}":
            depth -= 1
            if not depth:
                break
        buf.append(SYMBOLS.get(tokens[i][1:], tokens[i]) if tokens[i].startswith("\\") else tokens[i])
        i += 1
    return "".join(buf), i + 1


def _emit(node) -> str:
    kind = node[0]
    if kind == "run":
        # LaTeX writes subtraction as ASCII "-"; in an OMML math run it must be the minus sign
        # U+2212 (as the inline maths already is), or it renders as a short hyphen
        return f"<m:r><m:t xml:space='preserve'>{escape(node[1].replace('-', '−'))}</m:t></m:r>"
    if kind == "text":
        return (f"<m:r><m:rPr><m:nor/></m:rPr>"
                f"<m:t xml:space='preserve'>{escape(node[1])}</m:t></m:r>")
    if kind == "grp":
        return "".join(_emit(c) for c in node[1])
    if kind == "frac":
        return ("<m:f><m:fPr><m:type m:val='bar'/></m:fPr>"
                f"<m:num>{_emit(node[1])}</m:num><m:den>{_emit(node[2])}</m:den></m:f>")
    if kind == "rad":
        return ("<m:rad><m:radPr><m:degHide m:val='1'/></m:radPr><m:deg/>"
                f"<m:e>{_emit(node[1])}</m:e></m:rad>")
    if kind == "bar":
        return ("<m:bar><m:barPr><m:pos m:val='top'/></m:barPr>"
                f"<m:e>{_emit(node[1])}</m:e></m:bar>")
    if kind == "acc":
        return (f"<m:acc><m:accPr><m:chr m:val='{escape(node[1])}'/></m:accPr>"
                f"<m:e>{_emit(node[2])}</m:e></m:acc>")
    if kind == "sub":
        return f"<m:sSub><m:e>{_emit(node[1])}</m:e><m:sub>{_emit(node[2])}</m:sub></m:sSub>"
    if kind == "sup":
        return f"<m:sSup><m:e>{_emit(node[1])}</m:e><m:sup>{_emit(node[2])}</m:sup></m:sSup>"
    if kind == "subsup":
        return (f"<m:sSubSup><m:e>{_emit(node[1])}</m:e><m:sub>{_emit(node[2])}</m:sub>"
                f"<m:sup>{_emit(node[3])}</m:sup></m:sSubSup>")
    raise ValueError(kind)


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def latex_to_omml(latex: str, display: bool = True,
                  font: str | None = None, size_pt: float | None = None) -> str:
    """Return an OMML fragment (<m:oMathPara> if display, else <m:oMath>).

    An equation set inside a line of prose has to match the prose: without an explicit face and
    size every run falls back to the renderer's default maths font at the default size, which
    sits larger and lower than the surrounding text.
    """
    nodes, _ = _parse(_tokenize(latex))
    body = "".join(_emit(n) for n in nodes)
    if font or size_pt:
        rpr = "".join(
            ([f"<w:rFonts xmlns:w='{W}' w:ascii='{escape(font)}' w:hAnsi='{escape(font)}'/>"]
             if font else [])
            + ([f"<w:sz xmlns:w='{W}' w:val='{int(round(size_pt * 2))}'/>"] if size_pt else []))
        body = body.replace("<m:r>", f"<m:r><w:rPr xmlns:w='{W}'>{rpr}</w:rPr>")
    math = f"<m:oMath xmlns:m='{M}'>{body}</m:oMath>"
    if not display:
        return math
    return (f"<m:oMathPara xmlns:m='{M}'>"
            f"<m:oMathParaPr><m:jc m:val='center'/></m:oMathParaPr>{math}</m:oMathPara>")


def literal_chars(latex: str) -> str:
    """The visible characters a correct rendering must contain, in order."""
    out = []
    for t in _tokenize(latex):
        if t.startswith("\\"):
            cmd = t[1:]
            if cmd in SYMBOLS:
                out.append(SYMBOLS[cmd])
            continue
        if t in ("{", "}", "^", "_", " "):
            continue
        out.append(t)
    return "".join(out).replace(" ", "")


def check_roundtrip(latex: str) -> None:
    """Raise if any literal character of the LaTeX is missing from the emitted OMML.

    Compares character multisets, not order: OMML writes a sub-superscript's subscript before its
    superscript, so order legitimately differs. This is the guard that catches a silent truncation
    -- the failure mode that would otherwise put a subtly wrong equation into the manuscript
    without anything erroring.
    """
    frag = latex_to_omml(latex, display=False)
    emitted = "".join(re.findall(r"<m:t(?:\s[^>]*)?>(.*?)</m:t>", frag))
    # "-" is emitted as the minus sign U+2212, so the two count as the same character
    got = Counter(unescape(emitted).replace(" ", "").replace("−", "-"))
    want = Counter(literal_chars(latex).replace("−", "-"))
    lost = want - got
    if lost:
        raise ValueError(f"OMML lost {dict(lost)} from: {latex[:70]}\n  want: {sorted(want.elements())}"
                         f"\n  got:  {sorted(got.elements())}")


def demo() -> None:
    x = latex_to_omml(r"\frac{a}{b}", display=False)
    assert "<m:f>" in x and "<m:num><m:r><m:t xml:space='preserve'>a</m:t></m:r></m:num>" in x
    assert "<m:rad>" in latex_to_omml(r"\sqrt{x}", display=False)
    assert "<m:bar>" in latex_to_omml(r"\bar f", display=False)
    assert "m:chr m:val='̂'" in latex_to_omml(r"\hat f", display=False)
    # x_a^b must collapse to one sSubSup, not nested scripts
    y = latex_to_omml(r"x_a^b", display=False)
    assert y.count("<m:sSubSup>") == 1 and "<m:sSub>" not in y
    assert latex_to_omml(r"t_{n-2}", display=False).count("<m:sSub>") == 1
    assert "<m:nor/>" in latex_to_omml(r"\text{exactly}", display=False)
    assert "θ" in latex_to_omml(r"\theta", display=False)
    try:
        latex_to_omml(r"\bogus{x}")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported command should raise")
    # \bar must produce a bar over the *symbol*, never over a preceding space
    b = latex_to_omml(r"\bar Y", display=False)
    assert "<m:bar>" in b and "Y" in b and "<m:t xml:space='preserve'> </m:t>" not in b
    # \frac's denominator must capture its whole braced group
    f = latex_to_omml(r"\frac{a}{s\sqrt{x}}", display=False)
    assert "<m:rad>" in f and f.index("<m:rad>") < f.index("</m:den>")
    # nothing after a \frac may be dropped
    check_roundtrip(r"\frac{a}{s\sqrt{x}} \sim t_{n-2}")
    try:
        latex_to_omml(r"a}b")
    except ValueError:
        pass
    else:
        raise AssertionError("unbalanced brace should raise")
    print("omml demo ok")


if __name__ == "__main__":
    demo()
