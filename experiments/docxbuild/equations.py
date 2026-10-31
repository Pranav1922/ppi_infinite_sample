r"""Hand-written LaTeX for every display equation in docs/MANUSCRIPT.md.

Each entry is keyed by the *exact* source text of the fenced block it replaces. The builder looks
each block up by that key; a block whose text is not in this table is rendered as a monospace
listing rather than as an equation, and a key here that no longer appears in the manuscript raises.
That coupling is deliberate: it makes it impossible for an edit to the manuscript's mathematics to
be silently typeset from stale LaTeX.
"""
from __future__ import annotations

import re
import unicodedata

# key: source block text (lines joined by "\n", each stripped) -> list of (latex, eq_number|None)
EQUATIONS: dict[str, list[tuple[str, str | None]]] = {
    "P{|T₁| < v}  ≤  P{|V_γ| < v}  ≤  P{|T₂| < v}    for all γ ∈ [0,1],": [
        (r"P\{|T_1| < v\} \leq P\{|V_\gamma| < v\} \leq P\{|T_2| < v\}"
         r"\text{   for all }\gamma \in [0,1],", None),
    ],
    "S_ff = Σ_i (f_i − f̄_n)²   (a sum of squares, not a variance),   S_fY = Σ_i (f_i − f̄_n)(Y_i − Ȳ),\n"
    "b = S_fY / S_ff,           s² = RSS/(n−2),        r = n/N,       λ̂ = b/(1+r).": [
        (r"S_{ff} = \sum_i (f_i - \bar f_n)^2 \quad \text{(a sum of squares, not a variance)},"
         r"\qquad S_{fY} = \sum_i (f_i - \bar f_n)(Y_i - \bar Y),", None),
        (r"b = S_{fY}/S_{ff}, \qquad s^2 = \text{RSS}/(n-2), \qquad r = n/N,"
         r"\qquad \hat\lambda = b/(1+r).", None),
    ],
    "θ̂_LC − θ  =  ε̄  +  β(f̄_n − μ_f)  +  b(f̄_N − f̄_n).": [
        (r"\hat\theta_{LC} - \theta = \bar\varepsilon + \beta(\bar f_n - \mu_f)"
         r" + b(\bar f_N - \bar f_n).", None),
    ],
    "θ̂_LC − θ  =  ε̄  +  (b − β)(μ_f − f̄_n)  +  b·δ ,        θ̂_LC = Ȳ + b(f̄_N − f̄_n).      (1)": [
        (r"\hat\theta_{LC} - \theta = \bar\varepsilon + (b - \beta)(\mu_f - \bar f_n)"
         r" + b\cdot\delta, \qquad \hat\theta_{LC} = \bar Y + b(\bar f_N - \bar f_n).", "1"),
    ],
    "(θ̂_LC − θ) / ( s·√(1/n + (μ_f − f̄_n)²/S_ff) )  ~  t_{n−2}   exactly.                    (2)": [
        (r"\frac{\hat\theta_{LC} - \theta}{s\sqrt{1/n + (\mu_f - \bar f_n)^2/S_{ff}}}"
         r" \sim t_{n-2} \text{  exactly.}", "2"),
    ],
    "G = (θ̂_LC − θ − βδ) / ( s·√(1/n + Δ²/S_ff) )  ~  t_{n−2}": [
        (r"G = \frac{\hat\theta_{LC} - \theta - \beta\delta}"
         r"{s\sqrt{1/n + \Delta^2/S_{ff}}} \sim t_{n-2}", None),
    ],
    "Var(θ̂_LC | design)  =  σ²/n  +  σ²(μ_f − f̄_n)²/S_ff  +  (β² + σ²/S_ff)·σ_f²/N .         (3)": [
        (r"\text{Var}(\hat\theta_{LC} \mid \text{design}) = \sigma^2/n"
         r" + \sigma^2(\mu_f - \bar f_n)^2/S_{ff} + (\beta^2 + \sigma^2/S_{ff})\cdot\sigma_f^2/N.", "3"),
    ],
    "T  =  Z / √( (1−s)·Q/ν + s ) ,        Z ~ N(0,1) ⊥ Q ~ χ²_ν ,   ν = n−2.                 (4)": [
        (r"T = \frac{Z}{\sqrt{(1-s)\cdot Q/\nu + s}}, \qquad Z \sim N(0,1) \perp Q \sim \chi^2_\nu,"
         r" \qquad \nu = n-2.", "4"),
    ],
    "1 − α  ≤  coverage  ≤  2Φ(t_{n−2,1−α/2}) − 1 .                                            (5)": [
        (r"1 - \alpha \leq \text{coverage} \leq 2\Phi(t_{n-2,1-\alpha/2}) - 1.", "5"),
    ],
    "|Δ_cov|  =  |coverage_practical − coverage_(4)|  ≤  B(n, ρ, α, N).                       (6)": [
        (r"|\Delta_{\text{cov}}| = |\text{coverage}_{\text{practical}}"
         r" - \text{coverage}_{(4)}| \leq B(n, \rho, \alpha, N).", "6"),
    ],
    "T_practical  ⟶  W / √(χ²_{N−1}/(N−1))  ~  t_{N−1}                                          (7)": [
        (r"T_{\text{practical}} \to \frac{W}{\sqrt{\chi^2_{N-1}/(N-1)}} \sim t_{N-1}", "7"),
    ],
    "coverage  ⟶  P(|t_{N−1}| ≤ t_{n−2, 1−α/2}) ,     below 1−α exactly when N < n − 1.         (8)": [
        (r"\text{coverage} \to P(|t_{N-1}| \leq t_{n-2,\,1-\alpha/2}),"
         r"\quad \text{  below }1-\alpha\text{ exactly when }N < n-1.", "8"),
    ],
}


def demo() -> None:
    from omml import check_roundtrip, latex_to_omml
    for key, eqs in EQUATIONS.items():
        for latex, _ in eqs:
            frag = latex_to_omml(latex)
            assert frag.startswith("<m:oMathPara"), key[:40]
            check_roundtrip(latex)
    print(f"equations demo ok ({sum(len(v) for v in EQUATIONS.values())} equations compile)")


if __name__ == "__main__":
    demo()


#: Displayed blocks that are a *sequence* of labelled mathematical lines rather than a single
#: numbered equation: an algorithm whose steps are formulae, and a pair of labelled quantile rules.
#: Each row is (leading text, LaTeX, trailing text); the LaTeX becomes a native Word equation, so
#: no line is left as a string such as `t_{n-2}` or `sqrt(V)`.
MATH_BLOCKS: dict[str, list[tuple[str, str, str]]] = {
    "Input:  n gold pairs (Y_i, f_i) ;  N pool predictions f_j^u ;  level 1 − α\n"
    "1. Ybar, fbar_n from the gold sample ;  fbar_N from the pool\n"
    "2. S_ff = Σ_i (f_i − fbar_n)² ;  S_fY = Σ_i (f_i − fbar_n)(Y_i − Ybar) ;  b = S_fY/S_ff\n"
    "3. s² = RSS/(n−2) ,  RSS = Σ_i (Y_i − Ybar − b(f_i − fbar_n))²\n"
    "4. h = (fbar_N − fbar_n)² / S_ff\n"
    "5. sigma_hat_f² from the pool\n"
    "6. V = s²(1/n + h) + b²·sigma_hat_f²/N\n"
    "7. theta_hat = Ybar + b(fbar_N − fbar_n)\n"
    "8. CI = theta_hat ± t_{n−2, 1−α/2}·√V\n"
    "9. finite-pool safeguard: q = t_{min(n−2, N−1), 1−α/2} in place of the step-8 quantile\n"
    "10. suspected heteroskedasticity: V = Σ_i w_i²e_i²/(1−h_ii)² + b²·sigma_hat_f²/N": [
        ("Input:", r"n \text{ gold pairs } (Y_i, f_i)\ ;\ N \text{ pool predictions } f_j^u\ ;\ "
                   r"\text{level } 1 - \alpha", ""),
        ("1.", r"\bar Y,\ \bar f_n \text{ from the gold sample}\ ;\ "
               r"\bar f_N \text{ from the pool}", ""),
        ("2.", r"S_{ff} = \sum_i (f_i - \bar f_n)^2\ ;\ "
               r"S_{fY} = \sum_i (f_i - \bar f_n)(Y_i - \bar Y)\ ;\ b = S_{fY}/S_{ff}", ""),
        ("3.", r"s^2 = \text{RSS}/(n-2),\ "
               r"\text{RSS} = \sum_i (Y_i - \bar Y - b(f_i - \bar f_n))^2", ""),
        ("4.", r"h = (\bar f_N - \bar f_n)^2 / S_{ff}", "leverage"),
        ("5.", r"\hat\sigma_f^2", "from the pool"),
        ("6.", r"V = s^2(1/n + h) + b^2\cdot\hat\sigma_f^2/N", ""),
        ("7.", r"\hat\theta = \bar Y + b(\bar f_N - \bar f_n)", ""),
        ("8.", r"\text{CI} = \hat\theta \pm t_{n-2,\,1-\alpha/2}\cdot\sqrt V", ""),
        ("9. finite-pool safeguard:", r"q = t_{\text{min}(n-2,\,N-1),\,1-\alpha/2}",
         "in place of the step-8 quantile"),
        ("10. suspected heteroskedasticity:",
         r"V = \sum_i w_i^2 e_i^2/(1-h_{ii})^2 + b^2\cdot\hat\sigma_f^2/N", ""),
    ],
}


def demo_math_blocks() -> None:
    from omml import check_roundtrip, latex_to_omml
    n = 0
    for key, rows in MATH_BLOCKS.items():
        for _, latex, _ in rows:
            assert latex_to_omml(latex, display=False).startswith("<m:oMath"), key[:40]
            check_roundtrip(latex)
            n += 1
    print(f"math-block demo ok ({n} equation lines compile and round-trip)")


#: Inline spans whose source text reads as raw LaTeX (a braced sub- or superscript) and which are
#: therefore rendered as native Word math rather than as styled Unicode. The mapping is explicit,
#: one hand-checked entry per distinct span, because a general Unicode-to-LaTeX parser would risk
#: silently altering notation. Every other inline span is ordinary typographic maths and is left
#: exactly as it is.
INLINE_MATH: dict[str, str] = {
    "t_{n−2}":                   r"t_{n-2}",
    "t_{N−1}":                   r"t_{N-1}",
    "t_{min(f₁,f₂)}":            r"t_{\text{min}(f_1,f_2)}",
    "t_{min(n−2, N−1)}":         r"t_{\text{min}(n-2,\,N-1)}",
    "e_i = Y_i − Ȳ − b(f_i − f̄_n)":
        r"e_i = Y_i - \bar Y - b(f_i - \bar f_n)",
    "h_ii = 1/n + (f_i − f̄_n)²/S_ff":
        r"h_{ii} = 1/n + (f_i - \bar f_n)^2/S_{ff}",
    "E[Δ² | design] = (μ_f − f̄_n)² + σ_f²/N":
        r"E[\Delta^2 \mid \text{design}] = (\mu_f - \bar f_n)^2 + \sigma_f^2/N",
    "t_{ν_eff}":                 r"t_{\nu_{\text{eff}}}",
    "χ²_{n−2}":                  r"\chi^2_{n-2}",
    "χ²_{N−1}":                  r"\chi^2_{N-1}",
    "v = t_{n−2,1−α/2}":         r"v = t_{n-2,1-\alpha/2}",
    "f_iW_i ~ χ²_{f_i}":         r"f_i W_i \sim \chi^2_{f_i}",
    "(n−2)s²/σ² ~ χ²_{n−2}":     r"(n-2)s^2/\sigma^2 \sim \chi^2_{n-2}",
    "P(|t_{ν_eff}| ≤ t_{n−2,1−α/2})":
        r"P(|t_{\nu_{\text{eff}}}| \leq t_{n-2,1-\alpha/2})",
    "O_P(N^{−1/2})":             r"O_P(N^{-1/2})",
    "B = C(n,ρ,α)/N + O(N^{−3/2})":
        r"B = C(n,\rho,\alpha)/N + O(N^{-3/2})",
}


def demo_inline_math() -> None:
    from omml import check_roundtrip, latex_to_omml
    for src, latex in INLINE_MATH.items():
        assert latex_to_omml(latex, display=False).startswith("<m:oMath"), src
        check_roundtrip(latex)
    print(f"inline-math demo ok ({len(INLINE_MATH)} inline spans compile and round-trip)")


# ---------------------------------------------------------------------------------------------
# Inline maths written in the Markdown as Unicode inside a code span.
#
# The manuscript spells inline maths the way plain text has to: ASCII subscripts (`S_ff`,
# `sigma_f²`), combining accents (`f̄_N`), a radical character (`√(n−2)`).  Word must show these
# as real equations, so they are transcribed to the LaTeX subset and emitted as OMML like any
# display equation.  The transcription is mechanical -- it renames notation, it never rewrites
# it -- and every result is checked by omml.check_roundtrip.

_SUP = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9","⁻":"-","½":"1/2"}
_SUB = {"₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9","ᵢ":"i"}
_ACCENT = {"\u0302": "hat", "\u0304": "bar", "\u0303": "tilde"}
_FUNCS = ("Var", "Cov", "Corr", "exp", "log", "min", "max", "lim", "sd", "bsd", "coverage")

# Code spans that are code, not maths: file paths, DOIs, shell commands, status markers,
# column names and method IDs.  These stay in the monospace face.
# A path and a DOI both contain "/", which is also division, so they are excluded by shape
# rather than by the slash alone.
_NOT_MATH = re.compile(
    r"^(?:10\.\d|make\s|scipy|ddof|rho$|R2$|PARTIAL$|VERIFIED$|B9$|R[15]$)"
    r"|^(?:docs|experiments|results|figures|src|tests)/"
    r"|\.(?:py|csv|md|png)\b")


def is_math_variable(span: str) -> bool:
    """A lone variable, optionally accented: `n`, `ρ`, `λ̂`.  Notation, not an equation -- it is
    set as italic prose, which is both the journal convention and the only way it stays on the
    same baseline and size as the sentence around it."""
    return bool(re.fullmatch(r"[A-Za-zα-ωΑ-Ω][\u0300-\u036f]?",
                             unicodedata.normalize("NFD", span)))


def is_inline_math(span: str) -> bool:
    """True for a code span the reader should see as mathematics rather than as literal text.

    Mathematics is a relation, an operator with operands, a script, an accent, a radical, a
    Greek letter or a bare variable.  Numbers, file paths, DOIs, shell commands and method
    labels are data and identifiers, not equations, and keep the monospace face."""
    if _NOT_MATH.search(span) or is_math_variable(span):
        return False
    if span in INLINE_MATH:
        return True
    return bool(re.search(r"[=~≤≥<>∈→√_^±×·−+/|]|[⁰-⁹⁻½²³ᵢ₀-₉]|[α-ωΑ-Ω]", span)
                or any(a in unicodedata.normalize("NFD", span) for a in _ACCENT))


def needs_equation(span: str) -> bool:
    """True when a span must be a real equation object rather than styled text.

    A script, accent, radical or fraction needs Office Math to be set correctly. Plain notation
    -- a relation among named parameters and numbers, such as `n = 10` or `R = 200,000` -- does
    not: it is ordinary text, and setting it as text keeps it at the line height of the prose
    around it. That matters because Word grows a line to fit an inline equation object, so one
    equation per quoted parameter value costs whole pages of pagination that LibreOffice, which
    packs those lines more tightly, does not show.
    """
    if span in INLINE_MATH:
        return True
    # a bare slash is set as a literal slash, not a built-up fraction, so it needs no object
    if re.search(r"[√^_]", span) or re.search(r"[\u2070-\u209f\u00b2\u00b3\u00b9\u00bd]", span):
        return True
    return any(a in unicodedata.normalize("NFD", span) for a in _ACCENT)


def unicode_to_latex(span: str) -> str:
    """Transcribe one Unicode maths span into the LaTeX subset omml.py parses."""
    if span in INLINE_MATH:                       # hand-written where the automatic rule is coarse
        return INLINE_MATH[span]
    s = span.replace("{", r"\{").replace("}", r"\}")   # before any brace of ours is added
    s = unicodedata.normalize("NFD", s)
    for mark, cmd in _ACCENT.items():             # f + combining macron -> \bar{f}
        s = re.sub(r"([^\\])" + mark, rf"\\{cmd}{{\1}}", s)
    s = s.replace("^½", "^{1/2}")                 # a power ½: superscript, not a literal caret
    for u, d in _SUP.items():
        s = s.replace(u, "^{" + d + "}")
    for u, d in _SUB.items():
        s = s.replace(u, "_{" + d + "}")
    s = re.sub(r"√\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"√([A-Za-z0-9]+)", r"\\sqrt{\1}", s)
    s = re.sub(r"_(?!\{|\\)([A-Za-z0-9]+)", r"_{\1}", s)
    s = re.sub(r"\^(?!\{|\\)([A-Za-z0-9]+)", r"^{\1}", s)
    # a multi-letter subscript is a label, not an index, so it is set upright
    s = re.sub(r"_\{([A-Za-z]{2,})\}", r"_{\\text{\1}}", s)
    for fn in _FUNCS:
        # an accented function name has already been split, e.g. Cov-with-hat -> \hat{C}ov;
        # keep the accent on the first letter and set the whole name upright
        for cmd in _ACCENT.values():
            s = s.replace(rf"\{cmd}{{{fn[0]}}}{fn[1:]}",
                          rf"\{cmd}{{\text{{{fn[0]}}}}}\text{{{fn[1:]}}}")
        s = re.sub(rf"(?<![A-Za-z\\{{]){fn}(?![A-Za-z}}])", rf"\\text{{{fn}}}", s)
    return s


def demo_unicode_math() -> None:
    from omml import check_roundtrip, latex_to_omml
    cases = {
        "f̄_N": r"\bar{f}_{N}", "σ_f²": r"σ_{f}^{2}", "1/√(n−2)": r"1/\sqrt{n−2}",
        "ρ ∈ {0, 0.5, 0.9}": r"ρ ∈ \{0, 0.5, 0.9\}", "f_j^u": r"f_{j}^{u}",
        "ν_eff = (n−2)/(1−s)²": r"ν_{\text{eff}} = (n−2)/(1−s)^{2}",
    }
    for src, want in cases.items():
        got = unicode_to_latex(src)
        assert got == want, f"{src!r}: {got!r} != {want!r}"
        check_roundtrip(got)
        assert latex_to_omml(got, display=False).startswith("<m:oMath"), src
    assert not is_inline_math("experiments/make_tables.py")
    assert not is_inline_math("10.1214/aos/1176343594")
    assert not is_inline_math("PARTIAL")
    assert not is_inline_math("results/E_finite_N.csv, results/E_largeN.csv")
    assert is_inline_math("1/√(n−2)")          # division, not a path
    assert is_inline_math("N < n − 1")
    assert is_math_variable("n") and is_math_variable("λ̂") and not is_inline_math("n")
    assert is_inline_math("(n, ρ)")
    assert not is_inline_math("127") and not is_inline_math("[0.988, 1.025]")
    assert unicode_to_latex("Ĉov[h,f]") == r"\hat{\text{C}}\text{ov}[h,f]"
    assert is_inline_math("θ̂_LC = Ȳ + b(μ_f − f̄_n)")
    print(f"unicode-math demo ok ({len(cases)} transcriptions)")
