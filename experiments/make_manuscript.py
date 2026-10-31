#!/usr/bin/env python3
"""Assemble the final manuscript from the draft, the generated tables and the reference list.

    python experiments/make_manuscript.py > docs/MANUSCRIPT.md

The manuscript is **assembled, never hand-edited**: prose comes from `docs/PAPER_DRAFT.md`, tables
are spliced verbatim from `docs/17_PAPER_TABLES.md` (generated from `results/*.csv`), and
references from `docs/REFERENCES.md`. Status markers used for internal auditing are stripped.
Re-run after any change to the draft or the results.
"""
import re
import sys
from pathlib import Path

DRAFT = Path("docs/PAPER_DRAFT.md")
TABLES = Path("docs/17_PAPER_TABLES.md")
REFS = Path("docs/REFERENCES.md")

#: Where each generated table is spliced in, and which CSV produced it.
TABLE_SOURCES = {
    4: "results/repair_extension/summary.csv",
    1: "results/C_sample_size.csv, results/A_reproduction.csv",
    2: "results/A4_simple_fixes.csv, results/A5_mse.csv",
    3: "results/B_bootstrap.csv, results/B_final.csv",
    5: "results/E_finite_N.csv, results/E_largeN.csv",
    6: "results/F_heavy_tails.csv, F_heavy_f.csv, G_skew.csv, H_hetero.csv, H_step.csv",
    7: "results/I_real_data.csv",
    8: "results/J_mtbench.csv",
}

#: Captions follow the Statistics and Computing rules: "Fig. n" in bold (added by make_docx.py),
#: no punctuation after the number or at the end, every plotted element identified, and nothing
#: that the figure itself states in a title (the journal renderings carry none).
MAIN_FIGURES = [
    (1, "journal/Fig1.png", "Empirical coverage of nominal 95 % intervals against gold-sample size "
        "n = 4 to 50 (log scale), for predictor quality (a) ρ = 0, (b) ρ = 0.5 and (c) ρ = 0.9; "
        "bivariate Gaussian model, `N = 10⁶`, R = 200,000 replications. B1, classical t interval; "
        "B3, PPI++; B4, PPI++ with a Student t quantile; B5, cross-fitted PPI++; B9-nolev, "
        "Algorithm 1 without the leverage term; B9, Algorithm 1 (LC-PPI). The dashed line marks "
        "the nominal level; bars are ±1.96 Monte Carlo standard errors. PPI++ falls to 0.681 at "
        "n = 4, while B9 stays within 0.9495–0.9505 throughout"),
    (2, "journal/Fig2.png", "Empirical coverage against predictor quality ρ = Corr(Y, f) on a "
        "ten-point grid, for gold-sample size (a) n = 10, (b) n = 20 and (c) n = 50; R = 200,000 "
        "replications. Methods as in Fig. 1, plus B6, PPI++ with a hard threshold (|t| > 2). The "
        "dashed line marks the nominal level; bars are ±1.96 Monte Carlo standard errors. The "
        "PPI++ (B3) deficit is flat in ρ (spread ≤ 0.0018); the threshold rule B6 is visibly "
        "non-monotone"),
    (3, "journal/Fig3.png", "Empirical coverage of the published few-label repairs against "
        "gold-sample size n (log scale), for predictor quality (a) ρ = 0, (b) ρ = 0.5 and "
        "(c) ρ = 0.9, on the design of Table 4 (R = 10,000; all methods on the same draws). "
        "Ridge-PPI (Eyre and Madras 2025), PPBoot (Zrnic 2025) and Efron-PPboot2 (Efron 2026) are "
        "compared with the classical interval, PPI++, and Algorithm 1 (LC-PPI) with its `t_min` "
        "variant (step 9). The dashed line marks 0.95 and the shaded band ±0.005 about it. The "
        "asterisk on Ridge-PPI marks an interval that is ours: the paper proposing that estimator "
        "defines none"),
    (4, "journal/Fig4.png", "Empirical coverage against unlabelled-pool size N (log scale) at "
        "ρ = 0.9 in explicit unlabelled mode, for gold-sample size (a) n = 10, (b) n = 20 and "
        "(c) n = 50; R = 200,000 replications. B1, classical t interval; B3, PPI++; B9, Algorithm 1; "
        "B9-R1, B9 with the expectation-form variance; B9-R2, B9 with Welch–Satterthwaite degrees "
        "of freedom; B9-R5, B9 with the exact quantile of (4) (all defined with Table 5). The "
        "dashed line marks the nominal level; bars are ±1.96 Monte Carlo standard errors. B9 "
        "over-covers as N falls and never under-covers on this grid; B9-R2 removes the "
        "over-coverage; B9-R5 does not improve on it"),
    (5, "journal/Fig5.png", "Empirical coverage under multiplicative heteroskedasticity "
        "σ(f) = exp(γf), γ ∈ {0, 0.25, 0.5, 1}, at ρ = 0, for gold-sample size (a) n = 10, "
        "(b) n = 20 and (c) n = 50; R = 200,000 replications. B1, classical t interval; B3, PPI++; "
        "B9, Algorithm 1; B9-HC, Algorithm 1 with the HC3 variance of step 10. The dashed line "
        "marks the nominal level; bars are ±1.96 Monte Carlo standard errors. B9 degrades as γ "
        "grows; B9-HC does not"),
    (6, "journal/Fig6.png", "Galaxies: fraction of R = 20,000 repeated gold subsamples whose "
        "interval contains the population value θ, against gold-sample size n (log scale). B1, "
        "classical t interval; B2, PPI; B3, PPI++; B4, PPI++ with a Student t quantile; B5, "
        "cross-fitted PPI++; B8, BCa bootstrap; B9, Algorithm 1; B9-R2, B9 with "
        "Welch–Satterthwaite degrees of freedom; B9-HC, B9 with the HC3 variance. The dashed line "
        "marks 0.95; bars are ±1.96 Monte Carlo standard errors. This is containment under "
        "subsampling from one fixed finite population of 16,743 items, NOT superpopulation "
        "coverage"),
]

#: The six figures the argument needs. The remaining diagnostic figures in `figures/`
#: are regenerable with `make figures` and are cited in the availability section
#: rather than carried in the manuscript, which has a hard page budget.
SUPP_FIGURES: list[tuple[str, str, str]] = []


#: Statistics and Computing asks for 4 to 6 keywords; each is a term the paper itself uses.
KEYWORDS = ("**Keywords** Prediction-powered inference · Confidence intervals · Finite-sample "
            "coverage · Behrens–Fisher problem · Difference estimator · Survey sampling")

#: Title page. The journal also requires the corresponding author and an active e-mail, and asks for
#: ORCID iDs. None is recorded in this project and none is invented: the placeholder line was
#: removed by the author (Word edit, 2026-10-04 09:15) and the gap is tracked as MANUAL INPUT REQUIRED
#: in docs/STATISTICS_AND_COMPUTING_FINAL_CHECKLIST.md. Add the line here once supplied.
TITLE_PAGE = [
    "# Finite-Sample Confidence Intervals for Prediction-Powered Mean Estimation\n",
    "**Pranav Kumar, Mainak Ghosh, Vansh Garg**\n",
    "Vellore Institute of Technology, Vellore, Tamil Nadu, India\n",
]


def normalise(text: str) -> str:
    """Whitespace- and blockquote-normalised text, for the submission checklist's substring tests."""
    return " ".join(re.sub(r"^> ?", "", text, flags=re.M).split())


#: Tables 1 and 2 each carry a secondary panel — interval width, point-estimate MSE — that the main
#: text quotes but does not need beside it. The panel is split off at this lead-in line and placed
#: in the supplement; the primary coverage panel stays in the body.
TABLE_SPLIT = {
    1: "**Mean interval width ÷ classical**",
    2: "**Point-estimate MSE ÷ classical**",
}


def _table_text(n: int) -> tuple[str, str]:
    """(heading, body) for table `n`, verbatim from the generated file."""
    text = TABLES.read_text()
    parts = re.split(r"^## (Table \d+[^\n]*)$", text, flags=re.M)
    for i in range(1, len(parts), 2):
        if parts[i].startswith(f"Table {n} "):
            return parts[i], parts[i + 1].strip()
    raise KeyError(f"Table {n} not found in {TABLES}")



def table_block(n: int, part: str = "all") -> str:
    """Table `n`, or one panel of it: 'main' drops the secondary panel, 'extra' keeps only it."""
    head, body = _table_text(n)
    mark = TABLE_SPLIT.get(n)
    if part == "main" and mark:
        body = body.split(mark, 1)[0].strip()
    elif part == "extra":
        head = f"Table {n} (continued) — {head.split(' — ', 1)[1]}"
        body = mark + body.split(mark, 1)[1].strip()
    # The CSV that produced each table is recorded in docs/17_PAPER_TABLES.md and TABLE_SOURCES;
    # a results/ path is repository bookkeeping, not journal prose, so it is not printed here.
    return f"**{head}**\n\n{body}"


def main() -> int:
    draft = DRAFT.read_text()

    # strip the internal status markers and the draft's own front matter
    # A marker at the start of a line must lose only the marker: an earlier `\s*` here ate the
    # newline that closed the preceding fenced block, merging the paragraph into the fence and
    # silently dropping its first clause from the manuscript.
    marker = r"`\[(?:PROVED|DERIVED|SIM|OBS|PRIOR|OPEN)\]`"
    body = re.sub(rf"^{marker} ?", "", draft, flags=re.M)
    body = re.sub(rf"[ \t]*{marker}", "", body)
    body = re.sub(r"^\*Draft\..*?`\[OPEN\]`\.\n\n---\n\n", "", body, flags=re.S)
    body = body.split("\n---\n\n## 0. Abstract", 1)
    body = "## 0. Abstract" + body[1] if len(body) > 1 else draft
    # journal front matter: an unnumbered abstract followed by the 4-6 indexing keywords
    body = body.replace("## 0. Abstract", "## Abstract", 1)
    body = body.replace("\n---\n\n## 1. Introduction", "\n" + KEYWORDS + "\n\n---\n\n## 1. Introduction", 1)

    # splice the tables in where the prose refers to them
    out = body
    out = out.replace("**Table 1.** At ten gold labels",
                      table_block(1, "main") + "\n\n**Table 1.** At ten gold labels", 1)
    out = out.replace("**Table 2**, at `n = 10`.",
                      table_block(2, "main") + "\n\n**Table 2**, at `n = 10`.", 1)
    out = out.replace("**Table 3 — the bootstrap.**",
                      table_block(3) + "\n\n**Table 3 — the bootstrap.**", 1)
    out = out.replace("**Table 5**, at `n = 10, ρ = 0.9`",
                      table_block(5) + "\n\n**Table 5**, at `n = 10, ρ = 0.9`", 1)
    out = out.replace("**Table 6**, `R = 200,000`, 192 cells.",
                      "**Table 6**, `R = 200,000`, 192 cells.\n\n" + table_block(6), 1)
    out = out.replace("**Table 7.** PPI++ contains",
                      table_block(7) + "\n\n**Table 7.** PPI++ contains", 1)
    out = out.replace("**Table 4 placeholder**", table_block(4), 1)
    out = out.replace("**Table 8.** PPI++ contains",
                      table_block(8) + "\n\n**Table 8.** PPI++ contains", 1)
    # the draft carries a placeholder reference section; the real one is spliced below
    out = re.sub(r"\*See `docs/FINAL_ABSTRACT\.md`.*?continuity\.\*\n\n", "", out, flags=re.S)
    body_out = out.split("\n## 14. References", 1)[0].rstrip().rstrip("-").rstrip()


    # the draft's internal pointer to the abstract source is not manuscript text

    # main-figure captions: relocated into the body by experiments/make_docx.py
    figs = ["\n---\n\n## Figure manifest\n", "### Main figures\n"]
    for n, f, cap in MAIN_FIGURES:
        figs.append(f"**Figure {n}** — `figures/{f}`\n\n{cap}\n")

    # the clean list only: the access-status table and the notes below it are internal
    refs = REFS.read_text().split("---\n", 1)[1].split("\n## ", 1)[0]

    for line in TITLE_PAGE:
        print(line)
    print(body_out)
    print("\n".join(figs))
    print("\n---\n\n## References\n")
    print(refs.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
