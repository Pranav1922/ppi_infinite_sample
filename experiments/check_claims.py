#!/usr/bin/env python3
"""Claim-consistency check: scan the draft for phrasing the claim audit forbids, and verify that
every number quoted in the draft appears in the generated tables.

    python experiments/check_claims.py

Exits non-zero on a violation. This is the executable form of `docs/FINAL_CLAIM_AUDIT.md`.
"""
import re
import sys
from pathlib import Path

DRAFT = Path("docs/PAPER_DRAFT.md")
ABSTRACT = Path("docs/FINAL_ABSTRACT.md")
TABLES = Path("docs/17_PAPER_TABLES.md")

#: (regex, why it is forbidden). Matched case-insensitively against the draft's prose.
FORBIDDEN = [
    (r"\bnovel\b", "FINAL_CONTRIBUTION forbids 'novel' — nothing here supports it"),
    (r"first[- ]ever|\bunprecedented\b|state[- ]of[- ]the[- ]art",
     "banned superlative (paper-preparation brief)"),
    (r"we (prove|show) that the (interval|correction) is conservative",
     "audit §6: conservativeness is Mickey & Brown (1966), for the idealised statistic"),
    (r"provably (valid|conservative)", "audit §9: finite-N validity is OPEN"),
    (r"we (introduce|propose) a (new )?degrees[- ]of[- ]freedom",
     "audit §7: Welch-Satterthwaite, instantiated"),
    (r"real[- ]data experiment confirms", "audit §14: not superpopulation coverage"),
    (r"(galaxies|real[- ]data).{0,60}\b95\s?%\s?coverage", "audit §14"),
    (r"all fail to reach 0\.945|none reaches 0\.945",
     "audit §4 (added): B6 at |t|>3 reaches 0.9489 at rho=0"),
    # Added 2026-08-25 after the consolidated prior-art audit. Each of these is now prior art.
    (r"first (to )?(identify|show|document|demonstrate|establish)",
     "prior art: PPI++ small-n instability is Mani et al. (2025) and Eyre & Madras (2025)"),
    (r"first finite[- ]sample theory",
     "prior art: finite-sample PPI analyses exist (Mani et al. 2025)"),
    (r"we (introduce|propose) (a |the )?leverage[- ]corrected",
     "the regression-estimator construction is classical (Seber & Lee); we transfer it"),
    (r"novel t[_ ]?min|new t[_ ]?min",
     "t_min is Hsu (1938) / Mickey & Brown (1966)"),
    (r"we (establish|prove) (the )?optimality of t[_ ]?min",
     "t_min optimality is Wang & Liu (2026) Theorem 1"),
    (r"exact finite[- ]sample (PPI )?confidence interval",
     "no exact finite-N interval is claimed; Theorem 1 bounds the GAP"),
    (r"no one has studied|literature contains no finite[- ]sample interval analysis",
     "overstated: estimator-level finite-sample analyses exist"),
    # Added 2026-08-26 with the repair-extension results. Each method was evaluated on ONE design.
    (r"Ridge-PPI (is |are )?(invalid|fails) (in general|universally)",
     "scope: evaluated only on M1, n<=20, N=1000, rho in {0,0.5,0.9}, with OUR interval"),
    (r"PPBoot (is |are )?(invalid|fails) (in general|universally)",
     "scope: evaluated only on the design of Table 4"),
    (r"Efron('s)? (method )?(is )?(invalid|fails) (in general|universally)",
     "scope: evaluated only on the design of Table 4, with two adaptations"),
    (r"(Sigmoid-PPI|cross-PPBoot)[^.]{0,60}(failed|fails|under[- ]covers|was tested)",
     "neither was evaluated: both are NOT APPLICABLE to this design"),
    (r"(?<!nor that )(?<!not )all (possible )?(PPI )?repairs fail",
     "overclaim: only the named repairs on the stated design were evaluated"),
    (r"\bwe (prove|proved)\b(?!.{0,200}Proposition 1)", None),   # counted, not banned
]

#: Phrases that MUST be present — the disclaimers the audit requires.
REQUIRED = [
    (r"we claim neither", "the explicit disclaimer after Mickey & Brown / Welch-Satterthwaite"),
    (r"not (conventional )?superpopulation coverage", "the Galaxies caveat"),
    (r"Mickey and Brown", "attribution for conservativeness"),
    (r"Mozer", "attribution for the PPI/GREG equivalence"),
    (r"unable to obtain Sitter \(1997\)", "the Sitter limitation"),
    (r"uniformly in `?.?rho|uniformly in predictor quality", "the corrected H2 phrasing"),
    (r"Eyre and Madras", "the prior art for the OLS reading of lambda"),
    (r"is \*\*not\*\* our discovery|not our discovery",
     "the explicit disclaimer that small-n instability is prior art"),
    (r"we do not present it as new|not ours", "the LC-PPI construction disclaimer"),
    (r"the interval is \*\*ours\*\*|the interval is ours, not theirs",
     "the Ridge-PPI interval-provenance disclaimer"),
    (r"[Nn]ot applicable", "the Sigmoid-PPI / cross-PPBoot applicability wording"),
]


def prose(text):
    """Strip the status-marker legend and the audit tables, which legitimately quote banned words."""
    text = re.sub(r"^\| .*\|$", "", text, flags=re.M)          # markdown tables
    text = re.sub(r"^> .*$", "", text, flags=re.M)             # blockquote notes
    return text


def main() -> int:
    draft = DRAFT.read_text()
    body = prose(draft)
    fails = []

    for pat, why in FORBIDDEN:
        if why is None:
            continue
        for m in re.finditer(pat, body, re.I):
            line = body[:m.start()].count("\n") + 1
            fails.append(f"FORBIDDEN  {DRAFT}:{line}  /{pat}/  -> {why}\n"
                         f"           ...{body[max(0,m.start()-90):m.end()+90].strip()}...")

    for pat, why in REQUIRED:
        if not re.search(pat, draft, re.I):
            fails.append(f"MISSING    required phrasing /{pat}/ -> {why}")

    # "we prove" must appear at most twice and only near Proposition 1
    proofs = [m.start() for m in re.finditer(r"\bwe prove\b", draft, re.I)]
    for pos in proofs:
        ctx = draft[max(0, pos - 400):pos + 400]
        if "Proposition 1" not in ctx and "shifted" not in ctx and "pivotal" not in ctx:
            fails.append(f"OVERCLAIM  'we prove' at char {pos} is not about Proposition 1")

    # every 4-decimal number in the draft must appear in the generated tables or theory doc
    tables = TABLES.read_text() + Path("docs/theory_finite_N.md").read_text() \
        + Path("docs/16_RESULTS_SUMMARY.md").read_text()
    quoted = set(re.findall(r"0\.\d{4}\b", body))
    missing = sorted(q for q in quoted if q not in tables)
    for q in missing:
        fails.append(f"UNSOURCED  {q} appears in the draft but in no generated table")

    # the abstract in the draft and in FINAL_ABSTRACT.md must not drift
    a_draft = re.search(r"## 0\. Abstract\n\n.*?continuity\.\*\n\n(.*?)\n\n---", draft, re.S)
    a_final = re.search(r"## Abstract .*?\n\n(.*?)\n\n---", ABSTRACT.read_text(), re.S)
    if a_draft and a_final and a_draft.group(1).strip() != a_final.group(1).strip():
        fails.append("DRIFT      the abstract differs between PAPER_DRAFT.md and FINAL_ABSTRACT.md")

    print(f"claim check: {len(quoted)} numeric claims, {len(proofs)} 'we prove', "
          f"{len(fails)} problem(s)")
    for f in fails:
        print("  " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
