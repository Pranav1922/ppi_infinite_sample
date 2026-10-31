"""Executable form of docs/FINAL_SUBMISSION_CHECKLIST.md.

Checks the eight required verifications against the generated manuscript. Substring tests run on
whitespace- and blockquote-normalised text, so a line wrap inside a required phrase cannot produce
a false failure.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from make_manuscript import normalise  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def citations_resolve(raw: str) -> bool:
    """Every reference entry is cited in the body by first-author surname and year, and every
    author–year citation in the body names an entry. Prints the offenders."""
    body, refs = raw.split("\n## References\n", 1)
    body = normalise(body)
    keys = set()
    for para in re.split(r"\n\s*\n", refs.strip()):
        # Springer style puts the year last, "(2023)" then "." or the DOI link or end of entry;
        # the first such year is the publication year (a preprint year, if any, comes after it)
        year = re.search(r"\(((?:19|20)\d\d[ab]?)\)(?=[.\s]|$)", para)
        if year:
            keys.add((re.split(r"[,\s]", para.strip())[0], year.group(1)))
    uncited = [k for k in keys if not re.search(re.escape(k[0]) + r"[^)]{0,80}\b" + k[1] + r"\b", body)]
    cited = set()
    for m_ in re.finditer(r"([A-ZÄÖÜ][A-Za-zäöüé&\-]+(?:,? (?:[A-ZÄÖÜ][A-Za-zäöüé\-]+|&|and|et al\.))*),? \(?((?:19|20)\d\d[ab]?)", body):
        cited.add((m_.group(1).split(",")[0].split(" ")[0], m_.group(2)))
    unresolved = [c for c in cited if c not in keys]
    for k in sorted(uncited):
        print(f"      uncited reference: {k}")
    for c in sorted(unresolved):
        print(f"      citation without entry: {c}")
    return not uncited and not unresolved


def main() -> int:
    raw = (ROOT / "docs" / "MANUSCRIPT.md").read_text()
    m = normalise(raw)
    claims = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "check_claims.py")],
        capture_output=True, text=True, cwd=ROOT,
    )

    checks = {
        "1. every theorem correctly qualified": (
            "Proposition 1" in m and "Theorem A" not in m and "exactly Student-*t* pivotal" in m and "for a shifted quantity" in m
        ),
        "2. no novelty claim: PPI/GREG equivalence": "We claim nothing on this axis" in m,
        "3. no novelty claim: Behrens-Fisher conservativeness": "are theirs" in m and "We claim neither" in m,
        "4. no novelty claim: Welch-Satterthwaite": "that formula instantiated, not a new one" in m,
        "5. no 95% coverage claim from Galaxies": (
            "not conventional superpopulation coverage" in m.replace("**", "")
            and "confirms 95" not in m
            and "contains the population value in **0.8000**" in m
        ),
        "6. OPEN limitations preserved": all(
            x in m for x in ["not sign-definite", "asymptotic refinement", "make no claim to a right to do so",
                             "Means only", "remains open", "not uniform in"]
        ),
        "7. Sitter not implied to have been read, and no provenance tags in the manuscript":
            "We were unable to obtain Sitter (1997)" in m
            and not re.search(r"\b(PARTIAL|not read|verified)\b", m.split("## References ", 1)[-1]),
        # Journal format: each table's source CSV is recorded in the repository (TABLE_SOURCES in
        # make_manuscript.py, docs/17_PAPER_TABLES.md), not printed under the table, and no
        # repository path may appear in the manuscript prose.
        "8a. eight tables present, no repository paths in the manuscript": (
            all(f"**Table {i} —" in m for i in range(1, 9))
            and not re.search(r"`(results|experiments|docs|figures)/", m.split("## Figure manifest")[0])
        ),
        "9. title and abstract scoped to the mean": (
            "Prediction-Powered Mean Estimation" in m
            and "We study confidence intervals for a population mean" in m
        ),
        "10. author–year citations resolve both ways": citations_resolve(raw),
        "8b. 6 figures, single self-contained document": (
            len(re.findall(r"\*\*Figure \d+\*\* —", m)) == 6
            and "Figure S" not in m and "Supplementary Material" not in m
        ),
        "8c. figures referenced in the text": all(
            t in m for t in ("Fig. 1", "Figs. 1–2", "Fig. 3", "Fig. 4", "Fig. 5", "Fig. 6")
        ),
        "8d. numerical claims match generated tables": "0 problem(s)" in claims.stdout,
    }

    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    n_bad = sum(1 for ok in checks.values() if not ok)
    print(f"\nsubmission check: {len(checks)} items, {n_bad} failure(s)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
