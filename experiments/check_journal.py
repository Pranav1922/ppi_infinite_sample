#!/usr/bin/env python3
"""Statistics and Computing submission rules, checked against the built manuscript.

    python experiments/check_journal.py

Each rule is quoted from the journal's submission guidelines
(https://link.springer.com/journal/11222/submission-guidelines, read 2026-10-04; see
docs/STATISTICS_AND_COMPUTING_SUBMISSION_AUDIT.md). FAIL items block submission; MANUAL items are
author information the journal requires that the project does not hold (corresponding author,
competing interests, funding, ...). Nothing is invented and no placeholder is printed in the
manuscript, so while any MANUAL item remains the manuscript is NOT ready to submit.
"""
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MD = (ROOT / "docs" / "MANUSCRIPT.md").read_text()
FIGS = ROOT / "figures" / "journal"


def section(name: str) -> str:
    return MD.split(f"\n## {name}\n", 1)[1].split("\n## ", 1)[0]


def main() -> int:
    body, refs = MD.split("\n## References\n", 1)
    abstract = section("Abstract").split("**Keywords**")[0]
    words = len(re.sub(r"[*`_]", "", abstract).split())
    keywords = section("Abstract").split("**Keywords**", 1)[1].split("\n\n")[0].split("·")
    heads = re.findall(r"^(#{2,6}) (.+)$", body, re.M)
    numbered = [t for h, t in heads if t not in ("Abstract", "Statements and Declarations",
                                                  "Figure manifest", "Main figures")]
    decl = section("Statements and Declarations")
    entries = [e for e in refs.strip().split("\n\n") if e.strip()]
    first_authors = [e.split(",")[0] for e in entries]
    caps = re.findall(r"\*\*Figure (\d+)\*\* — `figures/journal/Fig\d\.png`\n\n(.+)", MD)
    text = body.split("\n## Figure manifest")[0]
    t_order = list(dict.fromkeys(re.findall(r"Tables? (\d)", text)))
    f_order = []
    for a, b in re.findall(r"Figs?\. (\d)(?:–(\d))?", text):      # "Figs. 1–2" cites both
        f_order += [str(i) for i in range(int(a), int(b or a) + 1)]
    f_order = list(dict.fromkeys(f_order))

    checks = {
        "abstract 150-250 words": 150 <= words <= 250,
        "no undefined abbreviation in the abstract (HC3, LLM, t_min)":
            "(HC3)" in abstract and "LLM" not in abstract and "t_min" not in abstract,
        "4 to 6 keywords": 4 <= len(keywords) <= 6,
        "decimal headings, at most three levels":
            all(len(h) <= 4 for h, _ in heads) and all(re.match(r"\d+(\.\d+)*\.? ", t) for t in numbered),
        "Statements and Declarations before the references, with every statement the project supports":
            all(f"**{k}.**" in decl for k in ("Ethics approval and consent", "Data availability",
                                              "Code availability", "Use of AI tools")),
        "LLM use documented in the Methods section (§6), not as copy-editing, AI not an author":
            "**Use of AI tools.**" in section("6. Experimental Methodology")
            and "beyond copy-editing" in section("6. Experimental Methodology")
            and "Claude" not in MD.split("\n## Abstract")[0],
        "no unfinished placeholder in the manuscript":
            not re.search(r"\[(To be completed|to be completed|Authors to confirm|MANUAL)", MD),
        "no repository paths, build commands or provenance jargon in the prose": not re.search(
            r"`(results|experiments|docs|figures)/|\bmake (reproduce|tables|figures|manuscript)\b|"
            r"sidecar|\bharness\b|\(`10` H|Experiment [A-J]\b", text),
        "Galaxies: no redistribution right claimed (Limitation 9, §12, data statement)":
            "make no claim to a right to do so" in MD
            and "we did not seek or obtain separate authorisation" in MD.replace("\n", " ")
            and "not redistributed" in decl,
        "author-year citations: no '&' and no comma before the year": not re.search(
            r" & |[A-Za-z.], (19|20)\d\d\)", text),
        "references alphabetised by first author": first_authors == sorted(
            first_authors, key=lambda a: a.replace("ä", "a").lower()),
        "references in Springer form 'Initials: Title ... (year)'":
            all(re.search(r"^[^:]+\.: .+\((19|20)\d\d\)", e) for e in entries),
        "DOIs as full https://doi.org links, never bare": "DOI `" not in refs
            and all("doi.org" not in e or "https://doi.org/" in e for e in entries),
        "tables cited in consecutive order": t_order == [str(i) for i in range(1, 9)],
        "figures cited in consecutive order": f_order == [str(i) for i in range(1, 7)],
        "six captions, none ending in punctuation":
            len(caps) == 6 and all(not c.rstrip().endswith((".", ";", ":")) for _, c in caps),
    }
    for n in range(1, 7):
        png, eps = FIGS / f"Fig{n}.png", FIGS / f"Fig{n}.eps"
        im = Image.open(png)
        checks[f"Fig{n}: RGB, >= 600 dpi, EPS with embedded fonts"] = (
            im.mode == "RGB" and im.info.get("dpi", (0,))[0] >= 599
            and b"/FontName" in eps.read_bytes())

    fails = 0
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        fails += not ok
    title_page = MD.split("\n## Abstract")[0]
    code = decl.split("**Code availability.**", 1)[1].split("\n\n")[0]
    manual = [name for name, missing in {
        "corresponding author and active e-mail (title page)": "orresponding author" not in title_page,
        "ORCID iDs, if available (title page)": "ORCID" not in title_page,
        "Competing interests statement (mandatory) + submission interface": "**Competing interests.**" not in decl,
        "Funding statement": "**Funding.**" not in decl,
        "Author contributions statement + submission interface": "**Author contributions.**" not in decl,
        "public URL or DOI of the code repository": "https://" not in code,
    }.items() if missing]
    for m in manual:
        print(f"MANUAL INPUT REQUIRED  {m}")
    print(f"\njournal check: {len(checks)} rules, {fails} failure(s), {len(manual)} manual item(s)")
    if manual:
        print("NOT READY FOR SUBMISSION until the manual items are supplied by the authors.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
