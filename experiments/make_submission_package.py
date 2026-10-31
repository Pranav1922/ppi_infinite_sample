#!/usr/bin/env python3
"""Assemble the Statistics and Computing submission files from the built manuscript.

    make submission-package

Copies only finished artefacts, never build intermediates:
  docs/STATISTICS_AND_COMPUTING_MANUSCRIPT.{docx,pdf}    the named final manuscript
  submission/statistics_and_computing/                  what is uploaded to Editorial Manager
    Manuscript.docx   editable source (the journal accepts Word; it provides no Word template)
    Manuscript.pdf    the same document, rendered
    Fig1.eps ... Fig6.eps   vector artwork, fonts embedded, named as the guidelines ask
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PKG = ROOT / "submission" / "statistics_and_computing"


def main() -> int:
    for ext in ("docx", "pdf"):
        shutil.copy(DOCS / f"MANUSCRIPT.{ext}", DOCS / f"STATISTICS_AND_COMPUTING_MANUSCRIPT.{ext}")
        print(f"wrote docs/STATISTICS_AND_COMPUTING_MANUSCRIPT.{ext}")
    PKG.mkdir(parents=True, exist_ok=True)
    keep = {"Manuscript.docx", "Manuscript.pdf", "README.md"} | {f"Fig{n}.eps" for n in range(1, 7)}
    for stale in PKG.iterdir():            # only this fixed directory, only unexpected files
        if stale.is_file() and stale.name not in keep:
            stale.unlink()
    shutil.copy(DOCS / "MANUSCRIPT.docx", PKG / "Manuscript.docx")
    shutil.copy(DOCS / "MANUSCRIPT.pdf", PKG / "Manuscript.pdf")
    for n in range(1, 7):
        shutil.copy(ROOT / "figures" / "journal" / f"Fig{n}.eps", PKG / f"Fig{n}.eps")
    print(f"wrote {PKG.relative_to(ROOT)}/ ({len(list(PKG.iterdir()))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
