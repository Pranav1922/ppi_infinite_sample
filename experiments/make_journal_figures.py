#!/usr/bin/env python3
"""The six manuscript figures, rendered to the Statistics and Computing artwork rules.

    python experiments/make_journal_figures.py        # -> figures/journal/Fig1.{eps,png} ... Fig6

Same plotting functions and same saved CSVs as `make figures`, so no statistic is recomputed;
only the rendering changes: no titles inside the figure (the captions carry them), lettered
panels, a marker per method, Arial lettering at 8 pt at the printed width, vector EPS with
embedded fonts, and a 600-dpi RGB PNG for the DOCX. The ordinary figures in `figures/` are untouched.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "repair_ext"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import figures2  # noqa: E402
from ppi_finite import plots  # noqa: E402

OUT = ROOT / "figures" / "journal"

#: manuscript figure number -> the figure `make figures` writes under that name
SOURCES = {1: "C_coverage_vs_n", 2: "D_coverage_vs_rho", 3: "repair_extension_coverage",
           4: "E_coverage_vs_N", 5: "H_coverage_vs_gamma", 6: "I_method_comparison"}


def main() -> int:
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "ps.fonttype": 42, "pdf.fonttype": 42,          # embed the fonts as TrueType
        "savefig.dpi": 600,
    })
    plots.JOURNAL = True
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        plots.make_all_experiments(ROOT / "results", tmp)
        plots.make_all(ROOT / "results" / "I_real_data.csv", tmp, "I")
        figures2.coverage_figure(pd.read_csv(ROOT / "results" / "repair_extension" / "summary.csv"),
                                 tmp / "repair_extension_coverage.png", journal=True)
        for n, stem in SOURCES.items():
            shutil.copy(tmp / f"{stem}.eps", OUT / f"Fig{n}.eps")
            # RGB, 8 bits per channel, as the artwork rules ask; matplotlib writes RGBA
            im = Image.open(tmp / f"{stem}.png")
            Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA")) \
                .convert("RGB").save(OUT / f"Fig{n}.png", dpi=(600, 600))
            print(f"wrote figures/journal/Fig{n}.eps and .png  <- {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
