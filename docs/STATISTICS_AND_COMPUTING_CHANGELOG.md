# Statistics and Computing — change report (internal)

**Compared:** the pre-journal manuscript (`docs/baseline_pre_journal_2026-10-04/MANUSCRIPT.md`, 16 pp,
the faculty-approved state after the 2026-10-04 faculty round) against the final journal build
(`docs/MANUSCRIPT.md` → `docs/STATISTICS_AND_COMPUTING_MANUSCRIPT.{docx,pdf}`, 18 pp in Word and
PDF), 2026-10-04. Two passes: the journal-formatting pass (items 1–19) and the final compliance
pass (items 20–36, after re-reading the guidelines that afternoon).
Requirement numbers (#) refer to `docs/STATISTICS_AND_COMPUTING_SUBMISSION_AUDIT.md`.

Classes: **A** purely formatting · **B** journal-required editorial change · **C** metadata or
declaration · **D** citation or reference update · **E** scientific or content change.

## Mechanical integrity checks (exact)

| object | result |
|---|---|
| Fenced blocks: every display equation (1)–(8) and Algorithm 1 | **13 of 13 byte-identical** |
| Blockquoted results: Proposition 1, its proof, Theorem 1, Corollaries 1a and 1b | **15 of 15 lines byte-identical** |
| Table rows, all eight tables | **63 of 64 rows byte-identical**; the one difference is Table 5's header cell `B9 (v482)` → `B9` (item 27); every data row and every value is identical |
| `docs/17_PAPER_TABLES.md` | only the four renumbered headings and the Table 5 column key differ; no cell changed |
| Result CSVs, configurations, seeds | untouched (no experiment was rerun) |
| Project figures `figures/*.png` | **byte-identical** after the plotting changes (`make figures` re-run and hashed) |
| Journal figures `figures/journal/Fig1–6` | drawn by the same functions from the same CSVs; only the rendering differs (see A below) |
| Numbers in the body | none removed except one "4" (the §4.2 pointer, below). Every added number is a section, table or figure label, a citation year, the `v482` name, or an identifier in the data statement (CC-BY-4.0, SHA-256, revision f7d2896) |
| Native Word equations | 138 → 141: +`N` in the abstract, +3 in the Table 5 key, −`t_min` in the abstract. No equation became an image (6 drawings = 6 figures) |

## Every substantive difference, classified

| # | where | change | class | justification / evidence |
|---|---|---|---|---|
| 1 | title page | affiliation "Vellore Institute of Technology (VIT), Vellore, India" → "…, Vellore, Tamil Nadu, India" | C | SG title page asks for "city, (state), country"; Vellore is in Tamil Nadu |
| 2 | title page | added "Corresponding author: […]" and "ORCID iDs: […]" placeholders | C | SG requires a corresponding author and e-mail; none is recorded in the project, so nothing was invented |
| 3 | abstract | heading "0. Abstract" → "Abstract" | A | the abstract is not a numbered section |
| 4 | abstract | "a large pool of machine-generated labels" → "a large pool of `N` machine-generated labels" | B | SG: no undefined symbols; `N` is used later in the abstract (`N < n − 1`) |
| 5 | abstract | "Reading the classical `t_min` bound with the degrees of freedom…" → "Reading the classical Behrens–Fisher bound, a Student quantile on the smaller of two degrees of freedom, with the degrees of freedom…" | B | defines `t_min` in words; same bound, same claim (§2.1: `t_{min(f₁,f₂)}`, Mickey and Brown 1966) |
| 6 | abstract | "an HC3 variant" → "a heteroskedasticity-consistent (HC3) variant" | B | abbreviation defined at first use |
| 7 | abstract | "LLM-judgment populations" → "large-language-model-judgment populations" | B | abbreviation removed rather than defined |
| 8 | after abstract | six keywords added | B | SG: 4 to 6 keywords; each term occurs in the paper |
| 9 | throughout | in-text citations to Springer style: "(Mozer 2026)", "Mickey and Brown (1966)", "Mani et al. (2025)", ";" between citations | D | SG citation examples; 17 "&"→"and", 20 year-commas removed, 6 author lists of three or more → "et al." |
| 10 | §2.1, §2.2 | "Zrnic & Candès (2023)" → "Zrnic and Candès (2024)" | D | now published, *PNAS* 121(15), e2322083121 (2024), DOI 10.1073/pnas.2322083121; the cited content (the cross-fitting device) is unchanged |
| 11 | §1, §7 | "Figure n" → "Fig. n"; "Figures 1–2" → "Figs. 1–2" | A | Springer house form; matches the caption labels |
| 12 | §4.2 | "*grows* (Table 4)" → "*grows* (§7.5)" | B | SG: tables cited in consecutive order. The pointer now names the section holding the same table (now Table 5); the claim is unchanged |
| 13 | §7.4–§9.1 | Tables renumbered in order of first citation: repair comparison 7 → **4**, finite-`N` 4 → **5**, robustness 5 → **6**, Galaxies 6 → **7**; every in-text reference updated | B | SG: "Tables should always be cited in text in consecutive numerical order"; the old first-citation order was 1, 4, 2, 3, 7, 5, 6, 8 |
| 14 | Table 5 intro | column key added: what B9 (v482), B9 V_full, B9-R1, B9-R2 and B9-R5 are | B | SG: a table caption must explain the table's components; these columns were undefined. Each description is the docstring of the mode in `src/ppi_finite/variance.py` (`var_lc_ppi`) and of the methods in `src/ppi_finite/lc_ppi.py`; no value or claim added |
| 15 | figure captions | "**Figure n.** …." → "**Fig. n** …" with no final full stop; panels named "(a) ρ = 0, …"; method codes keyed ("method codes as in §6", "(B3)", "(B9)", "(B9-R2)", "(B9-HC)"); dashed line and ±1.96 MC-s.e. bars identified; Fig. 2 and Figs. 4–5 gain "R = 200,000" and Fig. 5 gains "σ(f) = exp(γf) at ρ = 0" (both moved from the old in-figure titles); Fig. 3 gains the ±0.005 band and the Ridge-PPI asterisk note (moved from its old in-figure title); Fig. 6 gains "16,743 items" (moved from its old in-figure subtitle) | A | SG caption rules; "Do not include titles or captions within your illustrations", so their content moves into the captions. Every added fact was already printed on the figure. The Fig. 3 caption's "design of Table 7" is now "Table 4" (renumbering) |
| 16 | figures | journal renderings: no in-figure titles or subtitles, lettered panels, a distinct marker per method, Arial at 8 pt at printed size, grid drawn opaque (#ebebeb, the same tint as before), EPS with embedded fonts plus 600-dpi RGB PNG, placed at printed width (~15.6 cm). Fig. 4 tick labels written as integers (50, 100, …) instead of floats; Fig. 5 ticks show the γ value instead of the setting string; Fig. 3 ticks as plain n | A | SG artwork rules (lettering, accessibility "patterns … in addition to colors", resolution, RGB, EPS, naming). Data source and plotted values unchanged: same functions (`ppi_finite.plots`, `repair_ext/figures2.py`), same CSVs |
| 17 | before references | new section *Statements and Declarations*: Funding, Competing interests, Ethics approval and consent, Data availability, Code availability, Author contributions, Use of generative AI | C | SG: mandatory section. Placeholders wherever the project holds no information. The data statement repeats the faculty-approved Galaxies position verbatim in substance ("not redistributed", "no licence is posted for the data by the upstream portal", "we did not seek or obtain separate authorisation to redistribute it") and states MT-bench's CC-BY-4.0 source and pinned revision |
| 18 | references | heading "13. References" → "References"; all 27 entries rewritten in the Springer basic style with full `https://doi.org/` links; ISO-4 abbreviations where certain; DOIs added for the arXiv preprints (10.48550/arXiv.*) and the three books; Li and Ignatiadis and Eyre and Madras given their ICML 2025 / PMLR 267 publication details; Zheng et al. given NeurIPS 36 pages and DOI; Willett et al. listed with all 18 authors; PPI++ kept as *Ann. Appl. Stat.* 2026 with its 2023 arXiv origin | D | SG reference rules; re-verification table in the audit. Every DOI resolved on 2026-10-04 |
| 19 | Limitation 9; the faculty sentence of §12 Data | **unchanged** | — | faculty wording retained word for word (§12 received additions only; see item 31) |

### Final compliance pass (afternoon, 2026-10-04)

| # | where | change | class | justification / evidence |
|---|---|---|---|---|
| 20 | title page | corresponding-author/ORCID placeholder line removed | C | the author deleted it in Word at 09:15; carried into the source so the rebuild keeps it. The information is MANUAL INPUT REQUIRED (checklist) |
| 21 | declarations | bracketed placeholders for Funding, Competing interests and Author contributions removed; Ethics written as a plain statement | C | the author's instruction that the package contain no unfinished placeholders; nothing about them exists in the project, so nothing was written in their place. They are tracked as MANUAL INPUT REQUIRED and `make check-journal` reports the manuscript NOT READY until supplied |
| 22 | §6 | **Use of AI tools** paragraph added (LLM assistant used beyond copy-editing for code, literature search and summary, and drafting); the declarations entry now points to §6 | C | SG: LLM use "should be properly documented in the Methods section"; Springer AI policy: such use is permitted with disclosure; AI not an author. Facts from the git record (all 30 commits co-authored by Claude via Claude Code) |
| 23 | §6 | "**B9** Algorithm 1" → "**B9** Algorithm 1 (LC-PPI, the leverage-corrected interval)" | B | SG: abbreviations defined at first use; LC-PPI appears in five table headers. Algorithm 1's own title is "Leverage-corrected …" |
| 24 | §2.2, §7.1, §7.4 | "harness" → "simulation design" / "simulation set-up" (×3) | B | internal jargon |
| 25 | §4.2 | "The full algebra is in the repository's theory document (§12)" → "… is provided with the code (§12)" | B | internal reference |
| 26 | §4.5, §5 (step 9), §7.5, Limitation 2 | "Experiment E" → "the finite-`N` experiment / grid of §7.5"; Table 7 note "Experiment G" → "the skew experiment of §8" | B | the experiment letters are never defined in the paper; same experiments, same claims |
| 27 | Table 5 | header `B9 (v482)` → `B9`; key "B9 (v482) is Algorithm 1 as deployed" → "B9 is Algorithm 1 as deployed" | B | `v482` is an internal code name; the column and its values are unchanged |
| 28 | Tables 2, 3 | "MC s.e." → "Monte-Carlo s.e."; Table 2 note `rho` → `ρ`, ">=" → "≥" | B | undefined abbreviation; typography |
| 29 | Table 3, Table 6 | "Kill criterion K3 does not fire" → "… is not triggered"; "(`10` H4)" → "(pre-registered hypothesis H4)" | B | jargon; the old text pointed to an internal file |
| 30 | all tables | the line "*Source: `results/…`; generated by `experiments/make_tables.py`.*" under each table removed | B | repository bookkeeping, not journal prose; the mapping stays in `make_manuscript.TABLE_SOURCES` and `docs/17_PAPER_TABLES.md` |
| 31 | §12 | first paragraph rewritten without build commands, repository paths or "provenance sidecar" (same content: one configuration per experiment, ~20 min on one CPU thread, everything generated from saved results, per-file records of configuration, code version, seeds and software versions, and the material held in full). After the faculty Data sentence, the actual acquisition route (dataset "galaxies" from the ppi_py dataset loader) and the full SHA-256 checksum were **added** | B / C | journal prose; Limitation 9 says "§12 gives acquisition instructions and a checksum", and now it literally does. Checksum from `docs/tier2_galaxies.md`; loader from the ppi_py README |
| 32 | declarations | Data availability rewritten concisely (same facts: MT-bench CC-BY-4.0 link and pinned revision; Galaxies not redistributed, no posted licence, no authorisation sought); Code availability describes the repository | C | SG Data Availability Statement; faculty Galaxies position preserved |
| 33 | figures | panel titles "(a) ρ = 0.0" → "(a)" only; Fig. 6's in-figure "nominal 0.95" label removed (it touched a data point; the caption identifies the line) | A | SG "Do not include titles or captions within your illustrations"; same functions, same CSVs; project figures stay byte-identical |
| 34 | captions | every caption self-contained: what is plotted, each panel's setting, every method code, the dashed line, bars and bands, R, and the key reading | A / B | SG "Identify all elements found in the figure in the figure caption". Facts from §6 definitions, the registry (B6 = \|t\| > 2) and Tables 4, 5 and 7 (R values) |
| 35 | references | the seven arXiv-only entries written "arXiv preprint arXiv:… (year)" | D | the form used in the journal's own July 2026 articles (Crossref deposits); all statuses re-verified |
| 36 | equations (rendering only) | `V_γ = X/(…)^½` was typeset with a literal caret and is now a superscript ½; display-equation subtraction emitted as the minus sign U+2212 instead of ASCII "-"; `\quad` spacing emitted in a text run so LibreOffice keeps it (Eq. (1)) | A | the LaTeX of every equation is unchanged (the fenced sources are byte-identical); only the OMML rendering of characters already there changed. Round-trip guard updated to treat "-" and "−" as one character |

**Class E (scientific/content) changes: none.** No theorem, proof, assumption, equation, estimator,
definition, interval, design, seed, sample size, dataset, numerical result, table value, figure
value, limitation, novelty claim, prior-art attribution, Algorithm 1 step or conclusion changed.
Items 4–7 and 14 are the closest to content: they define notation already in use and add no claim.

## Scope statements re-confirmed in the journal build

Mean-estimation-only scope (title, abstract, Limitation 7); interval-level contribution (§2.2);
prior art attributed (Mozer 2026; Mickey and Brown 1966; Hsu 1938; Wang and Liu 2026; Welch 1947;
Satterthwaite 1946; Mani et al. 2025; Eyre and Madras 2025); Theorem 1 bounds the gap and is
non-asymptotic; Corollary 1a is fixed-ρ and not uniform in ρ (Limitation 10); G8 is stated within
M1 (§4.5); the Ridge-PPI interval is ours (§7.4, Fig. 3 caption); HC3 is asymptotic only
(Limitation 5); Galaxies and MT-bench report containment under subsampling from a fixed finite
population, not coverage (§9, Fig. 6 caption). There is no universal coverage claim, no universal
`C/N` claim and no global-conservativeness claim (`make check-claims`, `make check-submission`).

## Checker changes, and why

| file | change | reason |
|---|---|---|
| `experiments/check_claims.py` | required phrases "Mickey & Brown", "Eyre & Madras" → "… and …"; two messages "Table 7" → "Table 4" | the citation style and table numbering changed intentionally; the disclaimers they guard are unchanged |
| `experiments/check_submission.py` | references heading "## References"; reference years parsed in Springer position; "and" accepted between author names; item 8c looks for "Fig. n" | intentional format changes; the checks themselves are unchanged |
| `experiments/check_manuscript_pdf.py` | "References" heading; captions found as "Fig. n" not preceded by "("; reference heads matched with the existing dash-join normalisation | Springer captions have no full stop, and page ranges can line-break at the en dash |
| `experiments/check_journal.py` | **new**; final pass: 22 rules, including no placeholders, LLM disclosure in §6, and no repository paths or jargon; author-only information reported as MANUAL INPUT REQUIRED, with a NOT READY notice | journal-specific check |
| `experiments/check_submission.py` item 8a | required a "Source: `results/…`" line under each table; now requires eight tables **and no repository paths** in the prose | the old check enforced exactly the internal language the journal format removes (item 30) |
| `experiments/docxbuild/omml.py` round-trip guard | "-" and "−" counted as one character | item 36: the rendering now emits U+2212 |
