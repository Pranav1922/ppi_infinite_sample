# 10 — Hypotheses and Kill Criteria

These are **pre-registered scientific decision rules**, written down before the experiments are run
so that results cannot be reinterpreted after the fact to keep the project alive.

The purpose is not to kill the project. It is to make the project's conclusions mean something. A
triggered kill criterion usually **narrows** the contribution rather than ending it — each one below
specifies what the project becomes if it fires.

**Rule of engagement:** when a criterion fires, record it in `13_RESEARCH_LOG.md` with the evidence,
state the resulting change of scope, and continue. Do not quietly widen a tolerance, drop a cell,
change a metric, or add a condition that was not pre-registered.

---

## Hypotheses

### H1 — PPI++ under-covers at small n

**Statement.** Under the bivariate-Gaussian working model with a large unlabelled pool, the standard
PPI++ interval at nominal 95% has empirical coverage at or below 0.92 for n ≤ 20, approximately
uniformly in ρ.

**Status.** `PRELIMINARILY CONFIRMED — awaiting independent reproduction.` Observed 0.8784 / 0.8791
/ 0.8787 at n = 10 (ρ = 0 / 0.5 / 0.9) and ≈ 0.9194–0.9198 at n = 20.

**Exact test.** Experiment A, Table A1, row block n = 10 and n = 20. Verification checks V3, V4, V5.

**Success condition.** B3 coverage ≤ 0.92 at n = 10 and n = 20 for every ρ tested, with the spread
across ρ at fixed n no greater than 0.006.

**Failure condition.** B3 coverage above 0.94 at n = 10 → the preliminary result does not reproduce.
The most likely causes, in order: a different quantile (Student instead of normal), a different
`ddof`, or a different λ̂ convention. Resolve before doing anything else. If B3 genuinely covers
correctly, **the project has no premise** and must stop.

---

### H2 — Simple controls do not uniformly repair it

**Statement.** None of {Student-*t* quantile, cross-fitting, hard thresholding of λ̂} attains ≥ 0.945
coverage uniformly across ρ at n = 10.

**Status.** `CONFIRMED — independently reproduced 2026-08-19` (extension E1;
`results/A4_simple_fixes.csv`, n = 10, N = 10⁵, R = 60,000, MC s.e. 0.00089). The guard was met in
full: **three** cross-fitting schemes and **three** thresholds.

| control | coverage at n = 10, across ρ ∈ {0, 0.2, 0.5, 0.8, 0.95} | max | verdict |
|---|---|---|---|
| B4 (`t` quantile) | 0.9227 – 0.9246 | 0.9246 | below 0.945 everywhere |
| B5 (`06` 2-fold formula) | 0.9091 – 0.9113 | 0.9113 | below 0.945 everywhere |
| B5-kfold (K = 5) | 0.9165 – 0.9181 | 0.9181 | below 0.945 everywhere |
| B5-avglam (`cover.py` scheme) | 0.9243 – 0.9271 | 0.9271 | below 0.945 everywhere |
| B6 at \|t\| > 1.5 | 0.9203 – 0.9338 | 0.9338 | below 0.945 everywhere, spread 0.0135 |
| B6 at \|t\| > 2 | 0.9178 – 0.9403 | 0.9403 | below 0.945 everywhere, **spread 0.0225** |
| B6 at \|t\| > 3 | 0.9128 – 0.9489 | 0.9489 | below 0.945 everywhere, spread 0.0361 |

Every control fails at every ρ, by a margin of at least 0.005 (B6 at `|t|>3`, ρ = 0) and typically
0.02–0.03 — tens of Monte-Carlo standard errors. B6's non-monotonicity is confirmed and is
**worst where the predictor is good**: at `|t| > 3` it is 0.9489 at ρ = 0 and 0.9128 at ρ = 0.8.
The failure condition ("any simple control attains ≥ 0.945 uniformly") did not occur for any of the
seven settings tested.

*Previously:* `PRELIMINARILY CONFIRMED — awaiting independent reproduction.` Observed at n = 10:
B4 ≈ 0.9241–0.9249; B5 ≈ 0.9245–0.9261; B6 ranged 0.9194–0.9409 and was **non-monotone in ρ**, worst
at ρ = 0.8.

**Exact test.** Experiment A (Table A4) and Experiment D.

**Success condition.** Every one of B4, B5, B6 falls below 0.945 in at least one ρ cell at n = 10;
and B6 is demonstrably non-monotone in ρ (spread ≥ 0.015 across ρ at fixed n).

**Failure condition.** Any simple control attains ≥ 0.945 uniformly at n = 10 → that control is the
answer and the project's contribution collapses to explaining *why* it works. Note this would still
be worth writing up, but as a much shorter paper.

**Guard against a false confirmation.** B5's fold scheme and B6's threshold are both arbitrary
choices. The pre-registered requirement is that H2 be tested against **at least two** cross-fitting
schemes and **three** thresholds (`06`). If the simple control fails only for one arbitrary setting,
H2 is not confirmed.

---

### H3 — The leverage correction yields near-nominal coverage under the Gaussian model

**Statement.** With `N → ∞` and the bivariate-Gaussian working model, the LC-PPI interval has
coverage within 0.004 of nominal for every n ≥ 4 and every ρ ∈ [0, 0.99].

> **E4 EXTENSION (2026-08-20), stated separately because it is a different claim.** H3 is about
> `N → ∞`. At **finite** N the same interval is **conservative but not exact**: over 81 cells of
> Experiment E it never falls below nominal (minimum 0.9488, at the `N → ∞` end where theory says
> exactly 0.95) and rises to 0.9711 at `n = 10, ρ = 0.9, N = 50`. The Satterthwaite variant
> `B9-R2` restores ±0.0028 across the whole grid but is itself not conservative. H3 is **not**
> extended to finite N by this evidence; what E4 supports is a separate, weaker statement of
> conservativeness (R4), which remains unproved.

**Status.** **`CONFIRMED 2026-08-20`** (Experiment C, R = 200,000). The previously untested region
is now covered: coverage **0.9495–0.9505 at every n from 4 to 50** and every ρ ∈ {0, 0.5, 0.9},
worst deviation from nominal **0.0005** against the 0.004 the success condition allows. Experiment
D adds ρ up to 0.99 at n ∈ {10, 20, 50}: B9 stays in [0.9487, 0.9507]. For contrast, PPI++ at n = 4
covers **0.6809**.

*Previously:* `PRELIMINARILY CONFIRMED at n ∈ {10, 20, 50}, ρ ∈ {0, 0.5, 0.9}` — observed
0.9492–0.9501. Untested at n = 4, 5, 8 and at ρ near 1. The underlying pivot claim is
`DERIVED, PROOF PENDING` (`04` §4.7).

**Exact test.** Experiment A (V6), Experiment C (n down to 4), Experiment D (ρ up to 0.99), and
critically **verification check V12** — a Kolmogorov–Smirnov test of the studentised statistic
against `t_{n−2}`, which tests the whole distribution rather than one quantile.

**Success condition.** Coverage within 0.950 ± 0.004 at every (n ≥ 4, ρ) cell **and** V12 does not
reject.

**Failure condition.** Coverage outside tolerance at any n ≥ 4, or V12 rejects → the pivot argument
has an error or an unstated condition. Re-derive `04` §4.7 before writing any proof. Note that
failure at n = 3 specifically is expected and is not a failure of H3.

---

### H4 — Robustness beyond the Gaussian model

**Statement.** Under heavy-tailed (Student-*t*, ν ≥ 5), skewed, and mildly heteroskedastic errors,
LC-PPI remains **conservative or near-nominal** (coverage ≥ 0.94) at n ≥ 20, and in no case is its
coverage materially worse than the classical interval B1 at the same cell.

**Status.** `TESTED AND PARTLY REFUTED — E5, 2026-08-20.` 192 cells at R = 200,000 across
Experiments F (heavy tails, both residual- and predictor-heavy), G (skew and binary) and H
(heteroskedasticity, multiplicative and step).

> ## H4 VERDICT: **FAILS** on clause (b), and the failure is real but is not what it looks like.
>
> | experiment | cells | B9 min coverage | min(B9 − B1) | clause (b) violated |
> |---|---|---|---|---|
> | F, heavy residuals | 36 | 0.9495 | −0.0022 | **0/36** |
> | F, heavy predictor `f` | 36 | 0.9494 | −0.0013 | **0/36** |
> | G, skew + binary | 84 | 0.8474 | **−0.1028** | **45/84** |
> | H, `σ(f)=exp(γf)` | 36 | 0.9047 | **−0.0684** | **20/36** |
> | H, two-group step | 36 | 0.9411 | −0.0124 | 3/36 |
>
> **Clause (a) passes outright**: coverage ≥ 0.94 for ν ≥ 5 at n ≥ 20 — observed minimum 0.9495 in
> both heavy-tail arms, against a 0.94 requirement.
>
> **Clause (b) — `coverage(B9) ≥ coverage(B1) − 0.005` at every cell — fails at 68 of 192 cells.**
> H4 is therefore not satisfied as written, and the threshold was not moved.
>
> **What the failure actually is, in each case:**
>
> * **Heavy tails: no failure at all.** B9 tracks B1 to within 0.0022 everywhere, including the
>   heavy-predictor variant where the leverage term is most exposed. Both over-cover slightly as
>   ν falls (B1 0.9594, B9 0.9572 at ν = 3, n = 10) — a shared property of the `t` interval, which
>   is exactly the case H4's framing note says must not be blamed on the method.
> * **Skew: the gap is B1 *improving*, not B9 degrading.** B9's coverage under skew is nearly
>   **constant in ρ** (mean 0.9286 / 0.9283 / 0.9283 at ρ = 0 / 0.5 / 0.9) while B1's **rises**
>   (0.9282 / 0.9431 / 0.9508). The reason is structural: with `Y = ρf + √(1−ρ²)ε`, a strong
>   predictor makes `Y` itself nearly Gaussian, which rescues the classical interval — whereas B9
>   studentises the **residual**, which stays skewed at every ρ. Violations by ρ: 2/28 at ρ = 0,
>   20/28 at ρ = 0.5, 23/28 at ρ = 0.9. **At ρ = 0 the mean difference is +0.0004.**
> * **Heteroskedasticity: a genuine degradation, and HC3 repairs it.** Multiplicative
>   heteroskedasticity is where B9 really breaks — 0.9047 at γ = 1, n = 10. **B9-HC never falls
>   below 0.9517 in any of the 72 heteroskedastic cells** and is below 0.94 in none of them.
> * **The binary case — the one `08` calls the most practically important — is B9's best result in
>   all of E5.** Mean `|coverage − 0.95|` over its 12 cells: **B9 0.0019**, B1 0.0148, B3 0.0312.
>   B9 counts as a clause-(b) violation there only because B1 *over*-covers at 0.978.

*Previously:* `ENTIRELY UNTESTED.` No experiment in this class had been run.

**Exact test.** Experiments F (heavy tails), G (skew and binary labels), H (heteroskedasticity).

**Success condition.** Coverage ≥ 0.94 for ν ≥ 5 at n ≥ 20; and `coverage(B9) ≥ coverage(B1) − 0.005`
at every cell — i.e. the correction does not make things worse than ignoring the model.

**Failure condition.** See K4 below.

**Important framing.** Under non-Gaussian errors the **classical** interval B1 also departs from
nominal. The comparison that matters is B9 versus B1, not B9 versus 0.95. A method should not be
blamed for a failure of the *t* approximation that the classical interval shares.

---

### H5 — Extension to M-estimation

**Statement.** An analogous leverage correction restores approximately nominal coverage for
PPI-style estimation of M-estimands (logistic-regression coefficients, quantiles, GLM contrasts) at
n ≥ 30.

**Status.** `OPEN — not formulated, not tested.` The paper that documents the failure analyses mean
estimation only (`03`), so there is no external evidence about M-estimation either way.

**Exact test.** Not yet designed. **OPEN — requires research decision.** A plausible route is to
work with the estimating-equation form, where the analogue of the leverage term is a term in the
sandwich variance accounting for the estimated tuning matrix. This is genuinely harder than the mean
case and should not be started until the mean case is settled.

**Success condition.** Coverage within 0.01 of nominal at n ≥ 30 for a logistic-regression
coefficient, with the corresponding uncorrected PPI++ interval demonstrably below it.

**Failure condition.** No failure condition kills the project. H5 is scope tier 3 (`01`). If it does
not work out, the mean-case paper stands on its own.

---

## Kill criteria

### K1 — The survey-sampling / GREG literature already contains the result

**Why it exists.** The LC-PPI estimator is structurally the classical **regression (difference)
estimator** of survey sampling, and the effect of estimating the regression coefficient on its
variance has been studied there for decades. The same "estimated-coefficient penalty" is also
classical in the **control-variates** literature in Monte-Carlo simulation.

**Exact test.** A directed literature search, executed **before implementation begins**, covering at
minimum: Cochran (1977) on the regression estimator; Särndal, Swensson & Wretman (1992) on GREG
variance estimation and g-weights; and the control-variate literature on the finite-sample loss
factor from an estimated control coefficient. Search terms are listed in `03`.

**Trigger.** An existing source gives an equivalent finite-sample-exact variance or interval for the
regression estimator with an estimated coefficient under a normal model.

**What the project becomes if it fires.** It does **not** end. The contribution narrows and the
framing changes to an honest one:

> "The prediction-powered-inference literature ships an interval that under-covers at small n. The
> exact correction has been known in survey sampling / control-variate theory since [year]. We show
> the equivalence, quantify the error in the PPI setting, extend the correction to the case where
> the auxiliary mean is itself estimated from a finite unlabelled sample (which the survey literature
> does not cover, because there the frame mean is known exactly), and provide a drop-in
> implementation."

That is a legitimate and useful paper. What is **not** permitted is continuing to claim the
correction as new.

> ## K1 VERDICT (2026-08-20, after reading Robinson 1976, Mozer 2026 and the two-phase reviews)
>
> **K1 does not fire on its literal trigger, and the distinction matters.** The trigger requires a
> *single* existing source giving a finite-sample-exact variance or interval **for the regression
> estimator with an estimated coefficient** under a normal model. None does. But the pieces are all
> published separately: the estimator (Cassel et al. 1976 / Särndal et al. 2003, recognised for PPI
> by Mozer 2026), the distributional form (Behrens–Fisher with one variance known), the
> conservativeness (**Mickey & Brown 1966**), and the df correction (Welch–Satterthwaite). Assembling
> them is what is left, and that is a *framing* contribution plus the `O(1/N)` bound of Part II.
>
> **Consequence, applied:** every novelty claim around Theorems B and C is **withdrawn** in
> `docs/theory_finite_N.md` Part III. The reframing K1 describes below is now the operative framing.
>
> **(Earlier partial update retained for the record.)**
>
> **K1 PARTIAL UPDATE (2026-08-20, focused check — see `docs/prior_art_findings.md` §4b).**
> K1 **does not fire yet**, but half of what it was worried about has happened. Mozer,
> arXiv:2603.19160 (19 March 2026) publishes the equivalence — PPI = difference estimator,
> PPI++ = GREG — so the *translation* framing above is **no longer available as a contribution**
> and must be cited, not claimed. K1's actual trigger (an existing finite-sample-exact variance or
> interval for the regression estimator with an estimated coefficient under a normal model) is not
> met by that paper's abstract, which mentions no finite-sample interval, no degrees of freedom and
> no Student-*t* — **but only the abstract was read.** Separately, the **two-phase sampling**
> regression-estimator literature is now identified as the exact structural analogue of finite-N
> PPI, which *raises* the prior-art risk on the part `03` assumed was safest. Both must be
> retrieved and read before any write-up; searching further is not a substitute.

**Priority: highest. This is Stage 0 of the implementation order.**

---

### K2 — The finite-N correction is already solved, or is trivial

**Why it exists.** The finite-N case is what remains genuinely open if K1 fires. If it turns out to
be a two-line consequence of an existing result with no new content, the project has no remaining
theoretical core.

**Exact test.** Part of the same literature sweep as K1, plus the derivation work in Experiment E.

**Trigger.** Both of: (i) K1 has fired, **and** (ii) the finite-N pivot follows immediately from an
existing published result.

> ## K2 VERDICT (2026-08-20): **fires in substance.** Clause (ii) is now satisfied — the finite-N
> distributional result *does* follow immediately from an existing published result once the
> identification is made (Behrens–Fisher with `ν₂ = ∞`; Mickey & Brown 1966). Clause (i) is
> satisfied "in substance but not literally" per the K1 verdict above. Under `10`'s own decision
> rule this makes the project **"a careful empirical and software contribution: a demonstration
> paper plus a corrected reference implementation"** — with two additions the rule did not
> anticipate: the *identification* itself (which is what makes the classical theory usable in PPI),
> and the `O(1/N)` coverage-error bound of `theory_finite_N.md` Part II, for which no equivalent was
> located. **The project continues, with theoretical claims reduced to those two plus the
> empirics.** Robustness (E5) now carries more of the novel content, exactly as `10` K2 predicts.
>
> **(Earlier status retained for the record.)**
>
> **K2 STATUS AFTER E4c (2026-08-20): closer to firing than before, still not evaluable.** The
> mechanism is now identified as Behrens–Fisher with one degree of freedom infinite, and a 1976
> Annals paper is reported to prove the conservativeness that Theorem C proves
> (`prior_art_findings.md` §4c, F4, `UNVERIFIED`). If that holds, clause (ii) — "the finite-N pivot
> follows immediately from an existing published result" — is close to satisfied for Theorem C,
> though **not** for Theorem A′ or for Part II's error analysis. Retrieving Robinson (1976) is now
> the highest-priority prior-art action in the project.
>
> **K2 STATUS AFTER E4 (2026-08-20): does not fire, and cannot yet be evaluated.** Clause (i) is
> unresolved because K1 is open (`docs/prior_art_findings.md`). On clause (ii), E4 establishes that
> there **is** finite-N content to have: the exact conditional variance (`04` §4.8.3, E4 update),
> the refutation of the double-counting diagnosis, and a quantitative degrees-of-freedom
> explanation that predicts the over-coverage to 0.003 with no free parameters. Whether any of that
> is already published is a question for the K1 sweep — the Satterthwaite step in particular is
> classical, and a prior-art check should look specifically for effective-degrees-of-freedom
> treatments of the regression estimator with an estimated auxiliary mean.

**What the project becomes if it fires.** A careful empirical and software contribution: a
demonstration paper plus a corrected reference implementation, with the robustness study (F/G/H) as
the novel empirical content. Downgrade any claim of theoretical contribution accordingly and say so
in the abstract.

---

### K3 — BCa bootstrap solves the practical problem with no meaningful disadvantage

**Why it exists.** The bootstrap is the standard model-free answer to "a plug-in variance is wrong
because a parameter was estimated". It has not been run. It is the strongest untested competitor.

**Exact test.** Experiment B.

**Trigger.** B8 attains coverage ≥ 0.945 at n = 10 across **all** ρ tested, **and** it does so under
the heavy-tailed and skewed DGPs of F and G, **and** its interval widths are no wider than LC-PPI's.

> ## K3 VERDICT — DOES NOT FIRE (E3, 2026-08-19)
>
> **The first clause of the trigger fails outright, so the criterion cannot fire and the remaining
> clauses are moot.** Evidence: `results/B_bootstrap.csv` (B = 2000) and `results/B_final.csv`
> (B = 9999), N = 10⁶, R = 20,000, MC s.e. 0.0015, on draws shared with B1/B3/B9.
>
> | ρ at n = 10 | 0.0 | 0.3 | 0.5 | 0.7 | 0.9 |
> |---|---|---|---|---|---|
> | B8 BCa, B = 2000 | 0.9026 | 0.8996 | 0.9020 | 0.8983 | 0.9001 |
> | B8 BCa, B = 9999 | 0.9029 | 0.9011 | 0.9038 | 0.9023 | 0.9012 |
> | B7 percentile, B = 9999 | 0.9022 | 0.9016 | 0.9056 | 0.9026 | 0.8998 |
> | **B9 LC-PPI** | **0.9496** | **0.9492** | **0.9528** | **0.9496** | **0.9487** |
>
> Required: ≥ 0.945 at every ρ. Observed maximum: **0.9056**. The shortfall is ≈ 0.045 — about
> **30 Monte-Carlo standard errors** — and it is unchanged by a five-fold increase in `B`, so it is
> a property of the bootstrap at this sample size, not of the resampling budget. The third clause
> also fails in the opposite direction to the one anticipated: BCa's intervals are **narrower**
> than LC-PPI's (0.933 vs 1.081 relative to B1 at ρ = 0), which is under-coverage in different
> units, not an efficiency advantage.
>
> **The kill criterion was not changed after seeing the result**; the trigger above is the
> pre-registered text and is quoted unmodified.
>
> **Consequence.** The reframing described below is **not** invoked. The project's contribution
> stands as it was. The cost, determinism and exactness arguments listed below remain true but are
> no longer the *only* arguments — the validity argument survives, and B8's shortfall becomes a
> result in its own right (`07` Experiment B: "If B8 falls short → LC-PPI's case strengthens
> considerably").
>
> **K3's second clause, settled in E5 (2026-08-20).** It concerned the heavy-tailed and skewed
> DGPs of F and G. Measured (R = 20,000, MC s.e. 0.0015): under heavy tails BCa covers
> **0.876–0.900** at n = 10 — *worse* than its 0.898–0.903 under the Gaussian model, and worse as
> the tails get heavier. Under skew it covers **0.829–0.959**, beating B9 in only 1 of 9 cells at
> n = 10 (the binary case, where it over-covers at 0.9586 while B9 sits at 0.9503, nearer nominal).
> The trigger requires ≥ 0.945 across all ρ at n = 10 under F and G; the observed minima are 0.876
> and 0.829. **K3 does not fire under M1, under heavy tails, or under skew. It is now fully
> settled and closed.**

**What the project becomes if it fires.** The headline changes from "the only known fix" to a
narrower but still defensible claim, resting on three arguments that survive:

1. **Cost.** LC-PPI is `O(n + N)`; BCa is `O(B(n + N))` plus a jackknife. Three to four orders of
   magnitude.
2. **Determinism.** LC-PPI gives the same interval every time; a bootstrap interval has Monte-Carlo
   noise of its own, which matters when the reported number is a scientific finding.
3. **Exactness.** LC-PPI has (or will have) a finite-sample exactness proof under a stated model;
   BCa has an asymptotic-order argument.

These are real but weaker claims and **must be stated as such**. In particular, do not present a
cost argument as if it were a validity argument.

**Note:** the partial case matters too. If BCa works at n = 20 but not at n = 10, or works under
Gaussian but not heavy-tailed errors, that is a *finding*, and the honest framing is a
regime-dependent recommendation.

**Priority: run immediately after Experiment A.**

---

### K4 — Robustness collapses

**Why it exists.** The exactness argument depends on normality and homoskedasticity. If the method
is fragile to realistic departures, its practical scope is much smaller than the framing suggests.

**Exact test.** Experiments F, G, H.

**Trigger.** LC-PPI coverage falls below 0.93 at ν ≥ 5 and n ≥ 20 (heavy tails), **and** the failure
cannot be repaired by a studentisation, a robust scale estimate, or an HC-type variance — **and**
the classical interval B1 does not fail comparably at the same cell.

> ## K4 VERDICT: **DOES NOT FIRE** (E5, 2026-08-20).
>
> The first clause is not met and the others are therefore moot. Over the cells the trigger names
> (ν ≥ 5, n ≥ 20, heavy tails), **B9's minimum coverage is 0.9495** against a 0.93 threshold — it
> is not merely above the line, it is at nominal. The same holds in the heavy-predictor variant
> (0.9494). The trigger text was not modified after seeing the result, and `verify.py` computes the
> verdict from it rather than asserting it.
>
> **This does not mean robustness is unqualified.** K4 is a *heavy-tail* criterion and heavy tails
> turn out to be the easy case. The real limitations E5 found are elsewhere and are recorded under
> H4: severe right-skew (B9 0.847 at lognormal σ = 1, n = 10 — but B1 is 0.839 at ρ = 0, so this is
> largely a failure of the `t` interval in general) and multiplicative heteroskedasticity (B9
> 0.905 at γ = 1, n = 10, **repaired by B9-HC to 0.957**). Neither is covered by K4 as written; both
> are stated in the paper's scope rather than hidden.

The last clause is essential: a failure that B1 shares is a failure of the *t* approximation in
general, not of this method.

**What the project becomes if it fires.** A correctly scoped paper: "exact under a Gaussian working
model; here is precisely where it degrades and by how much; use the HC/bootstrap variant outside
that model." Reporting the boundary of validity honestly is a contribution. Suppressing it is not.

---

## Decision table

| Situation | Action |
|---|---|
| H1 fails to reproduce | **Stop.** No premise. Write up the non-reproduction. |
| H1 and H2 confirmed, H3 confirmed | Proceed to theory and to F/G/H. Strongest path. |
| H3 fails at some n | Re-derive `04` §4.7. Do not proceed to a proof until V12 passes. |
| K1 fires | Reframe as translation + extension. Continue. |
| K1 and K2 both fire | Reframe as empirical + software contribution. Continue, with reduced claims. |
| K3 fires fully | Reframe around cost/determinism/exactness. Continue, with reduced claims. |
| K3 fires partially | Report a regime-dependent recommendation. Continue. |
| K4 fires | Narrow the stated scope to the Gaussian working model. Continue. |
| K1, K2, K3 and K4 all fire | **Stop.** Write up an honest negative/replication note; it is still a real contribution to have measured all of this carefully. |

---

## Standing prohibitions

Carried over from the project-selection discipline that produced this choice, and binding on the
implementation:

1. **Do not reinterpret a failed result to make the project survive.** If a number comes out wrong,
   the number is the finding.
2. **Do not widen a tolerance to pass a check.**
3. **Do not drop an unfavourable cell, seed, or dataset from a table.** Report everything that was
   run.
4. **Do not present a hypothesis as an established fact.** Use the status tags of `04` §4.12.
5. **Do not claim novelty until K1 is resolved.**
6. **Do not claim publication is guaranteed.** It is not.
7. **Do not add methods to the comparison after seeing the results in order to improve the story.**
   New methods may be added, but they must be added to the pre-registered list in `06` with the
   addition and its reason logged in `13`.
