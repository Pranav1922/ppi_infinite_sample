r"""Applicability determinations, made from the primary sources before any code was written.

Each entry records what the published method is, and whether it transfers to *this* study's exact
target: the mean of `Y` under the bivariate Gaussian model T1.1, with a **fixed** predictor `f`
supplied directly (there are no covariates `X` and no model-fitting step).

Nothing here is an approximation of a published method. Where a method does not transfer, it is
excluded from the headline comparison and the mismatch is stated.
"""
from __future__ import annotations

SOURCES = {
    "eyre_madras": {
        "cite": "Eyre, B. & Madras, D. (2024). Regression for the Mean: Auto-Evaluation and "
                "Inference with Few Labels through Post-hoc Regression. arXiv:2411.12665v2 "
                "(v1 19 Nov 2024, v2 8 Jul 2025).",
        "read": "full text, PDF extracted",
    },
    "zrnic": {
        "cite": "Zrnic, T. (2024/2025). A Note on the Prediction-Powered Bootstrap. "
                "arXiv:2405.18379v3 (v1 28 May 2024, v3 24 Jan 2025).",
        "read": "full text, PDF extracted",
    },
    "efron_tibshirani": {
        "cite": "Efron, B. & Tibshirani, R. J. (1994). An Introduction to the Bootstrap. "
                "Chapman & Hall.",
        "read": "not read; cited by Zrnic only as a pointer to other bootstrap variants",
    },
    "efron2026": {
        "cite": "Efron, B. (2026). A bootstrap approach to prediction-powered inference. "
                "arXiv:2606.28621 (26 Jun 2026); listed as 2025 on the author's page.",
        "read": "full text, PDF extracted",
    },
}

DETERMINATIONS = {
    # ------------------------------------------------------------------ applicable
    "Ridge-PPI": {
        "status": "APPLICABLE, with a documented deviation",
        "source": "eyre_madras",
        "published_definition":
            "Eq. (8): lambda_hat_alpha = Cov_hat[h(X), f(X)] / (Var_hat[f(X)] + alpha), used in "
            "place of the PPI++ lambda. Section 4.4: 'We follow this methodology and scale both "
            "the PPI++ estimate lambda_hat_Opt and the Ridge-PPI estimate lambda_hat_alpha by "
            "(1 + n/N)^-1.' Section 5.2: 'we perform cross validation over the labelled set to "
            "determine the ridge parameter alpha.'",
        "transfers": "Yes. The target is mean estimation, which is the paper's own setting "
                     "(their Section 2.2 'Prediction Powered Inference for Mean Estimation'), and "
                     "the estimator needs only (Y, f) on the labelled sample plus the pool mean.",
        "deviation":
            "THE PAPER DEFINES NO CONFIDENCE INTERVAL. The strings 'confidence interval', "
            "'coverage' and 'CI' occur ZERO times in it; the reported metric is mean absolute "
            "error of the point estimate. To evaluate coverage at all we must supply an interval. "
            "We use the PPI++ interval with lambda_hat_alpha substituted -- the construction most "
            "favourable to the method, and the one its own framing implies. This interval is OURS, "
            "not theirs, and no coverage result here should be read as a claim the authors made.",
    },
    "PPBoot": {
        "status": "APPLICABLE",
        "source": "zrnic",
        "published_definition":
            "Algorithm 1. For b = 1..B: resample (X*, Y*) from the labelled data and X~* from the "
            "unlabelled data, both with replacement; compute "
            "theta*_b = theta_hat(X~*, f(X~*)) + theta_hat(X*, Y*) - theta_hat(X*, f(X*)). "
            "Output the percentile interval "
            "(quantile({theta*_b}; alpha/2), quantile({theta*_b}; 1-alpha/2)).",
        "transfers": "Yes, for theta_hat = sample mean: theta*_b = mean(f~*) + mean(Y*) - mean(f*). "
                     "Requires the unlabelled pool to be materialised, since it is resampled.",
        "deviation": "None in the algorithm. The pool must be materialised to be resampled, which "
                     "bounds the pool sizes at which it can be run in this harness (see run.py).",
    },
    "Sigmoid-PPI": {
        "status": "NOT APPLICABLE to the Gaussian DGP; APPLICABLE to the binary/rate setting",
        "source": "eyre_madras",
        "published_definition":
            "Eqs. (9)-(10): g(f(X)) = 1 / (1 + exp(-alpha*f(X) + beta)) with alpha, beta learned "
            "on the labelled data, and "
            "mu_hat = (1/N) sum g(f(X^u_i)) + (1/n) sum (h(X_i) - g(f(X_i))). "
            "Section 5.2: L2 regularisation with the penalty chosen by cross-validation.",
        "transfers":
            "NOT to model T1.1. Section 4.3 introduces it for 'the rate estimation case (i.e. "
            "binary h)', where 'it's somewhat unnatural to perform linear regression of "
            "f in [0,1] onto h in {0,1}'. The sigmoid maps into (0,1), so it cannot represent the "
            "mean of an unbounded Gaussian Y: under T1.1, Y is real-valued with mean 0 and the "
            "regression target is not a probability. Fitting it there would be our construction, "
            "not theirs. It IS applicable to the Galaxies population, where Y is binary in {0,1} "
            "and the predictions lie in [0,1] -- exactly the paper's stated setting -- so it is "
            "evaluated there and excluded from the Gaussian headline grid.",
        "deviation": "Same missing-interval issue as Ridge-PPI where it is run.",
    },
    # ------------------------------------------------------------------ not applicable
    "cross-PPBoot": {
        "status": "NOT APPLICABLE",
        "source": "zrnic",
        "published_definition":
            "Section 4.2. Partition the labelled data into K folds; train K models f^(1..K), each "
            "on all labelled data except its fold; apply PPBoot using f^(j)(X_i) for a labelled "
            "point in fold j and the average model fbar(X~_i) = (1/K) sum_j f^(j)(X~_i) for an "
            "unlabelled point.",
        "transfers":
            "NO. Zrnic introduces cross-PPBoot explicitly for the case where a pre-trained model "
            "is unavailable: it 'incorporates cross-fitting when a good pre-trained model for "
            "producing the predictions is not available a priori but needs to be trained or "
            "fine-tuned', and Section 4.2 opens 'we extend PPBoot to problems where we do not "
            "have access to a pre-trained model f'. This study is the opposite case. Under "
            "assumption A5 the predictor is a FIXED function not fitted on either sample, and the "
            "DGP supplies the prediction f_i directly -- there are no covariates X from which any "
            "f^(j) could be trained. There is therefore no model-fitting step to cross-fit, and "
            "constructing one would mean inventing a learning problem the study does not contain.",
        "deviation": "Excluded. Not run, not approximated, not substituted. Note that the paper's "
                     "own existing cross-fitted PPI++ variants (B5, B5-kfold, B5-avglam) address "
                     "the different question of cross-fitting lambda, and are already reported.",
    },
    "Efron-PPboot2": {
        "status": "APPLICABLE, with two documented adaptations",
        "source": "efron2026",
        "published_definition":
            "Section 5, algorithm PPboot2, for the quantitative-response model "
            "modelq (5.1): y_a(i) = mu_a(i) + sigma_a(i)*eps(i), mu = m(f), sigma = s(f), "
            "eps ~ p iid with mean 0 and variance 1. Figure 4's steps: (1) draw y*_a from the "
            "fitted model; (2) refit on (f_a, y*_a); (3) apply the refitted model to f_b; "
            "(4) draw y*_b; (5) theta*_b = t(x_b, y*_b). Over B replicates (default 1000), "
            "(3.2a) theta_hat = mean of theta*_b and (3.2b) bsd = their standard deviation; the "
            "interval is theta_hat +- bsd. Section 4 treats theta = E{y} specifically.",
        "transfers":
            "Yes. Model T1.1 satisfies modelq exactly, with m(f) linear and s(f) constant, and "
            "the target theta = E[Y] is the case Efron's Section 4 singles out.",
        "deviation":
            "TWO, both forced by the homoskedastic Gaussian model and neither optional. "
            "(a) s-hat: Efron estimates a VARYING s(f) from successive differences of the "
            "responses (Remark 13); under T1.1 s(f) is constant, so the corresponding "
            "low-dimensional estimate is the residual standard deviation on n-2 degrees of "
            "freedom. (b) p-hat: taken as the empirical distribution of standardised residuals, "
            "the standard reading of 'eps_i iid ~ p-hat', which the paper does not pin down "
            "further. Separately, for the mean the step-4 draw of y*_b need not be materialised: "
            "mean(y*_b) is obtained exactly from a multinomial over the n residual atoms. That "
            "is algebra, not approximation. "
            "CORRECTION: an earlier note in this project recorded that no Efron PPBoot method "
            "existed. That was wrong -- it was inferred from Zrnic's reference list rather than "
            "from a search. The method exists and is implemented here.",
    },
}


def as_rows() -> list[dict]:
    """Flatten for `method_definitions.csv`."""
    rows = []
    for name, d in DETERMINATIONS.items():
        src = SOURCES[d["source"]]
        rows.append({
            "method": name,
            "status": d["status"],
            "source": src["cite"],
            "source_access": src["read"],
            "published_definition": " ".join(d["published_definition"].split()),
            "applicability_to_this_study": " ".join(d["transfers"].split()),
            "deviation_from_published_method": " ".join(d["deviation"].split()),
        })
    return rows
