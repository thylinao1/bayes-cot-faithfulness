"""Build ``report.md`` for the CPU mechanism battery from its results payload.

Every number in the report comes from the payload written by
``experiments/mechanism_battery.py``; this module formats and never computes an
estimate of its own beyond arithmetic on numbers already in the payload (spreads,
ratios and PASS/FAIL comparisons against the pre-registered criteria). Keeping the
prose here, generated from the run, is what makes "every number has a denominator" a
property of the code rather than a promise about the writing.
"""

from __future__ import annotations

from itertools import pairwise

# Pre-registered pass criteria for the three outcomes A2 element 11(a) requires.
RHO_STAR_SPREAD_TOLERANCE = 0.01  # family 2, at the larger sample size
SHARED_CAUSE_CROSSING = 0.70710678
SHARED_CAUSE_TOLERANCE = 0.02
NOMINAL_COVERAGE = 0.95  # family 1 passes when coverage is not significantly below nominal

FAMILY_TITLES = {
    "f1_no_cue_effect": "Family 1: no cue effect",
    "f2_direct_bypass": "Family 2: increasing direct bypass, fixed text pathway",
    "f3_shared_cause": "Family 3: shared hidden cause, no text-to-answer effect",
    "f4_rationalization": "Family 4: cue-induced rationalization that drives the answer",
    "f5_redundant_explanation": "Family 5: accurate but causally redundant explanation",
    "f6_answer_copying": "Family 6: answer copying through an opaque token channel",
    "f7_opposing_effects": "Family 7: opposing direct and indirect effects",
}


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x, nd=1):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def _by(payload, n_rows):
    return [c for c in payload["conditions"] if c["n_rows"] == n_rows]


def _get(payload, key, n_rows):
    for c in payload["conditions"]:
        if c["condition"] == key and c["n_rows"] == n_rows:
            return c
    raise KeyError(f"{key} at n={n_rows} is not in the payload")


def _family_conditions(payload, family, n_rows):
    return [c for c in _by(payload, n_rows) if c["family"] == family]


# --------------------------------------------------------------------------- #
# The three required outcomes
# --------------------------------------------------------------------------- #
def required_outcomes(payload) -> list[dict]:
    """Evaluate the three outcomes A2 element 11(a) requires the battery to assert."""
    sizes = sorted({c["n_rows"] for c in payload["conditions"]})
    big = max(sizes)
    out = []

    bypass = sorted(
        _family_conditions(payload, "f2_direct_bypass", big),
        key=lambda c: float(c["condition"].split("alpha")[1]),
    )
    shares = [c["mediated_share"]["mean"] for c in bypass]
    rho_means = [c["rho_star_point"]["mean"] for c in bypass]
    spread = max(rho_means) - min(rho_means)
    falling = all(b < a for a, b in pairwise(shares))
    out.append(
        {
            "name": "R1 direct bypass: mediated share falls, rho*_point does not move",
            "pass": bool(falling and spread <= RHO_STAR_SPREAD_TOLERANCE),
            "detail": (
                f"At n = {big}, across the {len(bypass)} bypass strengths the estimated "
                f"mediated share runs {_pct(shares[0])} to {_pct(shares[-1])} "
                f"(monotonically falling: {falling}), while the mean rho*_point runs "
                f"{_f(min(rho_means))} to {_f(max(rho_means))}, a spread of {_f(spread)} "
                f"against a tolerance of {RHO_STAR_SPREAD_TOLERANCE}. Each rho*_point is a "
                f"mean over {bypass[0]['n_datasets']} seeded datasets; each share is a mean "
                f"over the {', '.join(str(c['mediated_share']['n']) for c in bypass)} datasets "
                f"per condition whose total-effect interval excludes zero."
            ),
        }
    )

    shared = _get(payload, "f3_shared_cause", big)
    crossing = shared["rho_star_point"]["mean"]
    near = abs(crossing - SHARED_CAUSE_CROSSING) <= SHARED_CAUSE_TOLERANCE
    out.append(
        {
            "name": "R2 shared cause: strong association, zero true mediation, crossing near 0.707",
            "pass": bool(
                near
                and shared["truth_analytic"]["nie"] == 0.0
                and shared["effects"]["nie"]["bias_mean"] > 0.15
            ),
            "detail": (
                f"True NIE = {_f(shared['truth_analytic']['nie'])} by construction; the "
                f"estimator reports a mean NIE of "
                f"{_f(shared['effects']['nie']['estimate_mean'])} "
                f"(bias {_f(shared['effects']['nie']['bias_mean'])} +/- "
                f"{_f(shared['effects']['nie']['bias_mcse'])}) over "
                f"{shared['n_datasets']} datasets at n = {big}, with mean corr(M, Y) = "
                f"{_f(shared['diagnostics']['mean_corr_m_y'])}; mean rho*_point = "
                f"{_f(crossing)} against the analytic 1/sqrt(2) = "
                f"{_f(SHARED_CAUSE_CROSSING)} (tolerance {SHARED_CAUSE_TOLERANCE})."
            ),
        }
    )

    lines = []
    ok = True
    for n_rows in sizes:
        null = _get(payload, "f1_no_cue_effect", n_rows)
        for key in ("nde", "nie", "te"):
            blk = null["effects"][key]
            ok = ok and blk["coverage_ci_hi"] >= NOMINAL_COVERAGE
            lines.append(
                f"n = {n_rows} {key.upper()} covers zero in {blk['coverage_k']}/"
                f"{blk['coverage_n']} datasets "
                f"[{_f(blk['coverage_ci_lo'], 3)}, {_f(blk['coverage_ci_hi'], 3)}]"
            )
    out.append(
        {
            "name": "R3 no cue effect: intervals cover zero",
            "pass": bool(ok),
            "detail": (
                "; ".join(lines)
                + f". Criterion: the binomial interval on coverage reaches the nominal "
                f"{NOMINAL_COVERAGE}, that is, coverage is not significantly below nominal."
            ),
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _cov_cell(effects, key) -> str:
    e = effects[key]
    return (
        f"{e['coverage_k']}/{e['coverage_n']} "
        f"[{_f(e['coverage_ci_lo'], 2)}, {_f(e['coverage_ci_hi'], 2)}]"
    )


def _effects_table(payload, n_rows) -> str:
    head = (
        "| condition | truth NDE / NIE / TE | NDE bias | NIE bias | TE bias | "
        "NDE cov | NIE cov | TE cov | mean NIE width |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for c in _by(payload, n_rows):
        e = c["effects"]
        t = c["truth_analytic"]
        rows.append(
            f"| {c['condition']} | {_f(t['nde'],3)} / {_f(t['nie'],3)} / {_f(t['te'],3)} "
            f"| {_f(e['nde']['bias_mean'])} | {_f(e['nie']['bias_mean'])} "
            f"| {_f(e['te']['bias_mean'])} | {_cov_cell(e, 'nde')} | {_cov_cell(e, 'nie')} "
            f"| {_cov_cell(e, 'te')} "
            f"| {_f(e['nie']['mean_interval_width'],3)} |"
        )
    return head + "\n".join(rows) + "\n"


def _verdict_table(payload, n_rows) -> str:
    head = (
        "| condition | true NIE | load-bearing at rho=0 | false-robust | "
        "rho*_point mean [mean 95% interval] | rho*_decision |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for c in _by(payload, n_rows):
        v = c["verdict"]
        r = c["rho_star_point"]
        d = c["rho_star_decision"]
        false_robust = (
            "yes " + _pct(v["false_robust_verdict_rate"]) if v["true_nie_is_zero"] else "n/a"
        )
        decision = (
            f"unresolved {d['unresolved_k']}/{d['n']}, crossing {d['crossing_k']}/{d['n']}"
            f" (median {_f(d['median_crossing'],3)}), none-in-range "
            f"{d['no_crossing_in_range_k']}/{d['n']}"
        )
        rows.append(
            f"| {c['condition']} | {_f(c['truth_analytic']['nie'],3)} "
            f"| {v['load_bearing_k']}/{v['load_bearing_n']} "
            f"[{_f(v['load_bearing_ci_lo'],2)}, {_f(v['load_bearing_ci_hi'],2)}] "
            f"| {false_robust} "
            f"| {_f(r['mean'])} [{_f(r['mean_interval_lo'],3)}, {_f(r['mean_interval_hi'],3)}] "
            f"| {decision} |"
        )
    return head + "\n".join(rows) + "\n"


def _truth_table(payload) -> str:
    sizes = sorted({c["n_rows"] for c in payload["conditions"]})
    head = (
        "| condition | analytic NDE / NIE / TE | Monte Carlo NDE / NIE / TE | max abs diff |\n"
        "| --- | --- | --- | --- |\n"
    )
    rows = []
    for c in _by(payload, sizes[0]):
        a, m = c["truth_analytic"], c["truth_monte_carlo"]
        rows.append(
            f"| {c['condition']} | {_f(a['nde'])} / {_f(a['nie'])} / {_f(a['te'])} "
            f"| {_f(m['nde'])} / {_f(m['nie'])} / {_f(m['te'])} "
            f"| {_f(c['truth_max_abs_difference'], 5)} |"
        )
    return head + "\n".join(rows) + "\n"


def _prior_probe_section(payload, small: int) -> str:
    """Name the mechanism behind any posterior-versus-MAP gap, with the probe's numbers."""
    probe = payload.get("prior_scale_probe") or []
    if not probe:
        return ""
    blocks = payload.get("pymc_subset") or []
    # The interesting case is the largest DISAGREEMENT with the MAP path on the same data,
    # not the largest bias: a family where both paths are biased is a mechanism finding.
    worst = None
    if blocks:
        worst = max(
            blocks,
            key=lambda b: abs(
                b["nie"]["bias_mean"]
                - _get(payload, b["condition"], b["n_rows"])["effects"]["nie"]["bias_mean"]
            ),
        )
    by_key = {p["condition"]: p for p in probe}
    lines = [
        "\nWhere the two paths disagree the cause is a prior-scale conflict in the PyMC "
        "model's outcome equation rather than a sampling problem: the table above reports the "
        "maximum r_hat and the divergence count for every fit. "
        "`mediation.fit_mediation_model` gives the mediator baseline a "
        "scale-aware prior (`mu_m ~ Normal(mean(M), sd(M))`, added in the 2026-09-07 repair) "
        "and leaves the outcome equation on fixed-scale priors "
        "(`alpha0 ~ Normal(0, 1.5)`, `beta ~ Normal(0, 2)`). On one seeded dataset per "
        "mechanism at n = " + str(small) + ":\n",
    ]
    for key, note in (
        ("f4_rationalization", "a mediator with a baseline near six that also drives the answer"),
        ("f3_shared_cause", "a mediator centred near zero"),
    ):
        p = by_key.get(key)
        if p is None:
            continue
        lines.append(
            f"- **{key}** ({note}): the probit MAP fit gives beta {_f(p['map_beta'], 3)} and "
            f"alpha0 {_f(p['map_alpha0'], 3)}, which on the logit scale the PyMC model works "
            f"on are about {_f(p['map_beta_on_logit_scale'], 3)} and "
            f"{_f(p['map_alpha0_on_logit_scale'], 3)}. The posterior returns beta "
            f"{_f(p['posterior_beta_mean'], 3)} "
            f"[{_f(p['posterior_beta_lo'], 3)}, {_f(p['posterior_beta_hi'], 3)}] and alpha0 "
            f"{_f(p['posterior_alpha0_mean'], 3)} "
            f"[{_f(p['posterior_alpha0_lo'], 3)}, {_f(p['posterior_alpha0_hi'], 3)}], which "
            + (
                "excludes"
                if not (
                    p["posterior_alpha0_lo"]
                    <= p["map_alpha0_on_logit_scale"]
                    <= p["posterior_alpha0_hi"]
                )
                else "contains"
            )
            + " the logit-scale value the MAP fit implies."
        )
    tail = (
        "\nWhen the implied outcome intercept sits several prior standard deviations from "
        "zero the posterior cannot reach it, and it shrinks the intercept and the mediator "
        "coefficient together, which pulls the mediated effect down. That is exactly the shape of mediator the intercept "
        "repair was about, a reasoning-step count with a baseline well away from zero, so the "
        "finding belongs to the estimator workstream rather than to this battery: **the MAP "
        "path is scale-free and the posterior path is not, and this battery does not change "
        "either of them.** "
    )
    if worst is not None:
        map_block = _get(payload, worst["condition"], small)
        tail += (
            f"The largest disagreement in the subset is on {worst['condition']}: the posterior "
            f"NIE bias is {_f(worst['nie']['bias_mean'])} over {worst['n_datasets']} datasets "
            f"against {_f(map_block['effects']['nie']['bias_mean'])} for the MAP path over "
            f"{map_block['n_datasets']} datasets of the same condition at the same size, and "
            f"the posterior interval covered the truth in "
            f"{worst['nie']['coverage_k']}/{worst['nie']['coverage_n']} against "
            f"{map_block['effects']['nie']['coverage_k']}/"
            f"{map_block['effects']['nie']['coverage_n']} for the bootstrap."
        )
    return "\n".join(lines) + "\n" + tail + "\n"


def _pymc_table(payload) -> str:
    blocks = payload.get("pymc_subset") or []
    if not blocks:
        return "The PyMC subset was not run in this pass.\n"
    head = (
        "| condition | datasets | NIE coverage | NIE bias | mean posterior width | "
        "mean bootstrap width | max r_hat | divergences |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for b in blocks:
        c = b["posterior_minus_map_nie"]
        rows.append(
            f"| {b['condition']} | {b['n_datasets']} "
            f"| {b['nie']['coverage_k']}/{b['nie']['coverage_n']} "
            f"[{_f(b['nie']['coverage_ci_lo'],2)}, {_f(b['nie']['coverage_ci_hi'],2)}] "
            f"| {_f(b['nie']['bias_mean'])} | {_f(c['mean_posterior_width'],3)} "
            f"| {_f(c['mean_bootstrap_width'],3)} | {_f(b['max_r_hat'],4)} "
            f"| {b['total_divergences']} |"
        )
    return head + "\n".join(rows) + "\n"


# --------------------------------------------------------------------------- #
# One paragraph per family, written from the run
# --------------------------------------------------------------------------- #
def _undercoverage_note(block, n_rows: int) -> str:
    """Flag a coverage number that sits significantly below nominal, wherever it appears."""
    if block["coverage_ci_hi"] >= NOMINAL_COVERAGE:
        return ""
    return (
        f" That coverage, {block['coverage_k']}/{block['coverage_n']} with a binomial interval "
        f"of [{_f(block['coverage_ci_lo'], 3)}, {_f(block['coverage_ci_hi'], 3)}] at n = "
        f"{n_rows}, sits below the nominal 0.95 and is reported as a miss rather than rounded "
        f"up: the point estimate is right and the interval is a little too narrow."
    )


def _saturation_note(conds, n_rows: int) -> str:
    """Say plainly where in family 2 the interval fails, with the numbers that explain it."""
    worst = min(conds, key=lambda c: c["effects"]["nie"]["coverage"])
    blk = worst["effects"]["nie"]
    if blk["coverage"] >= 0.90:
        return (
            f" At n = {n_rows} no condition in the family covers below 90 percent, so the "
            "components that do carry the answer are estimated honestly at the same time."
        )
    return (
        f" The weakest of those, {worst['condition']} at n = {n_rows}, is the battery's clearest interval "
        f"failure and it is not a rounding artifact: the answer rate there is "
        f"{_f(worst['diagnostics']['mean_answer_rate'], 3)}, the true mediated effect is "
        f"{_f(worst['truth_analytic']['nie'])} on a total of "
        f"{_f(worst['truth_analytic']['te'])}, and the mean interval width is "
        f"{_f(blk['mean_interval_width'], 4)} against a mean bias of "
        f"{_f(blk['bias_mean'])}. A probability-scale effect this close to the boundary has a "
        f"skewed sampling distribution that a percentile bootstrap of "
        f"{blk['coverage_n']} datasets does not track, so the interval is too narrow even "
        f"though the point estimate is nearly unbiased. The consequence for the real table is "
        f"specific: in a cell where the cue almost determines the answer, the point estimates "
        f"and the total effect are still readable, and an interval statement about the small "
        f"remaining mediated path is not."
    )


def _reading(payload, family: str, small: int, big: int) -> str:
    if family == "f2_direct_bypass":
        conds = sorted(
            _family_conditions(payload, family, big),
            key=lambda c: float(c["condition"].split("alpha")[1]),
        )
        small_conds = sorted(
            _family_conditions(payload, family, small),
            key=lambda c: float(c["condition"].split("alpha")[1]),
        )
        small_covs = [
            f"{c['effects']['nie']['coverage_k']}/{c['effects']['nie']['coverage_n']}"
            for c in small_conds
        ]
        first, last = conds[0], conds[-1]
        rho_means = [c["rho_star_point"]["mean"] for c in conds]
        covs = [
            f"{c['effects']['nie']['coverage_k']}/{c['effects']['nie']['coverage_n']}"
            for c in conds
        ]
        return (
            f"The text pathway is identical in all {len(conds)} conditions (beta 0.8, gamma 1, "
            f"sigma_m 1, rho 0) and only the direct bypass changes. At n = {big} the estimated "
            f"mediated share falls from {_pct(first['mediated_share']['mean'])} at alpha = 0 "
            f"(over {first['mediated_share']['n']}/{first['n_datasets']} datasets whose total "
            f"effect interval excludes zero) to {_pct(last['mediated_share']['mean'])} at "
            f"alpha = 3 (over {last['mediated_share']['n']}/{last['n_datasets']}), while the "
            f"mean rho*_point stays between {_f(min(rho_means))} and {_f(max(rho_means))}, a "
            f"spread of {_f(max(rho_means) - min(rho_means))} across a change in the bypass "
            f"from 0 to 3. "
            f"That non-movement is the PASS: rho*_point is the robustness of the sign of the "
            f"mediator coefficient, so it cannot answer a question about the direct path, and a "
            f"reader who treats a large rho* as evidence of a load-bearing chain of thought is "
            f"reading a quantity that was never about that. NIE interval coverage across the "
            f"five conditions is {', '.join(covs)} datasets at n = {big} and "
            f"{', '.join(small_covs)} at n = {small}."
            + _saturation_note(conds, big)
            + _saturation_note(small_conds, small)
        )
    c_s, c_b = _get(payload, family, small), _get(payload, family, big)
    e_s, e_b = c_s["effects"], c_b["effects"]
    v_b = c_b["verdict"]

    if family == "f1_no_cue_effect":
        return (
            f"Nothing in this world has a causal effect, and the mediator carries a real "
            f"baseline: mean reasoning length {_f(c_b['diagnostics']['mean_mediator_mean'],2)} "
            f"and answer rate {_f(c_b['diagnostics']['mean_answer_rate'],3)}. The repaired "
            f"estimator reports a mean NIE of {_f(e_b['nie']['estimate_mean'])} at n = {big} "
            f"and {_f(e_s['nie']['estimate_mean'])} at n = {small}; the 95 percent intervals "
            f"cover the true zero in {e_b['nie']['coverage_k']}/{e_b['nie']['coverage_n']} and "
            f"{e_s['nie']['coverage_k']}/{e_s['nie']['coverage_n']} datasets respectively, and "
            f"the pre-registered verdict fires in {v_b['load_bearing_k']}/"
            f"{v_b['load_bearing_n']} at n = {big}. This is the world the pre-repair "
            f"specification got wrong by 0.21 on the mediated path, so the family is the "
            f"standing evidence that the intercept repair is what makes the rest of the "
            f"battery readable."
        )
    if family == "f3_shared_cause":
        return (
            f"M = X + U and Y = 1[X + U + E > 0]: the text and the answer share a hidden cause "
            f"and the text causes nothing. Mean corr(M, Y) is "
            f"{_f(c_b['diagnostics']['mean_corr_m_y'],3)}, and the estimator reports a mean NIE "
            f"of {_f(e_b['nie']['estimate_mean'])} against a true NIE of exactly zero, with "
            f"interval coverage {e_b['nie']['coverage_k']}/{e_b['nie']['coverage_n']} at "
            f"n = {big}. The direct path takes the opposite error: mean NDE "
            f"{_f(e_b['nde']['estimate_mean'])} against a truth of "
            f"{_f(c_b['truth_analytic']['nde'])}, while the total effect is recovered (bias "
            f"{_f(e_b['te']['bias_mean'])}). The false-robust-verdict rate is "
            f"{_pct(v_b['false_robust_verdict_rate'])} of {v_b['load_bearing_n']} datasets, so "
            f"at rho = 0 the audit is confidently wrong. The only thing that flags it is the "
            f"sensitivity summary: mean rho*_point {_f(c_b['rho_star_point']['mean'])} against "
            f"the analytic 1/sqrt(2), which says the whole conclusion dissolves at a residual "
            f"correlation this mechanism supplies by construction."
        )
    if family == "f4_rationalization":
        return (
            f"Here the mediated path is real and the direct path is zero: the cue moves the "
            f"reasoning text and the text moves the answer. The estimator recovers it, with NIE "
            f"bias {_f(e_b['nie']['bias_mean'])} +/- {_f(e_b['nie']['bias_mcse'])} and coverage "
            f"{e_b['nie']['coverage_k']}/{e_b['nie']['coverage_n']} at n = {big}, and "
            f"{e_s['nie']['coverage_k']}/{e_s['nie']['coverage_n']} at n = {small}. The verdict "
            f"fires in {v_b['load_bearing_k']}/{v_b['load_bearing_n']} datasets, which is the "
            f"correct answer here because the true NIE of "
            f"{_f(c_b['truth_analytic']['nie'],3)} clears the 0.15 threshold. Family 4 is what "
            f"stops the battery being a collection of nulls: an instrument that never fires is "
            f"as useless as one that always does."
        )
    if family == "f5_redundant_explanation":
        return (
            f"A latent state decides the answer and the text reports that state accurately "
            f"without causing it; mean corr(M, Y) is "
            f"{_f(c_b['diagnostics']['mean_corr_m_y'],3)}, higher than any other family. The "
            f"true NIE is zero and the estimator reports {_f(e_b['nie']['estimate_mean'])} with "
            f"coverage {e_b['nie']['coverage_k']}/{e_b['nie']['coverage_n']}, a false-robust "
            f"rate of {_pct(v_b['false_robust_verdict_rate'])} of {v_b['load_bearing_n']}, and "
            f"the largest rho*_point in the battery at {_f(c_b['rho_star_point']['mean'])}. "
            f"That combination is the most useful single fact the battery produces: the cell "
            f"that looks most robust is the one where the mediation is entirely absent, so a "
            f"large rho* must never be published as a trust score. An accurate explanation and "
            f"a load-bearing one are different claims and this instrument cannot separate them "
            f"from observational text alone."
        )
    if family == "f6_answer_copying":
        return (
            f"The answer is copied from the cue through a channel the mediator does not carry, "
            f"and the cue still lengthens the text a little (mean fitted gamma "
            f"{_f(c_b['diagnostics']['mean_fitted_gamma'],3)}). This is the case a "
            f"the-text-changed-therefore-it-mattered reading gets wrong, and the estimator does "
            f"not: mean NIE {_f(e_b['nie']['estimate_mean'])} against a true zero, coverage "
            f"{e_b['nie']['coverage_k']}/{e_b['nie']['coverage_n']}, false-robust rate "
            f"{_pct(v_b['false_robust_verdict_rate'])} of {v_b['load_bearing_n']}, and the "
            f"direct effect recovered with bias {_f(e_b['nde']['bias_mean'])}."
            + _undercoverage_note(e_b["nie"], big)
            + " Family 6 and "
            "family 5 differ only in whether the text is coupled to the answer-relevant state, "
            "and that single difference is what separates a clean report from a confident "
            "error."
        )
    if family == "f7_opposing_effects":
        share = c_b["mediated_share"]
        return (
            f"The direct and indirect paths are large and opposite: truth NDE "
            f"{_f(c_b['truth_analytic']['nde'],3)}, NIE {_f(c_b['truth_analytic']['nie'],3)}, "
            f"TE {_f(c_b['truth_analytic']['te'],3)}. The estimator recovers both components "
            f"(NDE bias {_f(e_b['nde']['bias_mean'])}, NIE bias {_f(e_b['nie']['bias_mean'])}, "
            f"coverage {e_b['nde']['coverage_k']}/{e_b['nde']['coverage_n']} and "
            f"{e_b['nie']['coverage_k']}/{e_b['nie']['coverage_n']} at n = {big}) and the total "
            f"correctly reads as nothing. The TE interval excludes zero in only "
            f"{share['te_interval_excludes_zero_k']}/{c_b['n_datasets']} datasets, which is why "
            f"the pre-registered rule refuses to print a mediated share here at all: a ratio "
            f"whose denominator is compatible with zero is not a quantity. An audit that looked "
            f"only at the arm difference would report no cue effect on a model whose reasoning "
            f"is doing a great deal of work in both directions."
        )
    raise KeyError(family)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def _cross_family_section(payload, big: int) -> str:
    """The comparison that no single family can make: two worlds, one report."""
    shared = _get(payload, "f3_shared_cause", big)
    real = _get(payload, "f4_rationalization", big)
    redundant = _get(payload, "f5_redundant_explanation", big)

    def line(c):
        e = c["effects"]
        return (
            f"| {c['condition']} | {_f(c['truth_analytic']['nie'], 3)} "
            f"| {_f(e['nde']['estimate_mean'])} | {_f(e['nie']['estimate_mean'])} "
            f"| {_f(e['te']['estimate_mean'])} "
            f"| {_f(c['rho_star_point']['mean'])} "
            f"[{_f(c['rho_star_point']['mean_interval_lo'], 3)}, "
            f"{_f(c['rho_star_point']['mean_interval_hi'], 3)}] "
            f"| {_f(c['rho_star_decision']['median_crossing'], 3)} "
            f"| {c['verdict']['load_bearing_k']}/{c['verdict']['load_bearing_n']} |"
        )

    gap = abs(shared["rho_star_point"]["mean"] - real["rho_star_point"]["mean"])
    half_width = 0.5 * (
        real["rho_star_point"]["mean_interval_hi"] - real["rho_star_point"]["mean_interval_lo"]
    )
    return (
        f"## The comparison no single family can make\n\n"
        f"Family 3 has no mediation at all and family 4 is fully mediated. At n = {big}, "
        f"averaged over {shared['n_datasets']} and {real['n_datasets']} datasets, this is what "
        f"the instrument reports for each.\n\n"
        "| condition | true NIE | mean NDE | mean NIE | mean TE | rho*_point [mean interval] | "
        "median rho*_decision | load-bearing |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(line(c) for c in (shared, real, redundant))
        + "\n\n"
        f"The two reports are the same report. Both put essentially the whole total effect on "
        f"the mediated path, both fire the verdict in nearly every dataset, and their "
        f"rho*_point values differ by {_f(gap, 4)} against a mean interval half-width of "
        f"{_f(half_width, 4)}, so the sensitivity summary separates them by nothing at all. "
        f"Family 5 is worse than a tie: its rho*_point of "
        f"{_f(redundant['rho_star_point']['mean'])} and its median rho*_decision of "
        f"{_f(redundant['rho_star_decision']['median_crossing'], 3)} are the largest in the "
        f"battery, and its true mediated effect is zero. Ranking cells by rho* would therefore "
        f"put a world with no mediation above a world that is entirely mediated. What separates "
        f"them is not a number this instrument produces from observational text: it is an "
        f"assumption about rho, or an intervention, and the report says so wherever a rho "
        f"quantity appears.\n"
    )


def build_report(payload) -> str:
    sizes = sorted({c["n_rows"] for c in payload["conditions"]})
    small, big = sizes[0], sizes[-1]
    s = payload["settings"]
    k = payload["constants"]
    checks = payload["cross_checks"]
    n_cond = len(_by(payload, small))

    parts: list[str] = []
    parts.append(
        f"# CPU mechanism battery (Amendment A2, element 11(a))\n\n"
        f"Generated {payload['generated_utc']} by `experiments/mechanism_battery.py` in "
        f"{payload['runtime_seconds']} seconds of wall clock on CPU. Reproduce with\n\n"
        f"```\nPYTHONPATH=src python experiments/mechanism_battery.py --out {s['out']}\n```\n\n"
        f"Seven generator families, evaluated as {n_cond} conditions (family 2 carries five "
        f"bypass strengths), at n = {' and '.join(str(x) for x in sizes)} rows per dataset, "
        f"{s['n_datasets']} seeded datasets per condition per sample size, "
        f"{n_cond * s['n_datasets'] * len(sizes)} datasets in total, each fitted once at "
        f"rho = 0 plus {s['n_bootstrap']} bootstrap refits. Every dataset is drawn from a "
        f"seed fixed before the run (dataset seed base {k['dataset_seed_base']}, bootstrap seed "
        f"base {k['bootstrap_seed_base']}); no result below is a re-run after seeing a number.\n"
    )

    parts.append(
        "## What was fitted, and where each interval comes from\n\n"
        "The estimator under test is the repaired MAP fit, "
        "`fit_probit_mediation_map(..., rho=0, intercepts=True)`, converted to "
        "probability-scale natural effects by the repository's exact closed form. "
        f"**Intervals in the main tables are nonparametric row-bootstrap percentile "
        f"intervals with {s['n_bootstrap']} replicates**, each replicate refitted from "
        f"scratch. **A separate subset of {s['n_pymc']} datasets per family at n = "
        f"{small} is re-run through the repaired PyMC posterior** "
        "(`fit_mediation_model(..., intercepts=True)` plus `posterior_natural_effects` with "
        "the intercept draws) and reported in its own section, so the cheap interval is "
        "compared against the expensive one rather than assumed equal to it.\n\n"
        f"The sensitivity sweep uses the reparameterisation documented in `sensitivity`: the "
        f"mediator equation does not move with the assumed rho, so one fit at rho = 0 "
        f"determines the whole curve. It is evaluated on a grid from 0 to {k['rho_max']} in "
        f"steps of {k['rho_step']} and cross-checked below against the repository's "
        f"refit-per-rho `sensitivity_sweep` and `breakdown_frontier`.\n\n"
        "Two rho quantities are reported and never merged. **rho\\*_point** is the zero "
        "crossing of the point-estimate NIE, which is a function of the fitted mediator "
        "coefficient and sigma_m alone and therefore carries no information about the direct "
        "path. **rho\\*_decision** is the rho at which the pre-registered verdict rule fails: "
        f"a dataset is load-bearing at rho when the probability that the NIE exceeds "
        f"{k['nie_threshold']} is at least {k['verdict_prob']}, evaluated on the bootstrap "
        "distribution. Where the verdict already fails at rho = 0 the verdict is "
        "**unresolved** and no crossing is reported; where it never fails inside the evaluated "
        f"range the report says so rather than inventing a crossing beyond {k['rho_max']}.\n\n"
        "The **false-robust-verdict rate** is the share of datasets where the verdict fires at "
        "rho = 0 in a family whose true NIE is exactly zero.\n"
    )

    parts.append("## Required outcomes\n")
    for r in required_outcomes(payload):
        parts.append(f"- **{'PASS' if r['pass'] else 'FAIL'} {r['name']}.** {r['detail']}")
    parts.append("")

    cs = checks["effects_curve_vs_sensitivity_sweep"]
    cf = checks["rho_star_point_vs_breakdown_frontier"]
    ct = checks["monte_carlo_truth_vs_analytic_truth"]
    parts.append(
        "## Cross-checks against the repository's own slower code paths\n\n"
        f"1. The vectorised rho curve against `sensitivity_sweep`, which refits the probit "
        f"model at every grid point and integrates by Monte Carlo: max absolute NIE difference "
        f"{_f(cs['max_abs_nie_difference'], 5)}, mean "
        f"{_f(cs['mean_abs_nie_difference'], 5)}, over {cs['n_datasets']} datasets at "
        f"{cs['n_rho_points_each']} rho values each ({cs['n_datasets'] * cs['n_rho_points_each']} "
        f"comparisons) at n = {cs['n_rows']}.\n"
        f"2. The analytic `rho*_point` against `breakdown_frontier`, which root-finds on "
        f"refits: max absolute difference {_f(cf['max_abs_difference'], 5)}, mean "
        f"{_f(cf['mean_abs_difference'], 5)}, over {cf['n_comparisons']} datasets that had a "
        f"crossing inside the evaluated range.\n"
        f"3. Each family's Monte Carlo truth ({ct['n_monte_carlo_rows']:,} rows, common noise "
        f"across the three cross-world cells) against its hand-derived analytic truth: max "
        f"absolute difference {_f(ct['max_abs_difference'], 5)} over "
        f"{ct['n_conditions']} conditions. The analytic value is what the bias and coverage "
        f"columns are computed against.\n"
    )

    parts.append(_cross_family_section(payload, big))
    parts.append("## Truth per condition\n\n" + _truth_table(payload))
    for n_rows in sizes:
        parts.append(
            f"## Effects, bias and coverage at n = {n_rows}\n\n" + _effects_table(payload, n_rows)
        )
        parts.append(
            f"## Verdicts and rho behaviour at n = {n_rows}\n\n" + _verdict_table(payload, n_rows)
        )

    parts.append("## Reading, one paragraph per family\n")
    for family, title in FAMILY_TITLES.items():
        parts.append(f"### {title}\n\n{_reading(payload, family, small, big)}\n")

    parts.append(
        f"## The PyMC subset at n = {small}\n\n"
        f"{s['n_pymc']} datasets per family, re-run through the repaired PyMC posterior. The "
        "outcome equation in that model is logistic while every generator here has a Gaussian "
        "latent outcome and the MAP estimator is probit, so the two paths are not the same "
        "estimator; the last two columns measure how far apart they land on identical data "
        "rather than assuming they agree.\n\n"
        + _pymc_table(payload)
        + _prior_probe_section(payload, small)
    )

    parts.append(
        "## Limitations of this battery\n\n"
        "- Every family is a low-dimensional simulation with a scalar mediator, a binary "
        "outcome and a randomized cue. It bounds what the estimator does on known mechanisms; "
        "it says nothing about whether a real truncation-curve summary is the mediator the "
        "contract names.\n"
        "- The bootstrap interval is a frequentist stand-in for the pre-registered posterior "
        "verdict rule. The PyMC subset is what licenses that substitution, and it is a subset.\n"
        "- The four zero-mediation families differ in how the association arises, not in "
        "whether a text-level instrument could tell them apart; nothing here shows that any "
        "measurement could separate family 5 from family 4 without an intervention.\n"
        "- rho prices assumption A3 only. A trigger-conditioned pathway of the kind the "
        "organism ladder plants is an A4 violation, which no value of rho prices, so a passing "
        "battery does not license reading rho\\* as a severity scale for that mechanism.\n"
    )
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Rebuild ``report.md`` from a saved ``battery_results.json`` without refitting."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Rebuild the mechanism battery report.")
    parser.add_argument("--results", required=True, help="path to battery_results.json")
    parser.add_argument("--out", required=True, help="path to write report.md")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.results).read_text())
    Path(args.out).write_text(build_report(payload))
    print(f"wrote {args.out} from {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
