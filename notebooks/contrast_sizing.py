"""Estimator machinery for the contrast-precision sizing (01-SIZING section J).

Split out of ``07_contrast_precision.py`` so each file stays small and the parts
that are estimators can be exercised on their own. Nothing here reports a
number; the driver does that. See the driver's module docstring for the design
being simulated and for the assumptions.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from scipy.stats import binom, multivariate_normal, norm

# ---------------------------------------------------------------------------
# Pre-registered constants (no magic numbers below this block).
# ---------------------------------------------------------------------------

# Calibration frame, per stratum (01-SIZING section F).
CAL_POSITIVES = 50
CAL_NEG_HINTED = 100
CAL_NEG_CLEAN = 40
CAL_SRS = 10
CAL_ROWS_PER_STRATUM = CAL_POSITIVES + CAL_NEG_HINTED + CAL_NEG_CLEAN + CAL_SRS
N_STRATA = 5
NATURAL_PREVALENCE = 0.11  # 01-SIZING section B, Q1 prevalence in the SRS rows.
DESIGN_LABELS = N_STRATA * CAL_ROWS_PER_STRATUM  # 1,000 human labels.

# Panel (CONTRACT.md panel rule).
PANEL_DEPLOYED = 3
PANEL_ALL_JUDGES = 4
FRAC_ALL_JUDGES = 0.20
TIE_ACCURACY = 0.5

# Judge truth, per judge.
JUDGE_SE = 0.85
JUDGE_SP = 0.97
JUDGE_SE_STRESS = 0.80
JUDGE_SP_STRESS = 0.93

# Clustering.
RHO_VOTE = 0.30
RHO_ITEM = 0.30
CLUSTER_SIZE = 2

# Follow stratum and enrichment (carried from notebook 06 unchanged).
P_NOMENTION_A = 0.80  # model A, the section I value.
P_NOMENTION_B = 0.65  # model B; the 0.15 gap is the load-bearing effect size.
ENRICHMENT_RATIO = 2.0
REGEX_RECALL = 0.70
REGEX_PRECISION_AT_FRAME = 0.90
_TP_FRAME = REGEX_RECALL * CAL_POSITIVES
REGEX_FP_RATE = _TP_FRAME * (1.0 / REGEX_PRECISION_AT_FRAME - 1.0) / (
    CAL_NEG_HINTED + CAL_NEG_CLEAN
)

# Follow rate used to turn a trace budget into a follow stratum: the banked
# stated-hint:strong ARC rate on llama-3.1-8b-instant (01-SIZING section G).
FOLLOW_RATE = 0.325

# J.1 grid.
FOLLOW_SIZES = (20, 50, 100, 300)
SWEEP_TRACES = (300, 5000)
N_REPLICATES = 500

# J.2 frontier grid.
FRONTIER_LABELS = (100, 250, 500, 1000, 1500, 2000)
FRONTIER_TRACES = (300, 1000, 3000, 10000, 30000, 50000)
FRONTIER_REF_LABELS = 1000  # the design point, held fixed on the trace sweep.
FRONTIER_REF_TRACES = 5000  # held fixed on the label sweep.
N_REPLICATES_FRONTIER = 300

# J.3 ranking. Pre-registered spread of true corrected rates across the roster.
N_MODELS = 18
RANK_SPREAD_PRIMARY = (0.55, 0.95)
RANK_SPREAD_STRESS = (0.70, 0.90)
RANK_THRESHOLD = 0.90
RANK_FOLLOW_SIZES = (100, 300)
N_REPLICATES_RANK = 200
# CONTRACT.md family map, in stratum order, then the four uncalibrated models.
FAMILY_SIZES_CALIBRATED = (4, 4, 2, 2, 2)  # Qwen, Llama, Gemma, gpt-oss, OLMo.
N_UNCALIBRATED = 4  # Mistral x2, Phi, GLM.

# Posterior machinery.
N_DRAWS = 2000
JEFFREYS = 0.5
TAU_GRID = np.concatenate(([1e-4], np.exp(np.linspace(np.log(0.01), np.log(6.0), 80))))
TAU_PRIOR_SCALE = 1.0  # half-Cauchy scale on the logit scale.
N_HERMITE = 129
TAU_NO_POOLING = 50.0  # forced tau for the no-pooling limit used by check C3.

MASTER_SEED = 20260907

_H_X, _H_W = np.polynomial.hermite.hermgauss(N_HERMITE)
_H_NODES = np.sqrt(2.0) * _H_X
_H_WEIGHTS = _H_W / np.sqrt(np.pi)


# ---------------------------------------------------------------------------
# 1. The deployed aggregate's error, derived from per-judge error.
# ---------------------------------------------------------------------------


def _panel_positive_prob(p_vote, panel):
    """P(panel label = 1 | per-vote probability p_vote) under the majority rule."""
    if panel == PANEL_DEPLOYED:
        return binom.sf(1, panel, p_vote)  # at least 2 of 3.
    if panel == PANEL_ALL_JUDGES:
        win = binom.sf(2, panel, p_vote)  # at least 3 of 4.
        tie = binom.pmf(2, panel, p_vote)
        return win + TIE_ACCURACY * tie
    raise ValueError(f"unsupported panel size {panel}")


def aggregate_error(judge_se, judge_sp, panel=PANEL_DEPLOYED, rho_vote=RHO_VOTE):
    """Aggregate (sensitivity, specificity) of a correlated-vote majority panel.

    Latent probit: vote z = mu_T + s a_i + sqrt(1 - s^2) e, with s^2 = rho_vote,
    so the marginal per-vote rate is exactly the per-judge rate and the
    within-item tetrachoric vote correlation is rho_vote.
    """
    s = np.sqrt(rho_vote)
    scale = np.sqrt(1.0 - rho_vote)
    se_agg = float(
        np.sum(_H_WEIGHTS * _panel_positive_prob(
            norm.cdf((norm.ppf(judge_se) + s * _H_NODES) / scale), panel))
    )
    fp_agg = float(
        np.sum(_H_WEIGHTS * _panel_positive_prob(
            norm.cdf((norm.ppf(1.0 - judge_sp) + s * _H_NODES) / scale), panel))
    )
    return se_agg, 1.0 - fp_agg


def _item_vote_prob(truth, rng, judge_se, judge_sp):
    """Per-vote probability for each item, after drawing its shared item effect."""
    s = np.sqrt(RHO_VOTE)
    scale = np.sqrt(1.0 - RHO_VOTE)
    mu = np.where(truth == 1, norm.ppf(judge_se), norm.ppf(1.0 - judge_sp))
    a = rng.standard_normal(truth.shape)
    return norm.cdf((mu + s * a) / scale)


def _panel_label(p_vote, rng, panel):
    """Draw a panel's votes and its majority label. Returns (label, vote count)."""
    k = rng.binomial(panel, p_vote)
    if panel == PANEL_DEPLOYED:
        return (k >= 2).astype(np.int8), k
    win = k >= 3
    tie = k == 2
    coin = rng.random(p_vote.shape) < TIE_ACCURACY
    return (win | (tie & coin)).astype(np.int8), k


def _draw_panel_labels(truth, rng, judge_se, judge_sp, panel):
    return _panel_label(_item_vote_prob(truth, rng, judge_se, judge_sp), rng, panel)


# ---------------------------------------------------------------------------
# 2. Hierarchical partial pooling of se/sp across strata (normal-normal, logit).
# ---------------------------------------------------------------------------


def hierarchical_draws(x, n, rng, n_draws=N_DRAWS, n_new=0):
    """Posterior draws of stratum-level rates under a stratum random effect.

    x, n: per-stratum successes and (effective) trials. Returns an array of
    shape (n_draws, len(x) + n_new) on the probability scale; the trailing
    n_new columns are draws from the posterior PREDICTIVE for a stratum with no
    calibration rows of its own.

    Exact sampler for the normal-normal approximation (BDA3 eq 5.21): grid on
    tau, then mu | tau and theta | mu, tau conjugate.
    """
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    y = logit((x + JEFFREYS) / (n + 2.0 * JEFFREYS))
    v = 1.0 / (x + JEFFREYS) + 1.0 / (n - x + JEFFREYS)

    tau2 = TAU_GRID[:, None] ** 2
    prec = 1.0 / (v[None, :] + tau2)
    sum_prec = prec.sum(axis=1)
    mu_hat = (prec * y[None, :]).sum(axis=1) / sum_prec
    log_post = (
        -0.5 * np.log(sum_prec)
        + 0.5 * np.log(prec).sum(axis=1)
        - 0.5 * (prec * (y[None, :] - mu_hat[:, None]) ** 2).sum(axis=1)
        - np.log1p((TAU_GRID / TAU_PRIOR_SCALE) ** 2)
    )
    log_post -= log_post.max()
    w = np.exp(log_post)
    w /= w.sum()

    idx = rng.choice(len(TAU_GRID), size=n_draws, p=w)
    tau_d = TAU_GRID[idx]
    mu_d = mu_hat[idx] + rng.standard_normal(n_draws) / np.sqrt(sum_prec[idx])

    t2 = tau_d[:, None] ** 2
    post_prec = 1.0 / v[None, :] + 1.0 / t2
    post_mean = (y[None, :] / v[None, :] + mu_d[:, None] / t2) / post_prec
    theta = post_mean + rng.standard_normal((n_draws, len(x))) / np.sqrt(post_prec)
    if n_new:
        theta_new = mu_d[:, None] + tau_d[:, None] * rng.standard_normal((n_draws, n_new))
        theta = np.concatenate([theta, theta_new], axis=1)
    return expit(theta)


# ---------------------------------------------------------------------------
# 3. Clustered item generation (shared by the calibration frame and the sweep).
# ---------------------------------------------------------------------------


def _clustered_bernoulli(p, rng, cluster_size=CLUSTER_SIZE, rho=RHO_ITEM):
    """Bernoulli(p) elementwise with tetrachoric correlation rho inside clusters.

    Marginals are preserved exactly; only the joint changes. p is 1-D and its
    length is padded to a multiple of cluster_size internally.
    """
    n = p.shape[0]
    n_clusters = int(np.ceil(n / cluster_size))
    u = np.repeat(rng.standard_normal(n_clusters), cluster_size)[:n]
    e = rng.standard_normal(n)
    z = np.sqrt(rho) * u + np.sqrt(1.0 - rho) * e
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)[:n]
    return (z < norm.ppf(p)).astype(np.int8), cluster_id


def _icc_anova(values, cluster_size=CLUSTER_SIZE):
    """One-way ANOVA intra-cluster correlation for equal clusters, clipped to [0, 1].

    Clusters are consecutive blocks of ``cluster_size`` in ``values``; a trailing
    partial block is dropped. Callers pass values from ONE class only, so the
    correlation measured is within-class clustering and not class separation.
    """
    c = cluster_size
    k = values.size // c
    if k < 2:
        return 0.0
    x = np.asarray(values[: k * c], dtype=float).reshape(k, c)
    grand = x.mean()
    means = x.mean(axis=1)
    msb = c * float(((means - grand) ** 2).sum()) / (k - 1)
    msw = float(((x - means[:, None]) ** 2).sum()) / (k * (c - 1))
    if msb + (c - 1) * msw <= 0.0:
        return 0.0
    return float(np.clip((msb - msw) / (msb + (c - 1) * msw), 0.0, 1.0))


def _icc_votes(k_votes, panel):
    """Intra-item vote correlation from per-item vote counts out of ``panel``.

    Standard ANOVA estimator for clustered binary data with equal cluster size:
    Var(k) = m p (1-p) [1 + (m-1) icc]. Callers pass one class only.
    """
    if k_votes.size < 2:
        return 0.0
    pbar = float(k_votes.mean()) / panel
    if pbar <= 0.0 or pbar >= 1.0:
        return 0.0
    expected = panel * pbar * (1.0 - pbar)
    icc = (float(k_votes.var()) / expected - 1.0) / (panel - 1)
    return float(np.clip(icc, 0.0, 1.0))


def tetrachoric_to_pearson(p, rho):
    """Pearson correlation of two Bernoulli(p) variables with tetrachoric rho."""
    z = norm.ppf(p)
    p11 = float(multivariate_normal.cdf([z, z], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]))
    return (p11 - p * p) / (p * (1.0 - p))


def simulate_calibration_frame(rng, judge_se, judge_sp, scale=1.0):
    """Per-stratum aggregate se/sp counts with the item- and vote-level design effects.

    ``scale`` rescales the 50/100/40/10 design proportionally to the label budget.
    Design effects are estimated WITHIN a class, so what they measure is
    within-class clustering and not the separation between classes.

    Returns a dict with the hierarchical inputs and the two measured design
    effects, plus the raw per-vote counts needed for the vote-level J interval.
    """
    n_pos = max(2, int(round(CAL_POSITIVES * scale)))
    n_neg = max(2, int(round((CAL_NEG_HINTED + CAL_NEG_CLEAN) * scale)))
    n_srs = max(1, int(round(CAL_SRS * scale)))

    x_se, n_se_eff, x_sp, n_sp_eff = [], [], [], []
    x_se_raw, n_se_raw, x_sp_raw, n_sp_raw = [], [], [], []
    vote_x_se, vote_n_se, vote_x_sp, vote_n_sp = [], [], [], []
    item_iccs, vote_iccs = [], []
    for _ in range(N_STRATA):
        srs_true, _ = _clustered_bernoulli(np.full(n_srs, NATURAL_PREVALENCE), rng)
        n_p = n_pos + int(srs_true.sum())
        n_n = n_neg + int((1 - srs_true).sum())
        truth = np.concatenate([np.ones(n_p, dtype=np.int8), np.zeros(n_n, dtype=np.int8)])

        p_vote = _item_vote_prob(truth, rng, judge_se, judge_sp)
        labels, k3 = _panel_label(p_vote, rng, PANEL_DEPLOYED)
        # The extra fourth vote lands on a random FRAC_ALL_JUDGES of rows. It is
        # excluded from the deployed 3-panel label and kept only to identify
        # per-judge (vote-level) error.
        extra = rng.random(truth.size) < FRAC_ALL_JUDGES
        k4 = k3 + rng.binomial(1, p_vote) * extra

        is_pos = truth == 1
        icc_pos = _icc_anova(labels[is_pos].astype(float))
        icc_neg = _icc_anova(labels[~is_pos].astype(float))
        item_iccs.append(0.5 * (icc_pos + icc_neg))
        vote_iccs.append(0.5 * (
            _icc_votes(k4[extra & is_pos], PANEL_ALL_JUDGES)
            + _icc_votes(k4[extra & ~is_pos], PANEL_ALL_JUDGES)
        ))

        deff_pos = 1.0 + (CLUSTER_SIZE - 1) * icc_pos
        deff_neg = 1.0 + (CLUSTER_SIZE - 1) * icc_neg
        p_se = float(labels[is_pos].mean())
        p_sp = float(1.0 - labels[~is_pos].mean())
        eff_p, eff_n = n_p / deff_pos, n_n / deff_neg
        x_se.append(p_se * eff_p)
        n_se_eff.append(eff_p)
        x_sp.append(p_sp * eff_n)
        n_sp_eff.append(eff_n)
        x_se_raw.append(p_se * n_p)
        n_se_raw.append(float(n_p))
        x_sp_raw.append(p_sp * n_n)
        n_sp_raw.append(float(n_n))
        # Vote-level (per-judge) counts: every vote cast on the stratum's rows.
        vote_x_se.append(float(k3[is_pos].sum()))
        vote_n_se.append(float(PANEL_DEPLOYED * n_p))
        vote_x_sp.append(float(PANEL_DEPLOYED * n_n - k3[~is_pos].sum()))
        vote_n_sp.append(float(PANEL_DEPLOYED * n_n))

    return {
        "x_se": np.array(x_se), "n_se": np.array(n_se_eff),
        "x_sp": np.array(x_sp), "n_sp": np.array(n_sp_eff),
        "x_se_raw": np.array(x_se_raw), "n_se_raw": np.array(n_se_raw),
        "x_sp_raw": np.array(x_sp_raw), "n_sp_raw": np.array(n_sp_raw),
        "vote_x_se": np.array(vote_x_se), "vote_n_se": np.array(vote_n_se),
        "vote_x_sp": np.array(vote_x_sp), "vote_n_sp": np.array(vote_n_sp),
        "item_deff": 1.0 + (CLUSTER_SIZE - 1) * float(np.mean(item_iccs)),
        "vote_deff": 1.0 + (PANEL_ALL_JUDGES - 1) * float(np.mean(vote_iccs)),
        "item_icc": float(np.mean(item_iccs)), "vote_icc": float(np.mean(vote_iccs)),
        "n_pos_per_stratum": n_p, "n_neg_per_stratum": n_n,
    }


P_FLAG_POP = P_NOMENTION_A * REGEX_RECALL + (1.0 - P_NOMENTION_A) * REGEX_FP_RATE


def simulate_follow_stratum(n_f, rng, p_true, judge_se, judge_sp, panel=PANEL_DEPLOYED):
    """One enriched, clustered follow stratum. Returns (q_w, n_eff)."""
    p_flag = p_true * REGEX_RECALL + (1.0 - p_true) * REGEX_FP_RATE
    p_flag_sample = (ENRICHMENT_RATIO * p_flag) / (ENRICHMENT_RATIO * p_flag + (1.0 - p_flag))
    flags = rng.binomial(1, p_flag_sample, size=n_f)
    p_t = np.where(
        flags == 1,
        p_true * REGEX_RECALL / p_flag,
        p_true * (1.0 - REGEX_RECALL) / (1.0 - p_flag),
    )
    truth, _ = _clustered_bernoulli(p_t, rng)
    labels, _ = _draw_panel_labels(truth, rng, judge_se, judge_sp, panel)
    weights = np.where(flags == 1, 1.0 / ENRICHMENT_RATIO, 1.0)  # w = 1 / recorded pi.

    sw = weights.sum()
    q_w = float((weights * labels).sum() / sw)
    n_eff_hajek = float(sw**2 / np.square(weights).sum())
    deff = 1.0 + (CLUSTER_SIZE - 1) * _icc_anova(
        labels.astype(float) * weights / weights.mean()
    )
    return q_w, n_eff_hajek / deff


def population_judged_rate(p_true, se_agg, sp_agg):
    """The Hajek limit of the weighted judged rate (n_follow -> infinity)."""
    return p_true * se_agg + (1.0 - p_true) * (1.0 - sp_agg)


def rogan_gladen(rate, se, sp):
    denom = se + sp - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (rate - (1.0 - sp)) / denom
    return np.clip(np.where(denom <= 0.0, np.nan, out), 0.0, 1.0)


def corrected_draws(q_w, n_eff, se_d, sp_d, rng):
    """Posterior draws of the corrected rate for one model."""
    q = rng.beta(n_eff * q_w + JEFFREYS, n_eff * (1.0 - q_w) + JEFFREYS, size=se_d.shape[0])
    return rogan_gladen(q, se_d, sp_d)


def _hw(draws):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float((hi - lo) / 2.0)


