# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Paper figures — flow-accelerated PTA model comparison
#
# One master notebook producing every figure in the paper. Each figure is a
# self-contained section reading from the `outputs/` tree (`config.OUTPUT_DIR`).
# The shared paper style (`figs/paper.mplstyle`) is applied in the setup cell;
# figures are saved as `.pdf` + `.png` to `figs/` under the names the paper
# includes.
#
# Figures:
# - `pp_difference_overlay_{powerlaw,gaussian}` — PP-plot difference, flow+NS vs. traditional
# - `roc_bayes_factors` — ROC for the Bayes-factor classifier (with AUC)
# - `bayes_factor_distribution` — Bayes-factor distribution on Jeffreys' scale
# - `ns_vs_fulllik_flow` — nested-sampling vs. full-likelihood Bayes factors
# - `flow_correlations_{powerlaw,gaussian}` — cross-bin correlations, free-spectrum vs. flow
# - `gaussian_evidence_validation` — NS with the analytic likelihood vs. NS with a flow
# - `c2st_flow_freespec` — classifier two-sample test of the flow (data from `fig_c2st.py`)
# - `upper_tail_profile` — frozen vs. exact upper-tail profile (data from `scripts/make_upper_tail_profile.py`)

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = next(
    p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    if (p / "pyproject.toml").exists()
)
sys.path.insert(0, str(_ROOT / "notebooks"))
from _style import figsize, use_paper_style

from flows_model_comparison import config, full_likelihood

use_paper_style()

OUT = config.OUTPUT_DIR
NS_DIR = OUT / "nested_sampling"
SIM_DIR = OUT / "simulations"
RESULTS = OUT / "results"
DIAG = OUT / "diagnostics"
FIG_DIR = _ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["powerlaw", "gaussian"]
PARAMS = {
    "powerlaw": ["log10_A", "gamma"],
    "gaussian": ["log10_A", "log10_f_peak", "sigma"],
}
PARAM_LABELS = {
    "log10_A": r"$\log_{10}\!A$",
    "gamma": r"$\gamma$",
    "log10_f_peak": r"$\log_{10}\!f_{\mathrm{peak}}$",
    "sigma": r"$\sigma$, SD",
}
COLORS = {"powerlaw": "#1f77b4", "gaussian": "#d62728"}

LN10 = np.log(10.0)
bf = pd.read_csv(RESULTS / "bayes_factors.csv")
bf["log10_BF"] = bf["log_BF"] / LN10
bf["log10_BF_err"] = bf["log_BF_err"] / LN10
# Positive means the true model is favored.
bf["logB_truth"] = bf["log10_BF"].where(bf["sim_model"] == "powerlaw", -bf["log10_BF"])


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def pick_reps(model):
    """Best, median, and worst realizations by log B for the true model."""
    sub = bf[bf["sim_model"] == model].sort_values("logB_truth")
    return [
        (int(sub.iloc[-1]["realization"]), "best"),
        (int(sub.iloc[len(sub) // 2]["realization"]), "median"),
        (int(sub.iloc[0]["realization"]), "worst"),
    ]


# %% [markdown]
# ## PP-plot difference: flow+NS vs. traditional
#
# For each parameter we take the HDI rank of the injected truth in the posterior
# (uniform over realizations if calibrated), build the empirical CDF of those
# ranks for the flow+NS surrogate and for the traditional full-likelihood fit,
# and plot **their difference**. Both reference the same injection, so the shared
# deviation cancels: a flat difference inside the bands means the surrogate is as
# calibrated as the gold standard. Bands are Säilynoja simultaneous envelopes.
# All parameters of a model share one axes.

# %%
from arviz_stats.ecdf_utils import compute_ecdf
from scipy.stats import gaussian_kde

CI_LEVELS = [(0.997, r"$3\sigma$", 0.12), (0.95, r"$2\sigma$", 0.18), (0.68, r"$1\sigma$", 0.30)]
N_SIM = 2000
_RNG = np.random.default_rng(0)


def hdi_rank(samples, truth, weights=None, cap=4000):
    """Posterior mass at density >= density(truth); ~Uniform(0,1) if calibrated."""
    samples = np.asarray(samples, float)
    weights = np.ones(len(samples)) if weights is None else np.asarray(weights, float)
    weights = weights / weights.sum()
    # Resample to equal weights; a weighted KDE on every NS point is slow.
    if len(samples) > cap:
        samples = samples[_RNG.choice(len(samples), cap, p=weights)]
        weights = np.full(cap, 1.0 / cap)
        kde = gaussian_kde(samples)
    else:
        kde = gaussian_kde(samples, weights=weights)
    d_truth = float(kde(np.atleast_1d(float(truth)))[0])
    return float(np.sum(weights[np.asarray(kde(samples)) >= d_truth]))


def eval_grid(n):
    grid = np.arange(1, n) / n
    return grid, np.concatenate(([0.0], grid, [1.0]))


def pad_ecdf(ranks, grid):
    return np.concatenate(([0.0], compute_ecdf(np.asarray(ranks, float), grid), [1.0]))


def difference_band(n, prob, grid):
    """MC envelope for ECDF_a - ECDF_b under the uniform null."""
    sims = np.array([
        compute_ecdf(_RNG.uniform(size=n), grid) - compute_ecdf(_RNG.uniform(size=n), grid)
        for _ in range(N_SIM)
    ])
    # Bisect the pointwise level until whole curves stay inside with probability prob.
    lo_g, hi_g = 1e-4, 0.999
    for _ in range(40):
        mid = 0.5 * (lo_g + hi_g)
        lo = np.quantile(sims, mid / 2, axis=0)
        hi = np.quantile(sims, 1 - mid / 2, axis=0)
        if np.mean(np.all((sims >= lo) & (sims <= hi), axis=1)) > prob:
            lo_g = mid
        else:
            hi_g = mid
    mid = 0.5 * (lo_g + hi_g)
    lo, hi = np.quantile(sims, mid / 2, axis=0), np.quantile(sims, 1 - mid / 2, axis=0)
    return np.concatenate(([0.0], lo, [0.0])), np.concatenate(([0.0], hi, [0.0]))


def paired_ranks(model, param):
    """HDI ranks for realizations with both flow+NS and full-likelihood posteriors."""
    flow, trad = [], []
    for csv in sorted((NS_DIR / model).glob(f"realization-*/{model}_evidence.csv")):
        real = int(csv.parent.name.split("-")[1])
        trad_npz = full_likelihood.run_dir(model, real) / "posterior.npz"
        if not trad_npz.exists():
            continue
        truth = float(np.load(SIM_DIR / model / f"realization-{real}" / "params.npz")[f"gw_{param}"])
        post = pd.read_csv(csv)
        flow.append(hdi_rank(post[param].to_numpy(float), truth, post["weights"].to_numpy(float)))
        trad.append(hdi_rank(np.load(trad_npz)[param], truth))
    return np.asarray(flow), np.asarray(trad)


# %%
def make_pp_difference_overlay(model):
    params = PARAMS[model]
    paired = {p: paired_ranks(model, p) for p in params}
    n = max(len(v[0]) for v in paired.values())
    grid, gp = eval_grid(n)
    diff = {c: difference_band(n, c, grid) for c, _, _ in CI_LEVELS}

    fig, ax = plt.subplots(figsize=figsize(1.0), constrained_layout=True)
    band_handles = []
    for c, label, alpha in CI_LEVELS:
        lo, hi = diff[c]
        band_handles.append(ax.fill_between(gp, lo, hi, color="0.45", alpha=alpha, lw=0, label=label))
    ax.axhline(0, color="k", lw=0.8)
    line_handles = []
    for i, param in enumerate(params):
        ecdf_flow = pad_ecdf(paired[param][0], grid)
        ecdf_trad = pad_ecdf(paired[param][1], grid)
        line_handles.append(
            ax.plot(gp, ecdf_flow - ecdf_trad, lw=1.5, color=f"C{i}", label=PARAM_LABELS[param])[0]
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("credible level")
    ax.set_ylabel(r"$\Delta$ coverage")
    leg = ax.legend(handles=line_handles, fontsize=8, loc="upper center", ncol=3, columnspacing=1.0)
    ax.add_artist(leg)
    ax.legend(handles=band_handles, fontsize=8, loc="lower center", ncol=3)
    return fig


fig1 = make_pp_difference_overlay("powerlaw")
save(fig1, "pp_difference_overlay_powerlaw")
fig1

# %%
fig1g = make_pp_difference_overlay("gaussian")
save(fig1g, "pp_difference_overlay_gaussian")
fig1g

# %% [markdown]
# ## ROC for the Bayes-factor classifier
#
# Each pairwise comparison is a binary classifier scored by $\log\mathrm{BF}$
# (model 1 / model 2), positive class = data generated by model 1. Sweeping a
# threshold traces TPR vs. FPR. The decision points $\log_{10}\mathrm{BF}=0,\pm0.5$
# are marked; the AUC (threshold-free) is in the legend.

# %%
from sklearn.metrics import roc_auc_score, roc_curve

labels = (bf["sim_model"] == "powerlaw").astype(int).to_numpy()
scores = bf["log10_BF"].to_numpy()
fpr, tpr, _ = roc_curve(labels, scores)
auc = roc_auc_score(labels, scores)


def operating_point(threshold):
    pred = scores >= threshold
    return ((pred & (labels == 0)).sum() / max((labels == 0).sum(), 1),
            (pred & (labels == 1)).sum() / max((labels == 1).sum(), 1))


fig2, ax = plt.subplots(figsize=figsize(1.0, aspect=1.0))
ax.step(fpr, tpr, where="post", color="C0", lw=1.8,
        label=f"powerlaw vs gaussian (AUC $= {auc:.3f}$)")
for thr, marker, label in [(0.0, "o", r"$\log_{10} \mathrm{BF} = 0$"),
                           (0.5, "^", r"$\log_{10} \mathrm{BF} = +0.5$"),
                           (-0.5, "v", r"$\log_{10} \mathrm{BF} = -0.5$")]:
    ax.plot(*operating_point(thr), marker, color="k", ms=6, mec="k", mew=0.6, zorder=5, label=label)
ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.9, label="chance")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.legend(loc="lower right", framealpha=0.9)
save(fig2, "roc_bayes_factors")
fig2

# %% [markdown]
# ## Bayes-factor distribution on Jeffreys' scale
#
# Per-realization $\log_{10}\mathrm{BF}^{\mathrm{PL}}_{\mathrm{Gauss}}$ grouped by
# the true model, with the Jeffreys evidence bands shaded (heaviest =
# inconclusive). Positive favours powerlaw, negative favours gaussian.

# %%
import matplotlib.ticker as mticker

JEFFREYS = [
    (0.5, "0.55", 0.35),
    (1.0, "0.70", 0.22),
    (2.0, "0.85", 0.15),
    (None, "1.00", 0.00),
]
EDGES = [0.5, 1.0, 2.0]

fig3, ax = plt.subplots(figsize=figsize(1.0, aspect=1 / 1.3))
ax.set_yscale("symlog", linthresh=0.5)
ylim = float(np.ceil(bf["log10_BF"].abs().max() * 1.3))

lo = 0.0
for edge, grey, alpha in JEFFREYS:
    hi = ylim if edge is None else edge
    if alpha > 0:
        ax.fill_between([-0.5, 1.5], lo, hi, color=grey, alpha=alpha, lw=0, zorder=0)
        ax.fill_between([-0.5, 1.5], -hi, -lo, color=grey, alpha=alpha, lw=0, zorder=0)
    lo = hi
for edge in EDGES:
    ax.axhline(edge, color="0.55", ls=":", lw=0.6, zorder=1)
    ax.axhline(-edge, color="0.55", ls=":", lw=0.6, zorder=1)
ax.axhline(0, color="k", lw=0.8, zorder=1)

rng = np.random.default_rng(0)
for i, truth in enumerate(MODELS):
    sub = bf[bf["sim_model"] == truth]
    x = i + rng.uniform(-0.18, 0.18, size=len(sub))
    ax.errorbar(x, sub["log10_BF"], yerr=sub["log10_BF_err"], fmt="o", ms=4,
                color=COLORS[truth], alpha=0.9, ecolor="0.45", elinewidth=0.8,
                capsize=0, zorder=3, label=f"truth: {truth}")

for y, label in [(0.6, "inconclusive"), (0.7, "substantial"), (0.75, "strong"), (0.87, "decisive")]:
    ax.text(0.5, y, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color="0.4")

ax.set_xticks([])
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-ylim, ylim)
yt = np.array([-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10])
ax.set_yticks(yt[np.abs(yt) <= ylim])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_ylabel(r"$\log_{10} \mathrm{BF}^\mathrm{PL}_\mathrm{Gauss}$")
ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
save(fig3, "bayes_factor_distribution")
fig3

# %% [markdown]
# ## Nested sampling vs. full-likelihood Bayes factors
#
# Independent cross-check of the surrogate NS Bayes factor against the
# **full-likelihood** estimate: a Gaussian-container harmonic mean computed on
# full-PTA NUTS chains (no flow, no tails), stored in `bayes_factors_fulllik.csv`.
# Left: NS $\log_{10}\mathrm{BF}$ vs. full-likelihood $\log_{10}\mathrm{BF}$ with
# the $y=x$ line. Right: residual (NS $-$ full).

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable

full = pd.read_csv(RESULTS / "bayes_factors_fulllik.csv")
mf = bf[["sim_model", "realization", "log10_BF", "log10_BF_err"]].rename(
    columns={"log10_BF": "ns", "log10_BF_err": "ns_err"})
mf = mf.merge(full.assign(harm=full["log_BF"] / LN10, harm_err=full["log_BF_err"] / LN10)
              [["sim_model", "realization", "harm", "harm_err"]],
              on=["sim_model", "realization"])
mf["resid"] = mf["ns"] - mf["harm"]
print(f"{len(mf)} realizations with full-likelihood BFs; "
      f"resid: mean {mf.resid.mean():+.2f}, std {mf.resid.std():+.2f}, "
      f"median {mf.resid.median():+.2f}, RMS {np.sqrt((mf.resid**2).mean()):.2f} (log10)")

fig4b, ax = plt.subplots(figsize=figsize(1.0, aspect=0.78))
ax_r = make_axes_locatable(ax).append_axes("right", size="32%", pad=0.0, sharey=ax)

lim = [mf[["ns", "harm"]].min().min() - 3, mf[["ns", "harm"]].max().max() + 3]
ax.plot(lim, lim, ls="--", color="0.6", lw=0.9, zorder=1)
ax_r.axvline(0, color="0.6", ls="--", lw=0.9, zorder=1)
for truth in MODELS:
    sub = mf[mf["sim_model"] == truth]
    ax.errorbar(sub["harm"], sub["ns"], xerr=sub["harm_err"], yerr=sub["ns_err"],
                fmt="o", ms=4, color=COLORS[truth], alpha=0.9, ecolor="0.7",
                elinewidth=0.7, capsize=0, label=f"truth: {truth}")
    ax_r.errorbar(sub["resid"], sub["ns"], xerr=np.hypot(sub["ns_err"], sub["harm_err"]),
                  fmt="o", ms=4, color=COLORS[truth], alpha=0.9, ecolor="0.7",
                  elinewidth=0.7, capsize=0)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel(r"$\log_{10} \mathrm{BF}^{\mathrm{PL}}_{\mathrm{Gauss}}$ (full likelihood)")
ax.set_ylabel(r"$\log_{10} \mathrm{BF}^{\mathrm{PL}}_{\mathrm{Gauss}}$ (NS)")
ax.legend(loc="upper left", framealpha=0.9)
ax_r.set_xlabel(r"NS $-$ full")
plt.setp(ax_r.get_yticklabels(), visible=False)
save(fig4b, "ns_vs_fulllik_flow")
fig4b

# %% [markdown]
# ## Flow vs. free-spectrum cross-bin correlations
#
# Coppuccino flow against the free-spectrum MCMC it was trained on: the cross-bin
# rank-correlation structure (chain, flow, and their difference) for the median
# realization (by $\log_{10}\mathrm{BF}$ in favour of the truth) of each model —
# the joint structure a marginal check would miss.

# %%
import arviz as az
from scipy.stats import spearmanr

from coppuccino import load_flow, sample as flow_sample


def load_chain(model, real):
    path = OUT / "freespec_chains" / model / f"realization-{real}" / "chain.zarr"
    return az.extract(az.from_zarr(path), var_names="log10_rho", filter_vars="like").to_numpy().T


def load_flow_samples(model, real, n=4000):
    flow = load_flow(OUT / "flows" / "posterior" / model / f"realization-{real}" / "flow.pkl")
    return np.asarray(flow_sample(flow, n, rng_seed=real))


def make_correlations(model):
    real = pick_reps(model)[1][0]
    chain, flow = load_chain(model, real), load_flow_samples(model, real, n=40000)
    Cc, Cf = spearmanr(chain).correlation, spearmanr(flow).correlation
    diff = Cc - Cf
    iu = np.triu_indices(len(Cc), 1)

    vmax_cf = max(np.abs(Cc[iu]).max(), np.abs(Cf[iu]).max())
    vmax_d = np.abs(diff[iu]).max()
    panels = [("free-spec.", Cc, vmax_cf, "RdBu_r"),
              ("flow", Cf, vmax_cf, "RdBu_r"),
              ("diff.", diff, vmax_d, "PRGn")]

    fig = plt.figure(figsize=figsize(aspect=0.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[0.09, 1, 1, 1, 0.09])
    cax_s = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, j]) for j in (1, 2, 3)]
    cax_d = fig.add_subplot(gs[0, 4])

    def masked(M):
        Mm = M.copy()
        np.fill_diagonal(Mm, np.nan)
        return Mm

    ims = []
    for ax, (title, M, vm, cm) in zip(axes, panels):
        ims.append(ax.imshow(masked(M), vmin=-vm, vmax=vm, cmap=cm, origin="lower"))
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    cs = fig.colorbar(ims[0], cax=cax_s)
    cs.set_label(r"Spearman $\rho$")
    cs.ax.tick_params(labelsize=10)
    cs.ax.yaxis.set_ticks_position("left")
    cs.ax.yaxis.set_label_position("left")
    cd = fig.colorbar(ims[2], cax=cax_d)
    cd.set_label(r"$\Delta\rho$")
    cd.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    return fig


fig5c_pl = make_correlations("powerlaw")
save(fig5c_pl, "flow_correlations_powerlaw")
fig5c_pl

# %%
fig5c_g = make_correlations("gaussian")
save(fig5c_g, "flow_correlations_gaussian")
fig5c_g

# %% [markdown]
# ## Evidence validation on a Gaussian target
#
# The flow-as-likelihood machinery on a known analytic target (10-D standard
# Gaussian in a uniform box). Nested sampling with the **analytic** likelihood
# and with a **flow** trained on samples of the target both recover the analytic
# $\log Z$ (dashed) — the flow likelihood introduces no evidence bias.

# %%
g = np.load(OUT / "gaussian_validation" / "evidences.npz")
logZ_true = float(g["logZ_analytic"])

fig7, ax = plt.subplots(figsize=figsize(1.0))
labels = ["analytic\nintegral", "NS\n(analytic)", "NS\n(flow)"]
means = [logZ_true, float(g["logZ_ns_analytic_mean"]), float(g["logZ_ns_flow_mean"])]
errs = [0.0, float(g["logZ_ns_analytic_std"]), float(g["logZ_ns_flow_std"])]
x = np.arange(3)
ax.axhline(logZ_true, color="C3", ls="--", lw=0.8, alpha=0.8)
ax.errorbar(x, means, yerr=errs, fmt="o", color="k", capsize=3,
            label=r"$\log Z$: mean $\pm$ std. from NS.")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel(r"$\log Z$")
ax.legend(loc="best")
save(fig7, "gaussian_evidence_validation")
fig7

# %% [markdown]
# ## Classifier two-sample test (C2ST) on the flow
#
# A bounded, self-interpreting fidelity metric for the flow. A classifier is
# trained to tell two sample sets apart and scored by cross-validated accuracy:
# **0.5 = indistinguishable**, **1.0 = trivially distinguishable**.
#
# Four comparisons, all against the free-spectrum MCMC chain, form a ladder:
# **free-spectrum** (two disjoint halves of the chain — the empirical null);
# **flow** (coppuccino flow vs. chain — the validation result); **shuffled**
# (chain vs. its per-bin-permuted self: marginals exact, cross-bin correlations
# destroyed); **random** (chain vs. uniform noise over the matched support). Each
# half-violin is the distribution over realizations (powerlaw left, gaussian
# right). Data computed by `notebooks/fig_c2st.py`.

# %%
CASES = ["free-spectrum", "flow", "shuffled", "random"]
VW = 0.40
SIDE = {"powerlaw": -1, "gaussian": +1}

d = np.load(DIAG / "c2st_flow_freespec.npz", allow_pickle=True)
c2st_records = list(zip(d["model"], d["real"].astype(int),
                        d["case"].astype(int), d["acc"].astype(float)))

fig8, ax = plt.subplots(figsize=figsize(1.0))
ax.axhline(0.5, ls="--", color="0.5", lw=0.9, zorder=1)

rng_jit = np.random.default_rng(0)
for ci in range(len(CASES)):
    for model in MODELS:
        v = np.array([r[3] for r in c2st_records if r[2] == ci and r[0] == model])
        if len(v) < 2:
            continue
        sign, color = SIDE[model], COLORS[model]
        kde = gaussian_kde(v)
        yg = np.linspace(v.min(), v.max(), 128)
        norm = kde(yg).max()
        width = kde(yg) / norm * VW
        ax.fill_betweenx(yg, ci, ci + sign * width, color=color, alpha=0.22, lw=0, zorder=2)
        ax.plot(ci + sign * width, yg, color=color, lw=0.7, alpha=0.85, zorder=2)
        jit = rng_jit.uniform(0, 1, len(v)) * (kde(v) / norm * VW) * sign
        ax.scatter(ci + jit, v, s=5, color=color, alpha=0.65, edgecolor="none",
                   zorder=4, label=model if ci == 0 else None)
        med = float(np.median(v))
        print(CASES[ci], model, f"median {med:.4f}")
        med_w = float(kde([med])[0] / norm * VW)
        ax.plot([ci, ci + sign * med_w], [med, med], color="0.15", lw=1.1, zorder=5)

ax.set_xticks(range(len(CASES)))
ax.set_xticklabels(CASES, rotation=20, fontsize=10)
ax.set_ylabel("C2ST accuracy")
ax.set_ylim(0.45, 1.02)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
          fontsize=10, handletextpad=0.3, columnspacing=1.2)
save(fig8, "c2st_flow_freespec")
fig8

# %% [markdown]
# ## Upper-tail profile above the trusted region
#
# One signal-dominated bin of one realization, profiled above its trusted-region
# edge $\rho_i^{\mathrm{hi}}$. Freezing the nuisance parameters at their posterior
# medians (what production uses) against the exact IRN-marginalized profile from
# NUTS + Fisher's identity, with the asymptotic slope $2N_{\mathrm{psr}}\ln 10$
# for reference. Data written by `scripts/make_upper_tail_profile.py`
# (`slope` and `run`; ~20 min of NUTS per bin).

# %%
UPPER_TAIL = dict(model="powerlaw", real=34, bin=1)
_tag = f"{UPPER_TAIL['model']}_r{UPPER_TAIL['real']}"

z = np.load(DIAG / f"upper_frozen_vs_fisher_{_tag}.npz")
k = int(np.where(z["bins"] == UPPER_TAIL["bin"])[0][0])
offs, frozen, fisher, ferr = z["offs"], z["frozen"][k], z["fisher"][k], z["ferr"][k]
s_asym = float(z["s_asym"])

zs = np.load(DIAG / f"upper_frozen_slope_{_tag}.npz")
d_off, d_slope, d_bins = zs["offs"], zs["slopes"], zs["signal_bins"]

fig8u, (ax_a, ax_b) = plt.subplots(2, 1, figsize=figsize(1.0, aspect=1.05))

ax_a.plot(offs, -s_asym * offs, ls=":", color="0.5", lw=1.0, label="asymptotic slope")
ax_a.plot(offs, frozen, "o-", color=COLORS["powerlaw"], lw=1.0, label="frozen (used)")
ax_a.errorbar(offs, fisher, yerr=ferr, fmt="s--", color=COLORS["gaussian"], lw=1.0,
              ecolor="0.5", elinewidth=0.7, capsize=2, label="exact marginal")
ax_a.set_xlim(0, offs.max())
ax_a.set_ylim(1.05 * min(frozen.min(), fisher.min()), 0.02 * abs(frozen.min()))
ax_a.set_ylabel(r"$\Delta \log \mathcal{L}$ (nats)")
ax_a.legend(loc="lower left", framealpha=0.9)

for j, b in enumerate(d_bins):
    ax_b.plot(d_off, -d_slope[b] / s_asym, color="0.65", lw=0.7,
              label="signal bins" if j == 0 else None)
ax_b.plot(d_off, -d_slope[UPPER_TAIL["bin"]] / s_asym, color=COLORS["powerlaw"], lw=1.2,
          label="bin in (a)")
ax_b.axhline(1.0, ls=":", color="0.3", lw=1.0, label="asymptote")
ax_b.axvline(offs.max(), ls="--", color="0.5", lw=0.9,
             label=f"grid cutoff ({offs.max():g} dex)")
ax_b.set_xlim(0, d_off.max())
ax_b.set_ylim(0, 1.15)
ax_b.set_xlabel(r"$\Delta \log_{10} \rho_i$ above $\rho_i^{\mathrm{hi}}$ (dex)")
ax_b.set_ylabel(r"slope $/\, 2 N_{\mathrm{psr}}$")
ax_b.legend(loc="lower right", framealpha=0.9)

ax_a.text(0.975, 0.93, "(a)", transform=ax_a.transAxes, ha="right", va="top")
ax_b.text(0.025, 0.05, "(b)", transform=ax_b.transAxes, ha="left", va="bottom")
fig8u.tight_layout(pad=0.3)
save(fig8u, "upper_tail_profile")
