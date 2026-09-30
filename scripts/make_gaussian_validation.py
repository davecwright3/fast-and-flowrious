#!/usr/bin/env python
"""Evidence check on a D-dim standard Gaussian in a uniform box [-L, L]^D.

Compares the analytic log Z to nested sampling with the true likelihood and with a flow trained
on samples. They should agree. Writes outputs/gaussian_validation/evidences.npz for
paper_figures.py.

    pixi run -e gpu python scripts/make_gaussian_validation.py
    pixi run -e gpu python scripts/make_gaussian_validation.py --dim 5 --n-live 500
"""
import cyclopts
import jax.numpy as jnp
import numpy as np
from scipy import stats

from coppuccino import normalizing_flows_fit, normalizing_flows_fit_auto
from flows_model_comparison import config, nested_sampling

app = cyclopts.App()

OUT_PATH = config.OUTPUT_DIR / "gaussian_validation" / "evidences.npz"


def analytic_logZ(dim: int, half_width: float) -> float:
    """Analytic log Z: per-dim Gaussian mass in [-L, L] over the box volume."""
    mass = (2 * stats.norm.cdf(half_width) - 1) ** dim
    return float(np.log(mass) - dim * np.log(2 * half_width))


def gaussian_loglike(dim: int):
    """log N(x; 0, I) for the NS param dict."""
    log_norm = -0.5 * dim * jnp.log(2 * jnp.pi)

    def loglike(params: dict) -> float:
        x = jnp.stack([params[f"x{i}"] for i in range(dim)])
        return log_norm - 0.5 * jnp.dot(x, x)

    return loglike


def flow_loglike(flow, dim: int):
    """Flow log density for the NS param dict."""

    def loglike(params: dict) -> float:
        x = jnp.stack([params[f"x{i}"] for i in range(dim)]).reshape(1, -1)
        return flow.log_prob(x)[0]

    return loglike


@app.default
def main(
    dim: int = 10,
    half_width: float = 5.0,
    n_train: int = 20000,
    n_live: int = 1000,
    rng_seed: int = 42,
):
    """Run the comparison and save evidences.npz."""
    rng = np.random.default_rng(rng_seed)

    logZ_true = analytic_logZ(dim, half_width)
    print(f"analytic logZ = {logZ_true:.4f}")

    prior_bounds = {f"x{i}": (-half_width, half_width) for i in range(dim)}
    ns_config = {**config.NS_CONFIG, "n_live": n_live}

    print("[1/2] NS with analytic likelihood")
    res_analytic = nested_sampling.run_nested_sampling(
        gaussian_loglike(dim), prior_bounds, ns_config=ns_config, rng_seed=rng_seed,
    )

    print(f"[2/2] training flow on {n_train} samples, then NS with flow likelihood")
    train_samples = rng.standard_normal((n_train, dim))
    flow = normalizing_flows_fit_auto(
        train_samples,
        rng_seed=rng_seed,
        prior_bounds=np.tile([-half_width, half_width], (dim, 1)),
    )
    res_flow = nested_sampling.run_nested_sampling(
        flow_loglike(flow, dim), prior_bounds, ns_config=ns_config, rng_seed=rng_seed + 1,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        dim=dim,
        half_width=half_width,
        n_train=n_train,
        n_live=n_live,
        logZ_analytic=logZ_true,
        logZ_ns_analytic_mean=res_analytic["logZ_mean"],
        logZ_ns_analytic_std=res_analytic["logZ_std"],
        logZ_ns_flow_mean=res_flow["logZ_mean"],
        logZ_ns_flow_std=res_flow["logZ_std"],
    )
    print(
        f"\nlogZ analytic   = {logZ_true:.4f}\n"
        f"logZ NS (true)  = {res_analytic['logZ_mean']:.4f} +/- {res_analytic['logZ_std']:.4f}"
        f"  [d = {res_analytic['logZ_mean'] - logZ_true:+.4f}]\n"
        f"logZ NS (flow)  = {res_flow['logZ_mean']:.4f} +/- {res_flow['logZ_std']:.4f}"
        f"  [d = {res_flow['logZ_mean'] - logZ_true:+.4f}]\n"
        f"wrote {OUT_PATH}"
    )


if __name__ == "__main__":
    app()
