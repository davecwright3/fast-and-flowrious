"""Stage 05: nested sampling with the flow likelihood."""

from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from anesthetic import NestedSamples
from blackjax.ns.utils import finalise
from discovery import signals
from tqdm.auto import tqdm

from . import config


def get_frequencies(psrs, n_freqs: int | None = None):
    """Common-signal frequencies and bin widths."""
    if n_freqs is None:
        n_freqs = config.N_FREQS

    Tspan = signals.getspan(psrs)
    f = jnp.arange(1, n_freqs + 1, dtype=jnp.float64) / Tspan
    df = jnp.ones_like(f) / Tspan
    return f, df


def psd_to_log10_rho(psd_fn, f, df, params: dict) -> jnp.ndarray:
    """Convert discovery PSD output (rho^2, df already included) to log10_rho."""
    rho2 = psd_fn(f, df, **params)
    return 0.5 * jnp.log10(rho2)


def make_prior_sampler(prior_bounds: dict):
    """Uniform prior sampler and log density for blackjax nested sampling."""
    def sample_prior(key):
        keys = jax.random.split(key, len(prior_bounds))
        params = {}
        for i, (name, bounds) in enumerate(prior_bounds.items()):
            params[name] = jax.random.uniform(
                keys[i], minval=bounds[0], maxval=bounds[1]
            )
        return params

    def log_prior_fn(params):
        log_p = 0.0
        for name, bounds in prior_bounds.items():
            val = params[name]
            in_bounds = (val >= bounds[0]) & (val <= bounds[1])
            log_p = jnp.where(in_bounds, log_p, -jnp.inf)
        return log_p

    return sample_prior, log_prior_fn


def run_nested_sampling(
    likelihood_fn,
    prior_bounds: dict,
    ns_config: dict | None = None,
    rng_seed: int = 42,
    labels: dict | None = None,
) -> dict:
    """Run blackjax nested sampling.

    Returns a dict with 'samples', 'logZ_mean', 'logZ_std', and 'n_iterations'.
    """
    if ns_config is None:
        ns_config = config.NS_CONFIG

    rng_key = jax.random.key(rng_seed)
    sample_prior, log_prior_fn = make_prior_sampler(prior_bounds)

    n_live = ns_config["n_live"]
    num_delete = int(n_live * ns_config["num_delete_frac"])

    algo = blackjax.nss(
        logprior_fn=log_prior_fn,
        loglikelihood_fn=likelihood_fn,
        num_delete=num_delete,
        num_inner_steps=ns_config["num_inner_steps"],
    )

    keys = jax.random.split(rng_key, n_live + 1)
    rng_key = keys[0]
    initial_particles = jax.vmap(sample_prior)(keys[1:])

    state = algo.init(initial_particles)
    step = jax.jit(algo.step)

    dead_points = []
    termination_diff = ns_config["termination_logZ_diff"]

    pbar = tqdm(desc="Nested Sampling")
    while not (state.integrator.logZ_live - state.integrator.logZ < termination_diff):
        rng_key, subkey = jax.random.split(rng_key)
        state, dead = step(subkey, state)
        dead_points.append(dead)
        acc = float(dead.update_info.is_accepted.mean())
        pbar.update(num_delete)
        pbar.set_postfix(
            logZ=f"{state.integrator.logZ:.1f}",
            diff=f"{(state.integrator.logZ_live - state.integrator.logZ):.2E}",
            acc=f"{acc:.3f}",
        )
    pbar.close()

    final_state = finalise(state, dead_points)

    swl = final_state.particles
    samples = NestedSamples(
        swl.position,
        logL=swl.loglikelihood,
        logL_birth=swl.loglikelihood_birth,
        labels=labels,
    )

    # 100 draws over the prior-volume uncertainty.
    logZ_samples = np.asarray(samples.logZ(100))
    logZ_mean = float(logZ_samples.mean())
    logZ_std = float(logZ_samples.std())

    print(f"Nested sampling complete: logZ = {logZ_mean:.2f} +/- {logZ_std:.2f}")

    return {
        "samples": samples,
        "logZ_mean": logZ_mean,
        "logZ_std": logZ_std,
        "n_iterations": len(dead_points),
    }


def save_evidence(results: dict, output_path: Path) -> None:
    """Save the NS summary (.npz) and samples (.csv)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = results["samples"]
    csv_path = output_path.with_suffix(".csv")
    samples.to_csv(csv_path)

    np.savez(
        output_path,
        logZ_mean=results["logZ_mean"],
        logZ_std=results["logZ_std"],
        n_iterations=results["n_iterations"],
    )

    print(f"Saved evidence summary to {output_path}")
    print(f"Saved NS samples to {csv_path}")
