"""Splice the stage-04 tails onto the flow likelihood.

The flow is only good inside its training support. Outside it we use the stored tail profiles:

    log L(theta) = log q(clip(rho, b_lo, b_hi)) + sum_i T_i(rho_i)

The lower tail flattens to a floor; the upper tail goes to a slope of 2 N_psr ln(10) nats/dex.

Kept separate from fisher_tails because that module turns on float64 at import.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import config
from .nested_sampling import psd_to_log10_rho


def fisher_tail_meta_path(sim_model: str, realization: int,
                          output_dir: Path | None = None) -> Path:
    output_dir = config.OUTPUT_DIR if output_dir is None else Path(output_dir)
    return (output_dir / "flows" / "posterior" / sim_model /
            f"realization-{realization}" / "fisher_tail_meta.npz")


def load_fisher_tail_metadata(sim_model: str, realization: int,
                              output_dir: Path | None = None) -> dict:
    """Load the metadata written by ``fisher_tails``."""
    d = np.load(fisher_tail_meta_path(sim_model, realization, output_dir))
    return {k: d[k] for k in d.files}


def _tail_correction(log10_rho, meta):
    """Tail correction sum_i T_i(rho_i), zero inside the support.

    Interpolates the stored profiles. Past the last point the lower tail stops at ``dfloor`` and the
    upper tail continues with slope ``s_asym``.
    """
    b_lo = jnp.asarray(meta["b_lo"]); b_hi = jnp.asarray(meta["b_hi"])
    dfloor = jnp.asarray(meta["dfloor"])
    lo_d = jnp.asarray(meta["lo_d"]); lo_T = jnp.asarray(meta["lo_T"])
    up_d = jnp.asarray(meta["up_d"]); up_T = jnp.asarray(meta["up_T"])
    s_asym = float(meta["s_asym"])

    # dex below the lower edge
    x_lo = b_lo - log10_rho
    lo_interp = jax.vmap(lambda xi, Ti: jnp.interp(xi, lo_d, Ti))(x_lo, lo_T)
    lo_slope = (lo_T[:, -1] - lo_T[:, -2]) / (lo_d[-1] - lo_d[-2])
    lo_far = jnp.maximum(lo_T[:, -1] + lo_slope * (x_lo - lo_d[-1]), dfloor)
    t_low = jnp.where(x_lo > lo_d[-1], lo_far, jnp.where(x_lo > 0.0, lo_interp, 0.0))

    # dex above the upper edge
    x = log10_rho - b_hi
    up_interp = jax.vmap(lambda xi, Ti: jnp.interp(xi, up_d, Ti))(x, up_T)
    up_far = up_T[:, -1] - s_asym * (x - up_d[-1])
    t_up = jnp.where(x > up_d[-1], up_far, jnp.where(x > 0.0, up_interp, 0.0))
    return jnp.sum(t_low + t_up)


def make_flow_likelihood_with_metadata(posterior_flow, psd_fn, f, df, meta):
    """Flow likelihood with the stored tails spliced in."""
    b_lo = jnp.asarray(meta["b_lo"]); b_hi = jnp.asarray(meta["b_hi"])

    def likelihood_fn(params: dict) -> float:
        log10_rho = psd_to_log10_rho(psd_fn, f, df, params)
        clipped = jnp.clip(log10_rho, b_lo, b_hi)
        return posterior_flow.log_prob(clipped.reshape(1, -1))[0] + _tail_correction(log10_rho, meta)

    return likelihood_fn
