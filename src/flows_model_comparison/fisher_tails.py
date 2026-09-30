"""Stage 04: tail metadata for the flow likelihood.

The flow knows nothing outside its training support, so we profile the full likelihood along each
bin past the support edges and store that.

Upper tail: the frozen-noise profile. It's already accurate since the red noise can't absorb
extra common power.

Lower tail: for signal bins the frozen profile is far too deep, so we marginalize the red noise.
NUTS samples everything else with the bin clamped, Fisher's identity gives d log L / d rho as a
posterior mean, and we integrate the slopes. Noise bins get a flat floor.
"""
import os

# Fisher-slope gradients need float64; must be set before JAX initializes.
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist

from . import config, freespec, models, tail_metadata

numpyro.enable_x64()

# Dex past the trusted edge. Upper runs to 2 dex: at 1 dex the slope is only ~85% of s_asym.
UPPER_OFFSETS = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 1.5, 2.0])
LOWER_OFFSETS = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
SIGNAL_FLOOR_NATS = 2.0
NUTS_WARMUP = 200
NUTS_DRAWS = 400
NUTS_TREE_DEPTH = 7


def build_context(sim_model: str, realization: int, n_freqs: int | None = None) -> dict:
    """Free-spectrum likelihood plus parameter names, bounds, and chain medians."""
    if n_freqs is None:
        n_freqs = config.N_FREQS
    sim_dir = config.OUTPUT_DIR / "simulations" / sim_model / f"realization-{realization}"
    psrs = models.load_simulated_pulsars(sim_dir)
    n_psr = len(psrs)
    model = models.build_freespec_model(psrs, n_freqs=n_freqs)
    logl = model.logL

    params = list(logl.params)
    rho_name = next(p for p in params if "log10_rho" in p)
    nf = int(rho_name.split("(")[1].split(")")[0])
    if nf != n_freqs:
        raise ValueError(f"free-spec model has {nf} bins != n_freqs {n_freqs}")
    A_names = [p for p in params if "red_noise_log10_A" in p]
    g_names = [p for p in params if "red_noise_gamma" in p]

    bounds = models.uniform_bounds(params, models.priordict_for())

    chain = freespec.load_log10_rho_samples(sim_model, realization)
    med = np.median(chain, axis=0)
    # Trusted support is the 2-98% range of the chain.
    b_lo = np.quantile(chain, 0.02, axis=0)
    b_hi = np.quantile(chain, 0.98, axis=0)
    irn = _chain_irn_median(sim_model, realization)
    base = {}
    for p in A_names + g_names:
        psr = p.rsplit("_red_noise_", 1)[0]
        base[p] = (irn["red_noise_log10_A"][psr] if p.endswith("log10_A")
                   else irn["red_noise_gamma"][psr])

    logl_j = jax.jit(lambda rho: logl({**base, rho_name: rho}))

    return dict(logl=logl, logl_j=logl_j, rho_name=rho_name,
                A_names=A_names, g_names=g_names, bounds=bounds,
                nf=nf, n_psr=n_psr, med=med, b_lo=b_lo, b_hi=b_hi)


def _chain_irn_median(sim_model: str, realization: int) -> dict:
    """Per-pulsar red noise medians from the free-spectrum chain."""
    import arviz as az
    idata = az.from_zarr(
        config.OUTPUT_DIR / "freespec_chains" / sim_model / f"realization-{realization}" / "chain.zarr"
    )
    out = {}
    for var in ("red_noise_log10_A", "red_noise_gamma"):
        x = az.extract(idata, var_names=var)
        med = np.median(x.to_numpy(), axis=-1)
        out[var] = {str(p): float(m) for p, m in zip(x.coords["pulsar"].values, med)}
    return out


def _make_clamped(ctx: dict, j: int):
    """NUTS model and gradient function with bin ``j`` clamped."""
    logl, rho_name = ctx["logl"], ctx["rho_name"]
    A_names, g_names, bounds = ctx["A_names"], ctx["g_names"], ctx["bounds"]
    nf = ctx["nf"]

    A_lo = jnp.array([bounds[n][0] for n in A_names])
    A_hi = jnp.array([bounds[n][1] for n in A_names])
    g_lo = jnp.array([bounds[n][0] for n in g_names])
    g_hi = jnp.array([bounds[n][1] for n in g_names])
    r_lo = jnp.full(nf - 1, bounds[rho_name][0])
    r_hi = jnp.full(nf - 1, bounds[rho_name][1])

    def _pack(A, G, rho_free, v):
        pd_ = {n: A[k] for k, n in enumerate(A_names)}
        pd_.update({n: G[k] for k, n in enumerate(g_names)})
        pd_[rho_name] = jnp.concatenate([rho_free[:j], jnp.atleast_1d(v), rho_free[j:]])
        return pd_

    def nuts_model(v):
        A = numpyro.sample("red_noise_log10_A", dist.Uniform(A_lo, A_hi).to_event(1))
        G = numpyro.sample("red_noise_gamma", dist.Uniform(g_lo, g_hi).to_event(1))
        rho_free = numpyro.sample("log10_rho_free", dist.Uniform(r_lo, r_hi).to_event(1))
        numpyro.factor("logl", logl(_pack(A, G, rho_free, v)))

    def _grad_one(s, v):
        return jax.grad(
            lambda vv: logl(_pack(s["red_noise_log10_A"], s["red_noise_gamma"],
                                  s["log10_rho_free"], vv))
        )(v)

    grad_series_fn = jax.jit(lambda samples, v: jax.lax.map(lambda s: _grad_one(s, v), samples))
    return nuts_model, grad_series_fn


def fisher_slope(ctx: dict, j: int, v: float, seed: int, clamped=None,
                 num_warmup: int = NUTS_WARMUP, num_samples: int = NUTS_DRAWS,
                 max_tree_depth: int = NUTS_TREE_DEPTH) -> dict:
    """NUTS with bin ``j`` clamped at ``v``. Returns the slope E[dlogL/dv] and its MCSE."""
    import arviz as az
    if clamped is None:
        clamped = _make_clamped(ctx, j)
    nuts_model, grad_series_fn = clamped
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(nuts_model, max_tree_depth=max_tree_depth),
        num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False,
    )
    v = jnp.float64(v)
    mcmc.run(jax.random.key(seed), v)
    grads = np.asarray(grad_series_fn(mcmc.get_samples(), v))
    ess = max(float(az.ess(az.convert_to_dataset(grads[None, :])).x.values), 1.0)
    return dict(v=float(v), slope=float(np.mean(grads)),
                mcse=float(np.std(grads, ddof=1) / np.sqrt(ess)), ess=ess)


def _path_sample(v, slope, mcse):
    """Cumulative trapezoid of the slopes, with standard errors."""
    v, slope, mcse = np.asarray(v), np.asarray(slope), np.asarray(mcse)
    n = len(v)
    g = np.zeros(n)
    coeff = np.zeros((n, n))
    for k in range(1, n):
        dv = v[k] - v[k - 1]
        g[k] = g[k - 1] + 0.5 * (slope[k - 1] + slope[k]) * dv
        # coeff[k] holds g[k]'s weight on each slope, for propagating the MCSEs.
        coeff[k] = coeff[k - 1]
        coeff[k, k - 1] += 0.5 * dv
        coeff[k, k] += 0.5 * dv
    return g, np.sqrt(coeff ** 2 @ mcse ** 2)


def _fisher_curve(ctx: dict, j: int, edge: float, offsets: np.ndarray, sign: int,
                  seed0: int, **nuts_kw):
    """Exact profile at ``edge + sign*offsets``, zero at the edge.

    Returns (g, gerr, slope, mcse, ess).
    """
    clamped = _make_clamped(ctx, j)
    vs = edge + sign * offsets
    rows = [fisher_slope(ctx, j, float(v), seed=seed0 + k, clamped=clamped, **nuts_kw)
            for k, v in enumerate(vs)]
    slope = np.array([r["slope"] for r in rows])
    mcse = np.array([r["mcse"] for r in rows])
    g, gerr = _path_sample(vs, slope, mcse)
    return g, gerr, slope, mcse, np.array([r["ess"] for r in rows])


def _frozen_upper(ctx: dict, offsets: np.ndarray) -> np.ndarray:
    """Frozen-noise upper profile per bin, zero at the upper edge."""
    nf, logl_j, b_hi = ctx["nf"], ctx["logl_j"], ctx["b_hi"]
    med = jnp.asarray(ctx["med"])
    up_T = np.zeros((nf, len(offsets)))
    for i in range(nf):
        base = float(logl_j(med.at[i].set(float(b_hi[i]))))
        up_T[i] = [float(logl_j(med.at[i].set(float(b_hi[i] + d)))) - base for d in offsets]
    return up_T


def _frozen_floor(ctx: dict) -> np.ndarray:
    """Frozen-noise floor depth per bin (nats, <= 0), used to pick out signal bins."""
    nf, logl_j, b_lo = ctx["nf"], ctx["logl_j"], ctx["b_lo"]
    med = jnp.asarray(ctx["med"])
    # 8 dex below the median is well into the floor plateau.
    return np.array([float(logl_j(med.at[i].set(float(ctx["med"][i] - 8.0))))
                     - float(logl_j(med.at[i].set(float(b_lo[i])))) for i in range(nf)])


def build_fisher_tail_metadata(sim_model: str, realization: int, n_freqs: int | None = None,
                               output_dir: Path | None = None, save: bool = True,
                               seed: int = 0, verbose: bool = True) -> dict:
    """Build the tail metadata for one realization.

    Frozen upper profile for every bin; NUTS + Fisher lower floor for bins with
    |frozen floor| > SIGNAL_FLOOR_NATS. Optionally saves to ``fisher_tail_meta.npz``.
    """
    ctx = build_context(sim_model, realization, n_freqs)
    nf, n_psr = ctx["nf"], ctx["n_psr"]
    b_lo, b_hi = np.asarray(ctx["b_lo"]), np.asarray(ctx["b_hi"])

    up_T = _frozen_upper(ctx, UPPER_OFFSETS)
    frozen_floor = _frozen_floor(ctx)
    signal_bins = np.where(np.abs(frozen_floor) > SIGNAL_FLOOR_NATS)[0]
    if verbose:
        print(f"[fisher_tails] {sim_model} r{realization}: nf={nf}, signal bins={list(signal_bins)} "
              f"(frozen floor {np.array2string(frozen_floor, precision=1, max_line_width=200)})")

    lo_T = np.zeros((nf, len(LOWER_OFFSETS)))
    dfloor = np.zeros(nf)
    for i in signal_bins:
        g, gerr, slope, mcse, ess = _fisher_curve(ctx, int(i), float(b_lo[i]), LOWER_OFFSETS,
                                                  sign=-1, seed0=seed + 1000 * int(i) + 1)
        lo_T[i] = g
        dfloor[i] = g[-1]
        if verbose:
            print(f"  bin {i} LOWER exact floor = {g[-1]:.2f} +/- {gerr[-1]:.2f} nats "
                  f"(min ess {ess.min():.0f})")

    meta = dict(b_lo=b_lo, b_hi=b_hi, dfloor=dfloor,
                lo_d=LOWER_OFFSETS.copy(), lo_T=lo_T,
                up_d=UPPER_OFFSETS.copy(), up_T=up_T,
                # -0.5 log|C| over 2 N_psr modes, each scaling as 10^(2 rho).
                s_asym=float(2 * n_psr * np.log(10.0)))
    if save:
        path = tail_metadata.fisher_tail_meta_path(sim_model, realization, output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **meta)
        if verbose:
            print(f"  saved -> {path}")
    return meta
