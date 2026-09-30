"""Likelihoods and priors for every stage.

Everything builds its model here so the flow and full-likelihood analyses use the same
likelihood, albeit with different PSDs as needed. Priors are exact uniform boxes (not discovery's
makemodel_transformed) because the harmonic mean assumes an exactly uniform prior.
"""

import itertools
import re
from pathlib import Path

import discovery as ds
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from discovery.prior import getprior_uniform

from . import config
from .psd_models import PSD_FUNCTIONS

PLATES = {"pulsar": ["red_noise_log10_A", "red_noise_gamma"]}


def load_simulated_pulsars(sim_dir: Path) -> list:
    """Load the simulated pulsars of one realization directory."""
    feather_files = sorted(Path(sim_dir).glob("*.feather"))
    feather_files = [f for f in feather_files if not f.name.startswith("injection")]
    psrs = [ds.Pulsar.read_feather(f) for f in feather_files]
    print(f"Loaded {len(psrs)} simulated pulsars from {sim_dir}")
    return psrs


def physical_name(crn_name: str) -> str:
    """Strip the ``crn_`` prefix from a common parameter name."""
    return crn_name.removeprefix("crn_")


def build_curn_model(psrs: list, common_psd_fn, n_freqs: int, components=None):
    """CURN likelihood with per-pulsar red noise. Returns ``(model, crn_params)``."""
    Tspan = ds.getspan(psrs)
    psl_models = [
        ds.PulsarLikelihood(
            [
                psr.residuals,
                ds.makenoise_measurement(psr, psr.noisedict),
                ds.makegp_ecorr(psr, psr.noisedict),
                ds.makegp_timing(psr, svd=True),
            ]
        )
        for psr in psrs
    ]
    common_psd, crn_params = ds.make_combined_crn(n_freqs, ds.powerlaw, common_psd_fn)
    commongp = ds.makecommongp_fourier(
        psrs,
        common_psd,
        config.N_FREQS_IRN if components is None else components,
        T=Tspan,
        common=crn_params,
        name="red_noise",
    )
    return ds.ArrayLikelihood(psl_models, commongp=commongp), crn_params


def build_freespec_model(psrs: list, n_freqs: int | None = None):
    """Free-spectrum CURN model."""
    n_freqs = config.N_FREQS if n_freqs is None else n_freqs
    model, _ = build_curn_model(
        psrs,
        ds.freespectrum,
        n_freqs,
        # Red noise gets N_FREQS_IRN modes; the free spectrum only the first n_freqs.
        components={"irn": config.N_FREQS_IRN, "log10_rho": n_freqs},
    )
    return model


def build_parametric_model(psrs: list, test_model: str, n_freqs: int | None = None):
    """CURN model for ``test_model``. Returns ``(model, crn_params)``.

    Params shared with the red noise get a ``crn_`` prefix; see :func:`physical_name`.
    """
    n_freqs = config.N_FREQS if n_freqs is None else n_freqs
    return build_curn_model(psrs, PSD_FUNCTIONS[test_model], n_freqs)


def priordict_for(test_model: str | None = None, crn_params: tuple = ()) -> dict:
    """Prior overrides on top of discovery's priordict_standard.

    Keys have to match the standard keys exactly to replace them; a new regex key never wins.
    """
    rho_bounds = [config.FREESPEC_LOG10_RHO_LOWER, config.FREESPEC_LOG10_RHO_UPPER]
    pd = {
        # New key for make_combined_crn's bare log10_rho(N); the rest replace standard entries.
        r"log10_rho\(([0-9]*)\)": rho_bounds,
        r"(.*_)?red_noise_log10_rho\(([0-9]*)\)": rho_bounds,
        r"(.*_)?red_noise_crn_log10_rho\(([0-9]*)\)": rho_bounds,
        "crn_log10_rho": rho_bounds,
        "gw_(.*_)?log10_rho": rho_bounds,
    }
    if test_model is not None:
        bounds = {
            k.removeprefix("gw_"): list(v)
            for k, v in config.ANALYSIS_BOUNDS[test_model].items()
        }
        for c in crn_params:
            # Matches the standard "crn_log10_A.*"-style keys so these replace them.
            key = f"{c}.*" if c.startswith("crn_") else c
            pd[key] = bounds[physical_name(c)]
    return pd


_SHAPE_RE = re.compile(r"\(([\d,\s]+)\)$")


def shape_of(name: str) -> tuple:
    """Vector shape encoded in a parenthesized name suffix (``()`` if scalar)."""
    m = _SHAPE_RE.search(name)
    return tuple(int(d) for d in m.group(1).split(",")) if m else ()


def group_params(params, plates: dict) -> list:
    """Group ``params`` into sample sites.

    Returns ``(site_name, members, dim)`` tuples. Params matching a substring in ``plates`` are
    stacked into one vector site; the rest are scalar sites.
    """
    groups, used = [], set()
    for dim, keys in plates.items():
        for key in keys:
            members = [p for p in params if key in p and p not in used]
            if members:
                used.update(members)
                groups.append((key, members, dim))
    groups += [(p, [p], None) for p in params if p not in used]
    return groups


def make_numpyro_model(logl, priordict: dict):
    """numpyro model with uniform priors. Returns ``(numpyro_model, groups)``."""
    groups = group_params(list(logl.params), PLATES)

    def site_bounds(members, i):
        vshape = shape_of(members[0])
        arr = np.array(
            [np.broadcast_to(getprior_uniform(m, priordict)[i], vshape) for m in members],
            dtype=float,
        )
        return jnp.asarray(arr if len(members) > 1 else arr[0])

    sites = [(name, members, site_bounds(members, 0), site_bounds(members, 1))
             for name, members, _ in groups]

    def numpyro_model():
        params = {}
        for name, members, lo, hi in sites:
            x = numpyro.sample(name, dist.Uniform(lo, hi).to_event(lo.ndim))
            if len(members) > 1:
                params.update({m: x[i] for i, m in enumerate(members)})
            else:
                params[members[0]] = x
        numpyro.factor("logl", logl(params))

    def to_df(samples):
        cols = {}
        for name, members, _ in groups:
            arr = np.asarray(samples[name])
            for i, m in enumerate(members):
                v = arr[:, i] if len(members) > 1 else arr
                v = v.reshape(len(v), -1)
                shape = shape_of(m)
                if not shape:
                    cols[m] = v[:, 0]
                else:
                    for j, idx in enumerate(itertools.product(*map(range, shape))):
                        cols[f"{m}[{','.join(map(str, idx))}]"] = v[:, j]
        return pd.DataFrame(cols)

    numpyro_model.to_df = to_df
    return numpyro_model, groups


def uniform_bounds(params, priordict: dict):
    """Uniform bounds ``{name: (lo, hi)}`` for every parameter."""
    return {p: tuple(getprior_uniform(p, priordict)) for p in params}
