"""Stage 02: free-spectrum MCMC and its chain loader."""

from pathlib import Path

import arviz as az
import jax
import numpy as np
import numpyro

from . import config, models


def load_log10_rho_samples(sim_model: str, realization: int,
                           output_dir: Path | None = None) -> np.ndarray:
    """Free-spectrum log10_rho samples, shape (n_samples, n_freqs).

    These are what the flows are trained on.
    """
    output_dir = config.OUTPUT_DIR if output_dir is None else Path(output_dir)
    chain_path = (
        output_dir / "freespec_chains" / sim_model / f"realization-{realization}" / "chain.zarr"
    )
    return (
        az.extract(az.from_zarr(chain_path), var_names="log10_rho", filter_vars="like")
        .to_numpy()
        .T
    )


def run_freespec_mcmc(model, rng_seed: int = 42) -> az.InferenceData:
    """Run NUTS on the free-spectrum model. Returns arviz InferenceData."""
    logl = model.logL
    npmodel, _ = models.make_numpyro_model(logl, models.priordict_for())

    print(f"Running NUTS with {config.MCMC_CONFIG['num_samples']} samples...")
    sampler = numpyro.infer.MCMC(
        numpyro.infer.NUTS(npmodel),
        **config.MCMC_CONFIG,
        progress_bar=True,
    )
    sampler.run(jax.random.key(rng_seed))
    df = npmodel.to_df(sampler.get_samples())

    params = list(logl.params)
    rho_name = next(p for p in params if "log10_rho" in p)
    n_freqs = int(rho_name.split("(")[1].split(")")[0])
    A_names = [p for p in params if "red_noise_log10_A" in p]
    g_names = [p for p in params if "red_noise_gamma" in p]
    pulsar_names = [p.split("_")[0] for p in A_names]

    posterior = {
        "red_noise_log10_A": df[A_names].to_numpy()[None, ...],
        "red_noise_gamma": df[g_names].to_numpy()[None, ...],
        rho_name: df[[f"{rho_name}[{i}]" for i in range(n_freqs)]].to_numpy()[None, ...],
    }

    return az.from_dict(
        {"posterior": posterior},
        coords={
            "pulsar": pulsar_names,
            "freq": list(range(1, n_freqs + 1)),
        },
        dims={
            "red_noise_log10_A": ["pulsar"],
            "red_noise_gamma": ["pulsar"],
            rho_name: ["freq"],
        },
    )
