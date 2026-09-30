"""Simulate PTA data.

The amplitude is rescaled until the optimal-statistic SNR lands in [MIN_SNR, MAX_SNR].
"""

import json
from pathlib import Path

import discovery as ds
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from . import config
from .psd_models import PSD_FUNCTIONS


def load_max_like(filename: Path | None = None) -> dict:
    """Per-pulsar max-likelihood noise parameters."""
    if filename is None:
        filename = config.MAX_LIKE_FILE
    with filename.open("r") as f:
        return json.load(f)


def load_pulsars(
    data_dir: Path | None = None,
    n_pulsars: int | None = None,
) -> list:
    """Load the NANOGrav 15yr pulsars."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    feather_files = sorted(data_dir.glob("[JB]*.feather"))
    if n_pulsars is not None:
        feather_files = feather_files[:n_pulsars]
    psrs = [ds.Pulsar.read_feather(f) for f in feather_files]
    print(f"Loaded {len(psrs)} pulsars from {data_dir}")
    return psrs


def sample_parameters_from_prior(
    model_name: str,
    rng_seed: int,
    exclude_amplitude: bool = False,
) -> dict:
    """Draw PSD parameters from the injection prior, optionally skipping the amplitude."""
    if model_name not in config.MODELS:
        raise ValueError(f"Unknown model: {model_name}. Use one of: {config.MODELS}")

    rng = np.random.default_rng(rng_seed)
    prior_bounds = config.INJECTION_BOUNDS[model_name]

    params = {}
    for param_name, (lower, upper) in prior_bounds.items():
        if exclude_amplitude and param_name == "gw_log10_A":
            continue
        params[param_name] = rng.uniform(lower, upper)
    return params


def build_likelihood_for_sampling(
    psrs: list,
    model_name: str,
    n_freqs: int | None = None,
) -> ds.GlobalLikelihood:
    """Model used to draw simulated residuals, with an HD-correlated GWB."""
    if n_freqs is None:
        n_freqs = config.N_FREQS
    if model_name not in config.MODELS:
        raise ValueError(f"Unknown model: {model_name}. Use one of: {config.MODELS}")

    Tspan = ds.getspan(psrs)
    gwb_gp = ds.makeglobalgp_fourier(
        psrs,
        PSD_FUNCTIONS[model_name],
        ds.hd_orf,
        n_freqs,
        T=Tspan,
        name="gw",
    )

    psl_models = [
        ds.PulsarLikelihood(
            [
                psr.residuals,
                ds.makenoise_measurement(psr, psr.noisedict),
                ds.makegp_ecorr(psr, psr.noisedict),
                # Near-frozen timing model; sampling is unstable otherwise.
                ds.makegp_timing(psr, svd=True, variance=1e-12),
                ds.makegp_fourier(
                    psr,
                    ds.powerlaw,
                    T=Tspan,
                    components=config.N_FREQS_IRN,
                    name="red_noise",
                ),
            ]
        )
        for psr in psrs
    ]
    return ds.GlobalLikelihood(psl_models, globalgp=gwb_gp)


def build_os_likelihood(
    psrs: list,
    model_name: str,
    n_freqs: int | None = None,
) -> ds.GlobalLikelihood:
    """Model passed to ``discovery.OS`` to measure the SNR."""
    if n_freqs is None:
        n_freqs = config.N_FREQS
    if model_name not in config.MODELS:
        raise ValueError(f"Unknown model: {model_name}. Use one of: {config.MODELS}")

    # OS wants the GW as a per-pulsar common GP, not the HD global GP.
    common = {
        "powerlaw": ["gw_log10_A", "gw_gamma"],
        "gaussian": ["gw_log10_A", "gw_log10_f_peak", "gw_sigma"],
    }[model_name]

    Tspan = ds.getspan(psrs)
    psl_models = [
        ds.PulsarLikelihood(
            [
                psr.residuals,
                ds.makenoise_measurement(psr, psr.noisedict),
                ds.makegp_ecorr(psr, psr.noisedict, variable=True),
                ds.makegp_timing(psr, svd=True, variable=True),
                ds.makegp_fourier(
                    psr,
                    ds.powerlaw,
                    T=Tspan,
                    components=config.N_FREQS_IRN,
                    name="red_noise",
                ),
                ds.makegp_fourier(
                    psr,
                    PSD_FUNCTIONS[model_name],
                    n_freqs,
                    T=Tspan,
                    common=common,
                    name="gw",
                ),
            ]
        )
        for psr in psrs
    ]
    return ds.GlobalLikelihood(psl_models)


def simulate_pta(
    psrs: list,
    psr_params: dict,
    model_name: str,
    rng_seed: int,
    min_snr: float | None = None,
    max_snr: float | None = None,
    ref_log10_A: float = -14.5,
) -> dict:
    """Simulate one data set at a target SNR.

    Simulates at a reference amplitude, measures the OS SNR, then rescales the amplitude (SNR is
    linear in A) with the same noise until the SNR is in [min_snr, max_snr].
    """
    if model_name not in config.MODELS:
        raise ValueError(f"Unknown model: {model_name}. Use one of: {config.MODELS}")
    if min_snr is None:
        min_snr = config.MIN_SNR
    if max_snr is None:
        max_snr = config.MAX_SNR

    print("building os likelihood")
    model = build_os_likelihood(psrs, model_name)
    model_hd = build_likelihood_for_sampling(psrs, model_name)
    print("DONE building os likelihood")

    psr_names = [psr.name for psr in psrs]
    # Keep noise params for our pulsars plus any non-pulsar params.
    filtered_psr_params = {
        k: v
        for k, v in psr_params.items()
        if k.split("_")[0] in psr_names
        or not any(k.startswith(f"{name}_") for name in psr_names)
    }

    prior_bounds = config.INJECTION_BOUNDS[model_name]
    a_min, a_max = prior_bounds["gw_log10_A"]

    attempt_rng = np.random.default_rng(rng_seed)
    max_inner_iterations = 5
    attempt = 0
    while True:
        attempt += 1
        spec_seed = int(attempt_rng.integers(0, 2**31 - 1))
        noise_seed = int(attempt_rng.integers(0, 2**31 - 1))
        target_snr = float(attempt_rng.uniform(min_snr, max_snr))

        spectral_params = sample_parameters_from_prior(
            model_name,
            spec_seed,
            exclude_amplitude=True,
        )
        params = filtered_psr_params.copy()
        params.update(spectral_params)

        params["gw_log10_A"] = ref_log10_A
        key = jax.random.key(noise_seed)
        key, subkey = jax.random.split(key)
        _, ref_residuals = model_hd.sample(subkey, params)
        model.residuals = ref_residuals
        snr_ref = ds.OS(model).os(params)["snr"]

        if snr_ref <= 0:
            print(
                f"  attempt {attempt}: ref SNR non-positive ({snr_ref:.3f}); "
                f"redrawing spectral params and noise.",
            )
            continue

        # SNR is linear in A.
        log10_A_target = np.clip(
            ref_log10_A + np.log10(target_snr / snr_ref), a_min, a_max,
        )
        params["gw_log10_A"] = float(log10_A_target)

        achieved_snr = None
        bad_inner = False
        for iteration in range(max_inner_iterations):
            # Same noise seed every iteration, so only the amplitude changes.
            key = jax.random.key(noise_seed)
            key, subkey = jax.random.split(key)
            _, residuals = model_hd.sample(subkey, params)
            model.residuals = residuals
            achieved_snr = ds.OS(model).os(params)["snr"]
            print(f"  attempt {attempt} iter {iteration}: achieved_snr={achieved_snr:.2f}")

            if achieved_snr <= 0:
                bad_inner = True
                break

            snr_error = abs(achieved_snr - target_snr)
            if snr_error < 0.5 or snr_error / target_snr < 0.05:
                break

            correction = np.log10(target_snr / achieved_snr)
            params["gw_log10_A"] = float(
                np.clip(params["gw_log10_A"] + correction, a_min, a_max),
            )

        if bad_inner:
            print(
                f"  attempt {attempt}: inner SNR went non-positive; "
                f"redrawing spectral params and noise.",
            )
            continue

        if not (min_snr <= achieved_snr <= max_snr):
            print(
                f"  attempt {attempt}: achieved SNR {achieved_snr:.2f} outside "
                f"[{min_snr}, {max_snr}]; redrawing.",
            )
            continue

        break

    print(
        f"  Accepted on attempt {attempt}: target_snr={target_snr:.2f}, "
        f"achieved_snr={achieved_snr:.2f}, gw_log10_A={params['gw_log10_A']:.3f}, "
        f"gw_gamma={params.get('gw_gamma', 'N/A')}",
    )

    return {
        "residuals": residuals,
        "params": params,
        "snr": achieved_snr,
        "target_snr": target_snr,
        "metric": "SNR",
    }


def save_simulation(sim_data: dict, output_dir: Path, psrs: list) -> None:
    """Save residuals, injection parameters, and SNR."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if "params" in sim_data:
        np.savez(output_dir / "params.npz", **sim_data["params"])

    if "snr" in sim_data:
        metric_name = sim_data.get("metric", "SNR")
        np.savez(output_dir / "snr.npz", snr=sim_data["snr"], metric=metric_name)
        print(f"  Injection {metric_name}: {sim_data['snr']:.2f}")

    for psr, residuals in zip(psrs, sim_data["residuals"], strict=False):
        psr.residuals = np.array(residuals)
        psr.save_feather(output_dir / f"{psr.name}.feather")

    print(f"Saved simulation to {output_dir}")
