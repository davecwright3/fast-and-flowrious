"""Stage 06: learned harmonic mean evidence on the NS samples, as a check on NS."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import anesthetic
import harmonic as hm
import jax
import numpy as np

from . import config


PARAM_ORDER = {
    "powerlaw": ["gamma", "log10_A"],
    "gaussian": ["log10_A", "log10_f_peak", "sigma"],
}


def log_prior_const(model: str) -> float:
    """Log density of the uniform analysis prior."""
    bounds = config.ANALYSIS_BOUNDS[model]
    return -float(np.sum([np.log(b - a) for (a, b) in bounds.values()]))


def compute_harmonic_evidence(
    ns_csv: Path,
    test_model: Literal["powerlaw", "gaussian"],
    n_chains: int = 4,
    training_proportion: float = 0.5,
    n_scaled_layers: int = 4,
    n_unscaled_layers: int = 8,
    temperature: float = 0.8,
    epochs: int = 1000,
    verbose: bool = True,
    shuffle_seed: int = 0,
    fit_key: int = 0,
) -> dict:
    """Learned harmonic mean evidence from a stage-05 ``<test_model>_evidence.csv``."""
    samples = anesthetic.read_chains(ns_csv)
    columns = PARAM_ORDER[test_model]

    posterior_points = samples.posterior_points()
    post = posterior_points[columns].to_numpy()
    log_post = (
        posterior_points[["logL"]].to_numpy().squeeze()
        + log_prior_const(test_model)
    )

    # NS samples are sorted by logL; shuffle so the chain blocks are exchangeable.
    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(post.shape[0])
    post = post[perm]
    log_post = log_post[perm]

    chains = hm.Chains(post.shape[-1])
    chains.add_chain(post, log_post)
    chains.split_into_blocks(nblocks=n_chains)

    # split_data splits whole chains; the variance estimator needs >=2 inference chains.
    chains_train, chains_infer = hm.utils.split_data(
        chains, training_proportion=training_proportion
    )

    # Smoother fallbacks, used only if the harmonic sum diverges (overfit heavy tail).
    attempts = [
        (epochs, temperature, fit_key),
        (400, min(temperature, 0.6), fit_key),
        (250, 0.5, fit_key),
        (200, 0.4, fit_key),
        (200, 0.4, fit_key + 1),
        (120, 0.3, fit_key + 1),
    ]
    for rung, (attempt_epochs, attempt_temperature, attempt_key) in enumerate(attempts):
        model = hm.model.RealNVPModel(
            post.shape[-1],
            n_scaled_layers=n_scaled_layers,
            n_unscaled_layers=n_unscaled_layers,
            standardize=True,
            temperature=attempt_temperature,
        )
        model.fit(
            chains_train.samples,
            epochs=attempt_epochs,
            verbose=verbose,
            key=jax.random.PRNGKey(attempt_key),
        )
        ev = hm.Evidence(chains_infer.nchains, model)
        ev.add_chains(chains_infer)
        if np.isfinite(ev.ln_evidence_inv):
            break
    else:
        warnings.warn(f"harmonic evidence is non-finite after all {len(attempts)} fits: {ns_csv}")

    ln_inv = float(ev.ln_evidence_inv)
    err_neg, err_pos = ev.compute_ln_inv_evidence_errors()

    # Delta method; stays finite where compute_ln_inv_evidence_errors returns -inf.
    sigma_logZ = float(np.exp(0.5 * float(ev.ln_evidence_inv_var) - ln_inv))

    return {
        "ln_inv_evidence": ln_inv,
        "ln_inv_evidence_err_neg": float(err_neg),
        "ln_inv_evidence_err_pos": float(err_pos),
        "log_evidence": -ln_inv,
        "sigma_logZ": sigma_logZ,
        "n_posterior_samples": int(post.shape[0]),
        "fit_rung": rung,
    }


def save_harmonic_evidence(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **result)


def load_all_harmonic_evidences(evidence_dir: Path):
    """Collect the harmonic evidences into a DataFrame."""
    import pandas as pd

    rows = []
    for sim_model in config.MODELS:
        for realization in range(config.N_REALIZATIONS):
            for test_model in config.MODELS:
                path = (
                    evidence_dir
                    / sim_model
                    / f"realization-{realization}"
                    / f"{test_model}_harmonic.npz"
                )
                if not path.exists():
                    continue
                data = np.load(path)
                rows.append(
                    {
                        "sim_model": sim_model,
                        "realization": realization,
                        "test_model": test_model,
                        "logZ_mean": float(data["log_evidence"]),
                        "logZ_std": float(data["sigma_logZ"]),
                        # Sample count, stored under the NS column name so the tables line up.
                        "n_iterations": int(data["n_posterior_samples"]),
                    }
                )
    return pd.DataFrame(rows)
