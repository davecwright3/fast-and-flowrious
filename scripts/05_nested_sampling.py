#!/usr/bin/env python
"""Stage 05: nested sampling with the flow likelihood."""

from typing import Literal

import cyclopts

from coppuccino import load_flow
from flows_model_comparison import config, models, nested_sampling, tail_metadata
from flows_model_comparison.psd_models import PSD_FUNCTIONS

app = cyclopts.App()


@app.command
def run_nested_sampling(
    sim_model: Literal["powerlaw", "gaussian"],
    realization: int,
    test_model: Literal["powerlaw", "gaussian"],
) -> None:
    """Nested sampling for ``test_model`` on one realization.

    The likelihood is the flow plus the stage-04 tails; the prior is the ANALYSIS_BOUNDS box.
    """
    flow_dir = (config.OUTPUT_DIR / "flows" / "posterior" / sim_model
                / f"realization-{realization}")
    posterior_flow = load_flow(flow_dir / "flow.pkl")

    sim_dir = config.OUTPUT_DIR / "simulations" / sim_model / f"realization-{realization}"
    psrs = models.load_simulated_pulsars(sim_dir)
    f, df = nested_sampling.get_frequencies(psrs)

    # Tails belong to the flow, so they are keyed by sim model, not test model.
    meta = tail_metadata.load_fisher_tail_metadata(sim_model, realization)
    if meta["b_lo"].shape[0] != config.N_FREQS:
        raise ValueError(
            f"fisher_tail_meta bins {meta['b_lo'].shape[0]} != N_FREQS {config.N_FREQS}; "
            "rebuild stage 04 (fisher tails)."
        )
    likelihood_fn = tail_metadata.make_flow_likelihood_with_metadata(
        posterior_flow, PSD_FUNCTIONS[test_model], f, df, meta,
    )

    prior_bounds = {
        key.removeprefix("gw_"): val
        for key, val in config.ANALYSIS_BOUNDS[test_model].items()
    }

    print(f"Running NS for sim={sim_model}, real={realization}, test={test_model}")
    rng_seed = realization * 10 + list(config.MODELS).index(test_model)
    results = nested_sampling.run_nested_sampling(
        likelihood_fn, prior_bounds, rng_seed=rng_seed,
    )

    out_path = (config.OUTPUT_DIR / "nested_sampling" / sim_model / f"realization-{realization}"
                / f"{test_model}_evidence.npz")
    nested_sampling.save_evidence(results, out_path)


if __name__ == "__main__":
    app()
