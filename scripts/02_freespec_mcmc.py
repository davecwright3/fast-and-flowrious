#!/usr/bin/env python
"""Stage 02: free-spectrum MCMC."""

from typing import Literal

import cyclopts

from flows_model_comparison import config, freespec, models

app = cyclopts.App()


@app.command
def freespec_mcmc(
    sim_model: Literal["powerlaw", "gaussian"],
    realization: int,
) -> None:
    """Run free-spectrum MCMC on one realization."""
    sim_dir = (
        config.OUTPUT_DIR / "simulations" / sim_model / f"realization-{realization}"
    )
    psrs = models.load_simulated_pulsars(sim_dir)

    print(f"Building free-spectrum model for {sim_model} realization {realization} "
          f"with n_freqs={config.N_FREQS}...")
    model = models.build_freespec_model(psrs)

    inf_data = freespec.run_freespec_mcmc(model, rng_seed=realization)

    out_path = config.OUTPUT_DIR / "freespec_chains" / sim_model / f"realization-{realization}"
    out_path.mkdir(parents=True, exist_ok=True)
    inf_data.to_zarr(out_path / "chain.zarr")

    print(f"Saved chain to {out_path / 'chain.zarr'}")


if __name__ == "__main__":
    app()
