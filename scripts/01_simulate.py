#!/usr/bin/env python
"""Stage 01: simulate PTA data."""

from typing import Literal

import cyclopts

from flows_model_comparison import config, simulation

app = cyclopts.App()


@app.command
def simulate(
    model: Literal["powerlaw", "gaussian"],
    realization: int,
) -> None:
    """Simulate one realization. The realization index is the seed."""
    psrs = simulation.load_pulsars(n_pulsars=config.N_PULSARS)
    params = simulation.load_max_like()

    print(f"Simulating {model} realization {realization}...")
    sim_data = simulation.simulate_pta(psrs, params, model, rng_seed=realization)

    out_path = config.OUTPUT_DIR / "simulations" / model / f"realization-{realization}"
    simulation.save_simulation(sim_data, out_path, psrs)

    print("Done!")


if __name__ == "__main__":
    app()
