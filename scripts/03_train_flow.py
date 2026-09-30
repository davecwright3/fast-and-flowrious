#!/usr/bin/env python
"""Stage 03: train a flow on the free-spectrum chain."""

from typing import Literal

import cyclopts

from flows_model_comparison import config, flow_training, freespec

app = cyclopts.App()


@app.command
def train_posterior_flow(
    sim_model: Literal["powerlaw", "gaussian"],
    realization: int,
) -> None:
    """Train a flow on one realization's free-spectrum samples."""
    samples = freespec.load_log10_rho_samples(sim_model, realization)

    out_path = (config.OUTPUT_DIR / "flows" / "posterior" / sim_model
                / f"realization-{realization}" / "flow.pkl")
    flow_training.train_posterior_flow(samples, out_path)


if __name__ == "__main__":
    app()
