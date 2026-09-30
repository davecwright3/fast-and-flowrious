#!/usr/bin/env python
"""Stage 07: full-likelihood NUTS and Gaussian-container harmonic evidence.

``sample`` runs NUTS for one test model; ``evidence`` gets the Bayes factor from both chains.

    pixi run -e gpu python scripts/07_full_likelihood.py sample --sim-model powerlaw --realization 3 --test-model powerlaw
    pixi run -e gpu python scripts/07_full_likelihood.py sample --sim-model powerlaw --realization 3 --test-model gaussian
    pixi run -e gpu python scripts/07_full_likelihood.py evidence --sim-model powerlaw --realization 3
"""
from typing import Literal

import cyclopts

from flows_model_comparison import full_likelihood

app = cyclopts.App()


@app.command
def sample(
    sim_model: Literal["powerlaw", "gaussian"],
    realization: int,
    test_model: Literal["powerlaw", "gaussian"],
    seed: int | None = None,
) -> None:
    """Run NUTS on the full likelihood and save the chain. Seed defaults to the realization."""
    full_likelihood.sample(sim_model, realization, test_model, seed=seed)


@app.command
def evidence(
    sim_model: Literal["powerlaw", "gaussian"],
    realization: int,
) -> None:
    """Harmonic evidence for both test models and their Bayes factor."""
    full_likelihood.evidence(sim_model, realization)


if __name__ == "__main__":
    app()
