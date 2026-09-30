#!/usr/bin/env python
"""Stage 04: build the tail metadata that stage 05 splices onto the flow.

Only uses the free-spectrum chain and the likelihood, not the injection, so it works on real data.
Takes minutes to an hour per realization. Run one JAX process at a time.

    pixi run -e gpu python scripts/04_build_fisher_tails.py run --sim-model powerlaw --realization 38
"""
from typing import Literal

import cyclopts
import numpy as np

from flows_model_comparison import fisher_tails, tail_metadata

app = cyclopts.App()


@app.command
def run(sim_model: Literal["powerlaw", "gaussian"], realization: int, seed: int = 0) -> None:
    """Build and save the tail metadata for one realization."""
    meta = fisher_tails.build_fisher_tail_metadata(sim_model, realization, seed=seed)
    signal = np.where(np.abs(meta["dfloor"]) > 1e-3)[0]
    print(f"\nfisher_tail_meta for {sim_model} r{realization}:")
    print(f"  signal bins (nonzero floor): {list(signal)}")
    print(f"  dfloor (nats): {np.array2string(meta['dfloor'], precision=2, max_line_width=200)}")
    print(f"  s_asym = {meta['s_asym']:.1f} nats/dex")
    print(f"  saved -> {tail_metadata.fisher_tail_meta_path(sim_model, realization)}")


if __name__ == "__main__":
    app()
