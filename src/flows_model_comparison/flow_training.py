"""Flow training."""

from pathlib import Path

import numpy as np

from coppuccino import load_flow, normalizing_flows_fit_auto, save_flow

from . import config


def train_posterior_flow(samples: np.ndarray, output_path: Path,
                         flow_config: dict | None = None):
    """Train a flow on (n_samples, n_dims) samples and save it. Defaults to config.FLOW_CONFIG."""
    if flow_config is None:
        flow_config = config.FLOW_CONFIG

    print(f"Training flow on {samples.shape[0]} samples of dimension {samples.shape[1]}")

    fit_kwargs = {
        k: v for k, v in flow_config.items()
        if k in {"max_epochs", "patience", "prior_bounds", "tail_model"}
    }
    flow = normalizing_flows_fit_auto(
        samples, **fit_kwargs, copula_space="gaussian", n_workers=1, verbose=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_flow(flow, output_path)
    print(f"Saved trained flow to {output_path}")
    return flow


__all__ = ["train_posterior_flow", "load_flow"]
