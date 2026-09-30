"""Diagnostic plots for trained flows."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coppuccino import sample as flow_sample


def plot_marginals(training_samples: np.ndarray, flow, output_path: Path,
                   n_flow_samples: int | None = None, rng_seed: int = 0,
                   title: str | None = None) -> np.ndarray:
    """Overlay training and flow marginals. Returns the flow samples."""
    n_train, n_dims = training_samples.shape
    if n_flow_samples is None:
        n_flow_samples = n_train

    flow_samples = np.asarray(flow_sample(flow, n_flow_samples, rng_seed=rng_seed))

    ncols = min(4, n_dims)
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                             squeeze=False)

    for d in range(n_dims):
        ax = axes[d // ncols, d % ncols]
        lo = min(training_samples[:, d].min(), flow_samples[:, d].min())
        hi = max(training_samples[:, d].max(), flow_samples[:, d].max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(training_samples[:, d], bins=bins, density=True, histtype="step",
                color="black", lw=1.5, label="freespec MCMC")
        ax.hist(flow_samples[:, d], bins=bins, density=True, histtype="step",
                color="C0", lw=1.5, label="flow")
        ax.set_xlabel(f"log10_rho[{d}]")
        ax.set_yticks([])
    for d in range(n_dims, nrows * ncols):
        axes[d // ncols, d % ncols].axis("off")
    axes[0, 0].legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return flow_samples


def plot_corner(training_samples: np.ndarray, flow_samples: np.ndarray,
                output_path: Path, title: str | None = None,
                max_dims: int | None = None) -> None:
    """Corner plot of MCMC vs. flow samples for up to ``max_dims`` bins."""
    import corner

    n_dims = training_samples.shape[1]
    if max_dims is not None and n_dims > max_dims:
        n_dims = max_dims
    train = training_samples[:, :n_dims]
    flow = flow_samples[:, :n_dims]

    labels = [f"rho[{d}]" for d in range(n_dims)]
    ranges = [
        (min(train[:, d].min(), flow[:, d].min()),
         max(train[:, d].max(), flow[:, d].max()))
        for d in range(n_dims)
    ]

    corner_kwargs = dict(
        labels=labels, range=ranges, bins=40, plot_datapoints=False,
        plot_density=True, fill_contours=False, levels=(0.68, 0.95),
        smooth=1.0,
    )
    fig = corner.corner(train, color="black", **corner_kwargs)
    corner.corner(flow, color="C0", fig=fig, **corner_kwargs)

    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color="black", label="freespec MCMC"),
        mlines.Line2D([], [], color="C0", label="flow"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=12)
    if title:
        fig.suptitle(title, y=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qq(training_samples: np.ndarray, flow_samples: np.ndarray,
            output_path: Path, n_quantiles: int = 200,
            title: str | None = None) -> None:
    """Per-dimension Q-Q plots comparing flow samples to training samples."""
    n_dims = training_samples.shape[1]
    qs = np.linspace(0.001, 0.999, n_quantiles)

    ncols = min(4, n_dims)
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows),
                             squeeze=False)

    for d in range(n_dims):
        ax = axes[d // ncols, d % ncols]
        q_train = np.quantile(training_samples[:, d], qs)
        q_flow = np.quantile(flow_samples[:, d], qs)
        ax.plot(q_train, q_flow, lw=1.2, color="C0")
        lo = min(q_train.min(), q_flow.min())
        hi = max(q_train.max(), q_flow.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"MCMC q (dim {d})")
        ax.set_ylabel("flow q")
    for d in range(n_dims, nrows * ncols):
        axes[d // ncols, d % ncols].axis("off")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
