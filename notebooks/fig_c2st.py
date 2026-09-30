"""Classifier two-sample test (C2ST) of the flow against the free-spectrum chain.

Writes outputs/diagnostics/c2st_flow_freespec.npz for paper_figures.py. An accuracy of 0.5 means
indistinguishable, 1.0 means trivially separable. Each case is compared against the chain:

  free-spectrum  the other half of the chain (null)
  flow           flow samples
  shuffled       chain with each bin permuted separately (no correlations)
  random         uniform noise over the same support

    pixi run -e gpu python notebooks/fig_c2st.py [--out PATH]
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from coppuccino import load_flow, sample as flow_sample
from flows_model_comparison import config

OUT = config.OUTPUT_DIR
MODELS = ["powerlaw", "gaussian"]
N_SUB = 1000
FOLDS = 4


def load_chain(model, real):
    path = OUT / "freespec_chains" / model / f"realization-{real}" / "chain.zarr"
    return az.extract(
        az.from_zarr(path), var_names="log10_rho", filter_vars="like"
    ).to_numpy().T


def load_flow_samples(model, real, n):
    flow = load_flow(OUT / "flows" / "posterior" / model / f"realization-{real}" / "flow.pkl")
    return np.asarray(flow_sample(flow, n, rng_seed=real))


def bin_shuffle(x, rng):
    """Permute each column separately to remove cross-bin correlations."""
    xs = x.copy()
    for j in range(x.shape[1]):
        xs[:, j] = xs[rng.permutation(len(xs)), j]
    return xs


def c2st(xa, xb, seed):
    """Cross-validated classifier accuracy (0.5 = indistinguishable)."""
    rng = np.random.default_rng(seed)
    xa = xa[rng.choice(len(xa), min(N_SUB, len(xa)), replace=False)]
    xb = xb[rng.choice(len(xb), min(N_SUB, len(xb)), replace=False)]
    X = np.vstack([xa, xb])
    y = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean())


def realizations(model):
    base = OUT / "freespec_chains" / model
    return sorted(
        int(p.name.split("-")[1]) for p in base.glob("realization-*")
        if (p / "chain.zarr").exists()
        and (OUT / "flows" / "posterior" / model / p.name / "flow.pkl").exists()
    )


def main(out: Path) -> None:
    records = []
    for model in MODELS:
        for real in realizations(model):
            try:
                chain = load_chain(model, real)
                n = chain.shape[0] // 2
                rng = np.random.default_rng(real)
                half_a, half_b = chain[:n], chain[n:2 * n]
                flow = load_flow_samples(model, real, N_SUB)
                shuffled = bin_shuffle(chain, rng)
                lo, hi = chain.min(0), chain.max(0)
                rand = rng.uniform(lo, hi, size=(len(chain), chain.shape[1]))

                # Order sets the case index: free-spectrum, flow, shuffled, random.
                accs = [
                    c2st(half_a, half_b, real),
                    c2st(chain, flow, real),
                    c2st(chain, shuffled, real),
                    c2st(chain, rand, real),
                ]
                for ci, a in enumerate(accs):
                    records.append((model, real, ci, a))
            except Exception as e:  # noqa: BLE001
                print(f"skip {model} r{real}: {e}", flush=True)
        print(f"{model}: done", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        model=np.array([r[0] for r in records]),
        real=np.array([r[1] for r in records]),
        case=np.array([r[2] for r in records]),
        acc=np.array([r[3] for r in records]),
    )
    print(f"saved {len(records)} C2ST values -> {out}")


if __name__ == "__main__":
    import cyclopts
    app = cyclopts.App()

    @app.default
    def run(out: Path = OUT / "diagnostics" / "c2st_flow_freespec.npz") -> None:
        """Compute the C2ST accuracies and save them to ``out``."""
        main(out)

    app()
