#!/usr/bin/env python
"""Data for the upper-tail profile figure.

Shows that freezing the noise at its posterior median gives the right upper tail, since the
per-pulsar red noise can't absorb extra common power.

  slope  frozen profile out to ``max_dex`` and its slope vs. the 2 N_psr ln(10) asymptote
  run    frozen vs. exact (NUTS + Fisher) profile for the chosen bins, ~20 min per bin

The paper uses powerlaw realization 34, bin 1:

    pixi run -e gpu python scripts/make_upper_tail_profile.py slope --sim-model powerlaw --realization 34
    pixi run -e gpu python scripts/make_upper_tail_profile.py run --sim-model powerlaw --realization 34 --bins 1

Run one JAX process at a time.
"""
from typing import Literal

import cyclopts
import numpy as np

from flows_model_comparison import config, fisher_tails

app = cyclopts.App()

CACHE_DIR = config.OUTPUT_DIR / "diagnostics"


@app.command
def slope(sim_model: Literal["powerlaw", "gaussian"], realization: int,
          max_dex: float = 6.0, n: int = 121) -> None:
    """Frozen upper profile and its slope out to ``max_dex``."""
    ctx = fisher_tails.build_context(sim_model, realization)
    s_asym = float(2 * ctx["n_psr"] * np.log(10.0))
    offs = np.linspace(0.0, max_dex, n)
    up = fisher_tails._frozen_upper(ctx, offs)
    frozen_floor = fisher_tails._frozen_floor(ctx)
    signal_bins = np.where(np.abs(frozen_floor) > fisher_tails.SIGNAL_FLOOR_NATS)[0]

    print(f"{sim_model} r{realization}: asymptote {-s_asym:.1f} nats/dex; "
          f"signal bins {list(map(int, signal_bins))}")
    slopes = np.gradient(up, offs, axis=1)
    for j in map(int, signal_bins):
        frac = -slopes[j] / s_asym
        marks = [np.argmax(frac >= t) if (frac >= t).any() else -1 for t in (0.5, 0.9, 0.99)]
        where = [f"{offs[m]:.2f}" if m >= 0 else ">grid" for m in marks]
        print(f"  bin {j}: reaches 50/90/99% of asymptote at {where[0]}/{where[1]}/{where[2]} dex; "
              f"final {frac[-1]:.3f} at {offs[-1]:.1f} dex")

    path = CACHE_DIR / f"upper_frozen_slope_{sim_model}_r{realization}.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, offs=offs, frozen=up, slopes=slopes, s_asym=s_asym,
             signal_bins=signal_bins, n_psr=ctx["n_psr"])
    print(f"saved -> {path}")


@app.command
def run(sim_model: Literal["powerlaw", "gaussian"], realization: int, max_bins: int = 3,
        seed: int = 0, bins: list[int] | None = None) -> None:
    """Frozen vs. exact upper profile. Defaults to the ``max_bins`` signal bins with the deepest floors."""
    ctx = fisher_tails.build_context(sim_model, realization)
    offs = fisher_tails.UPPER_OFFSETS
    s_asym = float(2 * ctx["n_psr"] * np.log(10.0))

    frozen_all = fisher_tails._frozen_upper(ctx, offs)
    frozen_floor = fisher_tails._frozen_floor(ctx)
    signal_bins = np.where(np.abs(frozen_floor) > fisher_tails.SIGNAL_FLOOR_NATS)[0]
    signal_bins = signal_bins[np.argsort(frozen_floor[signal_bins])][:max_bins]
    if bins is not None:
        signal_bins = np.asarray(bins, dtype=int)
    b_hi = np.asarray(ctx["b_hi"])
    print(f"{sim_model} r{realization}: bins {list(map(int, signal_bins))} "
          f"(floors {np.round(frozen_floor[signal_bins], 1)}); s_asym={s_asym:.0f}")

    results = []
    for j in map(int, signal_bins):
        g, gerr, _slope, _mcse, ess = fisher_tails._fisher_curve(
            ctx, j, float(b_hi[j]), offs, sign=+1, seed0=seed + 1000 * j + 500)
        frozen = frozen_all[j]
        results.append(dict(bin=j, frozen=frozen, fisher=g, ferr=gerr))
        near = np.abs((frozen - g)[offs <= 0.2])
        print(f"bin {j}: max |frozen - fisher| within 0.2 dex of edge {near.max():.2f} nats "
              f"(min ess {ess.min():.0f})")

    _save_npz(sim_model, realization, offs, results, s_asym)


def _save_npz(sim_model, realization, offs, results, s_asym):
    """Save the profiles, merging bin by bin with any existing file."""
    path = CACHE_DIR / f"upper_frozen_vs_fisher_{sim_model}_r{realization}.npz"
    keep = {}
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["offs"], offs):
            keep = {int(b): (z["frozen"][k], z["fisher"][k], z["ferr"][k])
                    for k, b in enumerate(z["bins"])}
    for r in results:
        keep[int(r["bin"])] = (r["frozen"], r["fisher"], r["ferr"])
    bins = sorted(keep)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, bins=np.array(bins), offs=offs, s_asym=s_asym,
             frozen=np.array([keep[b][0] for b in bins]),
             fisher=np.array([keep[b][1] for b in bins]),
             ferr=np.array([keep[b][2] for b in bins]))
    print(f"saved profiles for bins {bins} -> {path}")


if __name__ == "__main__":
    app()
