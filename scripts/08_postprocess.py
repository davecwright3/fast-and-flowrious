#!/usr/bin/env python
"""Stage 08: flow diagnostics and Bayes factor tables.

  validate-flows  diagnostic plots for every trained flow
  bayes-factors   collect all evidences into the results/ CSVs
  all             both
"""
from pathlib import Path
from typing import Literal

import cyclopts
import numpy as np
import pandas as pd

from coppuccino import load_flow
from flows_model_comparison import (
    analysis,
    config,
    diagnostics,
    freespec,
    full_likelihood,
    harmonic_evidence,
)

app = cyclopts.App()


@app.command
def validate_flow(
    sim_model: Literal["powerlaw", "gaussian"],
    realization: int,
    output_dir: Path | str | None = None,
) -> None:
    """Marginal, Q-Q, and corner plots for one flow."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR / "diagnostics" / "flows"
    output_dir = Path(output_dir)

    samples = freespec.load_log10_rho_samples(sim_model, realization)
    flow = load_flow(
        config.OUTPUT_DIR / "flows" / "posterior" / sim_model
        / f"realization-{realization}" / "flow.pkl"
    )

    out_base = output_dir / sim_model / f"realization-{realization}"
    title = f"{sim_model} realization {realization}"

    flow_samples = diagnostics.plot_marginals(
        samples, flow, out_base / "marginals.png", title=title,
    )
    diagnostics.plot_qq(samples, flow_samples, out_base / "qq.png", title=title)
    diagnostics.plot_corner(samples, flow_samples, out_base / "corner.png", title=title)

    print(f"Wrote diagnostics to {out_base}/")


@app.command
def validate_flows() -> None:
    """Run validate_flow on every trained flow."""
    for sm in config.MODELS:
        for r in range(config.N_REALIZATIONS):
            flow_path = (
                config.OUTPUT_DIR / "flows" / "posterior" / sm
                / f"realization-{r}" / "flow.pkl"
            )
            if not flow_path.exists():
                continue
            validate_flow(sm, r)


def _aggregate_fulllik(output_dir: Path) -> None:
    """Write bayes_factors_fulllik.csv.

    log_BF is the T=0.7 estimate; the error is the two block errors in quadrature.
    """
    temps = [0.7]
    flag_nats = 8.0
    m1, m2 = config.MODELS

    rows = []
    for sim_model in config.MODELS:
        for realization in range(config.N_REALIZATIONS):
            p = full_likelihood.run_dir(sim_model, realization) / "gauss_harmonic_bf.npz"
            if not p.exists():
                continue
            d = np.load(p)
            bfs, errs = [], []
            for T in temps:
                a, b = d[f"{m1}_{T}"], d[f"{m2}_{T}"]
                bfs.append(float(a[0] - b[0]))
                errs.append(float(np.hypot(a[1], b[1])))
            mid = temps.index(0.7)
            tspread = max(bfs) - min(bfs)
            rows.append({
                "sim_model": sim_model, "realization": realization,
                "model_1": m1, "model_2": m2,
                "log_BF": bfs[mid],
                "log_BF_err": max(errs[mid], tspread),
                "T_spread": tspread,
                "flagged": tspread > flag_nats,
            })

    if not rows:
        print("\nNo full-likelihood harmonic BFs found; skipping bayes_factors_fulllik.csv.")
        return
    df = pd.DataFrame(rows)
    out = output_dir / "bayes_factors_fulllik.csv"
    df.to_csv(out, index=False)
    n_flag = int(df.flagged.sum())
    print(f"\n{len(df)} full-likelihood BFs -> {out} "
          f"({n_flag} flagged with T-spread > {flag_nats} nats)")
    if n_flag:
        print(df[df.flagged].to_string(index=False))


@app.command
def bayes_factors(output_dir: Path | str | None = None) -> None:
    """Collect all evidences into the results/ CSVs and plot them."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_dir = config.OUTPUT_DIR / "nested_sampling"
    evidence_df = analysis.load_all_evidences(evidence_dir)
    evidence_df.to_csv(output_dir / "all_evidences.csv", index=False)

    bf_df = analysis.compute_bayes_factors(evidence_df)
    bf_df.to_csv(output_dir / "bayes_factors.csv", index=False)

    summary = analysis.summarize_model_selection(bf_df)
    print("\nModel Selection Summary:")
    print(summary)
    summary.to_csv(output_dir / "model_selection_summary.csv")

    analysis.plot_bayes_factors(bf_df, output_dir)

    harm_df = harmonic_evidence.load_all_harmonic_evidences(evidence_dir)
    if not harm_df.empty:
        harm_df.to_csv(output_dir / "all_evidences_harmonic.csv", index=False)
        bf_harm = analysis.compute_bayes_factors(harm_df)
        bf_harm.to_csv(output_dir / "bayes_factors_harmonic.csv", index=False)

        summary_harm = analysis.summarize_model_selection(bf_harm)
        print("\nHarmonic Model Selection Summary:")
        print(summary_harm)
        summary_harm.to_csv(output_dir / "model_selection_summary_harmonic.csv")
    else:
        print("\nNo harmonic evidence files found; skipping cross-check.")

    _aggregate_fulllik(output_dir)

    print(f"\nResults saved to {output_dir}")


@app.command
def all() -> None:
    """Run validate-flows and bayes-factors."""
    validate_flows()
    bayes_factors()


if __name__ == "__main__":
    app()
