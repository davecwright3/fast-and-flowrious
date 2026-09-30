#!/usr/bin/env python
"""Stage 06: learned harmonic mean evidence on the stage-05 samples, as a check on NS."""
from typing import Literal

import cyclopts

from flows_model_comparison import config, harmonic_evidence

app = cyclopts.App()


@app.command
def run_harmonic(
    sim_model: Literal["powerlaw", "gaussian"],
    realization: int,
    test_model: Literal["powerlaw", "gaussian"],
) -> None:
    """Harmonic evidence for one (sim model, realization, test model)."""
    run_dir = config.OUTPUT_DIR / "nested_sampling" / sim_model / f"realization-{realization}"
    ns_csv = run_dir / f"{test_model}_evidence.csv"
    if not ns_csv.exists():
        raise FileNotFoundError(f"NS chain file not found: {ns_csv}. Run stage 05 first.")

    print(f"Harmonic evidence: sim={sim_model}, real={realization}, test={test_model}")
    result = harmonic_evidence.compute_harmonic_evidence(ns_csv, test_model)

    print(
        f"  log Z = {result['log_evidence']:.3f} "
        f"(+{result['ln_inv_evidence_err_neg']:.3f} / "
        f"{result['ln_inv_evidence_err_pos']:.3f})"
    )

    out_path = run_dir / f"{test_model}_harmonic.npz"
    harmonic_evidence.save_harmonic_evidence(result, out_path)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    app()
