"""Bayes factors and model-selection summaries."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def load_all_evidences(evidence_dir: Path) -> pd.DataFrame:
    """Load all NS evidences, one row per (sim model, realization, test model)."""
    results = []
    for sim_model in config.MODELS:
        for realization in range(config.N_REALIZATIONS):
            for test_model in config.MODELS:
                path = (evidence_dir / sim_model / f"realization-{realization}"
                        / f"{test_model}_evidence.npz")
                if path.exists():
                    data = np.load(path)
                    results.append({
                        "sim_model": sim_model,
                        "realization": realization,
                        "test_model": test_model,
                        "logZ_mean": float(data["logZ_mean"]),
                        "logZ_std": float(data["logZ_std"]),
                        "n_iterations": int(data["n_iterations"]),
                    })
                else:
                    print(f"Warning: Missing {path}")

    return pd.DataFrame(results)


def compute_bayes_factors(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Per-realization log Bayes factor with powerlaw over gaussian."""
    results = []
    m1, m2 = config.MODELS

    for sim_model in evidence_df["sim_model"].unique():
        for realization in evidence_df["realization"].unique():
            mask = (evidence_df["sim_model"] == sim_model) & (
                evidence_df["realization"] == realization
            )
            evidences = {
                row["test_model"]: (row["logZ_mean"], row["logZ_std"])
                for _, row in evidence_df[mask].iterrows()
            }
            if m1 in evidences and m2 in evidences:
                results.append({
                    "sim_model": sim_model,
                    "realization": realization,
                    "model_1": m1,
                    "model_2": m2,
                    "log_BF": evidences[m1][0] - evidences[m2][0],
                    "log_BF_err": np.sqrt(
                        evidences[m1][1] ** 2 + evidences[m2][1] ** 2
                    ),
                })

    return pd.DataFrame(results)


def summarize_model_selection(bf_df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each model is preferred per true simulation model."""
    results = []
    for _, row in bf_df.iterrows():
        preferred = row["model_1"] if row["log_BF"] > 0 else row["model_2"]
        results.append({
            "sim_model": row["sim_model"],
            "realization": row["realization"],
            "preferred_model": preferred,
        })

    summary_df = pd.DataFrame(results)
    return summary_df.groupby(["sim_model", "preferred_model"]).size().unstack(fill_value=0)


def plot_bayes_factors(bf_df: pd.DataFrame, output_dir: Path) -> None:
    """Per-realization Bayes factor plot one panel per true model."""
    fig, axes = plt.subplots(1, len(config.MODELS), figsize=(12, 5))

    for ax, sim_model in zip(axes, config.MODELS):
        sim_data = bf_df[bf_df["sim_model"] == sim_model]
        m1, m2 = config.MODELS
        ax.errorbar(
            range(len(sim_data)),
            sim_data["log_BF"].values,
            yerr=sim_data["log_BF_err"].values,
            label=f"{m1} vs {m2}",
            marker="o",
            capsize=3,
            linestyle="none",
        )
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Realization")
        ax.set_ylabel("log Bayes Factor")
        ax.set_title(f"True model: {sim_model}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "bayes_factors.png", dpi=150)
    plt.close()

    print(f"Saved plot to {output_dir / 'bayes_factors.png'}")
