"""Paper figure style: loads figs/paper.mplstyle and sizes figures to the text width."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TEXTWIDTH_PT = 246.0
INCHES_PER_PT = 1.0 / 72.27
GOLDEN_RATIO = (np.sqrt(5.0) - 1.0) / 2.0

_STYLE_PATH = Path(__file__).resolve().parents[1] / "figs" / "paper.mplstyle"


def figsize(scale: float = 1.0, aspect: float = GOLDEN_RATIO) -> tuple[float, float]:
    """Figure size in inches for ``scale`` of the text width."""
    w = TEXTWIDTH_PT * INCHES_PER_PT * scale
    return (w, w * aspect)


def use_paper_style(usetex: bool | None = None) -> None:
    """Apply the paper mplstyle. Set ``usetex=False`` if LaTeX isn't available."""
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use(_STYLE_PATH)
    if usetex is not None:
        plt.rcParams["text.usetex"] = usetex
