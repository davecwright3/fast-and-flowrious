"""Pipeline configuration."""

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(os.environ.get("FMC_DATA_DIR", PROJECT_ROOT / "data" / "psrs"))
OUTPUT_DIR = Path(os.environ.get("FMC_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
MAX_LIKE_FILE = DATA_DIR / "max-likelihood-estimate.json"

MODELS = ("powerlaw", "gaussian")

N_PULSARS = 67
# Fixed across every stage. Changing it invalidates all outputs.
N_FREQS = 14
N_FREQS_IRN = 30

N_REALIZATIONS = 50

MIN_SNR = 3.0
MAX_SNR = 8.0

INJECTION_BOUNDS = {
    "powerlaw": {
        "gw_log10_A": (-15.0, -13.5),
        "gw_gamma": (2.0, 6.0),
    },
    # log10_A is the strain at the peak
    # sigma is in natural-log units.
    "gaussian": {
        "gw_log10_A": (-14.0, -10.5),
        "gw_log10_f_peak": (-8.45, -8.05),
        "gw_sigma": (0.3, 0.6),
    },
}

# Must contain INJECTION_BOUNDS for every parameter.
ANALYSIS_BOUNDS = {
    "powerlaw": {
        "gw_log10_A": (-18.0, -11.0),
        "gw_gamma": (0.0, 7.0),
    },
    "gaussian": {
        "gw_log10_A": (-17.0, -10.0),
        "gw_log10_f_peak": (-8.55, -8.00),
        "gw_sigma": (0.3, 1.0),
    },
}

# Below discovery's -9 so noise-dominated bins sample the floor plateau instead of the prior wall.
FREESPEC_LOG10_RHO_LOWER = -14.0
FREESPEC_LOG10_RHO_UPPER = -4.0

FLOW_CONFIG = {
    "max_epochs": 2000,
    "patience": 20,
    "prior_bounds": np.array(
        [[FREESPEC_LOG10_RHO_LOWER, FREESPEC_LOG10_RHO_UPPER]] * N_FREQS
    ),
}

NS_CONFIG = {
    "n_live": 2000,
    "num_delete_frac": 0.5,
    "num_inner_steps": 20,
    "termination_logZ_diff": -3.0,
}

MCMC_CONFIG = {
    "num_warmup": 500,
    "num_samples": 4000,
    "num_chains": 1,
}
