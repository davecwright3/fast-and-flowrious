"""PSD models under comparison. Like discovery's, they return rho^2 = S(f) df."""

import discovery as ds
import jax.numpy as jnp

YR_S = 365.25 * 24 * 3600


def gaussian_psd(
    f: jnp.ndarray,
    df: jnp.ndarray,
    log10_A: float,
    log10_f_peak: float,
    sigma: float,
) -> jnp.ndarray:
    """Log-normal bump in the PSD.

        S(f) = A^2 / (12 pi^2) yr^3 exp(-ln(f/f_peak)^2 / (2 sigma^2))

    Same normalization as discovery's powerlaw, so ``log10_A`` is the strain at the peak.
    """
    A_sq = 10.0 ** (2.0 * log10_A)
    f_peak = 10.0**log10_f_peak

    normalization = A_sq / (12.0 * jnp.pi**2) * YR_S**3
    log_ratio = jnp.log(f / f_peak)
    psd = normalization * jnp.exp(-(log_ratio**2) / (2 * sigma**2))

    return psd * df


PSD_FUNCTIONS = {
    "powerlaw": ds.powerlaw,
    "gaussian": gaussian_psd,
}
