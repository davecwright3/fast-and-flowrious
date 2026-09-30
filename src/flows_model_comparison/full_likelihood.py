"""Stage 07: full-likelihood NUTS and Gaussian-container harmonic evidence.

We run NUTS on the full PTA likelihood for both test models. The injected model's chain is the
reference posterior for the PP plots, and the two chains give an independent Bayes factor.

A learned container blows up in ~136 dimensions, so we use a full-covariance Gaussian narrowed by
a temperature T < 1.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from scipy.special import logsumexp

from . import config, models

# Stage 08 only uses T=0.7; the others are diagnostics.
TEMPS = [0.2, 0.3, 0.5, 0.7, 0.9]


def run_dir(sim_model: str, realization: int) -> Path:
    return config.OUTPUT_DIR / "full_likelihood" / sim_model / f"realization-{realization}"


def _build(sim_model: str, realization: int, test_model: str):
    """(logl, crn_params, priordict) for one (data set, test model) pair."""
    sim_dir = config.OUTPUT_DIR / "simulations" / sim_model / f"realization-{realization}"
    psrs = models.load_simulated_pulsars(sim_dir)
    model, crn_params = models.build_parametric_model(psrs, test_model)
    return model.logL, crn_params, models.priordict_for(test_model, crn_params)


def sample(sim_model: str, realization: int, test_model: str,
           seed: int | None = None) -> None:
    """Run NUTS on the full likelihood and save the chain.

    Starts at the injected values, or the prior midpoint for the wrong model's common params.
    """
    seed = realization if seed is None else seed
    logl, crn_params, priordict = _build(sim_model, realization, test_model)
    npmodel, groups = models.make_numpyro_model(logl, priordict)
    bounds = models.uniform_bounds(logl.params, priordict)

    inj = dict(np.load(config.OUTPUT_DIR / "simulations" / sim_model
                       / f"realization-{realization}" / "params.npz"))

    def init_value(name):
        lo, hi = bounds[name]
        # Per-pulsar red noise is always injected; common params only for the right model.
        if name in inj:
            x = float(inj[name])
        elif sim_model == test_model and f"gw_{models.physical_name(name)}" in inj:
            x = float(inj[f"gw_{models.physical_name(name)}"])
        else:
            x = 0.5 * (lo + hi)
        w = hi - lo
        return float(np.clip(x, lo + 1e-3 * w, hi - 1e-3 * w))

    init_values = {
        site: (jnp.array([init_value(m) for m in members]) if dim is not None
               else jnp.float64(init_value(members[0])))
        for site, members, dim in groups
    }
    print("warm-starting at injected IRN + "
          + ("injected common" if sim_model == test_model else "midpoint common"))

    kern = numpyro.infer.NUTS(
        npmodel, init_strategy=numpyro.infer.init_to_value(values=init_values)
    )
    mcmc = numpyro.infer.MCMC(kern, **config.MCMC_CONFIG, progress_bar=True)
    mcmc.run(jax.random.key(seed), extra_fields=("diverging",))
    ndiv = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())

    df = npmodel.to_df(mcmc.get_samples())
    names = list(df.columns)
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])

    out_dir = run_dir(sim_model, realization)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chain_{test_model}.npz"
    np.savez(path, chain=df.to_numpy(), lo=lo, hi=hi, ndiv=ndiv,
             names=np.array(names), seed=seed, n_freqs=config.N_FREQS)
    print(f"divergences={ndiv}/{len(df)}; saved {path}")

    if test_model == sim_model:
        post = {models.physical_name(c): df[c].to_numpy() for c in crn_params}
        np.savez(out_dir / "posterior.npz", **post)
        print(f"saved injected-model posterior -> {out_dir / 'posterior.npz'}")

        print(f"{'param':>14} {'injected':>9} {'recovered (med, 16-84)':>30}")
        for phys, x in post.items():
            truth = float(inj[f"gw_{phys}"])
            q = np.percentile(x, [16, 50, 84])
            print(f"{phys:>14} {truth:>9.3f}   {q[1]:+8.3f}  [{q[0]:+.3f}, {q[2]:+.3f}]")


def _x64_logls(sim_model: str, realization: int, test_model: str, batch: int = 200) -> dict:
    """float64 logL at every chain sample, cached in the chain file."""
    path = run_dir(sim_model, realization) / f"chain_{test_model}.npz"
    d = dict(np.load(path))
    if "logl" not in d:
        logl, _, _ = _build(sim_model, realization, test_model)
        names = [str(n) for n in d["names"]]
        f = jax.jit(jax.vmap(lambda row: logl({n: row[i] for i, n in enumerate(names)})))
        chain = d["chain"].astype(np.float64)
        d["logl"] = np.concatenate([np.asarray(f(jnp.asarray(chain[i:i + batch])))
                                    for i in range(0, len(chain), batch)])
        np.savez(path, **d)
    return d


def gaussian_container_logZ(y, log_post, temps, seed=0):
    """Harmonic mean with a Gaussian container at temperature T.

    Returns {T: (logZ, block_std, ESS, max_weight)}.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    y, log_post = y[perm], log_post[perm]
    n = len(y) // 2
    # Fit the container on one half, evaluate on the other.
    ytr, yin, lpin = y[:n], y[n:], log_post[n:]
    mu = ytr.mean(axis=0)
    C = np.cov(ytr.T) + 1e-8 * np.eye(y.shape[1])
    evals, evecs = np.linalg.eigh(C)
    z = (yin - mu) @ evecs
    out = {}
    for T in temps:
        # log N(y; mu, T C), in C's eigenbasis.
        maha = np.sum(z**2 / (T * evals), axis=1)
        log_phi = -0.5 * (maha + np.sum(np.log(2 * np.pi * T * evals)))
        lw = log_phi - lpin
        ln_inv = logsumexp(lw) - np.log(len(lw))
        wn = np.exp(lw - logsumexp(lw))
        ess = 1.0 / np.sum(wn**2)
        blocks = [logsumexp(b) - np.log(len(b)) for b in np.array_split(lw, 4)]
        out[T] = (-ln_inv, float(np.std(blocks)), float(ess), float(wn.max()))
    return out


def evidence(sim_model: str, realization: int) -> None:
    """Harmonic evidence for both test models and their Bayes factor.

    Chains are logit-transformed first so the container has unbounded support.
    """
    R = {}
    for test_model in config.MODELS:
        d = _x64_logls(sim_model, realization, test_model)
        chain, lo, hi, lls = d["chain"].astype(np.float64), d["lo"], d["hi"], d["logl"]
        w = hi - lo
        u = np.clip((chain - lo) / w, 1e-12, 1 - 1e-12)
        y = np.log(u) - np.log1p(-u)
        # log |dx/dy| for the logit map.
        log_jac = np.sum(np.log(w) + np.log(u) + np.log1p(-u), axis=1)
        log_prior = -float(np.sum(np.log(w)))
        res = gaussian_container_logZ(y, lls + log_prior + log_jac, TEMPS)
        R[test_model] = res
        print(f"\n{test_model} (llmax_in_chain={lls.max():.1f}, ndiv={int(d['ndiv'])}):")
        for T, (logZ, err, ess, wmax) in res.items():
            print(f"  T={T}: logZ = {logZ:.2f} +- {err:.2f}  (ESS={ess:.0f}/{len(chain)//2}, "
                  f"max_w={wmax:.3f})", flush=True)

    m1, m2 = config.MODELS
    print(f"\n===== GAUSSIAN-CONTAINER HARMONIC BF: {sim_model} r{realization} =====")
    for T in TEMPS:
        bf = R[m1][T][0] - R[m2][T][0]
        err = float(np.hypot(R[m1][T][1], R[m2][T][1]))
        print(f"  T={T}: logBF({m1}-{m2}) = {bf:+.2f} +- {err:.2f} nats "
              f"(log10 {bf/np.log(10):+.2f})")

    out = run_dir(sim_model, realization) / "gauss_harmonic_bf.npz"
    np.savez(out, temps=np.array(TEMPS),
             **{f"{m}_{T}": np.array(v) for m, res in R.items() for T, v in res.items()})
    print(f"saved {out}")
