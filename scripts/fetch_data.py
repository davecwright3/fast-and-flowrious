#!/usr/bin/env python
"""Download the NANOGrav 15yr feather files into ``config.DATA_DIR``.

Files come from the pinned discovery revision and the pulsar list from
max-likelihood-estimate.json. Existing files are skipped.
"""

import urllib.request

import cyclopts

from flows_model_comparison import config, simulation

# Keep in sync with the discovery rev in pyproject.toml.
DISCOVERY_REV = "b26d2554df5540a7610eedf926659701daad01ad"
URL = (
    "https://raw.githubusercontent.com/nanograv/discovery/{rev}/data/"
    "v1p1_de440_pint_bipm2019-{psr}.feather"
)

app = cyclopts.App()


@app.default
def fetch(rev: str = DISCOVERY_REV) -> None:
    """Download missing feathers from discovery revision ``rev``."""
    pulsars = sorted(
        {k.split("_")[0] for k in simulation.load_max_like() if not k.startswith("crn_")}
    )
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for psr in pulsars:
        dest = config.DATA_DIR / f"{psr}.feather"
        if dest.exists():
            continue
        print(f"downloading {psr}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(rev=rev, psr=psr), tmp)
        tmp.rename(dest)
    print(f"{len(pulsars)} pulsars in {config.DATA_DIR}")


if __name__ == "__main__":
    app()
