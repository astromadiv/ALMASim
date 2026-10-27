"""The library's radiometer noise against the real one, split into Tsys and efficiency.

Per window: the real Tsys (SYSCAL, median over antennas, times and the window's
channels), the elevation it was observed at, and the efficiency the real noise
implies, eta = 2 k Tsys / (2 A sigma sqrt(t dnu)) for Stokes I (both hands);
the library's Tsys model (interferometry.noise, with CASA's ATM sky) for the
same window at the real elevation and at transit (what the library uses), and
its efficiency (aperture x correlator).

    python libtsys.py STOKESI.csv OUT.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NOISE_SET", "all")
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The library's modules without its package __init__s (numba, the simulator):
# this environment has casacore, the library's own does not.
import types  # noqa: E402

for _pkg in ("almasim", "almasim.services", "almasim.services.interferometry"):
    _module = types.ModuleType(_pkg)
    _module.__path__ = ["/home1/mdelliveneri/ALMASim-atm/src/" + _pkg.replace(".", "/")]
    sys.modules[_pkg] = _module
from counterpart import tsys_profile  # noqa: E402
from window import name, subset  # noqa: E402

K_B, JY = 1.380649e-23, 1e-26
NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")


def observed_elevation(ms: Path) -> float:
    """Median elevation of the target over the window's rows (degrees)."""
    from astropy import units as u
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time
    from casacore.tables import table

    from almasim.services.interferometry.observing import alma_location

    with table(str(ms), ack=False) as t:
        times = np.unique(t.getcol("TIME"))
    with table(str(ms / "FIELD"), ack=False) as f:
        ra, dec = np.asarray(f.getcell("PHASE_DIR", 0)).ravel()[:2]
    when = Time(np.percentile(times, [10, 50, 90]) / 86400.0, format="mjd")
    altaz = SkyCoord(ra * u.rad, dec * u.rad).transform_to(AltAz(obstime=when, location=alma_location()))
    return float(np.median(altaz.alt.deg))


def main(stokesi: str, out: str) -> None:
    from casacore.tables import table

    from almasim.services.interferometry.noise import (
        NoiseModelConfig,
        estimate_receiver_temperature_k,
    )
    from almasim.services.interferometry.atmosphere import system_temperature
    from almasim.services.observation_plan import estimate_transit_elevation

    s = pd.read_csv(stokesi).set_index("window")
    rows = pd.read_csv(NOISE / "libsim" / "rows.csv").set_index("window")
    windows = subset()
    eta_lib = NoiseModelConfig(pwv_mm=1.0).aperture_efficiency * NoiseModelConfig(pwv_mm=1.0).correlator_efficiency
    result = []
    for i, w in windows.iterrows():
        if i not in s.index:
            continue
        base = NOISE / "base" / f"{name(w)}.ms"
        real_ms = Path(w.ms)
        try:
            with table(str(base / "SPECTRAL_WINDOW"), ack=False) as sw:
                freq = np.asarray(sw.getcell("CHAN_FREQ", 0), dtype=float)
                bw_ratio = float(np.median(np.abs(sw.getcell("EFFECTIVE_BW", 0)) / np.abs(sw.getcell("CHAN_WIDTH", 0))))
            tsys_real = float(np.sqrt(np.median(tsys_profile(real_ms, int(w.spw), freq))))
            elevation = observed_elevation(base)
        except Exception as exc:  # noqa: BLE001 - one window must not stop the table
            print(f"{i}: {exc}", flush=True)
            continue
        row = rows.loc[i]
        diameter = 7.0 if str(row.Array_type).startswith("7m") else 12.0
        area = np.pi * (diameter / 2) ** 2
        trx = estimate_receiver_temperature_k(freq)
        lib_tsys = {}
        for key, el in (("real_el", elevation), ("transit", estimate_transit_elevation(float(row.Dec)))):
            airmass = 1.0 / np.sin(np.radians(max(5.0, min(90.0, el))))
            lib_tsys[key] = float(np.median(system_temperature(freq, trx, pwv_mm=float(row.PWV), airmass=airmass)))
        unit_real = float(s.loc[i, "real:unit"])
        # The noise follows each channel's noise bandwidth (EFFECTIVE_BW), not its
        # width, which the unit noise (sigma sqrt(t width)) is scaled by.
        eta_real = 2 * K_B * tsys_real / (2 * area * unit_real) / JY / np.sqrt(bw_ratio)
        result.append({
            "window": i, "band": int(row.Band), "array": "7m" if diameter == 7 else "12m",
            "freq_ghz": float(freq.mean() / 1e9), "pwv": float(row.PWV),
            "el_real": elevation, "el_transit": estimate_transit_elevation(float(row.Dec)),
            "tsys_real": tsys_real, "tsys_lib_real_el": lib_tsys["real_el"], "tsys_lib_transit": lib_tsys["transit"],
            "trx_lib": float(np.median(trx)), "bw_ratio": bw_ratio, "eta_real": eta_real, "eta_lib": eta_lib,
            "noise_ratio": float(s.loc[i, "ratio"]) if "ratio" in s.columns else float(s.loc[i, "lib:unit"] / unit_real),
        })
        print(f"done {i}", flush=True)
    d = pd.DataFrame(result)
    d["tsys_ratio"] = d.tsys_lib_transit / d.tsys_real
    d["eta_ratio"] = d.eta_real / d.eta_lib
    d["predicted"] = d.tsys_ratio * d.eta_ratio
    d.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["band", "array"])[["tsys_real", "tsys_lib_transit", "tsys_ratio", "eta_real", "eta_ratio", "predicted", "noise_ratio"]].median().round(3).to_string())
    print(d[["tsys_ratio", "eta_ratio", "predicted", "noise_ratio", "eta_real"]].median().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
