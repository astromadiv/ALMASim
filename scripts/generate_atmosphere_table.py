"""Write the zenith opacity table of the ALMA site from CASA's ATM model (MADIV-28).

python scripts/generate_atmosphere_table.py           # writes the table
python scripts/generate_atmosphere_table.py --check   # also compares it with ATM

Needs casatools (the ``casa`` extra) and its measures data. The table holds,
every 10 MHz from 30 to 1000 GHz, the dry and the per-mm-of-water zenith
opacity, and every 50 MHz the atmosphere's effective radiating temperature
for a range of water columns, so that
services.interferometry.atmosphere computes the sky without CASA.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "src" / "almasim" / "services" / "interferometry" / "data" / "atm_chajnantor.npz"
START_GHZ, STOP_GHZ, STEP_GHZ = 30.0, 1000.0, 0.01
CHUNK_GHZ = 5.0
REFERENCE_PWV_MM = 1.0
T_CMB_K = 2.725
# The effective temperature depends on the water column (water radiates from
# low, warm air; the dry air from higher and colder): tabulated per column,
# every TEFF_EVERY table steps.
PWV_GRID_MM = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TEFF_EVERY = 5

# Chajnantor, as the ALMA pipeline and the OT set it up.
SITE = {
    "altitude": {"value": 5059.0, "unit": "m"},
    "temperature": {"value": 270.0, "unit": "K"},
    "pressure": {"value": 555.0, "unit": "mbar"},
    "maxAltitude": {"value": 48.0, "unit": "km"},
    "humidity": 20.0,
    "dTem_dh": {"value": -5.6, "unit": "K/km"},
    "dP": {"value": 10.0, "unit": "mbar"},
    "dPm": 1.2,
    "h0": {"value": 2.0, "unit": "km"},
    "atmType": 1,  # tropical
}


def _atmosphere():
    from almasim.services.casa.environment import configure_casa_environment

    casa_data = os.environ.get("CASA_DATA")
    if casa_data:
        configure_casa_environment(Path(os.environ.get("CASA_RUN", ".casa-run")), casa_data)
    from casatools import atmosphere

    at = atmosphere()
    at.initAtmProfile(**SITE)
    return at


def _spectrum(at, centre_ghz: float, width_ghz: float, pwv_mm: float, airmass: float):
    """Frequencies (GHz), dry and wet opacity at zenith, and the sky's brightness."""
    nchan = int(round(width_ghz / STEP_GHZ))
    at.initSpectralWindow(
        nbands=1,
        fCenter={"value": centre_ghz, "unit": "GHz"},
        fWidth={"value": nchan * STEP_GHZ, "unit": "GHz"},
        fRes={"value": STEP_GHZ, "unit": "GHz"},
    )
    at.setUserWH2O({"value": pwv_mm, "unit": "mm"})
    at.setAirMass(airmass)
    freq = np.asarray(at.getSpectralWindow(0)["value"]) / 1e9
    dry = np.asarray(at.getDryOpacitySpec(spwid=0)[1])
    wet = np.asarray(at.getWetOpacitySpec(spwid=0)[1]["value"])
    tebb = np.asarray(at.getTebbSkySpec(spwid=0)[1]["value"])
    order = np.argsort(freq)
    return freq[order], dry[order], wet[order], tebb[order]


def build(at) -> dict[str, np.ndarray]:
    grid = np.round(np.arange(START_GHZ, STOP_GHZ + STEP_GHZ / 2, STEP_GHZ), 6)
    coarse = grid[::TEFF_EVERY]
    dry, wet = np.empty(grid.size), np.empty(grid.size)
    teff = np.empty((len(PWV_GRID_MM), coarse.size))
    for lo in np.arange(START_GHZ, STOP_GHZ, CHUNK_GHZ):
        sel = (grid >= lo) & (grid <= lo + CHUNK_GHZ)
        csel = (coarse >= lo) & (coarse <= lo + CHUNK_GHZ)
        for k, pwv in enumerate(PWV_GRID_MM):
            freq, d, w, tebb = _spectrum(at, lo + CHUNK_GHZ / 2, CHUNK_GHZ + 4 * STEP_GHZ, pwv, 1.0)
            if pwv == REFERENCE_PWV_MM:
                dry[sel] = np.interp(grid[sel], freq, d)
                wet[sel] = np.interp(grid[sel], freq, w) / REFERENCE_PWV_MM
            tau = d + w
            atmosphere_only = tebb - T_CMB_K * np.exp(-tau)  # the background, attenuated
            teff[k, csel] = np.interp(coarse[csel], freq, atmosphere_only / -np.expm1(-tau))
    return {
        "start_ghz": np.float64(START_GHZ),
        "step_ghz": np.float64(STEP_GHZ),
        "tau_dry": dry.astype(np.float32),
        "tau_wet_per_mm": wet.astype(np.float32),
        "teff_every": np.int64(TEFF_EVERY),
        "pwv_grid_mm": np.asarray(PWV_GRID_MM),
        "t_eff_k": teff.astype(np.float32),
    }


def check(at) -> None:
    """The table's sky against ATM's own, away from the reference water column."""
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from almasim.services.interferometry.atmosphere import sky_spectrum

    worst = 0.0
    for centre in (
        43.0,
        98.0,
        115.0,
        150.0,
        183.0,
        230.0,
        325.0,
        345.0,
        405.0,
        490.0,
        650.0,
        870.0,
    ):
        for pwv, airmass in ((0.3, 1.0), (1.0, 1.5), (3.0, 1.2), (6.0, 2.0)):
            freq, _, _, tebb = _spectrum(at, centre, 4.0, pwv, airmass)
            tsky, _ = sky_spectrum(freq * 1e9, pwv_mm=pwv, airmass=airmass)
            err = float(np.max(np.abs(tsky - tebb) / tebb))
            worst = max(worst, err)
            print(
                f"{centre:6.1f} GHz pwv {pwv:3.1f} airmass {airmass:3.1f}: "
                f"max |table - ATM| / ATM = {err:.4f}"
            )
    print(f"worst {worst:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="compare the table with ATM afterwards"
    )
    args = parser.parse_args()
    at = _atmosphere()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TARGET, **build(at))
    print(f"wrote {TARGET} ({TARGET.stat().st_size / 1e3:.0f} kB)")
    if args.check:
        check(at)


if __name__ == "__main__":
    main()
