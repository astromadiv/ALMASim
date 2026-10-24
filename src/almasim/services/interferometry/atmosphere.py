"""The sky and the system temperature across a spectral window (MADIV-28).

The noise of calibrated ALMA data varies across a window as ALMA's own Tsys
spectrum does (measured on the 100 MADIV-16 windows: the variance times
WEIGHT follows SYSCAL's Tsys squared). Its shape comes from atmospheric lines,
in the signal band and, through the receiver's finite sideband rejection, in
the image band, and from the receiver itself.

The sky comes from a table of CASA's ATM model for the ALMA site
(data/atm_chajnantor.npz, written by scripts/generate_atmosphere_table.py):
zenith opacity, dry and per mm of water, every 10 MHz from 30 to 1000 GHz,
and the atmosphere's effective radiating temperature for a range of water
columns, every 50 MHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

TABLE = Path(__file__).parent / "data" / "atm_chajnantor.npz"
TABLE_GROUND_K = 270.0  # the ground temperature the table was computed for
T_CMB_K = 2.725


@lru_cache(maxsize=1)
def _table() -> dict[str, np.ndarray]:
    with np.load(TABLE) as data:
        return {key: data[key] for key in data.files}


def _lookup(freqs_hz: np.ndarray, values: np.ndarray, step_ghz: float) -> np.ndarray:
    ghz = np.asarray(freqs_hz, dtype=np.float64) / 1e9
    grid = float(_table()["start_ghz"]) + step_ghz * np.arange(values.size)
    if np.any(ghz < grid[0]) or np.any(ghz > grid[-1]):
        raise ValueError(
            f"frequencies outside the atmosphere table ({grid[0]:.0f}-{grid[-1]:.0f} GHz)"
        )
    return np.interp(ghz, grid, values.astype(np.float64))


def _effective_temperature(freqs_hz: np.ndarray, pwv_mm: float) -> np.ndarray:
    """The atmosphere's radiating temperature, between the tabulated water columns
    (linear in log column, held at the ends)."""
    t = _table()
    step = float(t["step_ghz"]) * int(t["teff_every"])
    columns = np.log(t["pwv_grid_mm"])
    k = float(np.interp(np.log(max(pwv_mm, 1e-3)), columns, np.arange(columns.size)))
    lo = int(np.floor(k))
    hi = min(lo + 1, columns.size - 1)
    w = k - lo
    return (1 - w) * _lookup(freqs_hz, t["t_eff_k"][lo], step) + w * _lookup(
        freqs_hz, t["t_eff_k"][hi], step
    )


def sky_spectrum(
    freqs_hz: np.ndarray,
    *,
    pwv_mm: float,
    airmass: float,
    ground_temperature_k: float = TABLE_GROUND_K,
) -> tuple[np.ndarray, np.ndarray]:
    """The sky's brightness temperature (K) and the zenith opacity at ``freqs_hz``:
    the atmosphere's emission and the cosmic background seen through it."""
    t = _table()
    step = float(t["step_ghz"])
    pwv_mm = max(float(pwv_mm), 0.0)
    tau = _lookup(freqs_hz, t["tau_dry"], step) + pwv_mm * _lookup(
        freqs_hz, t["tau_wet_per_mm"], step
    )
    t_eff = _effective_temperature(freqs_hz, pwv_mm) * float(ground_temperature_k) / TABLE_GROUND_K
    path = float(airmass) * tau
    return t_eff * -np.expm1(-path) + T_CMB_K * np.exp(-path), tau


@dataclass(frozen=True)
class ReceiverBand:
    """An ALMA receiver band: its range, the centre of its IF and its image rejection
    (the image band's gain over the signal band's: 0 single sideband, ~0.1 for a
    sideband-separating receiver's 10 dB, 1 double sideband)."""

    name: str
    lo_ghz: float
    hi_ghz: float
    if_centre_ghz: float
    image_gain: float


BANDS = (
    ReceiverBand("1", 35.0, 50.0, 8.0, 0.0),
    ReceiverBand("2", 67.0, 84.0, 11.0, 0.1),
    ReceiverBand("3", 84.0, 116.0, 6.0, 0.1),
    ReceiverBand("4", 125.0, 163.0, 6.0, 0.1),
    ReceiverBand("5", 163.0, 211.0, 6.0, 0.1),
    ReceiverBand("6", 211.0, 275.0, 7.0, 0.1),
    ReceiverBand("7", 275.0, 373.0, 6.0, 0.1),
    ReceiverBand("8", 385.0, 500.0, 6.0, 0.1),
    ReceiverBand("9", 602.0, 720.0, 8.0, 1.0),
    ReceiverBand("10", 787.0, 950.0, 8.0, 1.0),
)


def receiver_band(freq_hz: float) -> ReceiverBand:
    """The band that covers ``freq_hz``, or the nearest one."""
    ghz = float(freq_hz) / 1e9
    for band in BANDS:
        if band.lo_ghz <= ghz <= band.hi_ghz:
            return band
    return min(BANDS, key=lambda b: min(abs(ghz - b.lo_ghz), abs(ghz - b.hi_ghz)))


def system_temperature(
    freqs_hz: np.ndarray,
    trx_k: np.ndarray | float,
    *,
    pwv_mm: float,
    airmass: float,
    ground_temperature_k: float = TABLE_GROUND_K,
    sideband: str = "auto",
) -> np.ndarray:
    """Tsys (K, above the atmosphere) across a window.

    Tsys = (1 + g) e^(A tau_s) (Trx + Tsky_s + g Tsky_i), with g the band's image
    gain and the image band mirrored about the LO, one IF centre from the
    window. ``sideband``: "usb" (LO below the window), "lsb", or "auto", the one
    whose image band gives the lower Tsys (as a tuning would choose).
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    centre = float(np.mean(freqs_hz))
    band = receiver_band(centre)
    sky_kwargs = {
        "pwv_mm": pwv_mm,
        "airmass": airmass,
        "ground_temperature_k": ground_temperature_k,
    }
    tsky, tau = sky_spectrum(freqs_hz, **sky_kwargs)
    signal = np.exp(airmass * tau) * (np.asarray(trx_k, dtype=np.float64) + tsky)
    if band.image_gain <= 0:
        return signal

    def with_image(lo_hz: float) -> np.ndarray:
        image = np.clip(2.0 * lo_hz - freqs_hz, 30e9, 1000e9)
        tsky_image, _ = sky_spectrum(image, **sky_kwargs)
        return (1.0 + band.image_gain) * (
            signal + band.image_gain * np.exp(airmass * tau) * tsky_image
        )

    offset = band.if_centre_ghz * 1e9
    choices = {"usb": centre - offset, "lsb": centre + offset}
    if sideband in choices:
        return with_image(choices[sideband])
    if sideband != "auto":
        raise ValueError(f"sideband must be 'usb', 'lsb' or 'auto', not {sideband!r}")
    return min((with_image(lo) for lo in choices.values()), key=lambda t: float(np.mean(t)))


def receiver_ripple(rng: np.random.Generator, nchan: int, rms: float, modes: int = 4) -> np.ndarray:
    """A smooth fractional ripple across ``nchan`` channels (Legendre modes 1 to
    ``modes`` with equal expected power), of expected rms ``rms`` across the window:
    how a receiver's temperature varies over its IF."""
    if rms <= 0 or nchan < 2:
        return np.zeros(nchan)
    x = np.linspace(-1.0, 1.0, nchan)
    basis = np.stack([np.polynomial.legendre.Legendre.basis(k)(x) for k in range(1, modes + 1)])
    basis /= np.sqrt(np.mean(basis**2, axis=1, keepdims=True))
    return rms / np.sqrt(modes) * (rng.standard_normal(modes) @ basis)
