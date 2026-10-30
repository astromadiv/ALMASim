"""Reduce many windows' measurements into a candidate noise model (MADIV-28).

``noise extract`` (window.json) and ``noise fit`` (fit.json) measure each
window; this takes a base model (the released one) and replaces each value
the windows measure by their median, recording how many windows it comes
from and their spread (``provenance.values``). A value fewer than
``min_windows`` measure keeps the base's, with a note. What is reduced, and
how:

- channel_correlation: rho1, rho2 of the windows whose native channels are
  Hanning-correlated (rho1 above 0.5, no averaging);
- receiver_ripple: the smooth part (Legendre modes 1 to 4) of the measured
  noise profile over SYSCAL's Tsys^2, halved (variance to Tsys);
- antenna_ripple: each antenna's noise shape, over the windows wider than
  1 GHz (narrow ones show finer structure);
- radiometer: the receiver temperature per band edge that ALMA's Tsys
  implies under the simulator's Tsys model (12 m windows, PWV and the
  target's elevation), and each dish size's Ruze aperture efficiency fitted
  to the efficiency the noise and Tsys imply;
- slow_phase, fast_phase, antenna_gains, baseline_errors: the fits.

The research that set alma@1 did the same by hand (research/noise_realism).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..noise_model import NoiseModel, Provenance, ValueStats
from .measurements import WindowFit, WindowNoise

HANNING_RHO1 = 0.5  # rho1 above this: native channels, not averaged
WIDE_WINDOW_HZ = 1e9
C_M_S = 299_792_458.0


@dataclass(frozen=True)
class Measured:
    """One window's measurements."""

    window: WindowNoise
    fit: WindowFit | None = None


def aggregate(
    measured: Sequence[Measured],
    base: NoiseModel,
    *,
    name: str | None = None,
    version: int | None = None,
    data: str | None = None,
    min_windows: int = 3,
) -> NoiseModel:
    """The candidate: ``base`` with every value the windows measure replaced
    by their median (see the module's description)."""
    from almasim.__version__ import __version__

    stats: dict[str, ValueStats] = {}
    notes: list[str] = []
    components = {name: c.model_dump() for name, c in base.components.items()}

    def reduce(
        key: str, measured: Sequence[float | None], set_value: Callable[[float], None]
    ) -> None:
        values = [float(v) for v in measured if v is not None and np.isfinite(v)]
        if len(values) < min_windows:
            notes.append(f"{key}: {len(values)} window(s), kept the base's")
            return
        stats[key] = _stats(values)
        set_value(stats[key].median)

    def setter(component: str, parameter: str) -> Callable[[float], None]:
        def set_value(value: float) -> None:
            if component in components:
                components[component][parameter] = value

        return set_value

    windows = [m.window for m in measured]
    native = [w for w in windows if w.thermal.rho1 > HANNING_RHO1]
    reduce(
        "channel_correlation.rho1",
        [w.thermal.rho1 for w in native],
        setter("channel_correlation", "rho1"),
    )
    reduce(
        "channel_correlation.rho2",
        [w.thermal.rho2 for w in native],
        setter("channel_correlation", "rho2"),
    )
    reduce(
        "receiver_ripple.rms",
        [_receiver_ripple(w) for w in windows],
        setter("receiver_ripple", "rms"),
    )
    wide = [
        w.antenna_shape
        for w in windows
        if w.antenna_shape is not None
        and w.exposure.channels * w.exposure.channel_width_hz > WIDE_WINDOW_HZ
    ]
    reduce("antenna_ripple.rms", [s.excess_rms for s in wide], setter("antenna_ripple", "rms"))
    reduce(
        "antenna_ripple.scale_mhz",
        [s.half_width_mhz for s in wide],
        setter("antenna_ripple", "scale_mhz"),
    )
    if "radiometer" in components:
        _radiometer(windows, base, components["radiometer"], stats, notes, min_windows)

    fits = [m.fit for m in measured if m.fit is not None]
    slow = [f.slow_phase for f in fits if f.slow_phase is not None]
    fast = [f.fast_phase for f in fits if f.fast_phase is not None]
    reduce(
        "slow_phase.path_um_1km", [s.path_um_1km for s in slow], setter("slow_phase", "path_um_1km")
    )
    reduce("slow_phase.slope", [s.slope for s in slow], setter("slow_phase", "slope"))
    reduce(
        "fast_phase.path_um_1km", [s.path_um_1km for s in fast], setter("fast_phase", "path_um_1km")
    )
    reduce("fast_phase.slope", [s.slope for s in fast], setter("fast_phase", "slope"))
    reduce(
        "antenna_gains.amp_rms", [f.gain_amp_rms for f in fits], setter("antenna_gains", "amp_rms")
    )
    reduce(
        "baseline_errors.amp_rms",
        [f.baseline_amp_rms for f in fits],
        setter("baseline_errors", "amp_rms"),
    )
    reduce(
        "baseline_errors.phase_rms_deg",
        [f.baseline_phase_rms_deg for f in fits],
        setter("baseline_errors", "phase_rms_deg"),
    )
    return NoiseModel.model_validate(
        {
            **base.model_dump(mode="json", exclude={"components", "provenance"}),
            "name": name or f"{base.name}-candidate",
            "version": version or base.version + 1,
            "description": base.description,
            "components": components,
            "provenance": Provenance(
                method="noise aggregate: medians over the windows, from "
                f"{base.label()} where too few measure a value",
                data=data,
                windows=len(windows),
                almasim_commit=__version__,
                date=dt.date.today().isoformat(),
                notes="; ".join(notes) or None,
                values=stats,
            ).model_dump(mode="json"),
        }
    )


def _stats(values: Sequence[float]) -> ValueStats:
    q16, median, q84 = np.percentile(values, [16, 50, 84])
    return ValueStats(n=len(values), median=float(median), q16=float(q16), q84=float(q84))


def _receiver_ripple(window: WindowNoise) -> float | None:
    """The receiver's ripple as a fraction of Tsys: the smooth part (Legendre
    modes 1 to 4) of the measured noise profile over SYSCAL's Tsys^2, which
    is a variance (twice the Tsys ripple). Edges (5% each side) left out."""
    if window.tsys is None or window.exposure.channels < 32:
        return None
    ratio = np.asarray(window.thermal.channel_profile) / np.asarray(window.tsys.profile)
    n = ratio.size
    edge = max(1, n // 20)
    x = np.linspace(-1.0, 1.0, n)[edge:-edge]
    y = ratio[edge:-edge] / np.mean(ratio[edge:-edge]) - 1.0
    smooth = np.polynomial.legendre.Legendre.fit(x, y, 4)(x)
    smooth -= smooth.mean()
    return float(np.sqrt(np.mean(smooth**2)) / 2.0)


def _radiometer(
    windows: Sequence[WindowNoise],
    base: NoiseModel,
    radiometer: dict[str, Any],
    stats: dict[str, ValueStats],
    notes: list[str],
    min_windows: int,
) -> None:
    """The receiver temperature per band edge and each dish's Ruze parameters."""
    from ..interferometry import noise

    config = noise.NoiseModelConfig(pwv_mm=1.0, **base.noise_config())
    table = config.receiver_temperatures or noise._RECEIVER_TEMPERATURE_K
    rows = radiometer.get("receiver_temperature_k") or [
        {"upper_ghz": None if np.isinf(edge) else edge, "trx_k": trx} for edge, trx in table
    ]
    edges = np.array([np.inf if r["upper_ghz"] is None else r["upper_ghz"] for r in rows])
    needed: dict[int, list[float]] = {}
    for w in windows:
        trx = _trx_needed(w, config.forward_efficiency)
        if trx is not None and round(w.exposure.dish_diameter_m) == 12:
            row = int(np.searchsorted(edges, w.exposure.frequency_ghz, side="right"))
            needed.setdefault(min(row, len(rows) - 1), []).append(trx)
    for k, row in enumerate(rows):
        values = needed.get(k, [])
        key = f"radiometer.receiver_temperature_k[{row.get('bands') or k}]"
        if len(values) < min_windows:
            notes.append(f"{key}: {len(values)} window(s), kept the base's")
            continue
        stats[key] = _stats(values)
        row["trx_k"] = max(stats[key].median, 1.0)
    radiometer["receiver_temperature_k"] = rows

    apertures = {float(d): (eta0, rms) for d, (eta0, rms) in (config.ruze or noise._RUZE).items()}
    correlator = config.correlator_efficiency
    for diameter in sorted(apertures):
        points = [
            (w.exposure.frequency_ghz * 1e9, w.visibility.efficiency / correlator)
            for w in windows
            if w.visibility.efficiency and round(w.exposure.dish_diameter_m) == round(diameter)
        ]
        fitted = (
            _ruze_fit(points, apertures[diameter]) if len(points) >= max(min_windows, 5) else None
        )
        key = f"radiometer.aperture[{diameter:g} m]"
        if fitted is None:
            notes.append(f"{key}: {len(points)} window(s), kept the base's")
            continue
        apertures[diameter] = fitted
        efficiencies = [p[1] for p in points]
        stats[key + ".eta0"] = ValueStats(
            n=len(points),
            median=fitted[0],
            q16=float(np.percentile(efficiencies, 16)),
            q84=float(np.percentile(efficiencies, 84)),
        )
        stats[key + ".surface_rms_m"] = ValueStats(
            n=len(points), median=fitted[1], q16=fitted[1], q84=fitted[1]
        )
    radiometer["aperture"] = [
        {"diameter_m": d, "eta0": e, "surface_rms_m": r}
        for d, (e, r) in sorted(apertures.items(), reverse=True)
    ]


def _trx_needed(window: WindowNoise, forward_efficiency: float | None = None) -> float | None:
    """The receiver temperature that gives the window's median SYSCAL Tsys under
    the simulator's Tsys model (linear in it; the Technical Handbook's with
    ``forward_efficiency``), at the window's PWV and the target's elevation;
    None without them."""
    from ..interferometry.atmosphere import system_temperature

    w = window
    if w.tsys is None or w.input.pwv_mm is None or w.exposure.elevation_deg is None:
        return None
    freq = np.linspace(-0.5, 0.5, 64) * w.exposure.channels * w.exposure.channel_width_hz
    freq += w.exposure.frequency_ghz * 1e9
    airmass = 1.0 / np.sin(np.radians(max(5.0, min(90.0, w.exposure.elevation_deg))))
    at = [
        float(
            np.median(
                system_temperature(
                    freq,
                    trx,
                    pwv_mm=w.input.pwv_mm,
                    airmass=airmass,
                    forward_efficiency=forward_efficiency,
                )
            )
        )
        for trx in (0.0, 100.0)
    ]
    if at[1] <= at[0]:
        return None
    return 100.0 * (w.tsys.median_k - at[0]) / (at[1] - at[0])


def _ruze_fit(
    points: list[tuple[float, float]], start: tuple[float, float]
) -> tuple[float, float] | None:
    """eta0 exp(-(4 pi eps nu / c)^2) fitted to (frequency, aperture
    efficiency) in log, robust to outliers (soft L1)."""
    from scipy.optimize import least_squares

    nu = np.array([p[0] for p in points])
    eta = np.array([p[1] for p in points])
    keep = (eta > 0) & np.isfinite(eta)
    if keep.sum() < 5:
        return None
    nu, eta = nu[keep], eta[keep]

    def residual(p: np.ndarray) -> np.ndarray:
        return np.log(p[0]) - (4 * np.pi * p[1] * 1e-6 * nu / C_M_S) ** 2 - np.log(eta)

    result = least_squares(
        residual,
        [start[0], start[1] * 1e6],
        loss="soft_l1",
        f_scale=0.1,
        bounds=([0.3, 0.0], [1.0, 80.0]),
    )
    return float(result.x[0]), float(result.x[1] * 1e-6)
