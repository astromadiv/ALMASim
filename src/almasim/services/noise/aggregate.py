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
- antenna_sefd: the spread of the antennas' own noise levels (log rms),
  added when the base has no such component (every array has the spread);
- radiometer: the receiver temperature per band edge that ALMA's Tsys
  implies under the simulator's Tsys model (12 m windows, PWV and the
  target's elevation), with the slope across the band that the windows'
  noise shows (their Tsys over their efficiency relative to the band's),
  and each dish size's Ruze aperture efficiency fitted
  to the efficiency the noise and Tsys imply, a factor for TDM windows
  where they depart from FDM windows of the same dish and band
  (tdm_efficiency), and a factor per dish and band where the band's windows
  depart from that curve (efficiency_scale);
- slow_phase, fast_phase, antenna_gains, baseline_errors: the fits; the in-scan
  screen's strengths at the windows' median coherence time, each keeping
  its phase change over one simulated sample, with a median per dish size
  (fast_phase.dishes) and the windows' log scatter about it
  (fast_phase.log_spread), which each simulated observation draws from;
  the frozen-flow screen's speed (fast_phase.wind_mps);
- baseline_errors.instrumental_*: the instrument's own share of the
  non-closing error, from the windows whose line channels tell it apart,
  with the continuum term keeping the rest;
- antenna_gains.bandpass: a dish size's residual ripple, set only when
  enough of its windows see one.

The research that set alma@1 did the same by hand (research/noise_realism).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..noise_model import NoiseModel, Provenance, ValueStats
from .measurements import PhaseFit, WindowFit, WindowNoise

HANNING_RHO1 = 0.5  # rho1 above this: native channels, not averaged
WIDE_WINDOW_HZ = 1e9
# The time over which windows' in-scan screens keep their phase change when
# their strengths are brought to one coherence time: the simulator's usual
# sample (observing.SCAN_TIMES_S).
FAST_REFERENCE_S = 18.144
# A bandpass shape with a longer period than this is smooth across the
# window (a residual delay or bandpass slope), not a ripple.
RIPPLE_MAX_MHZ = 50.0
# A band's receiver temperature takes a slope with frequency from at least
# this many windows, when the slope's 95% interval leaves zero out.
TRX_SLOPE_WINDOWS = 8
TRX_SLOPE_CONFIDENCE = 0.95
# A window's line channels need this share of continuum for its non-closing
# error to be told into the instrument's and the continuum's: below it, the
# window's signal may be the line itself, and the two errors are one.
MIN_CONTINUUM_SHARE = 0.05
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
    reduce(
        "antenna_sefd.log_rms",
        [w.antenna_noise.sefd_log_rms for w in windows if w.antenna_noise is not None],
        lambda value: components.setdefault("antenna_sefd", {}).update(log_rms=value),
    )
    fits = [m.fit for m in measured if m.fit is not None]
    slow = [f.slow_phase for f in fits if f.slow_phase is not None]
    fast = [f.fast_phase for f in fits if f.fast_phase is not None]
    # Each in-scan fit with its window's dish, for the per-dish strengths.
    fast_dish = [
        (m.window.exposure.dish_diameter_m, m.fit.fast_phase)
        for m in measured
        if m.fit is not None and m.fit.fast_phase is not None
    ]
    reduce(
        "slow_phase.path_um_1km", [s.path_um_1km for s in slow], setter("slow_phase", "path_um_1km")
    )
    reduce("slow_phase.slope", [s.slope for s in slow], setter("slow_phase", "slope"))
    reduce(
        "fast_phase.coherence_s",
        [s.coherence_s for s in fast if s.coherence_s is not None],
        setter("fast_phase", "coherence_s"),
    )
    coherence = stats.get("fast_phase.coherence_s")
    reduce(
        "fast_phase.path_um_1km",
        [
            s.path_um_1km if coherence is None else _path_at_coherence(s, coherence.median)
            for s in fast
        ],  # fmt: skip
        setter("fast_phase", "path_um_1km"),
    )
    reduce("fast_phase.slope", [s.slope for s in fast], setter("fast_phase", "slope"))
    reduce("fast_phase.wind_mps", [s.wind_mps for s in fast], setter("fast_phase", "wind_mps"))
    if "fast_phase" in components and "fast_phase.path_um_1km" in stats:
        _fast_dishes(
            fast_dish,
            None if coherence is None else coherence.median,
            components["fast_phase"],
            stats,
            notes,
            min_windows,
        )
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
    if "baseline_errors" in components:
        _instrumental(fits, components["baseline_errors"], stats, notes, min_windows)
    if "antenna_gains" in components:
        _bandpass(measured, components["antenna_gains"], stats, notes, min_windows)
    if "radiometer" in components:
        _radiometer(
            windows,
            base,
            components["radiometer"],
            stats,
            notes,
            min_windows,
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


def _fast_dishes(
    fits: Sequence[tuple[float, PhaseFit]],
    coherence_s: float | None,
    component: dict[str, Any],
    stats: dict[str, ValueStats],
    notes: list[str],
    min_windows: int,
) -> None:
    """The in-scan screen's strength per dish size and its spread between
    observations. Each dish size with ``min_windows`` fitted windows gets the
    median of their strengths (at the common coherence time); the spread is
    the rms of each window's log strength about its dish's median, over the
    windows of those dishes, with its 16-84% bootstrap interval."""
    paths: dict[float, list[float]] = {}
    for diameter, fit in fits:
        path = fit.path_um_1km if coherence_s is None else _path_at_coherence(fit, coherence_s)
        if path > 0:
            paths.setdefault(round(float(diameter), 1), []).append(path)
    rows: list[dict[str, float]] = []
    residuals: list[float] = []
    for diameter, values in sorted(paths.items(), reverse=True):
        key = f"fast_phase.dishes.{diameter:g}m"
        if len(values) < min_windows:
            notes.append(f"{key}: {len(values)} window(s), takes path_um_1km")
            continue
        stats[key] = _stats(values)
        rows.append({"diameter_m": diameter, "path_um_1km": stats[key].median})
        residuals += list(np.log(np.asarray(values) / stats[key].median))
    if rows:
        component["dishes"] = rows
    # Degrees of freedom: one median per dish taken out.
    dof = len(residuals) - len(rows)
    if dof < min_windows:
        notes.append(f"fast_phase.log_spread: {max(dof, 0)} degree(s) of freedom, kept the base's")
        return
    r = np.asarray(residuals)
    spread = float(np.sqrt(np.sum(r**2) / dof))
    # Bootstrap the windows (the per-dish medians held), seeded so a rerun agrees.
    rng = np.random.default_rng(0)
    draws = rng.choice(r, size=(2000, r.size))
    boot = np.sqrt(np.sum(draws**2, axis=1) / dof)
    q16, q84 = np.percentile(boot, [16, 84])
    stats["fast_phase.log_spread"] = ValueStats(
        n=r.size, median=spread, q16=float(q16), q84=float(q84)
    )
    component["log_spread"] = spread


def _instrumental(
    fits: Sequence[WindowFit],
    component: dict[str, Any],
    stats: dict[str, ValueStats],
    notes: list[str],
    min_windows: int,
) -> None:
    """The instrument's own share of the non-closing error, and the
    continuum's what remains. Each window whose line channels tell the two
    apart (``line_nonclosing``) gives its instrumental error over its whole
    window's; the median share ``s`` of the windows' non-closing error
    (``amp_rms``, already their median) is the instrument's, on all the
    signal, and the continuum term keeps ``sqrt(1 - s^2)`` of it, so the
    continuum still sees the windows' median in quadrature. Only windows
    whose line channels hold ``MIN_CONTINUUM_SHARE`` of continuum count."""
    parts = (
        ("amp_rms", "instrumental_amp_rms", "baseline_amp_rms"),
        ("phase_rms_deg", "instrumental_phase_rms_deg", "baseline_phase_rms_deg"),
    )
    for total_key, own_key, window_key in parts:
        shares = []
        for fit in fits:
            line, window = fit.line_nonclosing, getattr(fit, window_key)
            # Line channels with hardly any continuum say nothing: the window's
            # error may then be the line's own as well.
            if line is None or line.continuum_share < MIN_CONTINUUM_SHARE:
                continue
            own = getattr(line, own_key)
            if own is not None and window is not None and window > 0:
                # A share above 1 is noise in the split: the instrument has it all.
                shares.append(min(own / window, 1.0))
        key = f"baseline_errors.{own_key}"
        if len(shares) < min_windows:
            notes.append(f"{key}: {len(shares)} window(s) tell it apart, kept the base's")
            continue
        share = _stats(shares)
        stats[f"{key}_share"] = share
        total = float(component.get(total_key) or 0.0)
        component[own_key] = share.median * total
        component[total_key] = total * float(np.sqrt(1.0 - share.median**2))
        notes.append(
            f"baseline_errors.{total_key}: the continuum's part, "
            f"{np.sqrt(1.0 - share.median**2):.3g} of the windows' median"
        )


def _bandpass(
    measured: Sequence[Measured],
    component: dict[str, Any],
    stats: dict[str, ValueStats],
    notes: list[str],
    min_windows: int,
) -> None:
    """Each dish size's residual bandpass ripple, when at least ``min_windows``
    of its windows see one: a ripple period under ``RIPPLE_MAX_MHZ`` (a longer
    one is a smooth shape across the window, not a ripple) and an amplitude or
    phase larger than the noise's own rms per antenna and chunk. The row takes
    those windows' median period and their median rms as a sine's peak
    (x sqrt 2), up to the highest of their frequencies."""
    seen: dict[float, list[tuple[float, float, float, float]]] = {}
    for m in measured:
        bp = None if m.fit is None else m.fit.bandpass
        if bp is None or bp.period_mhz is None or bp.period_mhz >= RIPPLE_MAX_MHZ:
            continue
        # Seen: larger than the noise's own rms per antenna and chunk.
        if bp.amp_rms <= bp.thermal_amp_rms and bp.phase_rms_deg <= bp.thermal_phase_rms_deg:
            continue
        diameter = round(float(m.window.exposure.dish_diameter_m), 1)
        seen.setdefault(diameter, []).append(
            (bp.period_mhz, bp.amp_rms, bp.phase_rms_deg, m.window.exposure.frequency_ghz)
        )
    rows = []
    # The dish sizes with a bandpass fit at all: a run without one says nothing.
    dishes = {
        round(float(m.window.exposure.dish_diameter_m), 1)
        for m in measured
        if m.fit is not None and m.fit.bandpass is not None
    }
    for diameter in sorted(dishes, reverse=True):
        ripples = seen.get(diameter, [])
        key = f"antenna_gains.bandpass.{diameter:g}m"
        if len(ripples) < min_windows:
            notes.append(f"{key}: {len(ripples)} window(s) see a ripple, none set")
            continue
        period, amp, phase, freq = (np.asarray(v) for v in zip(*ripples, strict=True))
        stats[f"{key}.amp"] = _stats(list(np.sqrt(2.0) * amp))
        stats[f"{key}.phase_deg"] = _stats(list(np.sqrt(2.0) * phase))
        stats[f"{key}.period_mhz"] = _stats(list(period))
        rows.append(
            {
                "diameter_m": diameter,
                "upper_ghz": float(freq.max()),
                "period_mhz": stats[f"{key}.period_mhz"].median,
                "amp": stats[f"{key}.amp"].median,
                "phase_deg": stats[f"{key}.phase_deg"].median,
            }
        )
    if rows:
        component["bandpass"] = rows


def _path_at_coherence(fit: PhaseFit, coherence_s: float) -> float:
    """A window's in-scan screen strength at another coherence time, with the
    same phase change over ``FAST_REFERENCE_S``: its strength and coherence
    trade off, so medians of each over the windows would draw a faster screen
    than any of them. An AR(1) screen of variance P^2 changes by
    2 P^2 (1 - exp(-t / coherence)) over t."""
    if fit.coherence_s is None:
        return fit.path_um_1km
    own = 1.0 - np.exp(-FAST_REFERENCE_S / fit.coherence_s)
    other = 1.0 - np.exp(-FAST_REFERENCE_S / coherence_s)
    return float(fit.path_um_1km * np.sqrt(own / other))


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
    """The receiver temperature per band edge, each dish's Ruze parameters and,
    per dish and band, the factor by which the band's windows depart from the
    Ruze curve (``efficiency_scale``)."""
    from ..interferometry import noise

    config = noise.NoiseModelConfig(pwv_mm=1.0, **base.noise_config())
    table = config.receiver_temperatures or noise._RECEIVER_TEMPERATURE_K
    rows = radiometer.get("receiver_temperature_k") or [
        {"upper_ghz": None if np.isinf(row[0]) else row[0], "trx_k": row[1]} for row in table
    ]
    edges = np.array([np.inf if r["upper_ghz"] is None else r["upper_ghz"] for r in rows])
    apertures = {float(d): (eta0, rms) for d, (eta0, rms) in (config.ruze or noise._RUZE).items()}
    correlator = config.correlator_efficiency
    kept: set[float] = set()  # dishes whose aperture stays the base's
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
            kept.add(diameter)
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
    relative = _relative_efficiencies(windows, rows, edges, apertures, correlator)
    needed: dict[int, list[tuple[float, float, float | None]]] = {}
    for i, w in enumerate(windows):
        trx = _trx_needed(w, config.forward_efficiency)
        if trx is not None and round(w.exposure.dish_diameter_m) == 12:
            row = int(np.searchsorted(edges, w.exposure.frequency_ghz, side="right"))
            by_noise = (
                _trx_needed(w, config.forward_efficiency, w.tsys.median_k / relative[i])
                if i in relative and w.tsys is not None
                else None
            )
            needed.setdefault(min(row, len(rows) - 1), []).append(
                (w.exposure.frequency_ghz, trx, by_noise)
            )
    for k, row in enumerate(rows):
        trx_points = needed.get(k, [])
        key = f"radiometer.receiver_temperature_k[{row.get('bands') or k}]"
        if len(trx_points) < min_windows:
            notes.append(f"{key}: {len(trx_points)} window(s), kept the base's")
            continue
        ghz = np.array([p[0] for p in trx_points])
        values = np.array([p[1] for p in trx_points])
        row.pop("slope_k_per_ghz", None)
        row.pop("center_ghz", None)
        noise_points = [(p[0], p[2]) for p in trx_points if p[2] is not None]
        line = (
            _trx_slope(*(np.array(x) for x in zip(*noise_points, strict=True)))
            if len(noise_points) >= max(min_windows, TRX_SLOPE_WINDOWS)
            else None
        )
        if line is not None:
            slope, low, high, _ = line
            center = float(np.median(ghz))
            values = values - slope * (ghz - center)
            stats[f"{key}.slope_k_per_ghz"] = ValueStats(
                n=len(noise_points), median=slope, q16=low, q84=high
            )
            row["slope_k_per_ghz"], row["center_ghz"] = slope, center
        stats[key] = _stats(values.tolist())
        row["trx_k"] = max(stats[key].median, 1.0)
    radiometer["receiver_temperature_k"] = rows

    ratios = _efficiency_ratios(windows, rows, edges, apertures, correlator)
    tdm = _tdm_efficiency(ratios, max(min_windows, 3), stats, notes)
    if tdm is not None:
        radiometer["tdm_efficiency"] = tdm
    factor = radiometer.get("tdm_efficiency") or 1.0
    scales: dict[tuple[float, int], list[float]] = {}
    for diameter, row, is_tdm, ratio in ratios:
        scales.setdefault((diameter, row), []).append(ratio / (factor if is_tdm else 1.0))
    fitted_rows = _scale_rows(
        {key: v for key, v in scales.items() if key[0] not in kept}, rows, stats, notes, min_windows
    )
    # A scale is relative to its dish's Ruze curve: a dish that keeps the
    # base's aperture keeps the base's scale rows with it.
    base_rows = [
        r
        for r in radiometer.get("efficiency_scale") or []
        if any(round(r["diameter_m"]) == round(d) for d in kept)
    ]
    # Stable sort: each dish's rows keep their rising edges.
    merged = sorted(fitted_rows + base_rows, key=lambda r: -r["diameter_m"])
    radiometer["efficiency_scale"] = merged or None


def window_tdm(window: WindowNoise) -> bool:
    """Whether the window's channels are TDM (instrument.channels): from its
    resolution, or its channel width when it has none."""
    from ..instrument.channels import TDM_NATIVE_HZ, channels_from_resolution

    e = window.exposure
    if not e.resolution_hz:
        return e.channel_width_hz >= TDM_NATIVE_HZ * (1 - 1e-6)
    return channels_from_resolution(e.resolution_hz, e.channels * e.channel_width_hz).tdm


def _efficiency_ratios(
    windows: Sequence[WindowNoise],
    rows: list[dict[str, Any]],
    edges: np.ndarray,
    apertures: dict[float, tuple[float, float]],
    correlator: float,
) -> list[tuple[float, int, bool, float]]:
    """Each window's (dish, band row, TDM, measured efficiency over the dish's
    Ruze curve at the window's frequency): what one curve per dish leaves of a
    band's efficiency (the receivers' and correlator's own, band by band)."""
    from ..interferometry import noise

    ratios: list[tuple[float, int, bool, float]] = []
    for w in windows:
        diameter = next(
            (d for d in apertures if round(d) == round(w.exposure.dish_diameter_m)), None
        )
        if diameter is None or not w.visibility.efficiency:
            continue
        hz = w.exposure.frequency_ghz * 1e9
        curve = noise.aperture_efficiency(hz, diameter, {diameter: apertures[diameter]})
        row = min(
            int(np.searchsorted(edges, w.exposure.frequency_ghz, side="right")), len(rows) - 1
        )
        measured = w.visibility.efficiency / correlator
        ratios.append((diameter, row, window_tdm(w), measured / curve))
    return ratios


def _tdm_efficiency(
    ratios: list[tuple[float, int, bool, float]],
    needed: int,
    stats: dict[str, ValueStats],
    notes: list[str],
) -> float | None:
    """TDM windows' efficiency over FDM windows' of the same dish and band.
    Over the (dish, band) groups of at least ``needed`` windows with both
    kinds (in a group of TDM windows only, the band's own scale takes it),
    alternately each group's median with the TDM windows divided by the
    factor, and the factor as the median of the TDM windows over their
    group's. None, with a note, when fewer than ``needed`` TDM windows are in
    such groups."""
    key = "radiometer.tdm_efficiency"
    groups: dict[tuple[float, int], list[tuple[bool, float]]] = {}
    for diameter, row, is_tdm, ratio in ratios:
        if ratio > 0:
            groups.setdefault((diameter, row), []).append((is_tdm, float(np.log(ratio))))
    mixed = {
        g: v
        for g, v in groups.items()
        if len(v) >= needed and any(t for t, _ in v) and not all(t for t, _ in v)
    }
    count = sum(t for v in mixed.values() for t, _ in v)
    if count < needed:
        notes.append(f"{key}: {count} window(s), kept the base's")
        return None
    factor = 0.0
    residuals: list[float] = []
    for _ in range(50):
        residuals = []
        for v in mixed.values():
            centre = float(np.median([log - (factor if t else 0.0) for t, log in v]))
            residuals += [log - centre for t, log in v if t]
        new = float(np.median(residuals))
        if abs(new - factor) < 1e-9:
            break
        factor = new
    stats[key] = _stats(list(np.exp(residuals)))
    return float(np.exp(factor))


def _scale_rows(
    ratios: dict[tuple[float, int], list[float]],
    rows: list[dict[str, Any]],
    stats: dict[str, ValueStats],
    notes: list[str],
    min_windows: int,
) -> list[dict[str, Any]]:
    """The efficiency_scale rows: per dish with a band of at least
    ``max(min_windows, 3)`` windows, every band row, the band's median ratio
    where it has that many windows and 1 elsewhere."""
    needed = max(min_windows, 3)
    out: list[dict[str, Any]] = []
    for diameter in sorted({d for d, _ in ratios}, reverse=True):
        fitted: dict[int, float] = {}
        for k, row in enumerate(rows):
            values = ratios.get((diameter, k), [])
            key = f"radiometer.efficiency_scale[{diameter:g} m][{row.get('bands') or k}]"
            if len(values) < needed:
                if values:
                    notes.append(f"{key}: {len(values)} window(s), kept 1")
                continue
            stats[key] = _stats(values)
            fitted[k] = stats[key].median
        if not fitted:
            continue
        out += [
            {
                "diameter_m": diameter,
                "upper_ghz": row["upper_ghz"],
                "scale": fitted.get(k, 1.0),
                "bands": row.get("bands", ""),
            }
            for k, row in enumerate(rows)
        ]
    return out


def _relative_efficiencies(
    windows: Sequence[WindowNoise],
    rows: list[dict[str, Any]],
    edges: np.ndarray,
    apertures: dict[float, tuple[float, float]],
    correlator: float,
) -> dict[int, float]:
    """Per window (by its index), its measured efficiency over the Ruze curve
    relative to the median of the windows of its dish, band row and kind of
    channels (TDM or FDM): what of the window's noise a band's one efficiency
    leaves to its system temperature. Windows in a group of fewer than 3 have
    none."""
    from ..interferometry import noise

    groups: dict[tuple[float, int, bool], list[tuple[int, float]]] = {}
    for i, w in enumerate(windows):
        diameter = next(
            (d for d in apertures if round(d) == round(w.exposure.dish_diameter_m)), None
        )
        if diameter is None or not w.visibility.efficiency:
            continue
        hz = w.exposure.frequency_ghz * 1e9
        curve = noise.aperture_efficiency(hz, diameter, {diameter: apertures[diameter]})
        row = min(
            int(np.searchsorted(edges, w.exposure.frequency_ghz, side="right")), len(rows) - 1
        )
        ratio = w.visibility.efficiency / correlator / curve
        if ratio > 0 and np.isfinite(ratio):
            groups.setdefault((diameter, row, window_tdm(w)), []).append((i, float(ratio)))
    out: dict[int, float] = {}
    for members in groups.values():
        if len(members) < 3:
            continue
        centre = float(np.median([r for _, r in members]))
        out.update({i: r / centre for i, r in members})
    return out


def _trx_needed(
    window: WindowNoise, forward_efficiency: float | None = None, tsys_k: float | None = None
) -> float | None:
    """The receiver temperature that gives ``tsys_k`` (the window's median SYSCAL
    Tsys when None) under the simulator's Tsys model (linear in it; the
    Technical Handbook's with ``forward_efficiency``), at the window's PWV and
    the target's elevation; None without them."""
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
    target = w.tsys.median_k if tsys_k is None else tsys_k
    return 100.0 * (target - at[0]) / (at[1] - at[0])


def _trx_slope(ghz: np.ndarray, trx: np.ndarray) -> tuple[float, float, float, float] | None:
    """How a band's needed receiver temperatures change with frequency: the
    Theil-Sen slope (K/GHz), its 16% and 84% bounds and the windows' median
    frequency (GHz), which the band's temperature is then at; None when the
    slope's 95% interval holds zero (one temperature serves the band)."""
    from scipy.stats import theilslopes

    if np.ptp(ghz) <= 0:
        return None
    slope, _, low95, high95 = theilslopes(trx, ghz, alpha=TRX_SLOPE_CONFIDENCE)
    if not np.all(np.isfinite([slope, low95, high95])) or low95 <= 0.0 <= high95:
        return None
    _, _, low, high = theilslopes(trx, ghz, alpha=0.68)
    return float(slope), float(low), float(high), float(np.median(ghz))


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
