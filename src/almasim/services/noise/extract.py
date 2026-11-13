"""Measure a real window's noise from its calibrated visibilities (MADIV-28).

Everything comes from differences of channels: three channels apart they
remove the sky (smooth on that scale) and are uncorrelated even for
Hanning-smoothed channels; one and two apart they show the correlation of
neighbours, var(V_k - V_k+l) = 2 sigma^2 (1 - rho_l). No sky model is needed.

Rows are taken evenly over the whole observation (every n-th row, up to
about ``max_visibilities``), read in chunks so memory stays bounded whatever
the MS's size. The measurements are those of research/noise_realism
(counterpart.measure_noise, tsys_profile, stokesi_noise, uvmix and
gaps/ripplescale), which set the alma noise model's values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..imaging.ms_geometry import import_casacore_tables
from ..instrument.constants import DEFAULT_MAX_WEIGHT
from .measurements import (
    AntennaShape,
    Exposure,
    ThermalNoise,
    Tsys,
    VisibilityNoise,
    WindowInput,
    WindowNoise,
)

K_B = 1.380649e-23
JY = 1.0e-26
PARALLEL = {5, 8, 9, 12}  # RR, LL, XX, YY
CLIP = 20.0  # a channel difference above 20 times its channel's median is not noise
SHAPE_SMOOTH = 16  # channels an antenna's noise shape is smoothed over


@dataclass
class _Sums:
    """Running sums over the sampled rows."""

    nchan: int
    nant: int
    nhand: int
    lag_sum: dict[int, np.ndarray] = field(default_factory=dict)
    lag_count: dict[int, np.ndarray] = field(default_factory=dict)
    threshold: dict[int, np.ndarray] = field(default_factory=dict)
    stokes_sum: float = 0.0
    stokes_count: int = 0
    stokes_threshold: np.ndarray | None = None
    hands_identical: bool | None = None
    antenna_sum: np.ndarray = field(init=False)
    antenna_count: np.ndarray = field(init=False)
    antenna_rows: np.ndarray = field(init=False)
    # Per baseline (ANTENNA1 * nant + ANTENNA2): rows over the maximum WEIGHT,
    # flagged values and values, and the Stokes I sums (stokes_sum, stokes_count).
    baseline: dict[str, np.ndarray] = field(init=False)
    rows: int = 0

    def __post_init__(self) -> None:
        for lag in (1, 2, 3):
            self.lag_sum[lag] = np.zeros(self.nchan - lag)
            self.lag_count[lag] = np.zeros(self.nchan - lag)
        self.antenna_sum = np.zeros((self.nant, self.nchan - 3, self.nhand))
        self.antenna_count = np.zeros((self.nant, self.nchan - 3, self.nhand))
        self.antenna_rows = np.zeros(self.nant, dtype=np.int64)
        self.baseline = {
            key: np.zeros(self.nant * self.nant)
            for key in ("high", "flagged", "values", "stokes_sum", "stokes_count")
        }

    def shares(self) -> tuple[float, float]:
        """The flagged share of the values and the share that is unflagged in
        rows over the maximum WEIGHT, on the baselines between the array's
        antennas (antenna_rows): the rows of an antenna whose data are all
        flagged are not the array's. The two add up to the share the image
        does without; a flagged value of a row over the maximum is counted once."""
        array = self.antenna_rows > 0
        between = (array[:, None] & array[None, :]).ravel()
        b = {key: value[between].sum() for key, value in self.baseline.items()}
        return b["flagged"] / max(b["values"], 1), b["high"] / max(b["values"], 1)


def extract_window(
    ms: str | Path,
    spw: int,
    field_selection: str | int | None = None,
    *,
    name: str | None = None,
    syscal_ms: str | Path | None = None,
    line_channels: np.ndarray | None = None,
    max_visibilities: int = 40_000_000,
    chunk_values: int = 20_000_000,
    pwv_mm: float | None = None,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
) -> tuple[WindowNoise, dict[str, np.ndarray]]:
    """Measure one window. Returns the measurements and the arrays to keep
    beside them (``antenna_g`` (nant, nchan, nhand) and ``antenna_h``: each
    antenna's and hand's noise shape, empty when it could not be measured).
    The unflagged values of rows whose WEIGHT exceeds ``max_weight`` (the
    imaging's maximum) are counted as the exposure's ``high_weight_fraction``."""
    from almasim.__version__ import __version__

    table = import_casacore_tables()
    ms = Path(ms)
    window = _window(table, ms, spw)
    nchan = window["freq"].size
    if nchan < 8:
        raise ValueError(f"spw {spw} has {nchan} channels; the noise needs at least 8")
    ddids = _data_descriptions(table, ms, spw)
    fields = _fields(table, ms, field_selection)
    lines = (
        np.zeros(nchan, dtype=bool) if line_channels is None else np.asarray(line_channels, bool)
    )
    query = f"DATA_DESC_ID IN [{','.join(map(str, ddids))}] AND ANTENNA1 != ANTENNA2"
    if fields is not None:
        query += f" AND FIELD_ID IN [{','.join(map(str, fields))}]"
    notes: list[str] = []

    with table(str(ms), ack=False) as main, main.query(query) as sel:
        nrow = sel.nrows()
        if nrow == 0:
            raise ValueError(
                f"{ms}: no cross-correlation rows in spw {spw}"
                + (f", field {field_selection}" if fields else "")
            )
        ncorr = sel.getcell("DATA", 0).shape[1]
        corr_types = _correlations(table, ms, ddids[0])
        hands = [k for k, c in enumerate(corr_types) if c in PARALLEL] or list(range(ncorr))
        a1, a2 = sel.getcol("ANTENNA1"), sel.getcol("ANTENNA2")
        nant = int(max(a1.max(), a2.max())) + 1
        exposure_cols = {c: sel.getcol(c) for c in ("TIME", "INTERVAL")}
        step = max(1, int(np.ceil(nrow * nchan * ncorr / max_visibilities)))
        chunk = max(16, chunk_values // (nchan * ncorr))
        sums = _Sums(nchan=nchan, nant=nant, nhand=len(hands[:2]))
        sampled = np.arange(0, nrow, step)
        for start in range(0, sampled.size, chunk):
            first = int(sampled[start])
            count = int(min(chunk, sampled.size - start))
            _accumulate(
                sums,
                data=sel.getcol("DATA", first, count, step),
                flag=sel.getcol("FLAG", first, count, step),
                weight=sel.getcol("WEIGHT", first, count, step),
                ant1=a1[first : first + count * step : step],
                ant2=a2[first : first + count * step : step],
                hands=hands[:2],
                max_weight=max_weight,
            )

    thermal = _thermal(sums, lines, step)
    exposure = _exposure(
        table,
        ms,
        window,
        exposure_cols,
        a1,
        a2,
        sums.antenna_rows,
        fields,
        ncorr,
        *sums.shares(),
    )
    tsys = None
    try:
        tsys = _tsys(table, Path(syscal_ms) if syscal_ms else ms, spw, window["freq"])
    except (ValueError, RuntimeError, OSError) as exc:
        notes.append(f"tsys: {exc}")
    visibility = _visibility(sums, exposure, tsys, _own_dish_pairs(table, ms, sums, exposure))
    shape, arrays = _antenna_shape(sums, lines, exposure.channel_width_hz)
    if shape is None:
        notes.append("antenna_shape: too few antennas or channels with enough data")
    record = WindowNoise(
        name=name
        or f"{ms.name}_spw{spw}"
        + (f"_field{field_selection}" if field_selection is not None else ""),
        almasim_version=__version__,
        input=WindowInput(
            ms=str(ms),
            spw=int(spw),
            field=None if field_selection is None else str(field_selection),
            syscal_ms=None if syscal_ms is None else str(syscal_ms),
            line_channels=int(lines.sum()),
            pwv_mm=pwv_mm,
        ),
        exposure=exposure,
        thermal=thermal,
        visibility=visibility,
        tsys=tsys,
        antenna_shape=shape,
        notes=notes,
    )
    return record, arrays


# -- reading -----------------------------------------------------------------


def _window(table: Any, ms: Path, spw: int) -> dict[str, Any]:
    with table(str(ms / "SPECTRAL_WINDOW"), ack=False) as sw:
        if not 0 <= spw < sw.nrows():
            raise ValueError(f"{ms} has no spectral window {spw}")
        freq = np.asarray(sw.getcell("CHAN_FREQ", spw), dtype=np.float64)
        width = np.abs(np.asarray(sw.getcell("CHAN_WIDTH", spw), dtype=np.float64))
        effective = np.abs(np.asarray(sw.getcell("EFFECTIVE_BW", spw), dtype=np.float64))
        resolution = np.abs(np.asarray(sw.getcell("RESOLUTION", spw), dtype=np.float64))
    return {"freq": freq, "width": width, "effective": effective, "resolution": resolution}


def _data_descriptions(table: Any, ms: Path, spw: int) -> list[int]:
    with table(str(ms / "DATA_DESCRIPTION"), ack=False) as dd:
        ids = [i for i, s in enumerate(dd.getcol("SPECTRAL_WINDOW_ID")) if int(s) == spw]
    if not ids:
        raise ValueError(f"{ms}: spw {spw} has no data description")
    return ids


def _correlations(table: Any, ms: Path, ddid: int) -> list[int]:
    with table(str(ms / "DATA_DESCRIPTION"), ack=False) as dd:
        pol = int(dd.getcell("POLARIZATION_ID", ddid))
    with table(str(ms / "POLARIZATION"), ack=False) as p:
        return [int(c) for c in np.ravel(p.getcell("CORR_TYPE", pol))]


def _fields(table: Any, ms: Path, selection: str | int | None) -> list[int] | None:
    """The FIELD_IDs a field selection names (an id, or a name); None for all."""
    if selection is None or str(selection) == "":
        return None
    text = str(selection)
    with table(str(ms / "FIELD"), ack=False) as f:
        names = list(f.getcol("NAME"))
    if re.fullmatch(r"\d+", text):
        if int(text) >= len(names):
            raise ValueError(f"{ms} has no field {text}")
        return [int(text)]
    ids = [i for i, n in enumerate(names) if n == text]
    if not ids:
        raise ValueError(f"{ms} has no field named {text!r}")
    return ids


def _accumulate(
    sums: _Sums,
    *,
    data: np.ndarray,
    flag: np.ndarray,
    weight: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    hands: list[int],
    max_weight: float | None = None,
) -> None:
    """Add one chunk of rows (nrow, nchan, ncorr) to the sums."""
    imaged = np.ones(weight.shape[0], dtype=bool)
    if max_weight:
        imaged = weight.max(axis=1) <= max_weight
    pair = ant1 * sums.nant + ant2
    for key, value in (
        # The unflagged values of rows over the maximum WEIGHT: the archive
        # pipeline may have flagged such a row already, and flagged counts it.
        ("high", ((~imaged)[:, None, None] & ~flag).sum(axis=(1, 2)).astype(np.float64)),
        ("flagged", flag.sum(axis=(1, 2)).astype(np.float64)),
        ("values", np.full(pair.size, float(flag[0].size))),
    ):
        sums.baseline[key] += np.bincount(pair, value, sums.nant * sums.nant)
    if weight.shape[1] != data.shape[2]:
        weight = np.repeat(weight.mean(axis=1, keepdims=True), data.shape[2], axis=1)
    good = ~flag & (weight[:, None, :] > 0)
    imaged &= good.any(axis=(1, 2))
    np.add.at(sums.antenna_rows, ant1[imaged], 1)
    np.add.at(sums.antenna_rows, ant2[imaged], 1)
    sums.rows += data.shape[0]
    for lag in (1, 2, 3):
        diff = data[:, lag:] - data[:, :-lag]
        ok = good[:, lag:] & good[:, :-lag]
        v = np.where(ok, (diff.real**2 + diff.imag**2) / 4 * weight[:, None, :], np.nan)
        if lag not in sums.threshold:
            with np.errstate(all="ignore"):
                median = np.nanmedian(v, axis=(0, 2))
            sums.threshold[lag] = np.where(np.isfinite(median), CLIP * median, np.inf)
        keep = np.isfinite(v) & (v <= sums.threshold[lag][None, :, None])
        sums.lag_sum[lag] += np.where(keep, v, 0.0).sum(axis=(0, 2))
        sums.lag_count[lag] += keep.sum(axis=(0, 2))
        if lag == 3:
            _antenna_sums(sums, v[..., hands], keep[..., hands], ant1, ant2)
    p, q = hands[0], hands[-1]
    stokes = 0.5 * (data[..., p] + data[..., q])
    bad = flag[..., p] | flag[..., q]
    if sums.hands_identical is None:
        sums.hands_identical = bool(np.allclose(data[..., p], data[..., q]))
    bad |= (weight[:, p] <= 0)[:, None] | (weight[:, q] <= 0)[:, None]
    diff = (stokes[:, 3:] - stokes[:, :-3]) / np.sqrt(2.0)
    ok = ~(bad[:, 3:] | bad[:, :-3])
    v = np.where(ok, np.abs(diff) ** 2 / 2.0, np.nan)
    if sums.stokes_threshold is None:
        with np.errstate(all="ignore"):
            median = np.nanmedian(v, axis=0)
        sums.stokes_threshold = np.where(np.isfinite(median), CLIP * median, np.inf)
    keep = np.isfinite(v) & (v <= sums.stokes_threshold[None, :])
    kept = np.where(keep, v, 0.0)
    sums.stokes_sum += float(kept.sum())
    sums.stokes_count += int(keep.sum())
    size = sums.nant * sums.nant
    sums.baseline["stokes_sum"] += np.bincount(pair, kept.sum(axis=1), size)
    sums.baseline["stokes_count"] += np.bincount(pair, keep.sum(axis=1).astype(np.float64), size)


def _antenna_sums(
    sums: _Sums, v: np.ndarray, keep: np.ndarray, ant1: np.ndarray, ant2: np.ndarray
) -> None:
    """Each antenna's sum of its baselines' lag-3 variances, per channel and hand."""
    onehot = np.zeros((v.shape[0], sums.nant))
    rows = np.arange(v.shape[0])
    onehot[rows, ant1] += 1.0
    onehot[rows, ant2] += 1.0
    sums.antenna_sum += np.tensordot(onehot, np.where(keep, v, 0.0), axes=(0, 0))
    sums.antenna_count += np.tensordot(onehot, keep.astype(np.float64), axes=(0, 0))


# -- measurements ------------------------------------------------------------


def _per_channel(sums: _Sums, lag: int) -> np.ndarray:
    """The mean of a lag's differences, centred between their channels, per channel."""
    with np.errstate(all="ignore"):
        per = sums.lag_sum[lag] / sums.lag_count[lag]
    return np.pad(per, ((lag + 1) // 2, lag // 2), mode="edge")


def _thermal(sums: _Sums, lines: np.ndarray, step: int) -> ThermalNoise:
    """Means, not medians: an image's noise is a weighted mean of the
    visibilities' variances. Lag 3 gives the level and its variation across
    the window; lags 1 and 2 the correlation."""
    var1, var2, var3 = (_per_channel(sums, lag) for lag in (1, 2, 3))
    usable = np.isfinite(var1) & np.isfinite(var2) & np.isfinite(var3) & ~lines
    if usable.sum() < 4:
        usable = np.isfinite(var1) & np.isfinite(var2) & np.isfinite(var3)
    if not usable.any():
        raise ValueError("no unflagged channel differences to measure the noise on")
    scale = float(np.mean(var3[usable]))
    profile = np.where(np.isfinite(var3), var3 / scale, 1.0)
    return ThermalNoise(
        scale=scale,
        rho1=float(1 - np.mean(var1[usable]) / scale),
        rho2=float(1 - np.mean(var2[usable]) / scale),
        channel_profile=[round(float(p), 4) for p in profile],
        rows_sampled=sums.rows,
        row_step=step,
    )


def _exposure(
    table: Any,
    ms: Path,
    window: dict[str, Any],
    cols: dict[str, np.ndarray],
    a1: np.ndarray,
    a2: np.ndarray,
    antenna_rows: np.ndarray,
    fields: list[int] | None,
    ncorr: int,
    flagged: float,
    high_weight: float = 0.0,
) -> Exposure:
    """The antennas and baselines are those with data the image uses (unflagged,
    WEIGHT within the maximum) in the sampled rows: an antenna the MS lists
    whose data are all flagged is not part of the array that observed."""
    times, interval = cols["TIME"], cols["INTERVAL"]
    unique, first = np.unique(times, return_index=True)
    antennas = np.flatnonzero(antenna_rows > 0)
    both = np.isin(a1, antennas) & np.isin(a2, antennas)
    with table(str(ms / "ANTENNA"), ack=False) as ant:
        diameter = float(np.median(np.asarray(ant.getcol("DISH_DIAMETER"))[antennas]))
        names = ant.colnames()
        pads = (
            [f"{s}:{n}" for s, n in zip(ant.getcol("STATION"), ant.getcol("NAME"), strict=True)]
            if {"STATION", "NAME"} <= set(names)
            else []
        )
    width = float(np.median(window["width"]))
    effective = float(np.median(window["effective"])) or width
    return Exposure(
        antennas=int(antennas.size),
        dish_diameter_m=diameter,
        baselines=int(np.unique(a1[both] * 100_000 + a2[both]).size),
        integrations=int(unique.size),
        integration_s=float(np.median(interval)),
        time_on_source_s=float(interval[first].sum()),
        channels=int(window["freq"].size),
        channel_width_hz=width,
        noise_bandwidth_hz=effective,
        resolution_hz=float(np.median(window["resolution"])) or None,
        frequency_ghz=float(np.mean(window["freq"]) / 1e9),
        correlations=int(ncorr),
        flagged_fraction=float(flagged),
        high_weight_fraction=float(high_weight),
        elevation_deg=_elevation(table, ms, unique, fields),
        pads=[pads[a] for a in antennas] if pads else [],
        **_track(table, ms, unique, float(np.median(interval)), fields),
    )


def _track(
    table: Any, ms: Path, times: np.ndarray, integration_s: float, fields: list[int] | None
) -> dict[str, Any]:
    """The target's track: the hour angle at its middle, its span and its scans
    (runs of integrations without a gap longer than two integrations), as the
    simulation lays out one (``interferometry.observing.Track``)."""
    times = np.sort(np.asarray(times, dtype=np.float64))
    if times.size == 0:
        return {}
    scans = int(np.count_nonzero(np.diff(times) > 2.0 * integration_s)) + 1
    track: dict[str, Any] = {
        "track_span_s": float(times[-1] - times[0] + integration_s),
        "scans": scans,
    }
    try:
        from astropy import units as u
        from astropy.time import Time

        from ..interferometry.observing import alma_location

        with table(str(ms / "FIELD"), ack=False) as f:
            ra = float(np.asarray(f.getcell("PHASE_DIR", (fields or [0])[0])).ravel()[0])
        middle = Time(0.5 * (times[0] + times[-1]) / 86400.0, format="mjd")
        lst = middle.sidereal_time("apparent", longitude=alma_location().lon).to_value(u.rad)
        track["hour_angle_h"] = (
            float(np.remainder(lst - ra + np.pi, 2 * np.pi) - np.pi) * 12 / np.pi
        )
    except Exception:  # noqa: BLE001 - the hour angle is a by-product, not worth failing for
        pass
    return track


def _elevation(table: Any, ms: Path, times: np.ndarray, fields: list[int] | None) -> float | None:
    """The target's median elevation at ALMA over the observation (degrees)."""
    try:
        from astropy import units as u
        from astropy.coordinates import AltAz, SkyCoord
        from astropy.time import Time

        from ..interferometry.observing import alma_location

        with table(str(ms / "FIELD"), ack=False) as f:
            ra, dec = np.asarray(f.getcell("PHASE_DIR", (fields or [0])[0])).ravel()[:2]
        when = Time(np.percentile(times, [10, 50, 90]) / 86400.0, format="mjd")
        altaz = SkyCoord(ra * u.rad, dec * u.rad).transform_to(
            AltAz(obstime=when, location=alma_location())
        )
        return float(np.median(altaz.alt.deg))
    except Exception:  # noqa: BLE001 - the elevation is a by-product, not worth failing for
        return None


def _tsys(table: Any, ms: Path, spw: int, chan_freq: np.ndarray) -> Tsys:
    """ALMA's Tsys spectrum (SYSCAL) over the window: the median over antennas
    and times of the Tsys window on the same baseband that covers these
    channels, the mean of its hands; the Tsys window's rolled-off edge
    channels are left out."""
    if not (ms / "SYSCAL").exists():
        raise ValueError(f"{ms} has no SYSCAL table")
    with (
        table(str(ms / "SPECTRAL_WINDOW"), ack=False) as sw,
        table(str(ms / "SYSCAL"), ack=False) as sc,
    ):
        if sc.nrows() == 0 or "TSYS_SPECTRUM" not in sc.colnames():
            raise ValueError("SYSCAL holds no Tsys spectrum")
        baseband = sw.getcell("BBC_NO", spw)
        ids = [
            int(k)
            for k in np.unique(sc.getcol("SPECTRAL_WINDOW_ID"))
            if int(k) < sw.nrows() and sw.getcell("BBC_NO", int(k)) == baseband
        ]
        freqs = {k: np.asarray(sw.getcell("CHAN_FREQ", k)) for k in ids}
        ids = [
            k
            for k in ids
            if freqs[k].min() <= chan_freq.max() and freqs[k].max() >= chan_freq.min()
        ]
        if not ids:
            raise ValueError(f"no Tsys window covers spw {spw}")
        with sc.query(f"SPECTRAL_WINDOW_ID == {ids[0]}") as sel:
            raw = sel.getcol("TSYS_SPECTRUM")  # (nrow, nchan, npol)
    with np.errstate(all="ignore"):
        tsys = np.nanmedian(np.where(raw > 0, raw, np.nan), axis=0).mean(axis=-1)
    edge = max(2, tsys.size // 32)
    keep = np.zeros(tsys.size, dtype=bool)
    keep[edge:-edge] = True
    keep &= np.isfinite(tsys)
    if not keep.any():
        raise ValueError("SYSCAL's Tsys is empty over the window")
    f = freqs[ids[0]][keep]
    order = np.argsort(f)
    on_window = np.interp(chan_freq, f[order], tsys[keep][order])
    profile = on_window**2
    return Tsys(
        median_k=float(np.median(on_window)),
        profile=[round(float(p), 4) for p in profile / profile.mean()],
        spectral_window=ids[0],
    )


def rms_tsys_k(tsys: Tsys) -> float:
    """The rms of the window's Tsys spectrum: the Tsys the noise follows. The
    noise is the rms over the channels, each channel's noise following its own
    Tsys, so near an atmospheric line, where Tsys climbs across the window,
    the median Tsys understates it. The profile is Tsys^2 over its mean, so
    the median Tsys over the square root of the profile's median is the rms."""
    profile = np.asarray(tsys.profile, dtype=float)
    profile = profile[np.isfinite(profile) & (profile > 0)]
    if profile.size == 0:
        return tsys.median_k
    return float(tsys.median_k / np.sqrt(np.median(profile)))


def _own_dish_pairs(table: Any, ms: Path, sums: _Sums, exposure: Exposure) -> np.ndarray | None:
    """The baselines (ANTENNA1 * nant + ANTENNA2) between two of the array's
    own dishes (``exposure.dish_diameter_m``) when its antennas have dishes
    of two sizes (a 7 m array correlating the 12 m total-power dishes); None
    when they are all alike. A baseline with a larger dish is less noisy, and
    the efficiency is the array's own dish's."""
    with table(str(ms / "ANTENNA"), ack=False) as ant:
        diameters = np.asarray(ant.getcol("DISH_DIAMETER"), dtype=float)[: sums.nant]
    own = np.abs(diameters - exposure.dish_diameter_m) < 0.5
    array = sums.antenna_rows[: diameters.size] > 0
    if own[array].all():
        return None
    both = np.zeros((sums.nant, sums.nant), dtype=bool)
    both[: own.size, : own.size] = own[:, None] & own[None, :]
    return both.ravel()


def _visibility(
    sums: _Sums, exposure: Exposure, tsys: Tsys | None, pairs: np.ndarray | None = None
) -> VisibilityNoise:
    """One visibility's noise, over the baselines ``pairs`` selects (every
    baseline when None), and the efficiency it and Tsys imply."""
    total, count = sums.stokes_sum, sums.stokes_count
    if pairs is not None:
        total = float(sums.baseline["stokes_sum"][pairs].sum())
        count = int(sums.baseline["stokes_count"][pairs].sum())
    if count == 0:
        raise ValueError("no unflagged Stokes I channel differences")
    sigma = float(np.sqrt(total / count))
    unit = sigma * float(np.sqrt(exposure.integration_s * exposure.channel_width_hz))
    efficiency = None
    if tsys is not None and unit > 0:
        area = np.pi * (exposure.dish_diameter_m / 2.0) ** 2
        ratio = exposure.noise_bandwidth_hz / exposure.channel_width_hz
        efficiency = float(2 * K_B * rms_tsys_k(tsys) / (2 * area * unit) / JY / np.sqrt(ratio))
    return VisibilityNoise(
        sigma_jy=sigma,
        unit_jy=unit,
        hands_identical=bool(sums.hands_identical),
        efficiency=efficiency,
    )


def _antenna_shape(
    sums: _Sums, lines: np.ndarray, channel_width_hz: float
) -> tuple[AntennaShape | None, dict[str, np.ndarray]]:
    """Each antenna's and hand's noise shape: v = sum / count factors as
    A(a, p) C(c, p) L(a, c, p); its rms beyond sampling and the lag where its
    autocorrelation falls to half. Returns the summary and (g, h) as
    research/noise_realism's errors.antenna_noise_shape makes them."""
    from scipy.ndimage import uniform_filter1d

    z, n = sums.antenna_sum, sums.antenna_count
    pairs_ok = ~(lines[3:] | lines[:-3])
    total = n.sum(axis=1)
    seen = total.sum(axis=1) > 0
    if seen.sum() < 3:
        return None, {}
    used = seen & (total > 0.5 * np.median(total[seen], axis=0)).all(axis=1)
    ok = pairs_ok & (n[used] > 50).all(axis=(0, 2))
    if used.sum() < 3 or ok.sum() < 2 * SHAPE_SMOOTH:
        return None, {}
    v = z[used][:, ok] / n[used][:, ok]
    ant = np.ones((v.shape[0], 1, v.shape[2]))
    for _ in range(5):
        chan = (v / ant).mean(axis=0, keepdims=True)
        ant = (v / chan).mean(axis=1, keepdims=True)
    ant /= ant.mean()
    left = v / (ant * chan) - 1
    sampling = float(np.sqrt(np.mean(1.0 / n[used][:, ok])))
    excess = float(np.sqrt(max(float(np.var(left)) - sampling**2, 0.0)))
    x = left - left.mean(axis=1, keepdims=True)
    lags = np.arange(1, max(2, min(200, x.shape[1] // 2)))
    ac = np.array([np.mean(x[:, :-k] * x[:, k:]) for k in lags])
    below = ac < 0.5 * ac[0]
    half = int(lags[np.argmax(below)] if below.any() else lags[-1])

    def fill(values: np.ndarray, axis: int) -> np.ndarray:
        """``values`` on the good channels, smoothed and spread to all of them."""
        shape = list(values.shape)
        shape[axis] = len(ok)
        full, weight = np.zeros(shape), np.zeros(shape)
        index: list[Any] = [slice(None)] * len(shape)
        index[axis] = ok
        full[tuple(index)] = values
        weight[tuple(index)] = 1
        num = uniform_filter1d(full, SHAPE_SMOOTH, axis=axis, mode="nearest")
        den = uniform_filter1d(weight, SHAPE_SMOOTH, axis=axis, mode="nearest")
        return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)

    g = np.ones((z.shape[0], len(ok), z.shape[2]))
    g[used] = ant * np.exp(fill(np.log(np.clip(left + 1, 1e-6, None)), axis=1))
    hands = np.log(chan[0] / np.exp(np.log(chan[0]).mean(axis=1, keepdims=True)))
    h = np.exp(fill(hands, axis=0))
    summary = AntennaShape(
        antennas_used=int(used.sum()),
        channels_used=int(ok.sum()),
        excess_rms=excess,
        half_width_channels=half,
        half_width_mhz=float(half * channel_width_hz / 1e6),
    )
    pad = ((0, 0), (1, 2), (0, 0))  # the pairs are centred between channels k and k + 3
    return summary, {
        "antenna_g": np.pad(g, pad, mode="edge").astype(np.float32),
        "antenna_h": np.pad(h, pad[1:], mode="edge").astype(np.float32),
    }


def window_channels(ms: str | Path, spw: int) -> int:
    """How many channels a window has."""
    return int(_window(import_casacore_tables(), Path(ms), spw)["freq"].size)


def line_channels_from_log(log: Path, nchan: int) -> np.ndarray:
    """A window's line channels as ALMASim's imaging found them (its task log:
    "lines in channels a~b;c~d" or "none")."""
    lines = np.zeros(nchan, dtype=bool)
    for text in Path(log).read_text(errors="replace").splitlines():
        if "lines in channels" in text:
            spec = text.rsplit("lines in channels", 1)[1].strip()
            if spec != "none":
                lines[:] = False
                for part in spec.split(";"):
                    lo, _, hi = part.partition("~")
                    if lo.strip().isdigit():
                        lines[int(lo) : int(hi or lo) + 1] = True
    return lines


def parse_channels(spec: str, nchan: int) -> np.ndarray:
    """Channels from "a~b;c" (CASA's selection syntax)."""
    lines = np.zeros(nchan, dtype=bool)
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        lo, _, hi = part.partition("~")
        lines[int(lo) : int(hi or lo) + 1] = True
    return lines
