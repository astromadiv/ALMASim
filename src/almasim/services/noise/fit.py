"""Fit a real window's calibration residuals: the errors the noise model's
slow_phase, fast_phase, antenna_gains and baseline_errors components draw
(MADIV-28).

The data over the window's clean model, per (baseline, scan), where the model
is bright (S/N above 10 in the group): their phase scatter beyond what the
thermal noise alone gives, by baseline length, is the per-scan atmospheric
phase (a power law in baseline length); the loss of their amplitude below the
thermal one's is the phase that varies within a scan, |A| = exp(-sigma^2/2).
Split per scan into antenna terms, the amplitude scatter beyond what the
screens give is the antenna gains' error, and what does not close beyond the
screens and gains is the baselines' own.

One pass over the data. The references are built from it without reading it
again: a group's sum over its rows of w V M* (w = WEIGHT / scale) is, for data
G M plus thermal noise, sum G w |M|^2 plus a complex Gaussian of variance
kappa sum w |M|^2 per component, with kappa = 1 + 2 rho1 + 2 rho2 the noise's
correlation over neighbouring channels (window.json). The screens and gains
are drawn per real scan by the simulator's own engine
(interferometry.corruptions), so the fit and the simulation agree on what the
parameters mean. The statistics are research/noise_realism's (uvcompare,
fit_screen), which set the alma model's values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..imaging.ms_geometry import import_casacore_tables
from ..interferometry.corruptions import CorruptionModel
from .extract import PARALLEL, _correlations, _data_descriptions, _fields, _window
from .measurements import FitBin, PhaseFit, WindowFit, WindowNoise

C_M_S = 299_792_458.0
NBINS = 10
MIN_DEN = 100.0  # S/N 10 in a (baseline, scan) group
MIN_GROUPS = 20  # groups a uv bin needs to count in a fit
SLOPES = (0.05, 0.95)  # the screens' slopes; outside them the strength is refitted at the edge
FAST_MODEL_MISMATCH = 0.05  # data above the model by this much in ln|A|: the model does not match


@dataclass
class _Rows:
    """Per row (one baseline at one time): its group, time, antennas and sums."""

    group: list[np.ndarray] = field(default_factory=list)
    time: list[np.ndarray] = field(default_factory=list)
    scan: list[np.ndarray] = field(default_factory=list)
    a1: list[np.ndarray] = field(default_factory=list)
    a2: list[np.ndarray] = field(default_factory=list)
    den: list[np.ndarray] = field(default_factory=list)
    num: list[np.ndarray] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            k: np.concatenate(getattr(self, k))
            for k in ("group", "time", "scan", "a1", "a2", "den", "num")
        }


def fit_window(
    ms: str | Path,
    spw: int,
    field_selection: str | int | None,
    task_dir: str | Path,
    window: WindowNoise,
    *,
    continuum: str = "masked",
    line: str = "masked",
    realizations: int = 3,
    seed: int = 0,
    max_rows: int | None = None,
) -> WindowFit:
    """Fit one window's calibration residuals against its clean model
    (``task_dir``: an ALMASim imaging task, wsclean-model.fits and
    quality.json, and for lines wsclean-cube-model.fits and its line mask)."""
    table = import_casacore_tables()
    ms, task_dir = Path(ms), Path(task_dir)
    meta = _window(table, ms, spw)
    freq = meta["freq"]
    ddids = _data_descriptions(table, ms, spw)
    fields = _fields(table, ms, field_selection)
    planes, plane_of, px, py = model_planes(task_dir, freq, continuum=continuum, line=line)
    corr = _correlations(table, ms, ddids[0])
    hands = [k for k, c in enumerate(corr) if c in PARALLEL][:2] or [0]
    scale = window.thermal.scale
    kappa = max(1.0, 1.0 + 2.0 * window.thermal.rho1 + 2.0 * window.thermal.rho2)
    query = f"DATA_DESC_ID IN [{','.join(map(str, ddids))}] AND ANTENNA1 != ANTENNA2"
    if fields is not None:
        query += f" AND FIELD_ID IN [{','.join(map(str, fields))}]"

    rows = _Rows()
    with table(str(ms), ack=False) as main, main.query(query) as sel:
        nrow = sel.nrows()
        if nrow == 0:
            raise ValueError(f"{ms}: no cross-correlation rows in spw {spw}")
        uvw = sel.getcol("UVW")
        uvdist = np.hypot(uvw[:, 0], uvw[:, 1]) * float(np.mean(freq)) / C_M_S
        lo, hi = np.percentile(uvdist[uvdist > 0], [0.5, 99.5])
        edges = np.geomspace(lo, hi, NBINS + 1)
        step = 1 if not max_rows else max(1, int(np.ceil(nrow / max_rows)))
        chunk = max(16, 20_000_000 // max(1, freq.size * len(hands)))
        sampled = np.arange(0, nrow, step)
        for start in range(0, sampled.size, chunk):
            first, count = int(sampled[start]), int(min(chunk, sampled.size - start))
            take = slice(first, first + count * step, step)
            model = _predict(uvw[take], freq, planes, plane_of, px, py)
            _accumulate(
                rows,
                data=sel.getcol("DATA", first, count, step)[..., hands],
                flag=sel.getcol("FLAG", first, count, step)[..., hands],
                weight=_weights(sel.getcol("WEIGHT", first, count, step), hands),
                model=model,
                scale=scale,
                time=sel.getcol("TIME", first, count, step),
                scan=sel.getcol("SCAN_NUMBER", first, count, step),
                a1=sel.getcol("ANTENNA1", first, count, step),
                a2=sel.getcol("ANTENNA2", first, count, step),
            )
        with table(str(ms / "ANTENNA"), ack=False) as ant:
            xy = _enu(np.asarray(ant.getcol("POSITION"), dtype=np.float64))
        group_bin = np.clip(np.searchsorted(edges, uvdist) - 1, 0, NBINS - 1)
    r = rows.arrays()
    groups, inverse = np.unique(r["group"], return_inverse=True)
    den = np.bincount(inverse, r["den"], groups.size)
    num = np.bincount(inverse, r["num"].real, groups.size) + 1j * np.bincount(
        inverse, r["num"].imag, groups.size
    )
    first_row = np.zeros(groups.size, dtype=int)
    first_row[inverse[::-1]] = np.arange(inverse.size)[::-1]
    sampled_rows = sampled[: inverse.size]
    uvbin = group_bin[sampled_rows][first_row]
    group_scan = r["scan"][first_row]
    group_a1, group_a2 = r["a1"][first_row], r["a2"][first_row]
    bright = den > MIN_DEN
    notes: list[str] = []
    if bright.sum() < MIN_GROUPS:
        notes.append(f"{int(bright.sum())} bright (baseline, scan) groups: too faint to fit")

    freq_hz = float(np.mean(freq))
    baseline_m = np.sqrt(edges[:-1] * edges[1:]) * C_M_S / freq_hz
    keys = (group_scan[bright], group_a1[bright], group_a2[bright])
    real = _group_stats(num[bright], den[bright], uvbin[bright])
    real_solve = _antenna_solve(num[bright] / den[bright], *keys)
    ref = _Reference(
        rows=r,
        inverse=inverse,
        den=den,
        bright=bright,
        uvbin=uvbin[bright],
        keys=keys,
        xy=xy,
        wavelength_m=C_M_S / freq_hz,
        kappa=kappa,
        realizations=realizations,
        rng=np.random.default_rng(seed),
    )
    # In turn, each against a reference holding what was found before it: the
    # per-scan phase against the noise alone; the gains against that phase;
    # the in-scan phase against both (gain amplitude errors also lower the
    # mean ln|A|, on every baseline alike); the gains again with the in-scan
    # phase (its decorrelation also scatters the antennas' amplitudes); and
    # what does not close against all of them.
    thermal, _ = ref.stats(CorruptionModel())
    slow = _engine_fit(
        ref, real, thermal, CorruptionModel(),
        _power_law_fit(baseline_m, real, thermal, freq_hz, kind="phase"), kind="phase",
    )  # fmt: skip
    gain_amp = None
    if real_solve:
        gain_amp = _gain_excess(real_solve, ref.stats(_screens(slow, None, None))[1])
    elif bright.sum() >= MIN_GROUPS:
        notes.append("antenna terms: too few bright groups per scan to solve")
    fast = None
    if any(a is not None and a > FAST_MODEL_MISMATCH for a in real["lnamp_mean"]):
        notes.append("fast_phase: the data rise above the model, which does not match the source")
    else:
        base = _screens(slow, None, gain_amp)
        before_fast, _ = ref.stats(base)
        fast = _engine_fit(
            ref, real, before_fast, base,
            _power_law_fit(baseline_m, real, before_fast, freq_hz, kind="amplitude"),
            kind="amplitude",
        )  # fmt: skip
    baselines = None
    if real_solve:
        if fast is not None:
            gain_amp = _gain_excess(real_solve, ref.stats(_screens(slow, fast, None))[1])
        _, everything = ref.stats(_screens(slow, fast, gain_amp))
        baselines = {
            name: float(np.sqrt(max(real_solve[key] ** 2 - everything[key] ** 2, 0.0)))
            for key, name in (
                ("nonclosing_lnamp_rms", "amp_rms"),
                ("nonclosing_phase_rms_deg", "phase_rms_deg"),
            )
        }
    gains = gain_amp
    return WindowFit(
        name=window.name,
        bright_groups=int(bright.sum()),
        realizations=int(realizations),
        kappa=float(kappa),
        bins=[
            FitBin(
                baseline_m=float(b),
                groups=int(n),
                phase_rms_deg=pr,
                thermal_phase_rms_deg=pt,
                lnamp_mean=ar,
                thermal_lnamp_mean=at,
            )
            for b, n, pr, pt, ar, at in zip(
                baseline_m,
                real["n"],
                real["phase_rms_deg"],
                thermal["phase_rms_deg"],
                real["lnamp_mean"],
                thermal["lnamp_mean"],
                strict=True,
            )
        ],
        slow_phase=slow,
        fast_phase=fast,
        gain_amp_rms=gains,
        baseline_amp_rms=None if baselines is None else baselines["amp_rms"],
        baseline_phase_rms_deg=None if baselines is None else baselines["phase_rms_deg"],
        antenna_solve=real_solve or None,
        notes=notes,
    )


# -- the model ----------------------------------------------------------------


def model_planes(
    task_dir: Path, chan_freq: np.ndarray, *, continuum: str = "masked", line: str = "masked"
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The distinct sky planes (FITS order, even sizes) and each channel's plane:
    the continuum model plus the line plane at its frequency.

    ``continuum``: "all" the whole clean model; "masked" only where the real
    continuum image is above 5 sigma (widened by the beam): outside it the
    model is mostly noise peaks cleaning took for sources. ``line``: "all"
    the cube model; "masked" only in the line channels (the task log's) and
    inside the line mask; "none"."""
    from astropy.io import fits

    def planes(path: Path) -> tuple[np.ndarray, Any]:
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
            header = hdul[0].header
        while data.ndim > 3:
            data = data[0]
        return np.nan_to_num(data.reshape((-1,) + data.shape[-2:])), header

    cont, header = planes(task_dir / "wsclean-model.fits")
    if continuum == "masked":
        cont = cont * _emission_mask(task_dir)[None]
    distinct = [cont[0]]
    plane_of = np.zeros(len(chan_freq), dtype=int)
    cube_path = task_dir / "wsclean-cube-model.fits"
    if line != "none" and cube_path.exists():
        cube, lh = planes(cube_path)
        if cont.shape[-2:] != cube.shape[-2:]:
            raise ValueError("the continuum and cube models are on different grids")
        centres = lh["CRVAL3"] + (np.arange(cube.shape[0]) + 1 - lh["CRPIX3"]) * lh["CDELT3"]
        idx = np.abs(chan_freq[:, None] - centres[None, :]).argmin(axis=1)
        inside = np.abs(chan_freq - centres[idx]) <= abs(lh["CDELT3"]) / 2 + 1.0
        if line == "masked":
            from .extract import line_channels_from_log

            log = task_dir.parent / f"{task_dir.name}.log"
            lines = (
                line_channels_from_log(log, len(chan_freq))
                if log.exists()
                else np.zeros(len(chan_freq), bool)
            )
            mask_path = task_dir / "wsclean-cube-linemask.fits"
            inside &= lines & mask_path.exists()
            if mask_path.exists():
                cube = cube * (planes(mask_path)[0][:1] > 0)
        for k in np.unique(idx[inside]):
            if np.any(cube[k]):
                distinct.append(cont[0] + cube[k])
                plane_of[inside & (idx == k)] = len(distinct) - 1
    sky = _even(np.stack(distinct), header)
    deg = np.pi / 180
    return sky, plane_of, abs(header["CDELT1"]) * deg, abs(header["CDELT2"]) * deg


def _even(sky: np.ndarray, header: Any) -> np.ndarray:
    """Planes padded to even sizes with the reference pixel at n // 2, as the
    predictor takes them (an odd WSClean image has it at n // 2: one pixel
    before it keeps it there)."""
    pad = []
    for axis, key in ((1, "CRPIX2"), (2, "CRPIX1")):
        n = sky.shape[axis]
        ref = int(round(float(header.get(key, n // 2 + 1)))) - 1
        if n % 2 == 0 and ref == n // 2:
            pad.append((0, 0))
        elif n % 2 and ref == n // 2:
            pad.append((1, 0))
        elif n % 2 and ref == n // 2 + 1:
            pad.append((0, 1))
        else:
            raise ValueError(f"model image: reference pixel {ref} of {n} is not at its centre")
    return np.pad(sky, ((0, 0), pad[0], pad[1]))


def _emission_mask(task_dir: Path, sigma: float = 5.0) -> np.ndarray:
    """Where the real continuum image has emission above ``sigma`` times its
    noise (quality.json), widened by the beam's major axis."""
    from astropy.io import fits
    from scipy.ndimage import binary_dilation

    with fits.open(task_dir / "wsclean-image.fits") as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
        cell_arcsec = abs(hdul[0].header["CDELT2"]) * 3600
    while image.ndim > 2:
        image = image[0]
    quality = json.loads((task_dir / "quality.json").read_text())["continuum"]
    radius = max(1, int(round(quality["beam_arcsec"][0] / cell_arcsec)))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return binary_dilation(
        np.nan_to_num(image) > sigma * quality["noise_jy_beam"], x**2 + y**2 <= radius**2
    )


def _predict(uvw, freq, planes, plane_of, px, py) -> np.ndarray:
    """Model visibilities (nrows, nchan): each group of channels from its plane."""
    from ..interferometry.predict import predict_visibilities

    vis = np.zeros((len(uvw), len(freq)), dtype=np.complex128)
    for k in np.unique(plane_of):
        chans = np.flatnonzero(plane_of == k)
        chans = chans[np.argsort(freq[chans])]  # ducc0 takes rising frequencies
        if np.any(planes[k]):
            vis[:, chans] = predict_visibilities(uvw, freq[chans], planes[k], px, py)
    return vis


def _weights(weight: np.ndarray, hands: list[int]) -> np.ndarray:
    if weight.shape[1] <= max(hands):
        return np.repeat(weight.mean(axis=1, keepdims=True), len(hands), axis=1)
    return weight[:, hands]


def _enu(positions_itrf: np.ndarray) -> np.ndarray:
    """Antenna positions in a local east-north plane (m)."""
    centre = positions_itrf.mean(axis=0)
    lon = np.arctan2(centre[1], centre[0])
    lat = np.arctan2(centre[2], np.hypot(centre[0], centre[1]))
    d = positions_itrf - centre
    east = -np.sin(lon) * d[:, 0] + np.cos(lon) * d[:, 1]
    north = (
        -np.sin(lat) * np.cos(lon) * d[:, 0]
        - np.sin(lat) * np.sin(lon) * d[:, 1]
        + np.cos(lat) * d[:, 2]
    )
    return np.stack([east, north], axis=1)


def _accumulate(rows: _Rows, *, data, flag, weight, model, scale, time, scan, a1, a2) -> None:
    """One chunk's rows: sum over channels and hands of w V M* and w |M|^2."""
    w = np.where(~flag & (weight[:, None, :] > 0), weight[:, None, :] / scale, 0.0)
    m = model[:, :, None]
    rows.num.append((w * data * np.conj(m)).sum(axis=(1, 2)))
    rows.den.append((w * np.abs(m) ** 2).sum(axis=(1, 2)))
    rows.group.append((a1.astype(np.int64) * 10_000 + a2) * 1_000_000 + scan.astype(np.int64))
    rows.time.append(np.asarray(time, dtype=np.float64))
    rows.scan.append(np.asarray(scan, dtype=np.int64))
    rows.a1.append(np.asarray(a1, dtype=np.int64))
    rows.a2.append(np.asarray(a2, dtype=np.int64))


# -- statistics ---------------------------------------------------------------


def _with_noise(signal: np.ndarray, den: np.ndarray, kappa: float, rng) -> np.ndarray:
    """``signal`` (sum G w|M|^2 over a group) plus its thermal noise."""
    sigma = np.sqrt(kappa * den)
    return signal + sigma * (rng.standard_normal(den.size) + 1j * rng.standard_normal(den.size))


def _group_stats(num: np.ndarray, den: np.ndarray, uvbin: np.ndarray) -> dict[str, list]:
    """Per uv bin: the groups, their phase rms (deg) and mean ln|A|."""
    ratio = num / den
    out: dict[str, list] = {"n": [], "phase_rms_deg": [], "lnamp_mean": []}
    for b in range(NBINS):
        x = ratio[uvbin == b]
        out["n"].append(int(x.size))
        enough = x.size > 5
        out["phase_rms_deg"].append(
            float(np.degrees(np.sqrt(np.mean(np.angle(x) ** 2)))) if enough else None
        )
        out["lnamp_mean"].append(float(np.mean(np.log(np.abs(x)))) if enough else None)
    return out


def _mean_stats(runs: list[dict[str, list]]) -> dict[str, list]:
    """Realisations averaged: rms in quadrature, means as means."""
    out: dict[str, list] = {"n": runs[0]["n"]}
    for key, quadrature in (("phase_rms_deg", True), ("lnamp_mean", False)):
        values: list[float | None] = []
        for b in range(NBINS):
            xs = [r[key][b] for r in runs if r[key][b] is not None]
            if not xs:
                values.append(None)
            elif quadrature:
                values.append(float(np.sqrt(np.mean(np.square(xs)))))
            else:
                values.append(float(np.mean(xs)))
        out[key] = values
    return out


def _power_law_fit(
    baseline_m: np.ndarray, real: dict, reference: dict, freq_hz: float, *, kind: str
) -> PhaseFit | None:
    """A screen sigma(b) = sigma_1km (b / 1 km)^slope from the bins with enough
    groups. ``kind`` "phase": the phase rms beyond the reference's, in
    quadrature, fitted in log-log. "amplitude": the in-scan phase from the
    loss of ln|A| below the reference's, sigma^2 / 2, fitted with a constant
    besides (:func:`_decorrelation_fit`)."""
    b, y = [], []
    for k in range(NBINS):
        if real["n"][k] < MIN_GROUPS:
            continue
        if kind == "phase":
            pr, pt = real["phase_rms_deg"][k], reference["phase_rms_deg"][k]
            if pr is None or pt is None or pr <= pt:
                continue
            b.append(baseline_m[k])
            y.append(np.radians(np.sqrt(pr**2 - pt**2)))
        else:
            ar, at = real["lnamp_mean"][k], reference["lnamp_mean"][k]
            if ar is None or at is None:
                continue
            b.append(baseline_m[k])
            y.append(at - ar)
    if kind == "amplitude":
        fitted = _decorrelation_fit(np.asarray(b) / 1_000, np.asarray(y))
        if fitted is None:
            return None
        slope, sigma_1km = float(fitted[0]), float(fitted[1])
    else:
        if len(b) < 3:
            return None
        x, logs = np.log(np.asarray(b) / 1_000), np.log(y)
        slope, intercept = np.polyfit(x, logs, 1)
        if not SLOPES[0] <= slope <= SLOPES[1]:
            slope = min(max(slope, SLOPES[0]), SLOPES[1])
            intercept = float(np.mean(logs - slope * x))
        sigma_1km = float(np.exp(intercept))
    wavelength = C_M_S / freq_hz
    return PhaseFit(
        bins=len(b),
        slope=float(slope),
        phase_1km_deg=float(np.degrees(sigma_1km)),
        path_um_1km=float(sigma_1km * wavelength / (2 * np.pi) * 1e6),
    )


ENGINE_SLOPES = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)


def _engine_fit(
    ref: _Reference,
    real: dict,
    base: dict,
    base_model: CorruptionModel,
    initial: PhaseFit | None,
    *,
    kind: str,
) -> PhaseFit | None:
    """The screen the simulator's engine must draw to reproduce the real data:
    for each slope, the engine's screen at the initial strength P0 (on top of
    ``base_model``) gives each bin's excess over ``base`` (phase variance, or
    ln|A| loss), which grows as the strength squared; the strength is the
    least-squares scale of that to the real excess (with a constant beside it
    for the amplitude, as in :func:`_decorrelation_fit`), at the slope that
    fits best. None without an initial screen or a positive scale."""
    from dataclasses import replace

    if initial is None:
        return None
    p0 = initial.path_um_1km
    best: tuple[float, float, float, int] | None = None
    for slope in ENGINE_SLOPES:
        if kind == "phase":
            trial = replace(base_model, slow_path_um_1km=p0, slow_slope=slope)
        else:
            trial = replace(base_model, fast_path_um_1km=p0, fast_slope=slope)
        sim, _ = ref.stats(trial)
        y, x = _excess(real, base, sim, kind)
        if y.size < 3:
            continue
        design = np.stack([x, np.ones_like(x)], axis=1) if kind == "amplitude" else x[:, None]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        a = float(coef[0])
        if a <= 0:
            continue
        sse = float(np.sum((y - design @ coef) ** 2))
        if kind == "amplitude":
            scatter = np.sqrt(sse / max(y.size - 2, 1))
            if a * float(np.ptp(x)) <= 3 * scatter:
                continue
        if best is None or sse < best[0]:
            best = (sse, float(slope), a, int(y.size))
    if best is None:
        return None
    _, slope, a, bins = best
    path = p0 * np.sqrt(a)
    return PhaseFit(
        bins=bins,
        slope=slope,
        phase_1km_deg=float(np.degrees(2 * np.pi * path * 1e-6 / ref.wavelength_m)),
        path_um_1km=float(path),
    )


def _excess(real: dict, base: dict, sim: dict, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Per bin with enough groups: the real data's excess over ``base`` and the
    simulated screen's (phase variance in rad^2, or ln|A| loss)."""
    y, x = [], []
    for k in range(NBINS):
        if real["n"][k] < MIN_GROUPS:
            continue
        if kind == "phase":
            values = (real["phase_rms_deg"][k], base["phase_rms_deg"][k], sim["phase_rms_deg"][k])
            if None in values:
                continue
            r, b, m = (np.radians(v) for v in values)
            y.append(r**2 - b**2)
            x.append(m**2 - b**2)
        else:
            values = (real["lnamp_mean"][k], base["lnamp_mean"][k], sim["lnamp_mean"][k])
            if None in values:
                continue
            r, b, m = values
            y.append(b - r)
            x.append(b - m)
    return np.asarray(y, dtype=float), np.asarray(x, dtype=float)


def _decorrelation_fit(b_km: np.ndarray, loss: np.ndarray) -> tuple[float, float] | None:
    """(slope, sigma_1km) of loss(b) = c + sigma_1km^2 (b / 1 km)^(2 slope) / 2.

    The constant c takes what lowers ln|A| on every baseline alike (the
    antennas' amplitude errors, a model whose flux is a little off), which
    is not decorrelation; the slope is searched over the screens' range. None
    unless the loss grows with baseline by more than three times the fit's
    residual scatter."""
    if b_km.size < 4:
        return None
    best: tuple[float, float, float] | None = None
    for slope in np.linspace(SLOPES[0], SLOPES[1], 19):
        design = np.stack([np.ones_like(b_km), 0.5 * b_km ** (2 * slope)], axis=1)
        (c, v), *_ = np.linalg.lstsq(design, loss, rcond=None)
        if v <= 0:
            continue
        sse = float(np.sum((loss - design @ (c, v)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(slope), float(v))
    if best is None:
        return None
    best_sse, best_slope, best_v = best
    rise = 0.5 * best_v * (b_km.max() ** (2 * best_slope) - b_km.min() ** (2 * best_slope))
    scatter = np.sqrt(best_sse / max(b_km.size - 3, 1))
    if rise <= 3 * scatter:
        return None
    return best_slope, float(np.sqrt(best_v))


def _antenna_solve(ratio: np.ndarray, scan, a1, a2) -> dict[str, float]:
    """Per scan, ln of each bright group's data over model split into antenna
    terms by least squares, ln A_ij = ln g_i + conj(ln g_j): the rms of the
    antennas' amplitude and of what does not close (amplitude and phase)."""
    amp, res_a, res_p = [], [], []
    for s in np.unique(scan):
        on = scan == s
        ants = np.union1d(a1[on], a2[on])
        if on.sum() < 2 * ants.size:
            continue
        index = {a: k for k, a in enumerate(ants.tolist())}
        design = np.zeros((on.sum(), ants.size))
        dphase = np.zeros((on.sum(), ants.size))
        for k, (i, j) in enumerate(zip(a1[on].tolist(), a2[on].tolist(), strict=True)):
            design[k, index[i]] = design[k, index[j]] = 1
            dphase[k, index[i]], dphase[k, index[j]] = 1, -1
        la, lp = np.log(np.abs(ratio[on])), np.angle(ratio[on])
        ga = np.linalg.lstsq(design, la, rcond=None)[0]
        gp = np.linalg.lstsq(dphase, lp, rcond=None)[0]
        amp.extend(ga - ga.mean())
        res_a.extend(la - design @ ga)
        res_p.extend(np.angle(np.exp(1j * (lp - dphase @ gp))))
    if not amp:
        return {}
    return {
        "ant_lnamp_rms": float(np.std(amp)),
        "nonclosing_lnamp_rms": float(np.std(res_a)),
        "nonclosing_phase_rms_deg": float(np.degrees(np.std(res_p))),
    }


def _screens(
    slow: PhaseFit | None, fast: PhaseFit | None, gain_amp: float | None
) -> CorruptionModel:
    """The simulator's corruption model of what was fitted so far."""
    return CorruptionModel(
        slow_path_um_1km=slow.path_um_1km if slow else 0.0,
        slow_slope=slow.slope if slow else 5 / 6,
        fast_path_um_1km=fast.path_um_1km if fast else 0.0,
        fast_slope=fast.slope if fast else 5 / 6,
        gain_amp_rms=gain_amp or 0.0,
    )


def _gain_excess(real: dict[str, float], reference: dict[str, float]) -> float:
    """The antennas' amplitude scatter beyond the reference's, in quadrature."""
    excess = real["ant_lnamp_rms"] ** 2 - reference.get("ant_lnamp_rms", 0.0) ** 2
    return float(np.sqrt(excess)) if excess > 0 else 0.0


@dataclass
class _Reference:
    """The bright groups of the model times a corruption model's errors plus
    thermal noise, without reading the data again."""

    rows: dict[str, np.ndarray]
    inverse: np.ndarray
    den: np.ndarray
    bright: np.ndarray
    uvbin: np.ndarray
    keys: tuple[np.ndarray, np.ndarray, np.ndarray]
    xy: np.ndarray
    wavelength_m: float
    kappa: float
    realizations: int
    rng: np.random.Generator

    def stats(self, model: Any) -> tuple[dict[str, list], dict[str, float]]:
        """The per-bin statistics and the antenna solve, over the realisations
        (rms in quadrature, means as means)."""
        bins, solves = [], []
        for _ in range(self.realizations):
            num = _with_noise(self._signal(model), self.den[self.bright], self.kappa, self.rng)
            ratio = num / self.den[self.bright]
            bins.append(_group_stats(num, self.den[self.bright], self.uvbin))
            solves.append(_antenna_solve(ratio, *self.keys))
        solve = (
            {k: float(np.sqrt(np.mean([s[k] ** 2 for s in solves]))) for k in solves[0]}
            if solves[0]
            else {}
        )
        return _mean_stats(bins), solve

    def _signal(self, model: Any) -> np.ndarray:
        """Each bright group's sum over its rows of G w |M|^2."""
        from ..interferometry.corruptions import draw_antenna_corruptions

        r, bright_rows = self.rows, self.bright[self.inverse]
        if not model.has_antenna_errors:
            return self.den[self.bright].astype(np.complex128)
        gain = np.ones(self.inverse.size, dtype=np.complex128)
        for s in np.unique(r["scan"][bright_rows]):
            on = bright_rows & (r["scan"] == s)
            times, t_index = np.unique(r["time"][on], return_inverse=True)
            errors = draw_antenna_corruptions(model, self.xy, times - times[0], self.rng)
            phase = errors.phase + 2 * np.pi * errors.path_m / self.wavelength_m
            g = errors.amplitude * np.exp(1j * phase)
            gain[on] = g[r["a1"][on], t_index] * np.conj(g[r["a2"][on], t_index])
        weighted = gain * r["den"]
        size = self.den.size
        signal = np.bincount(self.inverse, weighted.real, size) + 1j * np.bincount(
            self.inverse, weighted.imag, size
        )
        return signal[self.bright]
