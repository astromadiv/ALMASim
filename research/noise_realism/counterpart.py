"""Make the simulated counterpart of a real spectral window's MeasurementSet.

    python counterpart.py base REAL_MS SPW FIELD BASE_MS
        Split the window and field out of the real MS (CASA split, flags kept),
        and measure the real data's noise: its scatter against WEIGHT, and the
        correlation of adjacent channels (written to BASE_MS/noise.json).

    python counterpart.py make BASE_MS TASK_DIR OUT_MS --errors ERRORS.json --seed N
        Copy BASE_MS, and replace its DATA by the sky the real imaging found
        (TASK_DIR's continuum model plus its line-cube model, at each MS
        channel), corrupted by the error model, plus thermal noise. The uv
        coverage, flags and weights stay the real ones.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from almasim.services.interferometry.predict import predict_visibilities  # noqa: E402
from errors import (  # noqa: E402
    C,
    AntennaGains,
    BaselineErrors,
    ErrorModel,
    FastScreen,
    PhaseScreen,
    ScanScreen,
    bandpass_factors,
    spectral_kernel,
    thermal_noise,
)

CASA_DATA = "/SKAO_data/cycle11/.casa-data"
PARALLEL = {5, 8, 9, 12}  # RR, LL, XX, YY
CHUNK = 100_000


def _tables():
    from casacore.tables import table

    return table


def split_window(real_ms: Path, spw: int, field: str, base_ms: Path) -> None:
    from almasim.services.casa.environment import configure_casa_environment

    configure_casa_environment(base_ms.parent / ".casa-run", CASA_DATA)
    from casatasks import split

    if base_ms.exists():
        shutil.rmtree(base_ms)
    split(
        vis=str(real_ms),
        outputvis=str(base_ms),
        spw=str(spw),
        field=str(field),
        datacolumn="data",
        keepflags=True,
    )


def measure_noise(base_ms: Path, samples: int = 40_000_000, exclude: np.ndarray | None = None) -> dict:
    """Per-component noise variance against WEIGHT, from channel differences.

    Differences three channels apart remove the sky (smooth on that scale)
    and are uncorrelated even for Hanning-smoothed channels; one and two
    apart they show the correlation of neighbours:
    var(V_k - V_k+l) = 2 sigma^2 (1 - rho_l).

    Rows are taken evenly over the whole observation (every n-th row, up to
    about ``samples`` visibilities): one block of rows can sit at another
    elevation or weather than the rest, where WEIGHT tracks the noise less well.

    ``exclude``: channels left out of the level and the correlation (the real
    line channels: there the differences hold sky as well as noise).
    """
    table = _tables()
    with table(str(base_ms), ack=False) as t:
        n = t.nrows()
        shape = t.getcell("DATA", 0).shape  # (nchan, ncorr)
        step = max(1, int(np.ceil(n * shape[0] * shape[1] / samples)))
        data = t.getcol("DATA", rowincr=step)  # (nrow, nchan, ncorr)
        flag = t.getcol("FLAG", rowincr=step)
        weight = t.getcol("WEIGHT", rowincr=step)  # (nrow, ncorr)
        a1, a2 = t.getcol("ANTENNA1", rowincr=step), t.getcol("ANTENNA2", rowincr=step)
    cross = a1 != a2
    data, flag, weight = data[cross], flag[cross], weight[cross]
    if weight.shape[1] != data.shape[2]:
        weight = np.repeat(weight.mean(axis=1, keepdims=True), data.shape[2], axis=1)
    good = ~flag & (weight[:, None, :] > 0)

    def lag_var(lag: int) -> np.ndarray:
        """Per channel, the mean of |V_k - V_k+lag|^2 / 4 x WEIGHT: what the
        noise variance would be at WEIGHT 1, less the correlation at that lag.
        Values above 20 times their channel's median (a line's edge, RFI;
        noise alone exceeds it once in a million) are left out."""
        diff = data[:, lag:] - data[:, :-lag]
        ok = good[:, lag:] & good[:, :-lag]
        v = np.where(ok, (diff.real**2 + diff.imag**2) / 4 * weight[:, None, :], np.nan)
        v = np.where(v > 20 * np.nanmedian(v, axis=(0, 2), keepdims=True), np.nan, v)
        per_chan = np.nanmean(v, axis=(0, 2))  # n - lag pairs, centred between their channels
        return np.pad(per_chan, ((lag + 1) // 2, lag // 2), mode="edge")

    # Means, not medians: an image's noise is a weighted mean of the visibilities'
    # variances. Lag 3 (uncorrelated even for Hanning-smoothed channels) gives the
    # level and its variation across the window; lags 1 and 2 the correlation.
    var1, var2, var3 = lag_var(1), lag_var(2), lag_var(3)
    usable = np.isfinite(var3) & np.isfinite(var1) & np.isfinite(var2)
    if exclude is not None:
        usable &= ~exclude
    scale = float(np.mean(var3[usable]))
    rho1 = float(1 - np.mean(var1[usable]) / scale)
    rho2 = float(1 - np.mean(var2[usable]) / scale)
    profile = np.where(np.isfinite(var3), var3 / scale, 1.0)
    return {
        "scale": scale,
        "rho1": rho1,
        "rho2": rho2,
        "rows_sampled": int(len(data)),
        "row_step": step,
        "estimator": "clipped mean",
        "channel_profile": [round(float(p), 4) for p in profile],
    }


def tsys_profile(real_ms: Path, spw: int, chan_freq: np.ndarray) -> np.ndarray:
    """ALMA's own Tsys spectrum (SYSCAL) over the window, squared: how the noise
    variance varies across it at fixed WEIGHT. The median over antennas and
    times of the Tsys window on the same baseband that covers these channels,
    the mean of its polarisations."""
    table = _tables()
    with table(str(real_ms / "SPECTRAL_WINDOW"), ack=False) as sw:
        baseband = sw.getcell("BBC_NO", spw)
        with table(str(real_ms / "SYSCAL"), ack=False) as sc:
            ids = [int(k) for k in np.unique(sc.getcol("SPECTRAL_WINDOW_ID")) if sw.getcell("BBC_NO", int(k)) == baseband]
            freqs = {k: np.asarray(sw.getcell("CHAN_FREQ", k)) for k in ids}
            ids = [k for k in ids if freqs[k].min() <= chan_freq.max() and freqs[k].max() >= chan_freq.min()]
            if not ids:
                raise ValueError(f"no Tsys window covers spw {spw}")
            with sc.query(f"SPECTRAL_WINDOW_ID == {ids[0]}") as sel:
                tsys = sel.getcol("TSYS_SPECTRUM")  # (nrow, nchan, npol)
    tsys = np.nanmedian(np.where(tsys > 0, tsys, np.nan), axis=0).mean(axis=-1)
    # The Tsys window's own edge channels roll off (to tens of times the rest):
    # left out, the science window's ends take the nearest kept value.
    edge = max(2, tsys.size // 32)
    keep = np.zeros(tsys.size, dtype=bool)
    keep[edge:-edge] = True
    f = freqs[ids[0]][keep]
    order = np.argsort(f)
    return np.interp(chan_freq, f[order], tsys[keep][order]) ** 2


def remeasure(base_ms: Path, task_dir: Path, real_ms: Path, spw: int) -> dict:
    """The noise outside the real line channels, its shape across the window from
    SYSCAL (noise.tsys.json): the level, correlation and profile without the sky
    that line channels add to the channel differences."""
    from scipy.ndimage import binary_dilation

    table = _tables()
    with table(str(base_ms / "SPECTRAL_WINDOW"), ack=False) as sw:
        chan_freq = np.asarray(sw.getcell("CHAN_FREQ", 0), dtype=np.float64)
    nchan = chan_freq.size
    lines = binary_dilation(_line_channels(task_dir, nchan), iterations=max(2, nchan // 100))
    noise = measure_noise(base_ms, exclude=lines)
    clean = ~lines
    if clean.sum() < max(8, nchan // 10):
        clean = np.ones(nchan, dtype=bool)  # a window that is line throughout: all of it
    profile = tsys_profile(real_ms, spw, chan_freq)
    var3 = np.asarray(noise["channel_profile"]) * noise["scale"]  # the measured level per channel
    profile = profile / np.mean(profile[clean])
    noise.update(
        scale=float(np.mean(var3[clean]) / np.mean(profile[clean])),
        channel_profile=[round(float(p), 4) for p in profile],
        profile_source="SYSCAL Tsys^2",
        line_channels_excluded=int(lines.sum()),
    )
    return noise


def _line_channels(task_dir: Path, nchan: int) -> np.ndarray:
    """The window's line channels, as the real run's continuum subtraction found them
    (its task log: "lines in channels a~b;c~d" or "none")."""
    lines = np.zeros(nchan, dtype=bool)
    log = task_dir.parent / f"{task_dir.name}.log"
    for text in log.read_text(errors="replace").splitlines():
        if "lines in channels" in text:
            spec = text.rsplit("lines in channels", 1)[1].strip()
            if spec != "none":
                for part in spec.split(";"):
                    lo, _, hi = part.partition("~")
                    lines[int(lo) : int(hi or lo) + 1] = True
    return lines


def _emission_mask(task_dir: Path, sigma: float = 5.0) -> np.ndarray:
    """Where the real continuum image has emission: above ``sigma`` times its noise
    (quality.json), widened by the beam's major axis. Outside it the clean model
    is mostly noise peaks that cleaning took for sources."""
    from astropy.io import fits
    from scipy.ndimage import binary_dilation

    with fits.open(task_dir / "wsclean-image.fits") as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
        cell_arcsec = abs(hdul[0].header["CDELT2"]) * 3600
    while image.ndim > 2:
        image = image[0]
    continuum = json.loads((task_dir / "quality.json").read_text())["continuum"]
    radius = max(1, int(round(continuum["beam_arcsec"][0] / cell_arcsec)))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return binary_dilation(np.nan_to_num(image) > sigma * continuum["noise_jy_beam"], x**2 + y**2 <= radius**2)


def _model_planes(
    task_dir: Path,
    chan_freq: np.ndarray,
    line: bool | str = True,
    continuum: bool | str = True,
    continuum_dir: Path | None = None,
) -> tuple[np.ndarray, float, float]:
    """The distinct sky planes (nplane, ny, nx) and each MS channel's plane:
    continuum plus the line plane at its frequency.

    ``line``: True, the whole cube model; False, none; "masked", only in the line
    channels and inside the line mask the real run found (the rest of the cube
    model is noise that cleaning took for signal, which the counterpart's own
    noise would add to). ``continuum``: True, the whole continuum model;
    "masked", only where the real image has emission (:func:`_emission_mask`).
    ``continuum_dir``: take the continuum model (and its emission mask) from
    another imaging of the real data, e.g. a deeper, multi-scale one.
    """
    from astropy.io import fits

    def planes(path):
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
            header = hdul[0].header
        while data.ndim > 3:
            data = data[0]
        return np.nan_to_num(data.reshape((-1,) + data.shape[-2:])), header

    cont_dir = continuum_dir or task_dir
    cont, ch = planes(cont_dir / "wsclean-model.fits")
    if continuum == "masked":
        cont = cont * _emission_mask(cont_dir)[None]
    # Each distinct plane once: the continuum (plane 0), and continuum plus each
    # line plane a channel uses; plane_of maps every MS channel to its plane
    # (one plane per native channel would take hundreds of GB on large windows).
    distinct = [cont[0]]
    plane_of = np.zeros(len(chan_freq), dtype=int)
    if line:
        cube, lh = planes(task_dir / "wsclean-cube-model.fits")
        if cont.shape[-2:] != cube.shape[-2:]:
            raise ValueError("continuum and cube models are on different grids")
        centres = lh["CRVAL3"] + (np.arange(cube.shape[0]) + 1 - lh["CRPIX3"]) * lh["CDELT3"]
        width = abs(lh["CDELT3"])
        idx = np.abs(chan_freq[:, None] - centres[None, :]).argmin(axis=1)
        inside = np.abs(chan_freq - centres[idx]) <= width / 2 + 1.0
        if line == "masked":
            mask_path = task_dir / "wsclean-cube-linemask.fits"
            inside &= _line_channels(task_dir, len(chan_freq)) & mask_path.exists()
            if mask_path.exists():
                cube = cube * (planes(mask_path)[0][:1] > 0)
        for k in np.unique(idx[inside]):
            if np.any(cube[k]):
                distinct.append(cont[0] + cube[k])
                plane_of[inside & (idx == k)] = len(distinct) - 1
    # ducc0 reads a plane with its axes the other way round from FITS: handed a
    # FITS (y, x) array it images it rotated by 90 degrees, (y, x) -> (n - x, y).
    # This is the plane that WSClean images back where the FITS model has it
    # (checked with a point source off centre; see orient/test.py in the notes).
    sky = np.roll(np.stack(distinct)[:, ::-1, :], 1, axis=1).transpose(0, 2, 1)
    deg = np.pi / 180
    return sky, plane_of, abs(ch["CDELT1"]) * deg, abs(ch["CDELT2"]) * deg


def _continuum_dir(root: str | None, ms: Path) -> Path | None:
    """extra.continuum_dir: an iteration's cubes/ holding the window's own imaging."""
    if not root:
        return None
    found = Path(root) / ms.name.removesuffix(".ms") / "SPW-0_FIELD-0"
    if not (found / "wsclean-model.fits").exists():
        raise FileNotFoundError(f"no continuum imaging of {ms.name} under {root}")
    return found


def _predict(uvw, chan_freq, sky, plane_of, px, py, nthreads) -> np.ndarray:
    """Model visibilities (nrows, nchan): each group of channels from its shared plane."""
    vis = np.zeros((len(uvw), len(chan_freq)), dtype=np.complex128)
    for k in np.unique(plane_of):
        chans = np.flatnonzero(plane_of == k)
        chans = chans[np.argsort(chan_freq[chans])]  # ducc0 takes frequencies in ascending order
        if np.any(sky[k]):
            vis[:, chans] = predict_visibilities(uvw, chan_freq[chans], sky[k], px, py, nthreads=nthreads)
    return vis


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


class Simulator:
    """The counterpart of one window, chunk by chunk: the model visibilities,
    and the same corrupted by the error model plus thermal noise.

    Opened on a table (the base MS, or a copy of it); every random draw comes
    from one Generator seeded by ``seed``, in a fixed order (the antenna
    gains, the screen, the bandpass, then each chunk's noise in row order).
    """

    def __init__(self, ms: Path, task_dir: Path, model: ErrorModel, seed: int, t) -> None:
        table = _tables()
        self.model = model
        # extra.noise "tsys": the level outside the line channels, the shape from SYSCAL.
        noise_file = "noise.tsys.json" if model.extra.get("noise") == "tsys" else "noise.json"
        self.noise = json.loads((ms / noise_file).read_text())
        self.rng = np.random.default_rng(seed)
        # The window and polarisation setup the rows use (the split MS has one data
        # description; its polarisation row need not be the first).
        with table(str(ms / "DATA_DESCRIPTION"), ack=False) as dd:
            spw_id, pol_id = int(dd.getcell("SPECTRAL_WINDOW_ID", 0)), int(dd.getcell("POLARIZATION_ID", 0))
        with table(str(ms / "SPECTRAL_WINDOW"), ack=False) as sw:
            self.chan_freq = np.asarray(sw.getcell("CHAN_FREQ", spw_id), dtype=np.float64)
        with table(str(ms / "POLARIZATION"), ack=False) as pol:
            self.corr = np.asarray(pol.getcell("CORR_TYPE", pol_id))
        with table(str(ms / "ANTENNA"), ack=False) as ant:
            self.xy = _enu(np.asarray(ant.getcol("POSITION"), dtype=np.float64))
        self.hands = np.flatnonzero(np.isin(self.corr, list(PARALLEL)))
        self.npol = len(self.hands)
        # extra.line_model false: the continuum model only (no line emission).
        self.sky, self.plane_of, self.px, self.py = _model_planes(
            task_dir,
            self.chan_freq,
            model.extra.get("line_model", True),
            model.extra.get("continuum_model", True),
            _continuum_dir(model.extra.get("continuum_dir"), ms),
        )
        self.nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
        self.kernel = (
            spectral_kernel(self.noise["rho1"], self.noise["rho2"])
            if model.spectral_kernel == "measured"
            else model.spectral_kernel
        )
        scans = np.unique(t.getcol("SCAN_NUMBER"))
        times = t.getcol("TIME")
        antennas = np.union1d(np.unique(t.getcol("ANTENNA1")), np.unique(t.getcol("ANTENNA2")))
        self.gains = AntennaGains.draw(self.rng, model, antennas, scans, self.npol)
        self.screen = (
            PhaseScreen(self.rng, model, self.xy, float(np.ptp(times)) + 1.0)
            if model.screen_path_um_1km > 0
            else None
        )
        self.bandpass = (
            bandpass_factors(self.rng, model, antennas, len(self.chan_freq), self.npol)
            if model.bandpass_amp_rms > 0 or model.bandpass_phase_rms_deg > 0
            else None
        )
        # Drawn last (the scan screen, then the fast one), so the other models' draws stay as they were.
        self.scan_screen = (
            ScanScreen(self.rng, model, self.xy, scans) if model.scan_screen_path_um_1km > 0 else None
        )
        if model.fast_screen_path_um_1km > 0:
            pairs = np.unique(np.stack([times, t.getcol("SCAN_NUMBER")]), axis=1)
            self.fast_screen = FastScreen(self.rng, model, self.xy, pairs[0], pairs[1].astype(int))
        else:
            self.fast_screen = None
        # extra.antenna_profile: each antenna's and hand's own noise shape across
        # the window (uvmix.py profile writes it next to noise.json).
        shape_file = ms / "noise.antenna.npz"
        if model.extra.get("antenna_profile") and shape_file.exists():
            with np.load(shape_file) as f:
                g, h = f["g"], f["h"]
            # Antennas with no unflagged data have no shape: theirs is flat.
            pad = len(self.xy) - g.shape[0]
            if pad > 0:
                g = np.concatenate([g, np.ones((pad,) + g.shape[1:], dtype=g.dtype)])
            # Normalised so each channel keeps its measured mean noise: per (channel,
            # hand), the mean of g_i g_j over baselines times h is 1, and h only
            # moves the hands apart.
            n = g.shape[0]
            pairs = (g.sum(axis=0) ** 2 - (g**2).sum(axis=0)) / (n * (n - 1))
            h = h / h.mean(axis=1, keepdims=True) / pairs
            self.antenna_shape = (g, h)
        else:
            self.antenna_shape = None
        # From a generator of its own, so every other draw stays as it was.
        self.baseline_errors = (
            BaselineErrors(int(np.random.default_rng([seed, 7]).integers(2**62)), model)
            if model.baseline_amp_rms > 0 or model.baseline_phase_rms_deg > 0
            else None
        )

    def weights(self, t, start: int, n: int) -> np.ndarray:
        weight = t.getcol("WEIGHT", startrow=start, nrow=n)
        if weight.shape[1] != len(self.corr):  # one weight per row, or per another setup
            weight = np.repeat(weight.mean(axis=1, keepdims=True), len(self.corr), axis=1)
        return weight

    def chunk(
        self, t, start: int, n: int, vis: np.ndarray | None = None
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """(the rows' columns, the model visibilities, the simulated data).

        ``vis``: the model's visibilities for these rows, if already predicted.
        """
        model = self.model
        cols = {
            "uvw": t.getcol("UVW", startrow=start, nrow=n).astype(np.float64),
            "a1": t.getcol("ANTENNA1", startrow=start, nrow=n),
            "a2": t.getcol("ANTENNA2", startrow=start, nrow=n),
            "scan": t.getcol("SCAN_NUMBER", startrow=start, nrow=n),
            "time": t.getcol("TIME", startrow=start, nrow=n),
            "weight": self.weights(t, start, n),
        }
        if vis is None:
            vis = _predict(
                cols["uvw"], self.chan_freq, self.sky, self.plane_of, self.px, self.py, self.nthreads
            )
        nchan, ncorr = len(self.chan_freq), len(self.corr)
        clean = np.zeros((n, nchan, ncorr), dtype=np.complex64)
        clean[:, :, self.hands] = vis[:, :, None]
        data = np.zeros((n, nchan, ncorr), dtype=np.complex64)
        factor = np.ones((n, nchan, self.npol), dtype=np.complex128)
        a1, a2, scan, time = cols["a1"], cols["a2"], cols["scan"], cols["time"]
        if model.gain_amp_rms > 0 or model.gain_phase_rms_deg > 0:
            g1 = self.gains.factors(a1, scan, time, self.npol)
            g2 = self.gains.factors(a2, scan, time, self.npol)
            factor *= (g1 * np.conj(g2))[:, None, :]
        if self.screen is not None:
            dpath = self.screen.residual_path_m(self.xy[a1], time) - self.screen.residual_path_m(
                self.xy[a2], time
            )
            factor *= np.exp(2j * np.pi * dpath[:, None] * self.chan_freq[None, :] / C)[:, :, None]
        if self.scan_screen is not None:
            dpath = self.scan_screen.path_m(a1, scan) - self.scan_screen.path_m(a2, scan)
            factor *= np.exp(2j * np.pi * dpath[:, None] * self.chan_freq[None, :] / C)[:, :, None]
        if self.fast_screen is not None:
            dpath = self.fast_screen.path_m(a1, time) - self.fast_screen.path_m(a2, time)
            factor *= np.exp(2j * np.pi * dpath[:, None] * self.chan_freq[None, :] / C)[:, :, None]
        if self.bandpass is not None:
            b1 = np.stack([self.bandpass[a] for a in a1])
            b2 = np.stack([self.bandpass[a] for a in a2])
            factor *= b1 * np.conj(b2)
        if self.baseline_errors is not None:
            factor *= self.baseline_errors.factors(a1, a2, scan)[:, None, None]
        data[:, :, self.hands] = (vis[:, :, None] * factor).astype(np.complex64)
        if model.thermal:
            weight = cols["weight"]
            sigma = np.sqrt(self.noise["scale"] / np.where(weight > 0, weight, np.inf))
            noise_cube = thermal_noise(self.rng, sigma * model.thermal_scale, nchan, self.kernel)
            profile = self.noise.get("channel_profile", [])
            if model.extra.get("channel_profile") and len(profile) == nchan:
                noise_cube *= np.sqrt(np.asarray(profile, dtype=np.float32))[None, :, None]
            if self.antenna_shape is not None:
                g, h = self.antenna_shape
                noise_cube[:, :, self.hands] *= np.sqrt(g[a1] * g[a2] * h[None])
            data += noise_cube
        return cols, clean, data


def make_counterpart(
    base_ms: Path, task_dir: Path, out_ms: Path, model: ErrorModel, seed: int
) -> dict:
    table = _tables()
    if out_ms.exists():
        shutil.rmtree(out_ms)
    shutil.copytree(base_ms, out_ms)
    with table(str(out_ms), readonly=False, ack=False) as t:
        sim = Simulator(out_ms, task_dir, model, seed, t)
        nrow = t.nrows()
        for start in range(0, nrow, CHUNK):
            n = min(CHUNK, nrow - start)
            _, _, data = sim.chunk(t, start, n)
            t.putcol("DATA", data, startrow=start, nrow=n)
    truth = {
        "seed": seed,
        "errors": model.__dict__,
        "noise": sim.noise,
        "kernel": np.asarray(sim.kernel).tolist() if not isinstance(sim.kernel, str) else sim.kernel,
        "rows": int(nrow),
    }
    (out_ms / "counterpart.json").write_text(json.dumps(truth, indent=1, default=str))
    return truth


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    base = sub.add_parser("base")
    base.add_argument("real_ms", type=Path)
    base.add_argument("spw", type=int)
    base.add_argument("field")
    base.add_argument("base_ms", type=Path)
    split = sub.add_parser("split")
    split.add_argument("real_ms", type=Path)
    split.add_argument("spw", type=int)
    split.add_argument("field")
    split.add_argument("base_ms", type=Path)
    measure = sub.add_parser("measure")
    measure.add_argument("base_ms", type=Path)
    re_measure = sub.add_parser("remeasure")
    re_measure.add_argument("base_ms", type=Path)
    re_measure.add_argument("task_dir", type=Path)
    re_measure.add_argument("real_ms", type=Path)
    re_measure.add_argument("spw", type=int)
    make = sub.add_parser("make")
    make.add_argument("base_ms", type=Path)
    make.add_argument("task_dir", type=Path)
    make.add_argument("out_ms", type=Path)
    make.add_argument("--errors", type=Path, required=True)
    make.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.step == "split":
        split_window(args.real_ms, args.spw, args.field, args.base_ms)
    elif args.step == "measure":
        noise_path = args.base_ms / "noise.json"
        if noise_path.exists():
            previous = args.base_ms / "noise.block.json"  # the one-block estimate
            if not previous.exists():
                noise_path.rename(previous)
        noise = measure_noise(args.base_ms)
        noise_path.write_text(json.dumps(noise, indent=1))
        print(json.dumps({k: noise[k] for k in ("scale", "rho1", "rho2", "rows_sampled", "row_step")}))
    elif args.step == "remeasure":
        noise = remeasure(args.base_ms, args.task_dir, args.real_ms, args.spw)
        (args.base_ms / "noise.tsys.json").write_text(json.dumps(noise, indent=1))
        print(json.dumps({k: noise[k] for k in ("scale", "rho1", "rho2", "line_channels_excluded")}))
    elif args.step == "base":
        if not (args.base_ms / "table.dat").exists():
            # CASA's split in a process of its own (it and python-casacore in one
            # process abort), into a temporary name, so a crash leaves no half base.
            partial = args.base_ms.with_name(args.base_ms.name + ".partial")
            cmd = [sys.executable, __file__, "split", str(args.real_ms), str(args.spw), args.field, str(partial)]
            subprocess.run(cmd, check=True)
            partial.rename(args.base_ms)
        noise = measure_noise(args.base_ms)
        (args.base_ms / "noise.json").write_text(json.dumps(noise, indent=1))
        print(json.dumps(noise))
    else:
        model = ErrorModel.from_dict(json.loads(args.errors.read_text()))
        truth = make_counterpart(args.base_ms, args.task_dir, args.out_ms, model, args.seed)
        print(json.dumps({k: truth[k] for k in ("seed", "rows")}))


if __name__ == "__main__":
    main()
