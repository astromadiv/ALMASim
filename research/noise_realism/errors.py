"""Corruptions of model visibilities, beyond thermal noise.

Each model takes the main table's columns for a chunk of rows and returns
per-row, per-channel factors or additive terms; ``corrupt`` applies them:

    V = g_i g_j* B_i B_j* V_model + n

- thermal noise ``n``: per visibility, from the real WEIGHT (scaled to the
  scatter measured in the real data), optionally correlated between
  adjacent channels as ALMA's spectral (Hanning) response makes it;
- antenna gains ``g``: residual amplitude and phase errors after
  calibration, constant per scan plus a slow drift within it;
- atmosphere: a Kolmogorov phase screen blown over the array (frozen
  flow), what phase referencing leaves of it (a high-pass over the
  calibration cycle), phase proportional to frequency;
- bandpass ``B``: residual smooth ripples across the window, per antenna;
- baseline-based errors ``b_ij``: an amplitude and phase per (baseline, scan)
  that does not factor into antenna terms, so self-calibration cannot take
  it out (correlator and closure errors, or source structure the model
  lacks): V = b_ij g_i g_j* ... .

Every random draw comes from one numpy Generator seeded per window and
iteration, so a counterpart can be remade exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C = 299_792_458.0


@dataclass
class ErrorModel:
    """Which corruptions to apply, and how strong."""

    thermal: bool = True
    thermal_scale: float = 1.0  # times the noise the real WEIGHT and scatter imply
    spectral_kernel: str = "none"  # "none" | "hanning" | "measured" (from the real data)
    gain_amp_rms: float = 0.0  # fractional, per antenna, per scan
    gain_phase_rms_deg: float = 0.0  # per antenna, per scan
    gain_drift_rms: float = 0.0  # fraction of the above, a random walk within a scan
    gain_drift_s: float = 60.0  # correlation time of the drift
    screen_path_um_1km: float = 0.0  # residual path rms between antennas 1 km apart
    screen_wind_mps: float = 10.0
    screen_cycle_s: float = 0.0  # phase-referencing cycle (0: no referencing)
    screen_outer_km: float = 3.0
    # What phase referencing leaves, held for a scan: per scan, a fresh screen
    # sampled at each antenna (the phase-transfer error, correlated in space).
    scan_screen_path_um_1km: float = 0.0
    scan_screen_slope: float = 5 / 6  # rms phase difference grows as (baseline)^slope
    # What varies within a scan, faster than referencing follows: a screen per
    # integration, correlated in time over fast_coherence_s, its per-scan mean
    # removed (that is the scan screen's); decorrelates the scan's average.
    fast_screen_path_um_1km: float = 0.0  # rms within-scan path difference, 1 km apart
    fast_screen_slope: float = 5 / 6
    fast_coherence_s: float = 10.0
    bandpass_amp_rms: float = 0.0  # fractional ripple across the window, per antenna
    bandpass_phase_rms_deg: float = 0.0
    bandpass_modes: int = 4  # Fourier modes of the ripple
    baseline_amp_rms: float = 0.0  # fractional, per (baseline, scan)
    baseline_phase_rms_deg: float = 0.0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict) -> ErrorModel:
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# Spectral response: white noise convolved with a Hanning window correlates
# adjacent channels by 2/3 and the next ones by 1/6, as ALMA's correlator
# leaves native channels; normalised so the per-channel variance is unchanged.
_HANNING = np.array([0.25, 0.5, 0.25])
_HANNING = _HANNING / np.sqrt(np.sum(_HANNING**2))


def spectral_kernel(rho1: float, rho2: float) -> np.ndarray:
    """A 3-tap filter whose output's channel autocorrelation is (1, rho1, rho2).

    White noise filtered by [h0, h1, h2] (unit norm) has lag-1 correlation
    h0 h1 + h1 h2 and lag-2 correlation h0 h2: solved from the measured ones.
    """
    from scipy.optimize import fsolve

    if rho1 < 0.05 and abs(rho2) < 0.05:
        return np.array([0.0, 1.0, 0.0])

    def equations(h):
        return [h @ h - 1, h[0] * h[1] + h[1] * h[2] - rho1, h[0] * h[2] - rho2]

    start = _HANNING if rho2 > 0.05 else np.array([0.0, 0.85, 0.5])
    h = fsolve(equations, start)
    if np.max(np.abs(equations(h))) < 1e-4:  # fsolve flags a start on the root as "no progress"
        return h
    # No 3-tap filter has these exactly: Hanning (2/3, 1/6) is on the edge of
    # what one can give, and measured values scatter past it. The nearest one,
    # unit norm. (Falling back to white noise here made a third of the windows
    # white: their continuum 15% quieter, adjacent channels uncorrelated.)
    from scipy.optimize import least_squares

    def correlations(h):
        g = h / np.linalg.norm(h)
        return [g[0] * g[1] + g[1] * g[2] - rho1, g[0] * g[2] - rho2]

    h = least_squares(correlations, _HANNING if rho2 > 0.05 else start).x
    return h / np.linalg.norm(h)


def thermal_noise(
    rng: np.random.Generator, sigma: np.ndarray, nchan: int, kernel: str | np.ndarray
) -> np.ndarray:
    """Complex noise (nrow, nchan, ncorr) with per-(row, corr) sigma per component.

    ``kernel``: "none" (independent channels), "hanning", or a 3-tap filter.
    """
    nrow, ncorr = sigma.shape
    shape = (nrow, nchan + 2, ncorr)
    noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    taps = _HANNING if isinstance(kernel, str) and kernel == "hanning" else kernel
    if isinstance(taps, np.ndarray):
        noise = taps[0] * noise[:, :-2] + taps[1] * noise[:, 1:-1] + taps[2] * noise[:, 2:]
    else:
        noise = noise[:, 1:-1]
    return (noise * sigma[:, None, :]).astype(np.complex64)


@dataclass
class AntennaGains:
    """Residual antenna gains: per (antenna, pol), constant per scan plus a drift."""

    amp: dict  # (ant, scan) -> (npol,) fractional amplitude error
    phase: dict  # (ant, scan) -> (npol,) phase error, radians
    drift_amp: float
    drift_phase: float
    drift_s: float
    seed: int

    @classmethod
    def draw(cls, rng, model: ErrorModel, antennas, scans, npol: int) -> AntennaGains:
        amp, phase = {}, {}
        for a in antennas:
            for s in scans:
                # Mostly common to the two polarisations, as the phase (the real
                # windows' per-antenna amplitudes are measured on their sum).
                amp[(a, s)] = rng.normal(0, model.gain_amp_rms) + rng.normal(0, model.gain_amp_rms / 4, npol)
                common = rng.normal(0, np.radians(model.gain_phase_rms_deg))
                phase[(a, s)] = common + rng.normal(0, np.radians(model.gain_phase_rms_deg) / 4, npol)
        return cls(
            amp,
            phase,
            model.gain_drift_rms * model.gain_amp_rms,
            model.gain_drift_rms * np.radians(model.gain_phase_rms_deg),
            model.gain_drift_s,
            int(rng.integers(2**31)),
        )

    def _drift(self, ant: np.ndarray, time: np.ndarray, which: int) -> np.ndarray:
        """A smooth zero-mean drift per antenna: sum of a few random sinusoids."""
        out = np.zeros(len(ant))
        for a in np.unique(ant):
            sel = ant == a
            rng = np.random.default_rng([self.seed, int(a), which])
            periods = self.drift_s * rng.uniform(0.5, 4, 3)
            phases = rng.uniform(0, 2 * np.pi, 3)
            out[sel] = np.sum(
                [np.sin(2 * np.pi * time[sel] / p + f) for p, f in zip(periods, phases, strict=True)],
                axis=0,
            ) / np.sqrt(1.5)
        return out

    def factors(self, ant: np.ndarray, scan: np.ndarray, time: np.ndarray, npol: int) -> np.ndarray:
        """g (nrow, npol) for these rows' antenna."""
        g = np.ones((len(ant), npol), dtype=np.complex128)
        da = self._drift(ant, time, 0) * self.drift_amp
        dp = self._drift(ant, time, 1) * self.drift_phase
        for i, (a, s) in enumerate(zip(ant, scan, strict=True)):
            amp = self.amp.get((int(a), int(s)))
            if amp is None:
                continue
            g[i] = (1 + amp + da[i]) * np.exp(1j * (self.phase[(int(a), int(s))] + dp[i]))
        return g


class PhaseScreen:
    """A frozen Kolmogorov screen of excess path, blown over the array."""

    def __init__(self, rng, model: ErrorModel, positions_m: np.ndarray, duration_s: float):
        self.wind = model.screen_wind_mps
        self.cycle = model.screen_cycle_s
        extent = np.ptp(positions_m[:, 0]) + self.wind * duration_s + 2_000.0
        extent_y = np.ptp(positions_m[:, 1]) + 2_000.0
        self.step = 20.0  # metres per screen pixel
        nx = int(2 ** np.ceil(np.log2(extent / self.step)))
        ny = int(2 ** np.ceil(np.log2(extent_y / self.step)))
        kx = np.fft.fftfreq(nx, self.step)
        ky = np.fft.fftfreq(ny, self.step)
        k2 = kx[None, :] ** 2 + ky[:, None] ** 2 + (1 / (model.screen_outer_km * 1e3)) ** 2
        spectrum = k2 ** (-11 / 12)  # amplitude of a k^-11/3 power spectrum
        screen = np.real(
            np.fft.ifft2(spectrum * (rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))))
        )
        self.origin = positions_m[:, :2].min(axis=0) - 1_000.0
        self.screen = screen
        # Scale so that the rms path difference 1 km apart is the model's.
        i0 = ny // 2
        sep = int(round(1_000 / self.step))
        diff = screen[i0, : nx - sep] - screen[i0, sep:]
        self.screen *= (model.screen_path_um_1km * 1e-6) / np.std(diff)

    def path_m(self, xy: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Excess path (m) over antennas at ``xy`` (n, 2) at ``time`` (n,), frozen flow."""
        x = (xy[:, 0] - self.origin[0] + self.wind * (time - time.min())) / self.step
        y = (xy[:, 1] - self.origin[1]) / self.step
        ix = np.clip(x.astype(int), 0, self.screen.shape[1] - 1)
        iy = np.clip(y.astype(int), 0, self.screen.shape[0] - 1)
        return self.screen[iy, ix]

    def residual_path_m(self, xy: np.ndarray, time: np.ndarray) -> np.ndarray:
        """What phase referencing leaves: the path minus its mean over each cycle."""
        path = self.path_m(xy, time)
        if self.cycle <= 0:
            return path
        # Referencing follows the screen at the cycle's pace: the residual is the
        # path minus the path a cycle-length later in the flow, halved (interpolated).
        later = self.path_m(xy, time + self.cycle)
        return (path - later) / 2


def kolmogorov_screen(
    rng, extent_m: float, outer_km: float, step: float = 10.0, slope: float = 5 / 6, ref_m: float = 150.0
) -> np.ndarray:
    """A square turbulent screen covering ``extent_m`` whose rms difference grows
    as separation^slope (power spectrum k^-(2 slope + 2): Kolmogorov's 5/3 + 2 =
    11/3 for slope 5/6), up to the outer scale; scaled so that the rms
    difference between points 1 km apart would be 1. The scale is set at
    ``ref_m``, well inside the screen: the screen is periodic, so at separations
    near its size its differences no longer grow."""
    n = int(2 ** np.ceil(np.log2(max(extent_m, 2_500.0) / step)))
    k = np.fft.fftfreq(n, step)
    k2 = k[None, :] ** 2 + k[:, None] ** 2 + (1 / (outer_km * 1e3)) ** 2
    amplitude = k2 ** (-(2 * slope + 2) / 4)  # sqrt of the power spectrum, as k^2 = |k|^2
    screen = np.real(np.fft.ifft2(amplitude * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))))
    sep = int(round(ref_m / step))
    rms = np.sqrt((np.mean((screen[:, : n - sep] - screen[:, sep:]) ** 2) + np.mean((screen[: n - sep] - screen[sep:]) ** 2)) / 2)
    return screen / rms * (ref_m / 1_000) ** slope


class ScanScreen:
    """Per scan and antenna, a residual path: a fresh screen per scan, sampled at the antennas."""

    def __init__(self, rng, model: ErrorModel, positions_m: np.ndarray, scans) -> None:
        origin = positions_m[:, :2].min(axis=0) - 500.0
        extent = float(np.ptp(positions_m[:, :2], axis=0).max()) + 1_000.0
        step = 10.0
        ix = ((positions_m[:, 0] - origin[0]) / step).astype(int)
        iy = ((positions_m[:, 1] - origin[1]) / step).astype(int)
        self.index = {int(s): i for i, s in enumerate(scans)}
        self.path = np.zeros((len(scans), len(positions_m)))
        for i, _ in enumerate(scans):
            screen = kolmogorov_screen(rng, extent, model.screen_outer_km, step, model.scan_screen_slope)
            self.path[i] = screen[iy, ix] * model.scan_screen_path_um_1km * 1e-6

    def path_m(self, ant: np.ndarray, scan: np.ndarray) -> np.ndarray:
        rows = np.array([self.index[int(s)] for s in scan])
        return self.path[rows, ant]


class FastScreen:
    """Per integration and antenna, the path that varies within a scan: a fresh
    screen per integration, blended with the last one (correlated over
    ``fast_coherence_s``), less each antenna's mean over the scan. Scaled so that
    the rms within-scan path difference of the array's baselines follows the
    model's power law (median over antenna pairs)."""

    def __init__(self, rng, model: ErrorModel, positions_m: np.ndarray, times: np.ndarray, scans: np.ndarray):
        origin = positions_m[:, :2].min(axis=0) - 500.0
        extent = float(np.ptp(positions_m[:, :2], axis=0).max()) + 1_000.0
        step = 10.0
        ix = ((positions_m[:, 0] - origin[0]) / step).astype(int)
        iy = ((positions_m[:, 1] - origin[1]) / step).astype(int)
        order = np.argsort(times)
        self.times, scans = times[order], scans[order]  # unique integrations, their scans
        path = np.zeros((len(self.times), len(positions_m)))
        for i, time in enumerate(self.times):
            fresh = kolmogorov_screen(rng, extent, model.screen_outer_km, step, model.fast_screen_slope)[iy, ix]
            if i and scans[i] == scans[i - 1]:
                a = np.exp(-(time - self.times[i - 1]) / model.fast_coherence_s)
                path[i] = a * path[i - 1] + np.sqrt(1 - a**2) * fresh
            else:
                path[i] = fresh
        for s in np.unique(scans):
            path[scans == s] -= path[scans == s].mean(axis=0)
        i, j = np.triu_indices(len(positions_m), 1)
        b = np.hypot(*(positions_m[i, :2] - positions_m[j, :2]).T)
        keep = b > step
        rms = np.sqrt(np.mean((path[:, i[keep]] - path[:, j[keep]]) ** 2, axis=0))
        scale = np.median(rms / (b[keep] / 1_000) ** model.fast_screen_slope)
        self.path = path * model.fast_screen_path_um_1km * 1e-6 / scale

    def path_m(self, ant: np.ndarray, time: np.ndarray) -> np.ndarray:
        rows = np.clip(np.searchsorted(self.times, time), 0, len(self.times) - 1)
        return self.path[rows, ant]


def bandpass_factors(rng, model: ErrorModel, antennas, nchan: int, npol: int) -> dict:
    """B per antenna (nchan, npol): smooth complex ripples across the window."""
    x = np.linspace(0, 1, nchan)
    out = {}
    for a in antennas:
        amp = np.zeros((nchan, npol))
        phase = np.zeros((nchan, npol))
        for m in range(1, model.bandpass_modes + 1):
            weight = 1 / m  # smoother: lower modes dominate
            for p in range(npol):
                amp[:, p] += weight * rng.normal() * np.sin(2 * np.pi * m * x + rng.uniform(0, 2 * np.pi))
                phase[:, p] += weight * rng.normal() * np.sin(2 * np.pi * m * x + rng.uniform(0, 2 * np.pi))
        norm = np.sqrt(np.sum([(1 / m) ** 2 for m in range(1, model.bandpass_modes + 1)]) / 2)
        out[a] = (1 + model.bandpass_amp_rms * amp / norm) * np.exp(
            1j * np.radians(model.bandpass_phase_rms_deg) * phase / norm
        )
    return out


class BaselineErrors:
    """b_ij per (baseline, scan): a complex factor common to the two hands,
    drawn from its own generator, keyed by the baseline and scan, so a chunk
    gets the same factor wherever its rows fall."""

    def __init__(self, seed: int, model: ErrorModel) -> None:
        self.seed = seed
        self.amp = model.baseline_amp_rms
        self.phase = np.radians(model.baseline_phase_rms_deg)

    def factors(self, a1: np.ndarray, a2: np.ndarray, scan: np.ndarray) -> np.ndarray:
        """b (nrow,) for these rows."""
        key = (a1.astype(np.int64) * 10_000 + a2) * 100_000 + scan.astype(np.int64)
        keys, inverse = np.unique(key, return_inverse=True)
        b = np.empty(len(keys), dtype=np.complex128)
        for k, value in enumerate(keys.tolist()):
            rng = np.random.default_rng([self.seed, value])
            b[k] = np.exp(rng.normal(0, self.amp) + 1j * rng.normal(0, self.phase))
        return b[inverse]


def antenna_noise_shape(z2: np.ndarray, n: np.ndarray, chans: np.ndarray, smooth: int = 16):
    """Each antenna's and each hand's own spectral shape of the noise, from the
    real residuals: ``z2``, ``n`` (nant, nchan, nhand) the sums of |z|^2 and
    their counts per antenna (rows it is in), with z scaled by WEIGHT; ``chans``
    the channels free of unmodelled sky.

    v = z2 / 2n factors as A(a, p) C(c, p) L(a, c, p). Returns (g, h): g
    (nant, nchan, nhand) = A L per antenna (L smoothed over ``smooth``
    channels, so the per-channel sampling noise stays out), h (nchan, nhand)
    the hands' profiles over their mean, smoothed alike. A baseline's noise
    variance is then g_i g_j h times the array-wide profile: an antenna's mean
    over its baselines is g_i, what ``v`` measures.
    """
    from scipy.ndimage import uniform_filter1d

    # Antennas with most of the data in every hand; the others keep g = 1.
    total = n.sum(axis=1)  # (nant, nhand)
    seen = total.sum(axis=1) > 0
    used = seen & (total > 0.5 * np.median(total[seen], axis=0)).all(axis=1)
    ok = chans & (n[used] > 50).all(axis=(0, 2))
    if ok.sum() < 2 * smooth:
        return np.ones(n.shape, dtype=np.float32), np.ones(n.shape[1:], dtype=np.float32)
    v = np.where(n > 0, z2 / np.maximum(2 * n, 1), np.nan)[:, ok]
    sub = v[used]
    ant = np.ones((sub.shape[0], 1, sub.shape[2]))
    for _ in range(5):
        chan = (sub / ant).mean(axis=0, keepdims=True)
        ant = (sub / chan).mean(axis=1, keepdims=True)
    ant /= ant.mean()

    def fill(x, axis):
        """x on the good channels (along ``axis``), smoothed and spread to all channels."""
        shape = list(x.shape)
        shape[axis] = len(ok)
        full = np.zeros(shape)
        weight = np.zeros(shape)
        index = [slice(None)] * len(shape)
        index[axis] = ok
        full[tuple(index)] = x
        weight[tuple(index)] = 1
        num = uniform_filter1d(full, smooth, axis=axis, mode="nearest")
        den = uniform_filter1d(weight, smooth, axis=axis, mode="nearest")
        return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)

    left = np.log(sub / (ant * chan))
    g = np.ones((n.shape[0], len(ok), n.shape[2]))
    g[used] = ant * np.exp(fill(left, axis=1))
    hands = np.log(chan[0] / np.exp(np.log(chan[0]).mean(axis=1, keepdims=True)))
    h = np.exp(fill(hands, axis=0))
    return g.astype(np.float32), h.astype(np.float32)
