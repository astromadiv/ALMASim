"""What calibrated ALMA visibilities carry besides white thermal noise (MADIV-28).

Measured on 100 MADIV-16 windows against their real calibrated data
(research/noise_realism): the noise of neighbouring channels is correlated
(ALMA's correlator Hanning-smooths native channels: rho1 ~ 2/3), and on bright
sources the calibration leaves antenna-based errors: a phase per scan that
grows with baseline (what phase referencing does not transfer), a faster
phase within each scan that decorrelates the scan's average, and an amplitude
per antenna and scan.

Everything here is off by default (:class:`CorruptionModel` with its zero
values corrupts nothing). The antenna errors are drawn once per array
configuration, from their own random stream, and applied identically to
every channel; the correlated noise is built per channel from white draws
keyed by channel, so each channel task makes its own without the others.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

SCREEN_STEP_M = 10.0  # metres per pixel of the turbulent screens
REFERENCE_SEPARATION_M = 150.0  # where a screen's strength is set (well inside it)


@dataclass(frozen=True)
class CorruptionModel:
    """Which corruptions to apply, and how strong. The defaults apply none."""

    # Correlation of the noise of channels one and two apart (0, 0: white).
    spectral_rho1: float = 0.0
    spectral_rho2: float = 0.0
    # Length of a scan: the slow phase and the gains hold for one (0: one scan).
    scan_length_s: float = 0.0
    # Antenna gain errors per scan: fractional amplitude and phase (degrees).
    gain_amp_rms: float = 0.0
    gain_phase_rms_deg: float = 0.0
    # The slow (per-scan) and fast (within-scan) atmospheric phase: the rms path
    # difference between antennas 1 km apart (microns), growing as
    # separation^slope (Kolmogorov: 5/6).
    slow_path_um_1km: float = 0.0
    slow_slope: float = 5 / 6
    fast_path_um_1km: float = 0.0
    fast_slope: float = 5 / 6
    fast_coherence_s: float = 10.0
    outer_scale_km: float = 3.0
    # The noise across a window follows Tsys: atmospheric lines in the signal
    # and the image band (atmosphere.system_temperature), and the receiver's
    # own smooth ripple, as a fraction of Tsys.
    atmospheric_tsys: bool = False
    receiver_ripple_rms: float = 0.0
    # Each antenna's own smooth ripple of its noise across the window, beyond
    # the array's (receivers and IF chains differ): the fractional rms of its
    # noise variance, and the frequency scale (MHz) over which it holds.
    antenna_ripple_rms: float = 0.0
    antenna_ripple_mhz: float = 100.0
    # Errors of a baseline that do not factor into its antennas' (closure
    # errors; self-calibration cannot take them out), per scan: fractional
    # amplitude and phase (degrees).
    baseline_amp_rms: float = 0.0
    baseline_phase_rms_deg: float = 0.0
    # Each visibility's noise from the radiometer equation (Tsys, channel width,
    # integration, dish; noise.visibility_noise), the source keeping its flux,
    # instead of a noise set from the run's target S/N (interferometric runs).
    radiometer_noise: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> CorruptionModel | None:
        """The model a mapping (e.g. a JSON object) describes; None for None or {}."""
        if not values:
            return None
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown corruption parameter(s): {', '.join(sorted(unknown))}")
        kinds = {f.name: f.type for f in fields(cls)}
        coerced: dict[str, Any] = {
            k: _flag(v) if kinds[k] == "bool" else float(v) for k, v in values.items()
        }
        return cls(**coerced)

    @property
    def correlates_channels(self) -> bool:
        return self.spectral_rho1 != 0.0 or self.spectral_rho2 != 0.0

    @property
    def has_antenna_errors(self) -> bool:
        return any(
            v > 0
            for v in (
                self.gain_amp_rms,
                self.gain_phase_rms_deg,
                self.slow_path_um_1km,
                self.fast_path_um_1km,
                self.baseline_amp_rms,
                self.baseline_phase_rms_deg,
            )
        )


def _flag(value: Any) -> bool:
    """A switch from JSON, a CLI string or a number."""
    if isinstance(value, str):
        if value.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if value.strip().lower() in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"not a switch: {value!r}")
    return bool(value)


# Named models. "alma": what the MADIV-16 windows' calibrated data show
# (research/noise_realism): native channels Hanning-correlated; on bright
# sources, after calibration, ~8% antenna amplitude errors per scan, a per-scan
# phase of ~80 um rms path at 1 km growing as baseline^0.45, and a faster
# in-scan phase of ~40 um. Faint sources show only the correlated noise. Their
# noise varies across a window as ALMA's Tsys does: the atmosphere's lines and a
# ~2% receiver ripple (SYSCAL against the measured channel profiles). Each
# antenna's noise has, besides, its own ~2% ripple over ~120 MHz (the median of
# the 73 windows wider than 1 GHz; the 27 narrow ones show ~1 MHz structure), and
# on bright sources each baseline keeps ~6% and ~2.5 deg per scan that do not
# close (the median of the 13 bright windows where that is measurable).
PRESETS: dict[str, dict[str, Any]] = {
    "none": {},
    "alma": {
        "spectral_rho1": 2 / 3,
        "spectral_rho2": 1 / 6,
        "scan_length_s": 300.0,
        "gain_amp_rms": 0.08,
        "slow_path_um_1km": 80.0,
        "slow_slope": 0.45,
        "fast_path_um_1km": 40.0,
        "fast_slope": 0.3,
        "fast_coherence_s": 10.0,
        "atmospheric_tsys": True,
        "receiver_ripple_rms": 0.02,
        "antenna_ripple_rms": 0.02,
        "antenna_ripple_mhz": 120.0,
        "baseline_amp_rms": 0.06,
        "baseline_phase_rms_deg": 2.5,
        "radiometer_noise": True,
    },
}


def resolve_corruptions(spec: str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A corruption model's fields from a preset name (PRESETS), a JSON object,
    a path to a JSON file, or a mapping; checked against CorruptionModel.
    None for none (None, "", "none", {})."""
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        mapping = dict(spec)
    else:
        text = str(spec).strip()
        if text in PRESETS:
            mapping = dict(PRESETS[text])
        elif text.startswith("{"):
            mapping = json.loads(text)
        elif text:
            path = Path(text).expanduser()
            if not path.is_file():
                raise ValueError(
                    f"corruptions {text!r}: not a preset ({', '.join(PRESETS)}), "
                    "a JSON object or a JSON file"
                )
            mapping = json.loads(path.read_text())
        else:
            mapping = {}
    CorruptionModel.from_mapping(mapping)  # unknown or non-numeric fields fail here
    return mapping or None


def spectral_kernel(rho1: float, rho2: float) -> np.ndarray:
    """A 3-tap symmetric kernel (a, b, a), unit norm, whose output's channel
    correlations are ``rho1`` and ``rho2``: rho1 = 2ab, rho2 = a^2 (with
    2a^2 + b^2 = 1). Hanning smoothing gives rho1 = 2/3, rho2 = 1/6."""
    if not 0.0 <= rho2 <= 0.25 or not 0.0 <= rho1 <= 1.0:
        raise ValueError(f"no 3-tap kernel has rho1={rho1}, rho2={rho2}")
    a = np.sqrt(rho2)
    b = np.sqrt(max(1.0 - 2.0 * rho2, 0.0))
    achieved = 2.0 * a * b
    if abs(achieved - rho1) > 0.05:
        # rho1 and rho2 together over-determine (a, b): take the pair that
        # matches rho1 exactly and is closest in rho2.
        # 2a sqrt(1 - 2a^2) = rho1  ->  a^2 = (1 - sqrt(1 - 2 rho1^2)) / 4
        a = np.sqrt((1.0 - np.sqrt(max(1.0 - 2.0 * rho1**2, 0.0))) / 4.0)
        b = np.sqrt(1.0 - 2.0 * a**2)
    return np.array([a, b, a])


def correlated_noise(
    entropy: tuple[int, ...],
    channel: int,
    shape: tuple[int, ...],
    sigma: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """Complex noise of ``channel`` (real and imaginary parts of rms ``sigma``),
    correlated with its neighbours through ``kernel``: the kernel applied to
    white draws of channels ``channel - 1 .. channel + 1``, each keyed by its
    channel, so neighbouring channel tasks share them without talking."""
    out = np.zeros(shape, dtype=np.complex128)
    for offset, weight in zip((-1, 0, 1), kernel, strict=True):
        rng = np.random.default_rng((*entropy, channel + offset + 1))
        out += weight * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    return (out * sigma).astype(np.complex64)


def turbulent_screen(
    rng: np.random.Generator, extent_m: float, outer_km: float, slope: float
) -> np.ndarray:
    """A square periodic screen covering ``extent_m`` whose rms difference grows as
    separation^slope (power spectrum k^-(2 slope + 2)) up to the outer scale,
    scaled so the rms difference 1 km apart would be 1 (set at
    REFERENCE_SEPARATION_M, where the periodic screen still follows its law)."""
    n = int(2 ** np.ceil(np.log2(max(extent_m, 2_500.0) / SCREEN_STEP_M)))
    k = np.fft.fftfreq(n, SCREEN_STEP_M)
    k2 = k[None, :] ** 2 + k[:, None] ** 2 + (1.0 / (outer_km * 1e3)) ** 2
    amplitude = k2 ** (-(2.0 * slope + 2.0) / 4.0)
    noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    screen = np.real(np.fft.ifft2(amplitude * noise))
    sep = int(round(REFERENCE_SEPARATION_M / SCREEN_STEP_M))
    rms = np.sqrt(
        (
            np.mean((screen[:, :-sep] - screen[:, sep:]) ** 2)
            + np.mean((screen[:-sep] - screen[sep:]) ** 2)
        )
        / 2.0
    )
    return screen / rms * (REFERENCE_SEPARATION_M / 1_000.0) ** slope


@dataclass(frozen=True)
class AntennaCorruptions:
    """Per antenna and integration: an amplitude factor, a phase (rad) and an
    excess path (m, whose phase scales with frequency)."""

    amplitude: np.ndarray  # (nant, ntime)
    phase: np.ndarray  # (nant, ntime)
    path_m: np.ndarray  # (nant, ntime)
    # Per (antenna1, antenna2, integration): what does not factor by antenna;
    # None: nothing.
    baseline: np.ndarray | None = None

    def baseline_gains(
        self, antnum: np.ndarray, wavelength_m: float, baseline_weight: float = 1.0
    ) -> np.ndarray:
        """g1 conj(g2) (times the baseline's own error) for each baseline (rows
        of ``antnum``) and integration.

        ``baseline_weight``: the share of the visibility the baseline's own error
        acts on, 1 + w (b - 1); the continuum's share of a channel, since what
        does not close in calibrated data is mostly continuum structure the
        model lacks, which leaves line emission alone."""
        g = self.amplitude * np.exp(1j * (self.phase + 2.0 * np.pi * self.path_m / wavelength_m))
        gains = g[antnum[:, 0]] * np.conj(g[antnum[:, 1]])
        if self.baseline is not None:
            own = self.baseline[antnum[:, 0], antnum[:, 1]]
            gains = gains * (1.0 + baseline_weight * (own - 1.0))
        return gains.astype(np.complex64)


def scan_index(times_s: np.ndarray, scan_length_s: float) -> np.ndarray:
    """The scan each integration falls in: blocks of ``scan_length_s`` (0: one)."""
    if scan_length_s <= 0:
        return np.zeros(len(times_s), dtype=int)
    return ((times_s - times_s[0]) // scan_length_s).astype(int)


def draw_antenna_corruptions(
    model: CorruptionModel,
    antenna_xy_m: np.ndarray,
    times_s: np.ndarray,
    rng: np.random.Generator,
) -> AntennaCorruptions:
    """The antenna errors of one array configuration over ``times_s``.

    Draws, in order: the gains per (antenna, scan), one slow screen per scan
    sampled at the antennas, then the fast screen per integration (each
    blended with the last over ``fast_coherence_s``, less each antenna's mean
    over its scan, and scaled so the median antenna pair's rms path difference
    follows the model's power law), then the baselines' own errors per scan.
    """
    xy = np.asarray(antenna_xy_m, dtype=np.float64)[:, :2]
    times_s = np.asarray(times_s, dtype=np.float64)
    nant, ntime = len(xy), len(times_s)
    scans = scan_index(times_s, model.scan_length_s)
    nscan = int(scans.max()) + 1
    amplitude = np.ones((nant, ntime))
    phase = np.zeros((nant, ntime))
    path = np.zeros((nant, ntime))

    if model.gain_amp_rms > 0 or model.gain_phase_rms_deg > 0:
        amp = rng.normal(0.0, model.gain_amp_rms, (nant, nscan))
        ph = rng.normal(0.0, np.radians(model.gain_phase_rms_deg), (nant, nscan))
        amplitude *= np.clip(1.0 + amp[:, scans], 0.0, None)
        phase += ph[:, scans]

    origin = xy.min(axis=0) - 500.0
    extent = float(np.ptp(xy, axis=0).max()) + 1_000.0
    ix = ((xy[:, 0] - origin[0]) / SCREEN_STEP_M).astype(int)
    iy = ((xy[:, 1] - origin[1]) / SCREEN_STEP_M).astype(int)

    if model.slow_path_um_1km > 0:
        for s in range(nscan):
            screen = turbulent_screen(rng, extent, model.outer_scale_km, model.slow_slope)
            path[:, scans == s] += (screen[iy, ix] * model.slow_path_um_1km * 1e-6)[:, None]

    if model.fast_path_um_1km > 0 and nant > 1:
        fast = np.zeros((ntime, nant))
        for t in range(ntime):
            fresh = turbulent_screen(rng, extent, model.outer_scale_km, model.fast_slope)[iy, ix]
            if t and scans[t] == scans[t - 1]:
                a = np.exp(-(times_s[t] - times_s[t - 1]) / model.fast_coherence_s)
                fast[t] = a * fast[t - 1] + np.sqrt(1.0 - a**2) * fresh
            else:
                fast[t] = fresh
        for s in range(nscan):
            fast[scans == s] -= fast[scans == s].mean(axis=0)
        i, j = np.triu_indices(nant, 1)
        separation = np.hypot(*(xy[i] - xy[j]).T)
        keep = separation > SCREEN_STEP_M
        if keep.any() and np.any(fast):
            rms = np.sqrt(np.mean((fast[:, i[keep]] - fast[:, j[keep]]) ** 2, axis=0))
            scale = np.median(rms / (separation[keep] / 1_000.0) ** model.fast_slope)
            if scale > 0:
                path += (fast * model.fast_path_um_1km * 1e-6 / scale).T

    # Drawn last, so the antenna errors stay as they were without it.
    baseline = _baseline_errors(model, nant, scans, rng)
    return AntennaCorruptions(amplitude=amplitude, phase=phase, path_m=path, baseline=baseline)


def _baseline_errors(
    model: CorruptionModel, nant: int, scans: np.ndarray, rng: np.random.Generator
) -> np.ndarray | None:
    """Each baseline's own complex error per scan, (nant, nant, ntime); None when off."""
    if model.baseline_amp_rms <= 0 and model.baseline_phase_rms_deg <= 0:
        return None
    nscan = int(scans.max()) + 1
    amp = rng.normal(0.0, model.baseline_amp_rms, (nant, nant, nscan))
    ph = rng.normal(0.0, np.radians(model.baseline_phase_rms_deg), (nant, nant, nscan))
    return np.exp(amp + 1j * ph)[:, :, scans]


def antenna_ripple(
    rng: np.random.Generator, nant: int, freq_hz: np.ndarray, rms: float, scale_mhz: float
) -> np.ndarray:
    """Each antenna's noise variance across the window, as a factor about 1
    (nant, nchan): white draws smoothed in frequency by a Gaussian whose
    autocorrelation falls to half at ``scale_mhz``, scaled to ``rms`` per
    antenna. A baseline's noise variance is then g_i g_j times the array's, so
    an antenna's mean over its baselines varies by ``rms``."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    nchan = freq_hz.size
    if rms <= 0 or nchan == 0:
        return np.ones((nant, nchan))
    step_mhz = float(np.median(np.abs(np.diff(freq_hz)))) / 1e6 if nchan > 1 else scale_mhz
    # A Gaussian kernel of width s gives an autocorrelation of width s*sqrt(2),
    # at half maximum sqrt(2 ln 2) * s * sqrt(2) = 2 sqrt(ln 2) s.
    sigma = scale_mhz / (2.0 * np.sqrt(np.log(2.0))) / max(step_mhz, 1e-12)
    pad = int(np.ceil(4 * sigma))
    white = rng.standard_normal((nant, nchan + 2 * pad))
    x = np.arange(-pad, pad + 1)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    smooth = np.stack([np.convolve(row, kernel, mode="valid") for row in white])
    smooth /= np.sqrt(np.sum(kernel**2))  # unit variance, whatever the width
    return np.clip(1.0 + rms * smooth, 0.05, None)


def baseline_antennas(nant: int) -> np.ndarray:
    """The (antenna1, antenna2) of each baseline, in the simulator's order."""
    i, j = np.triu_indices(nant, 1)
    return np.stack([i, j], axis=1)
