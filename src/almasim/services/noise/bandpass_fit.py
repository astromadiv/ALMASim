"""The residual bandpass a real window's calibration leaves on its antennas
(MADIV-28): what the noise model's ``antenna_gains.bandpass`` draws.

Over the whole observation, each baseline's data over its clean model is
summed per chunk of channels (a few MHz, enough to resolve a ripple of tens
of MHz): sum w V M* over sum w |M|^2, each row turned by its own phase over
the whole window first so that the atmosphere's phase, changing from one
integration to the next, does not decorrelate the sums. The gains are common
to a baseline's chunks, so each baseline's chunks over its mean across them
leave what varies across frequency. A single baseline's chunk rarely has an
S/N much above 1, so per chunk the ratio is split into antenna terms by
weighted linear least squares, which pools each antenna's baselines and
stays unbiased at low S/N: Re(ratio) - 1 = a_i + a_j (fractional amplitude),
Im(ratio) = p_i - p_j (phase, its mean held at zero), each baseline weighted
by its sum w |M|^2 times |mean|^2. Each antenna's mean and slope across the
window are removed (a residual delay or a smooth bandpass, which the ripple
is not), and the rms of what remains is the ripple's, less the rms the
thermal noise alone gives (the same solve on noise of the baselines' own
variance, kappa / sum w |M|^2 per component, over the baseline's mean). The
ripple's period is where the antennas' summed power spectrum, less the
noise's, peaks.
"""

from __future__ import annotations

import numpy as np

from .measurements import BandpassFit

MAX_CHUNK_HZ = 4e6  # a chunk at most this wide: a 25 MHz period has six or more chunks
MAX_CHUNKS = 512
MIN_CHUNKS = 16
MIN_DATA_SHARE = 0.8  # a baseline counts when it has data in this share of its chunks
MIN_ANTENNAS = 4
# The thermal noise's rms per antenna and chunk, above which a ripple of a few
# percent is lost in it (the excess's error is about thermal^2 sqrt(2 / n)).
MAX_THERMAL_RMS = 0.1
LOWEST_HARMONIC = 2  # the spectrum's lowest bins hold what the slope removal leaves


class BandpassSums:
    """Each baseline's sums per chunk of channels, over the whole observation."""

    def __init__(self, nant: int, freq: np.ndarray):
        freq = np.asarray(freq, dtype=np.float64)
        order = np.argsort(freq)
        width = float(np.median(np.abs(np.diff(freq[order])))) if freq.size > 1 else 0.0
        per = max(1, int(MAX_CHUNK_HZ // width)) if width > 0 else freq.size
        per = max(per, int(np.ceil(freq.size / MAX_CHUNKS)))
        self.nant = nant
        self.chunks = int(np.ceil(freq.size / per))
        self.chunk_hz = per * width
        self.chunk_of = np.empty(freq.size, dtype=int)
        self.chunk_of[order] = np.arange(freq.size) // per
        self.onehot = np.zeros((freq.size, self.chunks))
        self.onehot[np.arange(freq.size), self.chunk_of] = 1.0
        self.num = np.zeros((nant * nant, self.chunks), dtype=np.complex128)
        self.den = np.zeros((nant * nant, self.chunks))

    def add(self, w: np.ndarray, data: np.ndarray, model: np.ndarray, a1, a2) -> None:
        """One chunk of rows: ``w`` and ``data`` (nrow, nchan, nhand), ``model`` (nrow, nchan)."""
        m = model[:, :, None]
        num = (w * data * np.conj(m)).sum(axis=2) @ self.onehot
        den = (w * np.abs(m) ** 2).sum(axis=2) @ self.onehot
        # Each row's phase across the whole window taken out: the atmosphere's
        # phase, which changes from one integration to the next, would otherwise
        # decorrelate the sums over time.
        total = num.sum(axis=1)
        safe = np.where(total != 0, total, 1.0)
        num *= np.where(total != 0, np.conj(safe) / np.abs(safe), 1.0)[:, None]
        pair = np.asarray(a1, dtype=np.int64) * self.nant + np.asarray(a2, dtype=np.int64)
        np.add.at(self.num, pair, num)
        np.add.at(self.den, pair, den)


def _ripple(
    ratio: np.ndarray, weight: np.ndarray, a1: np.ndarray, a2: np.ndarray, nant: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each antenna's fractional amplitude and phase per chunk (nant, nchunk),
    by weighted linear least squares per chunk, less each antenna's mean and
    slope across the chunks."""
    rows = np.arange(a1.size)
    design = np.zeros((a1.size, nant))
    dphase = np.zeros((a1.size + 1, nant))
    design[rows, a1] = design[rows, a2] = 1.0
    dphase[rows, a1], dphase[rows, a2] = 1.0, -1.0
    dphase[-1] = 1.0  # the phases' mean held at zero
    nchunk = ratio.shape[1]
    amp, phase = np.zeros((nant, nchunk)), np.zeros((nant, nchunk))
    for k in range(nchunk):
        root = np.sqrt(weight[:, k])
        re = (ratio[:, k].real - 1.0) * root
        amp[:, k] = np.linalg.lstsq(design * root[:, None], re, rcond=None)[0]
        # The constraint row's weight: the chunk's mean, so it binds as the data do.
        rp = np.append(root, root.mean())
        lp = np.append(ratio[:, k].imag, 0.0)
        phase[:, k] = np.linalg.lstsq(dphase * rp[:, None], lp * rp, rcond=None)[0]
    x = np.linspace(-1.0, 1.0, nchunk)
    basis = np.vstack([np.ones_like(x), x]).T
    detrend = np.eye(x.size) - basis @ np.linalg.pinv(basis)
    return amp @ detrend.T, phase @ detrend.T


def _power(series: np.ndarray) -> np.ndarray:
    """The antennas' summed power spectrum across the chunks."""
    return (np.abs(np.fft.rfft(series, axis=1)) ** 2).sum(axis=0)


def fit_bandpass(
    sums: BandpassSums, kappa: float, rng: np.random.Generator, realizations: int = 3
) -> tuple[BandpassFit | None, str | None]:
    """The window's residual bandpass, or None and why not."""
    if sums.chunks < MIN_CHUNKS:
        return None, f"bandpass: {sums.chunks} chunks of channels, too few to see a ripple"
    has = sums.den > 0
    keep = np.flatnonzero(has.mean(axis=1) >= MIN_DATA_SHARE)
    a1, a2 = np.divmod(keep, sums.nant)
    used = np.union1d(a1, a2)
    if used.size < MIN_ANTENNAS or keep.size < 2 * used.size:
        return None, "bandpass: too few baselines with data in each chunk of channels"
    index = np.full(sums.nant, -1)
    index[used] = np.arange(used.size)
    a1, a2 = index[a1], index[a2]
    den = sums.den[keep]
    # Each baseline over its mean across the chunks: the gains drop out, and
    # the noise grows as that mean falls below 1 (what the sums over time lose).
    mean = sums.num[keep].sum(axis=1, keepdims=True) / den.sum(axis=1, keepdims=True)
    good = np.abs(mean[:, 0]) > 0
    if good.sum() < 2 * used.size:
        return None, "bandpass: too few baselines with signal"
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(den > 0, sums.num[keep] / np.where(den > 0, den, 1.0), 0.0) / np.where(
            good[:, None], mean, 1.0
        )
    ratio = np.where(den > 0, ratio, 1.0)
    # A component's noise variance is kappa / den / |mean|^2: weight its inverse.
    weight = np.where(good[:, None], den * np.abs(mean) ** 2, 0.0) / kappa
    amp, phase = _ripple(ratio, weight, a1, a2, used.size)
    power = _power(amp) + _power(phase)
    noise_amp, noise_phase, noise_power = [], [], np.zeros_like(power)
    with np.errstate(divide="ignore"):
        sigma = np.where(weight > 0, 1.0 / np.sqrt(np.where(weight > 0, weight, 1.0)), 0.0)
    for _ in range(max(1, realizations)):
        noisy = 1.0 + sigma * (rng.standard_normal(den.shape) + 1j * rng.standard_normal(den.shape))
        na, nph = _ripple(noisy, weight, a1, a2, used.size)
        noise_amp.append(float(np.std(na)))
        noise_phase.append(float(np.std(nph)))
        noise_power += (_power(na) + _power(nph)) / max(1, realizations)
    thermal_amp = float(np.sqrt(np.mean(np.square(noise_amp))))
    thermal_phase = float(np.sqrt(np.mean(np.square(noise_phase))))
    if max(thermal_amp, thermal_phase) > MAX_THERMAL_RMS:
        return None, (
            f"bandpass: thermal rms {max(thermal_amp, thermal_phase):.2f} per antenna and chunk, "
            "too faint to see a ripple"
        )
    excess = power - noise_power
    excess[:LOWEST_HARMONIC] = -np.inf
    peak = int(np.argmax(excess))
    period = float(sums.chunks * sums.chunk_hz / peak / 1e6) if peak and excess[peak] > 0 else None
    return (
        BandpassFit(
            chunks=int(sums.chunks),
            chunk_mhz=float(sums.chunk_hz / 1e6),
            antennas=int(used.size),
            baselines=int(keep.size),
            amp_rms=float(np.sqrt(max(np.std(amp) ** 2 - thermal_amp**2, 0.0))),
            phase_rms_deg=float(
                np.degrees(np.sqrt(max(np.std(phase) ** 2 - thermal_phase**2, 0.0)))
            ),
            thermal_amp_rms=thermal_amp,
            thermal_phase_rms_deg=float(np.degrees(thermal_phase)),
            period_mhz=period,
        ),
        None,
    )
