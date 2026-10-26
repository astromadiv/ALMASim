"""ALMA's channels from the archive's spectral resolution (MADIV-28).

The archive's frequency support gives each spectral window's resolution, not
its channel width. ALMA's correlator Hanning-smooths its native channels (the
resolution is then twice the width, and the noise of neighbouring channels is
correlated 2/3 and 1/6) and may average N of them into each channel. Every
ALMA channel width is 15.625 MHz / 2^k, and the resolution of N averaged
channels sits on that grid (N = 1, twice the width), 1.156 times above it
(N = 2) or just below it (N >= 4, about the width). Measured on the 100
MADIV-16 windows (the RESOLUTION, CHAN_WIDTH and EFFECTIVE_BW of their MSs
against their channels' noise correlation) and on the 10,052 windows of
cycle11.csv, every one of which falls in one of the three classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GRID_HZ = 15.625e6  # ALMA's channel widths: this over a power of two
NATIVE_CHANNELS = 3840  # a window's native (FDM) channels
ON_GRID_HZ = 10.0  # the archive gives resolutions to 0.01 kHz
HANNING_RHO = (2 / 3, 1 / 6)  # native channels' noise correlation, lags 1 and 2
HANNING_VARIANCE = 0.375  # (1/4)^2 + (1/2)^2 + (1/4)^2: a native channel's noise


@dataclass(frozen=True)
class Channels:
    """A window's channels: their width and the native channels each averages."""

    width_hz: float
    averaged: int = 1
    # The bandwidth of the radiometer equation (the MS's EFFECTIVE_BW): wider
    # than the channel, since smoothing and averaging keep less noise.
    noise_bandwidth_hz: float | None = None


def _on_grid(hz: float) -> float:
    return GRID_HZ * 2.0 ** round(math.log2(hz / GRID_HZ))


def channels_from_resolution(resolution_hz: float, bandwidth_hz: float) -> Channels:
    """The channels of an ALMA window of ``bandwidth_hz`` whose resolution the
    archive gives as ``resolution_hz``. A resolution off ALMA's grid is taken as
    the channel width, with nothing averaged and no noise bandwidth."""
    grid = _on_grid(resolution_hz)
    ratio = resolution_hz / grid
    if abs(resolution_hz - grid) <= ON_GRID_HZ:
        averaged, width = 1, grid / 2.0
    elif 1.1 < ratio < 1.25:
        averaged, width = 2, grid
    elif 0.98 < ratio < 1.0:
        native = _on_grid(bandwidth_hz / NATIVE_CHANNELS)
        averaged, width = max(4, round(grid / native)), grid
    else:
        return Channels(resolution_hz)
    return Channels(width, averaged, width * averaged / (HANNING_VARIANCE * _within(averaged)))


def _lags(n: int, offset: int) -> np.ndarray:
    """|lag| between native channel j of an n-block ``offset`` blocks on and channel i."""
    i = np.arange(n)
    return np.abs(offset * n + i[None, :] - i[:, None])


def _within(n: int, rho: tuple[float, float] = HANNING_RHO) -> float:
    return _block_sum(n, 0, rho)


def _block_sum(n: int, offset: int, rho: tuple[float, float]) -> float:
    r = np.array([1.0, *rho, 0.0])
    return float(r[np.minimum(_lags(n, offset), 3)].sum())


def averaged_correlation(
    n: int, rho1: float = HANNING_RHO[0], rho2: float = HANNING_RHO[1]
) -> tuple[float, float]:
    """The noise correlation of channels one and two apart when each averages
    ``n`` native channels correlated ``rho1`` and ``rho2``: Hanning's 2/3 falls
    to 0.30, 0.115, 0.052 and 0.026 for n = 2, 4, 8, 16."""
    rho = (float(rho1), float(rho2))
    within = _block_sum(n, 0, rho)
    return _block_sum(n, 1, rho) / within, _block_sum(n, 2, rho) / within
