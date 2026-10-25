"""Channel-by-channel interferometric imaging of a model cube (MADIV-28 T5.11).

:class:`ChannelImagingConfig` holds what every channel shares, so a channel
task is the config, the channel's image, its wavelength and its noise; the
22 positional arguments of :func:`imaging.image_channel` are filled from it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.io import fits

from .corruptions import (
    AntennaCorruptions,
    CorruptionModel,
    baseline_antennas,
    correlated_noise,
    spectral_kernel,
)
from .imaging import image_channel
from .utils import sampling_to_uv_mask

# The per-channel outputs of image_channel, in order (the ninth is the raw
# visibility rows, kept as a list).
CHANNEL_OUTPUTS = ("model", "dirty", "model_vis", "dirty_vis", "u", "v", "beam", "totsampling")


def picklable_header(header: Any) -> dict:
    """A FITS header as a plain dict of basic values (Headers hold thread locks)."""
    try:
        if isinstance(header, (fits.Header, dict)):
            return {
                str(k): (v if isinstance(v, (int, float, bool, type(None))) else str(v))
                for k, v in header.items()
            }
    except Exception:  # noqa: BLE001 - an unlistable header is left out
        pass
    return {}


@dataclass(frozen=True)
class ChannelImagingConfig:
    """Everything :func:`image_channel` needs that is the same for every channel."""

    Npix: int
    Nant: int
    Hcov: list[float]
    nH: int
    antPos: list
    robfac: float
    trlat: list[float]
    trdec: list[float]
    Diameters: list[float]
    imsize: float
    Xmax: float
    lfac: float
    distmat: np.ndarray
    Nphf: int
    Np4: int
    zooming: int
    header: dict
    robust: float
    scan_time_s: float
    # Seeds each channel's noise (MADIV-28 T5.1); None draws fresh entropy.
    noise_entropy: tuple[int, ...] | None = None
    # What calibrated data carry besides white noise (corruptions.py): the
    # model, and its antenna errors drawn once for all channels.
    corruptions: CorruptionModel | None = None
    antenna_corruptions: AntennaCorruptions | None = None
    # Keys the white draws correlated noise is built from (set whenever the
    # model correlates channels, seeded or not: every channel task needs them).
    corruption_entropy: tuple[int, ...] | None = None
    # Each antenna's noise variance per channel (corruptions.antenna_ripple),
    # (nant, nchan); None: every antenna alike.
    antenna_noise: np.ndarray | None = None
    # Each channel's continuum share of the source: the per-baseline errors act on
    # that part of its visibilities only. None: on all of them.
    continuum_fraction: np.ndarray | None = None

    def channel_rng(self, channel: int) -> np.random.Generator:
        """The generator of ``channel``: the same whichever worker images it."""
        if self.noise_entropy is None:
            return np.random.default_rng()
        return np.random.default_rng((*self.noise_entropy, int(channel)))


def image_configured_channel(
    config: ChannelImagingConfig, image: Any, wavelength: Any, noise: float, channel: int = 0
) -> tuple:
    """:func:`image_channel` of one channel with the shared ``config``."""
    corrupted: dict[str, np.ndarray] = {}  # passed only when there are corruptions
    if config.antenna_corruptions is not None:
        wavelength_m = float(wavelength[2]) * 1e3  # the simulator's wavelengths are in km
        fraction = config.continuum_fraction
        corrupted["gains"] = config.antenna_corruptions.baseline_gains(
            baseline_antennas(config.Nant),
            wavelength_m,
            baseline_weight=1.0 if fraction is None else float(fraction[channel]),
        )
    model = config.corruptions
    if model is not None and model.correlates_channels:
        shape = (config.Nant * (config.Nant - 1) // 2, config.nH)
        kernel = spectral_kernel(model.spectral_rho1, model.spectral_rho2)
        if config.corruption_entropy is None:
            raise ValueError(
                "correlated channel noise needs the configuration's corruption_entropy"
            )
        corrupted["noise_samples"] = correlated_noise(
            config.corruption_entropy, channel, shape, float(noise), kernel
        )
    if config.antenna_noise is not None:
        pairs = baseline_antennas(config.Nant)
        g = config.antenna_noise[:, channel]
        corrupted["noise_scale"] = np.sqrt(g[pairs[:, 0]] * g[pairs[:, 1]])
    return image_channel(
        image,
        wavelength,
        config.Npix,
        config.Nant,
        config.Hcov,
        config.nH,
        noise,
        config.antPos,
        config.robfac,
        config.trlat,
        config.trdec,
        config.Diameters,
        config.imsize,
        config.Xmax,
        config.lfac,
        config.distmat,
        config.Nphf,
        config.Np4,
        config.zooming,
        config.header,
        config.robust,
        config.scan_time_s,
        rng=config.channel_rng(channel),
        **corrupted,
    )


def channel_noise(noise: Any, channel: int) -> float:
    """The noise of ``channel``: one value per channel, or one for all."""
    if isinstance(noise, (list, tuple, np.ndarray)):
        return float(noise[channel])
    return float(noise)


def image_channels(
    backend: Any,
    config: ChannelImagingConfig,
    skymodel: np.ndarray,
    wavelengths: Sequence[Any],
    noise: Any,
    track_progress: Callable[[list], None],
) -> list:
    """Image every channel of ``skymodel`` on ``backend``; the per-channel results in order."""
    scattered = [backend.scatter(skymodel[i]) for i in range(skymodel.shape[0])]
    delayed = backend.delayed(image_configured_channel)
    tasks = [
        delayed(config, scattered[i], wavelengths[i], channel_noise(noise, i), i)
        for i in range(skymodel.shape[0])
    ]
    futures = [backend.compute(task, sync=False) for task in tasks]
    track_progress(futures)
    return backend.gather(futures)


def stack_channels(results: list, *, start: int, size: int) -> tuple[dict, list]:
    """The channel results as cubes cropped to the science field, and the raw visibility rows.

    The model sits in the central ``size`` box of the zero-padded FFT grid in
    both domains (the source is centred in the image, the sampled
    visibilities are DC-centred), so the ring around it is the guard band.
    """
    stop = start + size
    cubes = {
        name: np.stack([res[index] for res in results], axis=0)
        for index, name in enumerate(CHANNEL_OUTPUTS)
    }
    for name in ("model", "dirty", "beam", "model_vis", "dirty_vis", "totsampling"):
        cubes[name] = cubes[name][:, start:stop, start:stop]
    cubes["uv_mask"] = sampling_to_uv_mask(cubes["totsampling"])
    return cubes, [res[8] for res in results]
